// SPDX-License-Identifier: Apache-2.0
// Positional reads of selected fixed-size rows of a file region: a
// non-blocking page-cache probe, then threaded preads with bounded retries.
// The core of the PLE row reader (ple_read.cpp) and the Engram row fetch
// (engram_fetch.cpp).
#pragma once

#include <fcntl.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <memory>
#include <thread>
#include <vector>

namespace exl3_rows {

struct Failure {
  int64_t index;
  int64_t error;  // errno, or 0 for end of file
  int64_t offset;
};

struct Table {
  int fd;
  int64_t base;
  int64_t row_bytes;
  int64_t owner_lo;
  int64_t owner_hi;
  const int64_t* head_offsets;
  const int64_t* head_ends;  // offsets[h] + vocab_sizes[h]
  int64_t heads;
  int64_t retries;
};

// Exl3PLERowReader.owns: inside the owner range and inside its head's
// vocabulary (padding rows after a head belong to nobody).
inline bool owns(const Table& t, int64_t row) {
  if (row < t.owner_lo || row >= t.owner_hi) return false;
  const int64_t* end = t.head_offsets + t.heads;
  const int64_t* after = std::upper_bound(t.head_offsets, end, row);
  if (after == t.head_offsets) return false;
  return row < t.head_ends[after - t.head_offsets - 1];
}

// pread_exact_into: partial reads continue; EINTR is retried as os.preadv
// retries it; any other error or end of file is returned with the offset
// reached. Returns -1 on success.
inline int64_t read_exact(int fd, uint8_t* dest, int64_t length, int64_t offset,
                   int64_t* reached) {
  int64_t done = 0;
  while (done < length) {
    ssize_t count = pread(fd, dest + done, length - done, offset + done);
    if (count < 0) {
      if (errno == EINTR) continue;
      *reached = offset + done;
      return errno;
    }
    if (count == 0) {
      *reached = offset + done;
      return 0;
    }
    done += count;
  }
  return -1;
}

// Exl3PLERowReader.read_exact: the same bounded retries and backoff (1 ms,
// doubling).
inline bool read_row(const Table& t, uint8_t* dest, int64_t row, Failure* failure) {
  const int64_t offset = t.base + row * t.row_bytes;
  int64_t reached = offset;
  int64_t error = -1;
  for (int64_t attempt = 0; attempt <= t.retries; ++attempt) {
    error = read_exact(t.fd, dest, t.row_bytes, offset, &reached);
    if (error < 0) return true;
    if (attempt == t.retries) break;
    const int64_t ms = int64_t{1} << std::min<int64_t>(attempt, 40);
    timespec wait{static_cast<time_t>(ms / 1000),
                  static_cast<long>(ms % 1000) * 1000000L};
    while (nanosleep(&wait, &wait) != 0 && errno == EINTR) {
    }
  }
  failure->error = error;
  failure->offset = reached;
  return false;
}

// Reads the rows at indices[begin, end); stores their failures in order from
// failures[0] and returns how many there are.
inline int64_t read_rows(const Table& t, uint8_t* dest, const int64_t* ids,
                  const int64_t* indices, int64_t begin, int64_t end,
                  Failure* failures) {
  int64_t failed = 0;
  for (int64_t i = begin; i < end; ++i) {
    const int64_t index = indices[i];
    Failure failure{index, 0, 0};
    if (!read_row(t, dest + index * t.row_bytes, ids[index], &failure)) {
      failures[failed++] = failure;
    }
  }
  return failed;
}

// What a batch read needs besides its rows, allocated for up to `rows` rows
// and max_threads threads before the read starts.
struct Scratch {
  Scratch(int64_t rows, int64_t max_threads)
      : failures(new Failure[std::max<int64_t>(rows, 1)]),
        failed(new int64_t[max_threads]) {
    pending.reserve(rows);
    workers.reserve(max_threads);
    inline_chunks.reserve(max_threads);
  }
  std::vector<int64_t> pending;
  std::unique_ptr<Failure[]> failures;
  std::unique_ptr<int64_t[]> failed;  // per chunk
  std::vector<std::thread> workers;
  std::vector<int64_t> inline_chunks;
};

// Reads rows ids[0:count) into out as ple_read_rows (ple.h) describes and
// returns how many failed; their failures are scratch.failures[0, n), sorted
// by index. Only starting a thread allocates, and a thread that cannot start
// leaves its chunk to the calling thread, so no started thread is left
// joinable by an exception.
inline int64_t read_batch(const Table& t, uint8_t* out, const int64_t* id,
                   int64_t count, int64_t max_threads, Scratch& s) {
  // Rows already in the page cache are read here without waiting for
  // storage; the rest are read below, on up to max_threads threads, so their
  // storage latencies overlap. For a row that is not cached the probe also
  // starts the read of its page, so such a row gets one more read attempt
  // than Exl3PLERowReader.read_rows makes.
  s.pending.clear();
  bool nowait = true;
  for (int64_t index = 0; index < count; ++index) {
    uint8_t* row = out + index * t.row_bytes;
    if (!owns(t, id[index])) {
      std::memset(row, 0, t.row_bytes);
      continue;
    }
    if (nowait) {
      iovec vector{row, static_cast<size_t>(t.row_bytes)};
      ssize_t done = preadv2(t.fd, &vector, 1, t.base + id[index] * t.row_bytes,
                             RWF_NOWAIT);
      if (done == t.row_bytes) continue;
      if (done < 0 &&
          (errno == EOPNOTSUPP || errno == EINVAL || errno == ENOSYS)) {
        nowait = false;
      }
    }
    s.pending.push_back(index);
  }

  const int64_t rows = static_cast<int64_t>(s.pending.size());
  if (rows == 0) return 0;
  const int64_t threads = std::min(max_threads, rows);
  const int64_t step = (rows + threads - 1) / threads;
  const int64_t* pending = s.pending.data();
  Failure* failures = s.failures.get();
  int64_t* failed = s.failed.get();
  s.workers.clear();
  s.inline_chunks.clear();
  int64_t chunks = 1;
  for (; chunks < threads && chunks * step < rows; ++chunks) {
    const int64_t chunk = chunks;
    const int64_t begin = chunk * step;
    const int64_t end = std::min(rows, begin + step);
    try {
      s.workers.emplace_back(
          [&t, out, id, pending, begin, end, failures, failed, chunk] {
            failed[chunk] =
                read_rows(t, out, id, pending, begin, end, failures + begin);
          });
    } catch (...) {
      s.inline_chunks.push_back(chunk);
    }
  }
  failed[0] = read_rows(t, out, id, pending, 0, std::min(rows, step), failures);
  for (int64_t chunk : s.inline_chunks) {
    const int64_t begin = chunk * step;
    failed[chunk] = read_rows(t, out, id, pending, begin,
                              std::min(rows, begin + step), failures + begin);
  }
  for (std::thread& worker : s.workers) worker.join();

  // Chunk c stored its failures from failures[c * step]; chunks hold
  // increasing indices, so packing them in chunk order keeps index order.
  int64_t total = failed[0];
  for (int64_t chunk = 1; chunk < chunks; ++chunk) {
    std::memmove(failures + total, failures + chunk * step,
                 failed[chunk] * sizeof(Failure));
    total += failed[chunk];
  }
  return total;
}

}  // namespace exl3_rows
