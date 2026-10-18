// SPDX-License-Identifier: Apache-2.0
// The host fetch of one rank's DeepSeek V4.1 Engram rows (engram_fetch.h).
#include "engram_fetch.h"

#include "row_read.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <torch/all.h>
#include <torch/custom_class.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace {

using namespace exl3_rows;

// Indices of EngramHostFetch::counters(), in this order.
enum Counter : int64_t {
  kLookups,
  kRowsRequested,
  kRowsUnowned,
  kRowsUnique,
  kCacheHits,
  kCacheMisses,
  kCacheInserts,
  kCacheEvictions,
  kRowsFailed,
  kPageCacheRows,
  kStorageRows,
  kReadNs,
  kFetchNs,
  kCounters
};

// A row that could not be read: an E4M3 NaN as its first value, then zeros.
constexpr uint8_t kPoison = 0x7F;
// The E8M0 byte of 1.0, the scale of rows another rank owns.
constexpr uint8_t kUnownedScale = 127;
// Tag and LRU stamp bytes per cache entry, counted in the cache bound.
constexpr int64_t kEntryOverhead = 16;
// Fibonacci hashing of the row index onto the sets.
constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ull;

enum : int { kFree, kQueued, kDone };
enum : int64_t { kFetched = 0, kStopped = 1 };
constexpr int64_t kResultColumns = 7;

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct FetchState;

// A fixed staging of one graph size: the host function captured into a CUDA
// graph runs on these buffers on every replay, so nothing is chosen when it
// is queued.
struct FixedSlot {
  FetchState* state = nullptr;
  torch::Tensor ids, values, scales_out;
  int64_t* id = nullptr;
  uint8_t* out_values = nullptr;
  uint8_t* out_scales = nullptr;
  int64_t count = 0;
};

struct FetchJob {
  FetchState* state = nullptr;
  int64_t sequence = 0;
  int64_t count = 0;
  int64_t status = kFetched;
  int64_t failed = 0;
  int64_t first_row = 0;
  int64_t first_error = 0;
  int64_t first_offset = 0;
  std::atomic<int> phase{kFree};
};

// Bounded set-associative LRU cache of rows, keyed by local row index.
// Stamps come from one clock that advances once per probe that hits and
// once per insert that stores, so a set evicts its least recently used way
// (empty ways, stamp 0, first; ties to the lower way).
struct RowCache {
  RowCache(int64_t capacity_bytes, int64_t row_bytes, int64_t ways)
      : row_bytes(row_bytes),
        ways(ways),
        num_sets(std::max<int64_t>(capacity_bytes, 0) /
                 (ways * (row_bytes + kEntryOverhead))),
        tags(num_sets * ways, -1),
        stamps(num_sets * ways, 0) {
    data_bytes = num_sets * ways * row_bytes;
    if (data_bytes > 0) {
      void* map = mmap(nullptr, data_bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      TORCH_CHECK(map != MAP_FAILED, "Engram row cache mapping failed: ",
                  std::strerror(errno));
      data = static_cast<uint8_t*>(map);
    }
  }
  ~RowCache() {
    if (data != nullptr) munmap(data, data_bytes);
  }

  int64_t set_of(int64_t row) const {
    const uint64_t mixed = (static_cast<uint64_t>(row) * kGolden) >> 32;
    return static_cast<int64_t>(mixed % static_cast<uint64_t>(num_sets));
  }

  int64_t nbytes() const {
    return data_bytes + num_sets * ways * kEntryOverhead;
  }

  int64_t row_bytes;
  int64_t ways;
  int64_t num_sets;
  std::vector<int64_t> tags;
  std::vector<int64_t> stamps;
  uint8_t* data = nullptr;
  int64_t data_bytes = 0;
  int64_t clock = 0;
};

// Everything a fetch touches; a host function holds it by pointer.
struct FetchState {
  FetchState(int fd, int64_t base, int64_t row_bytes, int64_t scale_bytes,
             int64_t owner_lo, int64_t owner_hi, torch::Tensor scales,
             torch::Tensor table, torch::Tensor ids, torch::Tensor values,
             torch::Tensor scales_out, int64_t cache_bytes, int64_t ways,
             int64_t retries, int64_t max_threads, int64_t slots)
      : fd(fd),
        scales(std::move(scales)),
        table(std::move(table)),
        ids(std::move(ids)),
        values(std::move(values)),
        scales_out(std::move(scales_out)),
        row_bytes(row_bytes),
        scale_bytes(scale_bytes),
        owner_lo(owner_lo),
        owner_hi(owner_hi),
        capacity(this->ids.numel()),
        max_threads(max_threads),
        slots(slots),
        cache(this->table.numel() ? 0 : cache_bytes, row_bytes, ways),
        scratch(capacity, max_threads),
        jobs(new FetchJob[slots]),
        positions(capacity),
        keys(capacity),
        order(capacity),
        slot_of(capacity),
        unique(capacity),
        hit_way(capacity),
        miss_slot(capacity),
        miss_ids(capacity),
        good(capacity),
        groups(capacity),
        rows(capacity * row_bytes),
        missed(capacity * row_bytes) {
    region = Table{fd, base, row_bytes, owner_lo, owner_hi,
                   &head_offset, &head_end, 1, retries};
    head_offset = owner_lo;
    head_end = owner_hi;
    for (int64_t i = 0; i < slots; ++i) jobs[i].state = this;
    for (auto& counter : counters) counter.store(0);
  }
  ~FetchState() { ::close(fd); }

  // The fetch of ids[0:count); returns (failed rows, first failure).
  int64_t fetch(int64_t count, const int64_t* id, uint8_t* out_values,
                uint8_t* out_scales, int64_t* first_row, Failure* first);

  int fd;
  torch::Tensor scales, table, ids, values, scales_out;
  int64_t row_bytes, scale_bytes, owner_lo, owner_hi, capacity, max_threads,
      slots;
  int64_t head_offset = 0, head_end = 0;
  Table region{};
  RowCache cache;
  Scratch scratch;
  std::unique_ptr<FetchJob[]> jobs;
  // Fixed stagings (graph sizes), added before any is queued; their failures
  // since the last take_fixed, the first one's row, errno and offset, and
  // fetches stopped by an unexpected error. Atomics: take_fixed must not wait
  // for a fetch in flight.
  std::vector<std::unique_ptr<FixedSlot>> fixed;
  std::atomic<int64_t> fixed_failed{0};
  std::atomic<int64_t> fixed_row{0};
  std::atomic<int64_t> fixed_error{0};
  std::atomic<int64_t> fixed_offset{0};
  std::atomic<int64_t> fixed_stopped{0};
  // A fixed fetch was queued: a captured graph may run it at any time.
  std::atomic<bool> fixed_queued{false};
  std::mutex mutex;
  std::atomic<int64_t> counters[kCounters];
  // Per-fetch working storage, allocated once for `capacity` ids.
  std::vector<int64_t> positions, keys, order, slot_of, unique, hit_way,
      miss_slot, miss_ids;
  std::vector<uint8_t> good;
  std::vector<std::pair<int64_t, int64_t>> groups;
  std::vector<uint8_t> rows, missed;
  // The caller's thread only.
  int64_t next = 0;
  int64_t oldest = 0;
};

int64_t FetchState::fetch(int64_t count, const int64_t* id,
                          uint8_t* out_values, uint8_t* out_scales,
                          int64_t* first_row, Failure* first) {
  std::lock_guard<std::mutex> lock(mutex);
  const int64_t start = now_ns();
  int64_t c[kCounters] = {};
  c[kLookups] = 1;
  c[kRowsRequested] = count;
  const uint8_t* resident_scales = static_cast<const uint8_t*>(scales.data_ptr());

  // Owned ids take their scales; the others zero values and scale 1.0.
  int64_t owned = 0;
  for (int64_t i = 0; i < count; ++i) {
    const int64_t row = id[i];
    if (row < owner_lo || row >= owner_hi) {
      std::memset(out_values + i * row_bytes, 0, row_bytes);
      std::memset(out_scales + i * scale_bytes, kUnownedScale, scale_bytes);
      continue;
    }
    const int64_t local = row - owner_lo;
    std::memcpy(out_scales + i * scale_bytes,
                resident_scales + local * scale_bytes, scale_bytes);
    positions[owned] = i;
    keys[owned] = local;
    ++owned;
  }
  c[kRowsUnowned] = count - owned;

  int64_t failed = 0;
  if (table.numel()) {
    const uint8_t* resident = static_cast<const uint8_t*>(table.data_ptr());
    for (int64_t k = 0; k < owned; ++k) {
      std::memcpy(out_values + positions[k] * row_bytes,
                  resident + keys[k] * row_bytes, row_bytes);
    }
  } else {
    // Unique local rows in ascending order, and each owned id's slot.
    for (int64_t k = 0; k < owned; ++k) order[k] = k;
    std::sort(order.begin(), order.begin() + owned, [&](int64_t a, int64_t b) {
      return keys[a] < keys[b];
    });
    int64_t num_unique = 0;
    for (int64_t k = 0; k < owned; ++k) {
      const int64_t key = keys[order[k]];
      if (num_unique == 0 || unique[num_unique - 1] != key) {
        unique[num_unique++] = key;
      }
      slot_of[order[k]] = num_unique - 1;
    }
    c[kRowsUnique] = num_unique;

    // Probe: hits are copied out and stamped with one new clock value.
    const int64_t ways = cache.ways;
    int64_t hits = 0;
    int64_t misses = 0;
    for (int64_t j = 0; j < num_unique; ++j) {
      hit_way[j] = -1;
      if (cache.num_sets) {
        const int64_t base = cache.set_of(unique[j]) * ways;
        for (int64_t w = 0; w < ways; ++w) {
          if (cache.tags[base + w] == unique[j]) {
            hit_way[j] = w;
            std::memcpy(rows.data() + j * row_bytes,
                        cache.data + (base + w) * row_bytes, row_bytes);
            ++hits;
            break;
          }
        }
      }
      if (hit_way[j] < 0) {
        miss_slot[misses] = j;
        miss_ids[misses] = unique[j] + owner_lo;
        ++misses;
      }
    }
    if (hits) {
      ++cache.clock;
      for (int64_t j = 0; j < num_unique; ++j) {
        if (hit_way[j] >= 0) {
          cache.stamps[cache.set_of(unique[j]) * ways + hit_way[j]] =
              cache.clock;
        }
      }
    }
    c[kCacheHits] = hits;
    c[kCacheMisses] = misses;

    if (misses) {
      // Reads: page-cache rows on this thread, the rest on up to
      // max_threads threads; failures come back sorted by index.
      const int64_t read_start = now_ns();
      failed = read_batch(region, missed.data(), miss_ids.data(), misses,
                          max_threads, scratch);
      c[kReadNs] = now_ns() - read_start;
      const int64_t pending = static_cast<int64_t>(scratch.pending.size());
      c[kStorageRows] = pending;
      c[kPageCacheRows] = misses - pending;
      std::fill(good.begin(), good.begin() + misses, 1);
      for (int64_t f = 0; f < failed; ++f) {
        const Failure& failure = scratch.failures[f];
        uint8_t* row = missed.data() + failure.index * row_bytes;
        std::memset(row, 0, row_bytes);
        row[0] = kPoison;
        good[failure.index] = 0;
      }
      if (failed) {
        *first = scratch.failures[0];
        *first_row = miss_ids[first->index];
      }
      c[kRowsFailed] = failed;
      for (int64_t m = 0; m < misses; ++m) {
        std::memcpy(rows.data() + miss_slot[m] * row_bytes,
                    missed.data() + m * row_bytes, row_bytes);
      }

      // Insert the rows read, by set and then row order; a set takes at
      // most `ways` of them, into its ways in LRU order of the stamps
      // before this insert.
      int64_t candidates = 0;
      if (cache.num_sets) {
        for (int64_t m = 0; m < misses; ++m) {
          if (good[m]) {
            groups[candidates++] = {cache.set_of(miss_ids[m] - owner_lo), m};
          }
        }
      }
      if (candidates) {
        std::sort(groups.begin(), groups.begin() + candidates);
        ++cache.clock;
        int64_t inserted = 0;
        int64_t evicted = 0;
        int64_t lru[64];
        TORCH_CHECK(ways <= 64, "Engram row cache ways exceed 64");
        for (int64_t g = 0; g < candidates;) {
          const int64_t set = groups[g].first;
          const int64_t base = set * ways;
          for (int64_t w = 0; w < ways; ++w) lru[w] = w;
          std::stable_sort(lru, lru + ways, [&](int64_t a, int64_t b) {
            return cache.stamps[base + a] < cache.stamps[base + b];
          });
          int64_t rank = 0;
          for (; g < candidates && groups[g].first == set; ++g, ++rank) {
            if (rank >= ways) continue;
            const int64_t m = groups[g].second;
            const int64_t entry = base + lru[rank];
            evicted += cache.tags[entry] >= 0;
            cache.tags[entry] = miss_ids[m] - owner_lo;
            cache.stamps[entry] = cache.clock;
            std::memcpy(cache.data + entry * row_bytes,
                        missed.data() + m * row_bytes, row_bytes);
            ++inserted;
          }
        }
        c[kCacheInserts] = inserted;
        c[kCacheEvictions] = evicted;
      }
    }
    for (int64_t k = 0; k < owned; ++k) {
      std::memcpy(out_values + positions[k] * row_bytes,
                  rows.data() + slot_of[k] * row_bytes, row_bytes);
    }
  }
  c[kFetchNs] = now_ns() - start;
  for (int64_t i = 0; i < kCounters; ++i) {
    counters[i].fetch_add(c[i], std::memory_order_relaxed);
  }
  return failed;
}

// The host function. It runs without the GIL on a thread of the CUDA driver
// (or on the caller's thread for run()), makes no CUDA call and lets no
// exception out.
void execute(FetchJob& job) noexcept {
  static_cast<void>(job.phase.load(std::memory_order_acquire));
  FetchState& s = *job.state;
  try {
    Failure first{0, 0, 0};
    int64_t first_row = 0;
    job.failed = s.fetch(job.count, s.ids.data_ptr<int64_t>(),
                         static_cast<uint8_t*>(s.values.data_ptr()),
                         static_cast<uint8_t*>(s.scales_out.data_ptr()),
                         &first_row, &first);
    if (job.failed) {
      job.first_row = first_row;
      job.first_error = first.error;
      job.first_offset = first.offset;
    }
  } catch (...) {
    for (std::thread& worker : s.scratch.workers) {
      if (worker.joinable()) {
        try {
          worker.join();
        } catch (...) {
        }
      }
    }
    uint8_t* values = static_cast<uint8_t*>(s.values.data_ptr());
    for (int64_t i = 0; i < job.count; ++i) {
      std::memset(values + i * s.row_bytes, 0, s.row_bytes);
      values[i * s.row_bytes] = kPoison;
    }
    job.status = kStopped;
    job.failed = job.count;
  }
  job.phase.store(kDone, std::memory_order_release);
}

void CUDART_CB run_queued(void* job) { execute(*static_cast<FetchJob*>(job)); }

// The host function of a fixed staging: the same fetch, with its failures
// added to the state's fixed-failure record. Runs on every replay of the
// graph that captured it; makes no CUDA call and lets no exception out.
void execute_fixed(FixedSlot& f) noexcept {
  FetchState& s = *f.state;
  try {
    Failure first{0, 0, 0};
    int64_t first_row = 0;
    const int64_t failed = s.fetch(f.count, f.id, f.out_values, f.out_scales,
                                   &first_row, &first);
    if (failed && s.fixed_failed.fetch_add(failed) == 0) {
      s.fixed_row.store(first_row);
      s.fixed_error.store(first.error);
      s.fixed_offset.store(first.offset);
    }
  } catch (...) {
    for (std::thread& worker : s.scratch.workers) {
      if (worker.joinable()) {
        try {
          worker.join();
        } catch (...) {
        }
      }
    }
    for (int64_t i = 0; i < f.count; ++i) {
      std::memset(f.out_values + i * s.row_bytes, 0, s.row_bytes);
      f.out_values[i * s.row_bytes] = kPoison;
    }
    s.fixed_failed.fetch_add(f.count);
    s.fixed_stopped.fetch_add(1);
  }
}

void CUDART_CB run_fixed_queued(void* slot) {
  execute_fixed(*static_cast<FixedSlot*>(slot));
}

void check_cuda(const torch::Tensor& t, int64_t bytes, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.nbytes() >= bytes,
              "Engram fixed fetch: ", name,
              " must be contiguous CUDA memory of at least ", bytes, " bytes");
}

void check_copy(cudaError_t error, const char* what) {
  if (error != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    TORCH_CHECK(false, "Engram fixed fetch: ", what, ": ",
                cudaGetErrorString(error));
  }
}

void check_cpu(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_cpu() && t.is_contiguous() &&
                  t.scalar_type() == torch::kUInt8,
              "Engram ", name, " must be contiguous CPU uint8");
}

class EngramHostFetch : public torch::CustomClassHolder {
 public:
  EngramHostFetch(int64_t fd, int64_t base, int64_t row_bytes,
                  int64_t owner_lo, int64_t owner_hi, torch::Tensor scales,
                  torch::Tensor table, torch::Tensor ids, torch::Tensor values,
                  torch::Tensor scales_out, int64_t cache_bytes, int64_t ways,
                  int64_t retries, int64_t max_threads, int64_t slots) {
    TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type() == torch::kInt64 &&
                    ids.dim() == 1 && ids.is_contiguous(),
                "Engram row ids must be a contiguous 1-D CPU int64 tensor");
    check_cpu(scales, "scales");
    check_cpu(values, "values");
    check_cpu(scales_out, "scale staging");
    const int64_t rows = owner_hi - owner_lo;
    TORCH_CHECK(row_bytes > 0 && base >= 0 && 0 <= owner_lo &&
                    owner_lo <= owner_hi && retries >= 0 && max_threads > 0 &&
                    slots > 0 && ways > 0 && ways <= 64,
                "Invalid Engram fetch arguments");
    TORCH_CHECK(owner_hi <= (INT64_MAX - base) / row_bytes,
                "Engram file offset exceeds signed 64-bit range");
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == rows,
                "Engram scales must hold the owned rows");
    const int64_t scale_bytes = scales.size(1);
    const int64_t capacity = ids.numel();
    TORCH_CHECK(values.numel() == capacity * row_bytes &&
                    scales_out.numel() == capacity * scale_bytes,
                "Engram staging does not match the id capacity");
    if (table.numel()) {
      check_cpu(table, "table");
      TORCH_CHECK(table.numel() == rows * row_bytes,
                  "Engram resident table must hold the owned rows");
    }
    const int own = fcntl(static_cast<int>(fd), F_DUPFD_CLOEXEC, 0);
    TORCH_CHECK(own >= 0, "Cannot duplicate the Engram table descriptor: ",
                std::strerror(errno));
    try {
      state_ = std::make_unique<FetchState>(
          own, base, row_bytes, scale_bytes, owner_lo, owner_hi,
          std::move(scales), std::move(table), std::move(ids),
          std::move(values), std::move(scales_out), cache_bytes, ways,
          retries, max_threads, slots);
    } catch (...) {
      ::close(own);
      throw;
    }
  }

  ~EngramHostFetch() override {
    // A captured graph may still replay a fixed fetch: keep its state.
    if (state_->fixed_queued.load()) {
      static_cast<void>(state_.release());
      return;
    }
    for (int64_t i = 0; i < state_->slots; ++i) {
      if (state_->jobs[i].phase.load(std::memory_order_acquire) == kQueued) {
        static_cast<void>(state_.release());
        return;
      }
    }
  }

  void enqueue(int64_t count) {
    FetchJob& job = claim(count);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaError_t error = cudaStreamIsCapturing(stream, &capture);
    TORCH_CHECK(error != cudaSuccess || capture == cudaStreamCaptureStatusNone,
                "Engram host fetches cannot be captured");
    if (error == cudaSuccess) {
      job.phase.store(kQueued, std::memory_order_release);
      error = cudaLaunchHostFunc(stream, &run_queued, &job);
      if (error != cudaSuccess) job.phase.store(kFree, std::memory_order_relaxed);
    }
    if (error != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      TORCH_CHECK(false, "Cannot queue the Engram host fetch: ",
                  cudaGetErrorString(error));
    }
    ++state_->next;
  }

  void run(int64_t count) {
    FetchJob& job = claim(count);
    job.phase.store(kQueued, std::memory_order_release);
    ++state_->next;
    execute(job);
  }

  // Finished fetches not yet taken, in queue order; stops at the first one
  // still queued.
  torch::Tensor take() {
    FetchState& s = *state_;
    int64_t n = 0;
    while (s.oldest + n < s.next &&
           s.jobs[(s.oldest + n) % s.slots].phase.load(
               std::memory_order_acquire) == kDone) {
      ++n;
    }
    torch::Tensor result =
        torch::empty({n, kResultColumns}, torch::dtype(torch::kInt64).device(torch::kCPU));
    int64_t* data = result.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
      FetchJob& job = s.jobs[(s.oldest + i) % s.slots];
      *data++ = job.sequence;
      *data++ = job.count;
      *data++ = job.status;
      *data++ = job.failed;
      *data++ = job.first_row;
      *data++ = job.first_error;
      *data++ = job.first_offset;
      job.phase.store(kFree, std::memory_order_relaxed);
    }
    s.oldest += n;
    return result;
  }

  torch::Tensor counters() {
    torch::Tensor result =
        torch::empty({kCounters}, torch::dtype(torch::kInt64).device(torch::kCPU));
    int64_t* data = result.data_ptr<int64_t>();
    for (int64_t i = 0; i < kCounters; ++i) {
      data[i] = state_->counters[i].load(std::memory_order_relaxed);
    }
    return result;
  }

  void reset_counters() {
    for (auto& counter : state_->counters) counter.store(0);
  }

  int64_t cache_rows() { return state_->cache.num_sets * state_->cache.ways; }

  // A fixed staging for fetches of exactly ids.numel() rows (one graph
  // size); returns its index. Add every one before queueing any.
  int64_t add_fixed(torch::Tensor ids, torch::Tensor values,
                    torch::Tensor scales_out) {
    FetchState& s = *state_;
    TORCH_CHECK(!s.fixed_queued.load(),
                "Engram fixed stagings must be added before any is queued");
    TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type() == torch::kInt64 &&
                    ids.dim() == 1 && ids.is_contiguous(),
                "Engram fixed ids must be a contiguous 1-D CPU int64 tensor");
    check_cpu(values, "fixed values");
    check_cpu(scales_out, "fixed scale staging");
    const int64_t count = ids.numel();
    TORCH_CHECK(0 < count && count <= s.capacity,
                "Engram fixed staging exceeds the fetch capacity");
    TORCH_CHECK(values.numel() == count * s.row_bytes &&
                    scales_out.numel() == count * s.scale_bytes,
                "Engram fixed staging does not match its id count");
    auto slot = std::make_unique<FixedSlot>();
    slot->state = &s;
    slot->ids = std::move(ids);
    slot->values = std::move(values);
    slot->scales_out = std::move(scales_out);
    slot->id = slot->ids.data_ptr<int64_t>();
    slot->out_values = static_cast<uint8_t*>(slot->values.data_ptr());
    slot->out_scales = static_cast<uint8_t*>(slot->scales_out.data_ptr());
    slot->count = count;
    s.fixed.push_back(std::move(slot));
    return static_cast<int64_t>(s.fixed.size()) - 1;
  }

  // Queue on the current CUDA stream, capture allowed: the copy of dev_ids
  // to the slot's host ids, the fetch as a host function on the slot's
  // staging, and the copies of its values and scales to dev_values and
  // dev_scales. Every argument is fixed, so a graph replays it as is.
  void enqueue_fixed(int64_t index, torch::Tensor dev_ids,
                     torch::Tensor dev_values, torch::Tensor dev_scales) {
    FetchState& s = *state_;
    TORCH_CHECK(0 <= index && index < static_cast<int64_t>(s.fixed.size()),
                "No such Engram fixed staging");
    FixedSlot& f = *s.fixed[index];
    check_cuda(dev_ids, f.count * 8, "device ids");
    check_cuda(dev_values, f.count * s.row_bytes, "device values");
    check_cuda(dev_scales, f.count * s.scale_bytes, "device scales");
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    s.fixed_queued.store(true);
    check_copy(cudaMemcpyAsync(f.id, dev_ids.data_ptr(), f.count * 8,
                               cudaMemcpyDeviceToHost, stream),
               "copy of the ids to the host");
    check_copy(cudaLaunchHostFunc(stream, &run_fixed_queued, &f),
               "queue the host fetch");
    check_copy(cudaMemcpyAsync(dev_values.data_ptr(), f.out_values,
                               f.count * s.row_bytes, cudaMemcpyHostToDevice,
                               stream),
               "copy of the values to the device");
    check_copy(cudaMemcpyAsync(dev_scales.data_ptr(), f.out_scales,
                               f.count * s.scale_bytes, cudaMemcpyHostToDevice,
                               stream),
               "copy of the scales to the device");
  }

  // The fixed fetch of slot `index` on the calling thread (no copies).
  void run_fixed(int64_t index) {
    FetchState& s = *state_;
    TORCH_CHECK(0 <= index && index < static_cast<int64_t>(s.fixed.size()),
                "No such Engram fixed staging");
    execute_fixed(*s.fixed[index]);
  }

  // [failed rows, first failed row id, its errno (0: end of file), byte
  // offset reached, fetches stopped] of the fixed fetches since the last
  // call; never waits for a fetch in flight.
  torch::Tensor take_fixed() {
    FetchState& s = *state_;
    torch::Tensor result =
        torch::empty({5}, torch::dtype(torch::kInt64).device(torch::kCPU));
    int64_t* data = result.data_ptr<int64_t>();
    data[1] = s.fixed_row.load();
    data[2] = s.fixed_error.load();
    data[3] = s.fixed_offset.load();
    data[0] = s.fixed_failed.exchange(0);
    data[4] = s.fixed_stopped.exchange(0);
    return result;
  }

  int64_t cache_bytes() { return state_->cache.nbytes(); }

  int64_t prefault() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const int64_t start = now_ns();
    RowCache& cache = state_->cache;
    const int64_t page = sysconf(_SC_PAGESIZE);
    volatile uint8_t* data = cache.data;
    for (int64_t offset = 0; offset < cache.data_bytes; offset += page) {
      data[offset] = 0;
    }
    return now_ns() - start;
  }

 private:
  FetchJob& claim(int64_t count) {
    FetchState& s = *state_;
    TORCH_CHECK(0 <= count && count <= s.capacity,
                "Engram fetch exceeds the staging capacity");
    FetchJob& job = s.jobs[s.next % s.slots];
    TORCH_CHECK(job.phase.load(std::memory_order_acquire) == kFree,
                "Every Engram fetch slot is in flight or not taken");
    job.sequence = s.next;
    job.count = count;
    job.status = kFetched;
    job.failed = 0;
    job.first_row = 0;
    job.first_error = 0;
    job.first_offset = 0;
    return job;
  }

  std::unique_ptr<FetchState> state_;
};

}  // namespace

void register_engram_host_fetch(torch::Library& m) {
  m.class_<EngramHostFetch>("EngramHostFetch")
      .def(torch::init<int64_t, int64_t, int64_t, int64_t, int64_t,
                       torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t,
                       int64_t, int64_t>())
      .def("enqueue", &EngramHostFetch::enqueue)
      .def("run", &EngramHostFetch::run)
      .def("take", &EngramHostFetch::take)
      .def("counters", &EngramHostFetch::counters)
      .def("reset_counters", &EngramHostFetch::reset_counters)
      .def("cache_rows", &EngramHostFetch::cache_rows)
      .def("cache_bytes", &EngramHostFetch::cache_bytes)
      .def("prefault", &EngramHostFetch::prefault)
      .def("add_fixed", &EngramHostFetch::add_fixed)
      .def("enqueue_fixed", &EngramHostFetch::enqueue_fixed)
      .def("run_fixed", &EngramHostFetch::run_fixed)
      .def("take_fixed", &EngramHostFetch::take_fixed);
}
