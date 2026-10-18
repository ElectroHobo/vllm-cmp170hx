# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram embedding storage on sm_80: SSD rows, a bounded host row cache and
resident scales.

A DeepSeek V4.1 Engram table holds ~384M rows of 256 E4M3 bytes (~92 GiB)
plus one E8M0 byte per 32 values (~2.9 GiB). The EXL3 checkpoint carries no
``engram.embed.*``; the tables are read from the base checkpoint's
safetensors shards in the directory named by the model config field
``engram_table_dir`` (through its ``model.safetensors.index.json``).

Each TP rank owns whole hash heads, as upstream ``ParallelEngramEmbedding``
shards them, i.e. one contiguous row range ``[vocab_start, vocab_end)``:

- its scales are read once into host memory (pinned when CUDA is present);
- its values stay in the file (placement ``ssd``) and are read on demand at
  absolute int64 byte offsets through a bounded set-associative LRU host row
  cache, or are copied into pinned host memory (placement ``ram``, the
  comparison path);
- a lookup stages the requested rows' values and scales in pinned host
  memory, copies them to the device and dequantizes them there exactly as
  the reference (``ops/sm80_engram_dequant.py``).

The module registers no parameter or buffer, so no loader looks for the
table in the checkpoint.

Lookups run on the host and cannot be captured: the upstream Engram calls
them from ``_start_prefetch``, an eager break of the breakable CUDA graph,
so each replay runs them eagerly with that step's ids. A lookup copies the
ids to the host and waits for them, fills the host staging, then queues the
copies and the dequant on the caller's stream. Staging reuse is ordered by
two events: the host waits for the previous host-to-device copy before it
overwrites the host staging, and the stream waits for the previous dequant
before it overwrites the device staging.

A row that cannot be read (after the retries) fails only the requests that
use it, as the EXL3 PLE rows do (``vllm/sm80/failed_inputs.py``): the
owning rank stages it poisoned, an E4M3 NaN as its first value and zeros
after, which survives the TP gather of the rows. After the gather every rank
marks the tokens holding a NaN first value as failed, zeros those tokens'
rows, and reports them through ``take_failed_tokens``; the runner finishes
their requests with an error. Only startup and the table locator raise.
"""

import atexit
import ctypes
import errno
import functools
import json
import os
import struct
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

ENGRAM_QUANT_BLOCK = 32
# Read attempts after the first for a failed or truncated row read (1 ms
# backoff, doubling), as the PLE reader's default.
READ_RETRIES = 3
# Threads that read the rows not in the page cache.
READ_THREADS = 8
# Associativity of the host row cache.
CACHE_WAYS = 8
# Tag and LRU stamp bytes per cache entry.
CACHE_ENTRY_OVERHEAD = 16
# Chunk of the sequential startup reads (scales, ram placement).
_LOAD_CHUNK_BYTES = 256 << 20
_RWF_NOWAIT = 0x8
_MAX_HEADER_BYTES = 100 << 20
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
GiB = 1 << 30
# A row that could not be read: E4M3 NaN as its first value, then zeros.
_POISON_VALUE = 0x7F
# The native reader's operator, as csrc/exl3/module.cpp declares it.
PLE_READ_ROWS_SCHEMA = (
    "_exl3_C::ple_read_rows(int fd, Tensor ids, Tensor(a!) dest, int count, "
    "int base, int row_bytes, int owner_lo, int owner_hi, Tensor head_offsets, "
    "Tensor head_vocab_sizes, int retries, int max_threads) -> Tensor"
)


# ---------------------------------------------------------------------------
# Table files
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngramTableFiles:
    """Where one Engram table's values and scales live (absolute offsets)."""

    values_path: str
    values_offset: int
    scales_path: str
    scales_offset: int
    rows: int
    dim: int


def read_safetensors_header(path: str | os.PathLike) -> tuple[dict, int]:
    """Return a safetensors file's header and the absolute offset of its data."""
    with open(path, "rb") as file:
        raw = file.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file")
        (length,) = struct.unpack("<Q", raw)
        if length > _MAX_HEADER_BYTES:
            raise ValueError(f"{path}: safetensors header of {length} bytes")
        text = file.read(length)
        if len(text) != length:
            raise ValueError(f"{path}: truncated safetensors header")
    return json.loads(text), 8 + length


def locate_engram_table(
    table_dir: str | os.PathLike, layer_id: int, rows: int, dim: int
) -> EngramTableFiles:
    """Find and validate ``layers.<layer_id>.engram.embed.{weight,scale}``."""
    root = Path(table_dir)
    index_path = root / "model.safetensors.index.json"
    try:
        weight_map = json.loads(index_path.read_text())["weight_map"]
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Engram table index {index_path} unreadable: {exc}") from exc
    found = {}
    for suffix, dtype, width in (
        ("weight", "F8_E4M3", dim),
        ("scale", "F8_E8M0", dim // ENGRAM_QUANT_BLOCK),
    ):
        name = f"layers.{layer_id}.engram.embed.{suffix}"
        if name not in weight_map:
            raise ValueError(f"{index_path} lists no {name}")
        path = root / weight_map[name]
        header, data_start = read_safetensors_header(path)
        info = header.get(name)
        if info is None:
            raise ValueError(f"{path} holds no {name}")
        begin, end = info["data_offsets"]
        if (
            info["dtype"] != dtype
            or list(info["shape"]) != [rows, width]
            or end - begin != rows * width
        ):
            raise ValueError(
                f"{name} in {path} is {info['dtype']} {info['shape']}, "
                f"expected {dtype} {[rows, width]}"
            )
        offset = data_start + begin
        if os.stat(path).st_size < offset + rows * width:
            raise ValueError(f"{path} is truncated before the end of {name}")
        found[suffix] = (str(path), offset)
    return EngramTableFiles(
        values_path=found["weight"][0],
        values_offset=found["weight"][1],
        scales_path=found["scale"][0],
        scales_offset=found["scale"][1],
        rows=rows,
        dim=dim,
    )


def engram_owner_range(
    head_sizes: Sequence[int], tp_rank: int, tp_size: int
) -> tuple[int, int, int, int]:
    """(heads per rank, first head, vocab_start, vocab_end) of a TP rank, as
    upstream ``ParallelEngramEmbedding`` shards whole hash heads."""
    heads = len(head_sizes)
    part = -(-heads // tp_size)
    if (tp_size - 1) * part >= heads:
        raise ValueError(
            f"Engram sharding leaves ranks without hash heads: {heads} heads "
            f"over {tp_size} shards"
        )
    head_start = tp_rank * part
    return (
        part,
        head_start,
        sum(head_sizes[:head_start]),
        sum(head_sizes[: head_start + part]),
    )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class EngramTelemetry:
    """Counters of one table's lookups, read with :meth:`snapshot`.

    Row reads are classified by the NOWAIT probe that precedes them: rows the
    probe returned came from the page cache, the others from storage. The
    native reader and a filesystem without the probe leave them unclassified.
    Byte counts are row bytes delivered; storage reads move whole pages.
    """

    FIELDS = (
        "lookups",
        "rows_requested",
        "rows_unowned",
        "rows_unique",
        "cache_hits",
        "cache_misses",
        "cache_inserts",
        "cache_evictions",
        "rows_failed",
        "page_cache_rows",
        "storage_rows",
        "unclassified_rows",
        "page_cache_bytes",
        "storage_bytes",
        "unclassified_bytes",
        "read_ns",
        "fetch_ns",
        "gpu_wait_ns",
        "host_ns",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(self.FIELDS, 0)

    def add(self, **counts: int) -> None:
        with self._lock:
            for name, value in counts.items():
                self._counts[name] += value

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def reset(self) -> None:
        with self._lock:
            self._counts = dict.fromkeys(self.FIELDS, 0)


def format_engram_telemetry(counts: dict[str, int]) -> str:
    """One line of counters: rows, cache, page-cache and storage bytes, and
    host time (fetch, read, wait for the GPU, whole lookup)."""
    mib = 1 << 20
    hits, misses = counts["cache_hits"], counts["cache_misses"]
    return (
        f"lookups {counts['lookups']} rows {counts['rows_requested']} "
        f"(unowned {counts['rows_unowned']}, unique {counts['rows_unique']}) "
        f"cache hits {hits} misses {misses} "
        f"({100.0 * hits / max(hits + misses, 1):.1f}% hit) "
        f"evictions {counts['cache_evictions']}; "
        f"failed rows {counts['rows_failed']}; read MiB page cache "
        f"{counts['page_cache_bytes'] / mib:.1f} storage "
        f"{counts['storage_bytes'] / mib:.1f} unclassified "
        f"{counts['unclassified_bytes'] / mib:.1f}; host ms fetch "
        f"{counts['fetch_ns'] / 1e6:.1f} read {counts['read_ns'] / 1e6:.1f} "
        f"gpu wait {counts['gpu_wait_ns'] / 1e6:.1f} "
        f"lookup {counts['host_ns'] / 1e6:.1f}"
    )


# The live sm_80 Engram tables of this process.
_TABLES: "weakref.WeakSet[Sm80EngramEmbedding]" = weakref.WeakSet()


def engram_telemetry() -> dict[str, dict[str, int]]:
    """The counters of every live sm_80 Engram table in this process, by
    label (``layer <id>``)."""
    return {table.label: table.telemetry() for table in list(_TABLES)}


def _native_reader_schema() -> str:
    """The ple_read_rows schema of the loaded native EXL3 extension."""
    from vllm.sm80.kernels.exl3.ops import ensure_extension

    ensure_extension()
    return str(torch.ops._exl3_C.ple_read_rows.default._schema)


def check_native_reader() -> None:
    """Refuse the native reader unless the loaded extension has the
    ple_read_rows this module calls (at startup, not at the first miss)."""
    try:
        schema = _native_reader_schema()
    except Exception as exc:
        raise RuntimeError(
            "VLLM_ENGRAM_NATIVE_READ=1 needs the native EXL3 extension "
            f"(vllm._exl3_C) with ple_read_rows: {exc}"
        ) from exc
    if schema != PLE_READ_ROWS_SCHEMA:
        raise RuntimeError(
            "VLLM_ENGRAM_NATIVE_READ=1: the loaded vllm._exl3_C declares "
            f"{schema!r}, expected the schema {PLE_READ_ROWS_SCHEMA!r}; "
            "rebuild the native extension for this tree"
        )


# ---------------------------------------------------------------------------
# Host row cache
# ---------------------------------------------------------------------------


class EngramRowCache:
    """Bounded set-associative LRU cache of table rows in host memory.

    Rows are keyed by their local index. Entries are committed only as they
    fill, so the bound is the most the cache can ever hold. Thread-safe.
    """

    def __init__(self, capacity_bytes: int, row_bytes: int, ways: int = CACHE_WAYS):
        self.row_bytes = row_bytes
        self.ways = ways
        self.num_sets = max(capacity_bytes, 0) // (
            ways * (row_bytes + CACHE_ENTRY_OVERHEAD)
        )
        self._lock = threading.Lock()
        self._clock = 0
        self._tags = np.full((self.num_sets, ways), -1, dtype=np.int64)
        self._stamps = np.zeros((self.num_sets, ways), dtype=np.int64)
        self._data = np.empty((self.num_sets * ways, row_bytes), dtype=np.uint8)

    @property
    def capacity_rows(self) -> int:
        return self.num_sets * self.ways

    @property
    def nbytes(self) -> int:
        return self._data.nbytes + self._tags.nbytes + self._stamps.nbytes

    def set_of(self, rows: np.ndarray) -> np.ndarray:
        """Fibonacci hash of the row index, so strided ids spread over sets."""
        mixed = (rows.astype(np.uint64) * _GOLDEN) >> np.uint64(32)
        return (mixed % np.uint64(self.num_sets)).astype(np.int64)

    def lookup(self, rows: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Copy the cached ones of ``rows`` into ``out``; return the hit mask."""
        if not self.num_sets or not len(rows):
            return np.zeros(len(rows), dtype=bool)
        with self._lock:
            sets = self.set_of(rows)
            match = self._tags[sets] == rows[:, None]
            hit = match.any(axis=1)
            if hit.any():
                way = match.argmax(axis=1)[hit]
                hit_sets = sets[hit]
                out[hit] = self._data[hit_sets * self.ways + way]
                self._clock += 1
                self._stamps[hit_sets, way] = self._clock
        return hit

    def insert(self, rows: np.ndarray, data: np.ndarray) -> tuple[int, int]:
        """Insert unique ``rows`` with their ``data``; return (inserted,
        evicted). Rows of one set beyond its ways are not inserted; rows
        already present (another caller's) are skipped."""
        if not self.num_sets or not len(rows):
            return 0, 0
        with self._lock:
            sets = self.set_of(rows)
            absent = np.flatnonzero(~(self._tags[sets] == rows[:, None]).any(axis=1))
            order = absent[np.argsort(sets[absent], kind="stable")]
            sets = sets[order]
            if not len(sets):
                return 0, 0
            starts = np.r_[True, sets[1:] != sets[:-1]]
            first = np.flatnonzero(starts)[np.cumsum(starts) - 1]
            rank = np.arange(len(sets)) - first
            keep = rank < self.ways
            order, sets, rank = order[keep], sets[keep], rank[keep]
            # Least recently used first; empty ways (stamp 0) before any.
            lru = np.argsort(self._stamps[sets], axis=1, kind="stable")
            way = lru[np.arange(len(sets)), rank]
            evicted = int((self._tags[sets, way] >= 0).sum())
            self._clock += 1
            self._tags[sets, way] = rows[order]
            self._stamps[sets, way] = self._clock
            self._data[sets * self.ways + way] = data[order]
            return len(order), evicted


# ---------------------------------------------------------------------------
# Row source
# ---------------------------------------------------------------------------


class _IoVec(ctypes.Structure):
    _fields_ = [("base", ctypes.c_void_p), ("length", ctypes.c_size_t)]


@functools.cache
def _preadv2() -> Callable | None:
    """The C library's preadv2 (os.preadv may lack flags), or None."""
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "preadv2", None)
    if function is None:
        return None
    function.argtypes = (
        ctypes.c_int,
        ctypes.POINTER(_IoVec),
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int,
    )
    function.restype = ctypes.c_ssize_t
    return function


@dataclass
class EngramReadStats:
    page_cache_rows: int = 0
    storage_rows: int = 0
    unclassified_rows: int = 0
    # (index into the rows read, error) of the rows that still failed after
    # the retries; their bytes are unspecified.
    failed: list[tuple[int, BaseException]] = field(default_factory=list)


class EngramRowSource:
    """Positional reads of owned rows of one table extent in a file.

    The Python path probes each row with a NOWAIT read (rows in the page
    cache return at once, as the native PLE reader core does), then reads
    the rest on up to ``threads`` threads with bounded retries. The native
    path hands the whole batch to ``ple_read_rows`` (csrc/exl3/ple_read.cpp).
    Thread-safe; a failed row raises after every row was attempted.
    """

    def __init__(
        self,
        path: str,
        base: int,
        row_bytes: int,
        owner_lo: int,
        owner_hi: int,
        *,
        native: bool = False,
        threads: int = READ_THREADS,
        retries: int = READ_RETRIES,
    ):
        from vllm.models.qwen4_exp.nvidia.exl3_ple import (
            _read_error,
            page_cache_residency,
            pread_exact_into,
        )

        self._pread_exact_into = pread_exact_into
        self._read_error = _read_error
        self._residency = page_cache_residency
        if not 0 <= owner_lo <= owner_hi or base < 0 or row_bytes <= 0:
            raise ValueError("Invalid Engram row source extent")
        if owner_hi > ((1 << 63) - 1 - base) // row_bytes:
            raise ValueError("Engram file offset exceeds signed 64-bit range")
        self.path = path
        self.base = base
        self.row_bytes = row_bytes
        self.owner_lo, self.owner_hi = owner_lo, owner_hi
        self.native = native
        self.threads = threads
        self.retries = retries
        self._nowait = _preadv2() is not None
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        if native:
            check_native_reader()
        self.fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        # Rows are read at random; readahead would only fetch unused rows.
        os.posix_fadvise(
            self.fd, self.extent_offset, self.extent_bytes, os.POSIX_FADV_RANDOM
        )
        if native:
            # ple_read_rows runs on CPU tensors only. Model construction may
            # set a default device, so every tensor it gets names the CPU
            # (numpy-backed ones always live there).
            self._heads = (
                torch.tensor([owner_lo], dtype=torch.int64, device="cpu"),
                torch.tensor([owner_hi - owner_lo], dtype=torch.int64, device="cpu"),
            )
            try:
                # One empty batch: the operator dispatches with these arguments.
                self._read_native(
                    np.empty(0, dtype=np.int64),
                    np.empty((0, row_bytes), dtype=np.uint8),
                )
            except BaseException as exc:
                self.close()
                raise RuntimeError(
                    f"VLLM_ENGRAM_NATIVE_READ=1: ple_read_rows refused an empty "
                    f"read of {path}: {exc}"
                ) from exc

    @property
    def extent_offset(self) -> int:
        return self.base + self.owner_lo * self.row_bytes

    @property
    def extent_bytes(self) -> int:
        return (self.owner_hi - self.owner_lo) * self.row_bytes

    def page_cache_residency(self) -> tuple[int, int]:
        """(resident, total) page-cache pages of the owned extent; sampling
        does not change residency."""
        return self._residency(self.fd, self.extent_offset, self.extent_bytes)

    def drop_page_cache(self) -> None:
        """Evict the owned extent's clean page-cache pages (no root needed)."""
        os.posix_fadvise(
            self.fd, self.extent_offset, self.extent_bytes, os.POSIX_FADV_DONTNEED
        )

    def read_extent(self, first: int, count: int, dest: memoryview) -> None:
        """Read ``count`` consecutive rows from local row ``first``."""
        offset = self.extent_offset + first * self.row_bytes
        for begin in range(0, count * self.row_bytes, _LOAD_CHUNK_BYTES):
            end = min(begin + _LOAD_CHUNK_BYTES, count * self.row_bytes)
            self._read_exact(dest[begin:end], offset + begin)

    def _read_exact(self, target: memoryview, offset: int) -> None:
        for attempt in range(self.retries + 1):
            try:
                self._pread_exact_into(self.fd, target, offset)
                return
            except (OSError, EOFError):
                if attempt == self.retries:
                    raise
                time.sleep(0.001 * 2**attempt)

    def read(self, rows: np.ndarray, dest: np.ndarray) -> EngramReadStats:
        """Read global ``rows`` (int64, owned) into ``dest`` ([n, row_bytes]).

        Rows that still fail after the retries are returned in ``failed``.
        """
        count = len(rows)
        if dest.shape != (count, self.row_bytes) or not dest.flags.c_contiguous:
            raise ValueError("Engram row destination does not match the rows")
        if count and (rows.min() < self.owner_lo or rows.max() >= self.owner_hi):
            raise ValueError("Engram row outside this rank's owned range")
        if not count:
            return EngramReadStats()
        if self.native:
            return self._read_native(rows, dest)
        return self._read_python(rows, dest)

    def _read_native(self, rows: np.ndarray, dest: np.ndarray) -> EngramReadStats:
        from vllm.sm80.kernels.exl3.ops import ple_read_rows

        ids = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.int64))
        failed = ple_read_rows(
            self.fd,
            ids,
            torch.from_numpy(dest),
            len(rows),
            self.base,
            self.row_bytes,
            self.owner_lo,
            self.owner_hi,
            *self._heads,
            self.retries,
            self.threads,
        ).tolist()
        return EngramReadStats(
            unclassified_rows=len(rows),
            failed=[
                (index, self._read_error(code, offset))
                for index, code, offset in failed
            ],
        )

    def _read_python(self, rows: np.ndarray, dest: np.ndarray) -> EngramReadStats:
        row_bytes = self.row_bytes
        address = dest.ctypes.data
        pending = []
        probe = self._nowait
        if probe:
            preadv2 = _preadv2()
            vector = _IoVec(None, row_bytes)
            for index, row in enumerate(rows.tolist()):
                vector.base = address + index * row_bytes
                done = preadv2(
                    self.fd,
                    ctypes.byref(vector),
                    1,
                    self.base + row * row_bytes,
                    _RWF_NOWAIT,
                )
                if done == row_bytes:
                    continue
                if done < 0 and ctypes.get_errno() in (
                    errno.EOPNOTSUPP,
                    errno.EINVAL,
                    errno.ENOSYS,
                ):
                    # No NOWAIT reads here: read every row below, unclassified.
                    self._nowait = probe = False
                    pending = list(range(len(rows)))
                    break
                pending.append(index)
        else:
            pending = list(range(len(rows)))
        cached = len(rows) - len(pending)
        failures = []
        if pending:
            view = memoryview(dest).cast("B")
            failures = self._read_pending(rows, view, pending)
        if probe:
            return EngramReadStats(
                page_cache_rows=cached, storage_rows=len(pending), failed=failures
            )
        return EngramReadStats(
            page_cache_rows=cached, unclassified_rows=len(pending), failed=failures
        )

    def _read_pending(
        self, rows: np.ndarray, view: memoryview, pending: list[int]
    ) -> list[tuple[int, BaseException]]:
        row_bytes = self.row_bytes

        def read_chunk(chunk: list[int]) -> list[tuple[int, BaseException]]:
            failed = []
            for index in chunk:
                try:
                    self._read_exact(
                        view[index * row_bytes : (index + 1) * row_bytes],
                        self.base + int(rows[index]) * row_bytes,
                    )
                except (OSError, EOFError) as error:
                    failed.append((index, error))
            return failed

        threads = min(self.threads, len(pending))
        if threads <= 1:
            return read_chunk(pending)
        step = -(-len(pending) // threads)
        chunks = [pending[i : i + step] for i in range(0, len(pending), step)]
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.threads, thread_name_prefix="engram-read"
                )
            pool = self._pool
        futures = [pool.submit(read_chunk, chunk) for chunk in chunks[1:]]
        failed = read_chunk(chunks[0])
        errors = []
        for future in futures:
            try:
                failed.extend(future.result())
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]
        return sorted(failed, key=lambda item: item[0])

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


# ---------------------------------------------------------------------------
# Owned rows of one table
# ---------------------------------------------------------------------------


def _host_empty(shape: tuple[int, ...], dtype: torch.dtype, pin: bool) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin)


class EngramRowStore:
    """One rank's rows of one Engram table on the host.

    Scales are always resident. Values come from the file through the row
    cache (``ssd``) or from a resident copy (``ram``). :meth:`fetch` fills
    caller staging for global row ids: unowned ids get zero values and the
    scale byte of 1.0, which dequantize to +0 as the reference's mask does.
    """

    def __init__(
        self,
        files: EngramTableFiles,
        owner_lo: int,
        owner_hi: int,
        *,
        placement: str,
        cache_bytes: int,
        native_read: bool = False,
        pin: bool = False,
    ):
        if placement not in ("ssd", "ram"):
            raise ValueError(f"Engram placement must be ssd or ram, not {placement!r}")
        if not 0 <= owner_lo <= owner_hi <= files.rows:
            raise ValueError("Engram owner range exceeds the table rows")
        if files.dim % ENGRAM_QUANT_BLOCK:
            raise ValueError("Engram row width must be a multiple of 32")
        self.placement = placement
        self.owner_lo, self.owner_hi = owner_lo, owner_hi
        self.row_bytes = files.dim
        self.scale_bytes = files.dim // ENGRAM_QUANT_BLOCK
        self.telemetry = EngramTelemetry()
        rows = owner_hi - owner_lo
        self.source = EngramRowSource(
            files.values_path,
            files.values_offset,
            self.row_bytes,
            owner_lo,
            owner_hi,
            native=native_read,
        )
        try:
            scale_source = EngramRowSource(
                files.scales_path,
                files.scales_offset,
                self.scale_bytes,
                owner_lo,
                owner_hi,
            )
            try:
                self.scales = _host_empty((rows, self.scale_bytes), torch.uint8, pin)
                scale_source.read_extent(
                    0, rows, memoryview(self.scales.numpy()).cast("B")
                )
            finally:
                scale_source.close()
            self._scales_np = self.scales.numpy()
            self.table: torch.Tensor | None = None
            self.cache: EngramRowCache | None = None
            if placement == "ram":
                self.table = _host_empty((rows, self.row_bytes), torch.uint8, pin)
                self.source.read_extent(
                    0, rows, memoryview(self.table.numpy()).cast("B")
                )
                self._table_np = self.table.numpy()
            else:
                self.cache = EngramRowCache(cache_bytes, self.row_bytes)
        except BaseException:
            self.source.close()
            raise

    @property
    def host_bytes(self) -> int:
        """Host memory this store can hold: scales, resident table or cache."""
        resident = self.scales.nbytes
        if self.table is not None:
            resident += self.table.nbytes
        if self.cache is not None:
            resident += self.cache.nbytes
        return resident

    def fetch(self, ids: np.ndarray, values: np.ndarray, scales: np.ndarray) -> None:
        """Fill ``values`` [n, row_bytes] and ``scales`` [n, scale_bytes] for
        global row ids [n]. Rows that cannot be read are poisoned (an E4M3
        NaN, then zeros) and never cached. Thread-safe."""
        start = time.perf_counter_ns()
        owned = (ids >= self.owner_lo) & (ids < self.owner_hi)
        unowned = ~owned
        local = ids[owned] - self.owner_lo
        values[unowned] = 0
        scales[unowned] = 127
        scales[owned] = self._scales_np[local]
        counts = {
            "lookups": 1,
            "rows_requested": len(ids),
            "rows_unowned": int(unowned.sum()),
        }
        if self.table is not None:
            values[owned] = self._table_np[local]
        else:
            assert self.cache is not None
            unique, inverse = np.unique(local, return_inverse=True)
            rows = np.empty((len(unique), self.row_bytes), dtype=np.uint8)
            hit = self.cache.lookup(unique, rows)
            miss = unique[~hit]
            counts.update(
                rows_unique=len(unique),
                cache_hits=int(hit.sum()),
                cache_misses=len(miss),
            )
            if len(miss):
                missed = np.empty((len(miss), self.row_bytes), dtype=np.uint8)
                read_start = time.perf_counter_ns()
                stats = self.source.read(miss + self.owner_lo, missed)
                counts["read_ns"] = time.perf_counter_ns() - read_start
                good = np.ones(len(miss), dtype=bool)
                if stats.failed:
                    bad = np.array([index for index, _ in stats.failed])
                    good[bad] = False
                    missed[bad] = 0
                    missed[bad, 0] = _POISON_VALUE
                    counts["rows_failed"] = len(bad)
                    index, error = stats.failed[0]
                    logger.error(
                        "sm_80 Engram table %s: %d row read(s) failed after %d "
                        "retries (first: row %d: %s); the requests using them "
                        "will be finished with an error",
                        self.source.path,
                        len(bad),
                        self.source.retries,
                        int(miss[index]) + self.owner_lo,
                        error,
                    )
                rows[~hit] = missed
                inserted, evicted = self.cache.insert(miss[good], missed[good])
                counts.update(
                    cache_inserts=inserted,
                    cache_evictions=evicted,
                    page_cache_rows=stats.page_cache_rows,
                    storage_rows=stats.storage_rows,
                    unclassified_rows=stats.unclassified_rows,
                    page_cache_bytes=stats.page_cache_rows * self.row_bytes,
                    storage_bytes=stats.storage_rows * self.row_bytes,
                    unclassified_bytes=stats.unclassified_rows * self.row_bytes,
                )
            values[owned] = rows[inverse.reshape(-1)]
        counts["fetch_ns"] = time.perf_counter_ns() - start
        self.telemetry.add(**counts)

    def close(self) -> None:
        self.source.close()


# ---------------------------------------------------------------------------
# The embedding module
# ---------------------------------------------------------------------------


class Sm80EngramEmbedding(nn.Module):
    """``ParallelEngramEmbedding`` for sm_80 with SSD rows (or the ram
    comparison path), no table parameters and the same lookup interface.

    The attributes the upstream ``Engram`` reads keep their meaning;
    ``cpu_offload`` is set so ``Engram`` runs lookups on its prefetch stream
    from its eager graph break.
    """

    cpu_offload = True
    dp_shared_memory = False
    use_thp = False
    dp_size = 1

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        files: EngramTableFiles,
        *,
        tp_rank: int,
        tp_size: int,
        max_tokens: int,
        placement: str,
        cache_bytes: int,
        native_read: bool = False,
        device: torch.device | str = "cpu",
        block_size: int = ENGRAM_QUANT_BLOCK,
        label: str = "engram",
        log_every: int = 0,
    ) -> None:
        super().__init__()
        if block_size != ENGRAM_QUANT_BLOCK or dim % block_size:
            raise ValueError("Engram rows must hold 32-wide scale blocks")
        if (files.rows, files.dim) != (num_embeddings, dim):
            raise ValueError(
                f"Engram table is [{files.rows}, {files.dim}], the model expects "
                f"[{num_embeddings}, {dim}]"
            )
        if not head_sizes or sum(head_sizes) > num_embeddings:
            raise ValueError("Engram hash heads exceed the table rows")
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        self.n_hash_cols = len(head_sizes)
        self.tp_size = tp_size
        (
            self.part_n_hash_cols,
            self.head_start,
            self.vocab_start_idx,
            self.vocab_end_idx,
        ) = engram_owner_range(head_sizes, tp_rank, tp_size)
        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
        self.device = torch.device(device)
        self.label = label
        # Lookups between telemetry log lines; 0 never logs.
        self.log_every = log_every
        self._since_log = 0
        self._logged = dict.fromkeys(EngramTelemetry.FIELDS, 0)
        cuda = self.device.type == "cuda"
        self.store = EngramRowStore(
            files,
            self.vocab_start_idx,
            self.vocab_end_idx,
            placement=placement,
            cache_bytes=cache_bytes,
            native_read=native_read,
            pin=cuda,
        )
        self.capacity_rows = max_tokens * self.part_n_hash_cols
        scale_dim = dim // block_size
        cap = self.capacity_rows
        self._host_ids = _host_empty((cap,), torch.int64, cuda)
        self._host_values = _host_empty((cap, dim), torch.uint8, cuda)
        self._host_scales = _host_empty((cap, scale_dim), torch.uint8, cuda)
        self._host_ids_np = self._host_ids.numpy()
        self._host_values_np = self._host_values.numpy()
        self._host_scales_np = self._host_scales.numpy()
        # Tokens of the last forward whose rows failed, on any rank.
        self._failed_tokens = torch.zeros(
            max_tokens, dtype=torch.bool, device=self.device
        )
        if cuda:
            self._dev_ids = torch.empty(cap, dtype=torch.int64, device=self.device)
            self._dev_values = torch.empty(
                cap, dim, dtype=torch.uint8, device=self.device
            )
            self._dev_scales = torch.empty(
                cap, scale_dim, dtype=torch.uint8, device=self.device
            )
            self._ids_ready = torch.cuda.Event()
            # The last host-to-device copy of the host staging finished.
            self._staged = torch.cuda.Event()
            # The last dequant finished reading the device staging.
            self._consumed = torch.cuda.Event()

    @property
    def placement(self) -> str:
        return self.store.placement

    def host_bytes(self) -> int:
        """Host memory of this table: the store plus the host staging."""
        return (
            self.store.host_bytes
            + self._host_ids.nbytes
            + self._host_values.nbytes
            + self._host_scales.nbytes
        )

    def device_bytes(self) -> int:
        """Device memory of this table's staging (the tables use none)."""
        if self.device.type != "cuda":
            return 0
        return self._dev_ids.nbytes + self._dev_values.nbytes + self._dev_scales.nbytes

    def telemetry(self) -> dict[str, int]:
        """Counters since start or the last :meth:`reset_telemetry`."""
        return self.store.telemetry.snapshot()

    def reset_telemetry(self) -> None:
        self.store.telemetry.reset()
        self._logged = dict.fromkeys(EngramTelemetry.FIELDS, 0)

    def log_telemetry(self, when: str = "") -> None:
        """Log the counters since start and since the last line."""
        total = self.telemetry()
        window = {name: total[name] - self._logged[name] for name in total}
        self._logged = total
        logger.info(
            "sm_80 Engram %s telemetry%s: %s; last %s",
            self.label,
            f" {when}" if when else "",
            format_engram_telemetry(total),
            format_engram_telemetry(window),
        )

    def _count_lookup(self, **counts: int) -> None:
        self.store.telemetry.add(**counts)
        if self.log_every:
            self._since_log += 1
            if self._since_log >= self.log_every:
                self._since_log = 0
                self.log_telemetry()

    def page_cache_residency(self) -> tuple[int, int]:
        """(resident, total) page-cache pages of this rank's value extent."""
        return self.store.source.page_cache_residency()

    def _stage_ids(self, indices: torch.Tensor, num_tokens: int) -> int:
        """Write this rank's head columns of ``indices`` as int64 into the
        id staging (-1 for padded heads); return the row count."""
        part = self.part_n_hash_cols
        count = num_tokens * part
        columns = indices[:, self.head_start : self.head_start + part]
        width = columns.shape[1]
        target = self._dev_ids if self.device.type == "cuda" else self._host_ids
        staged = target[:count].view(num_tokens, part)
        staged[:, :width].copy_(columns)
        if width < part:
            staged[:, width:].fill_(-1)
        return count

    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        """Look up this rank's heads of ``indices`` [T, n_hash_cols] into
        ``out`` [T, part_n_hash_cols, dim] BF16 on the current stream.

        ``background`` is accepted for the upstream interface; the dequant
        kernel is small either way.
        """
        if out.dtype != torch.bfloat16:
            raise ValueError("sm_80 Engram lookups produce BF16 rows only")
        num_tokens = indices.shape[0]
        count = num_tokens * self.part_n_hash_cols
        if not count:
            return
        if count > self.capacity_rows:
            raise ValueError("Engram lookup exceeds the staging capacity")
        rows_out = out.view(count, self.dim)
        ids = self._host_ids_np[:count]
        values = self._host_values_np[:count]
        scales = self._host_scales_np[:count]
        if self.device.type != "cuda":
            start = time.perf_counter_ns()
            self._stage_ids(indices, num_tokens)
            self.store.fetch(ids, values, scales)
            from vllm.models.deepseek_v41.nvidia.ops.sm80_engram_dequant import (
                dequant_engram_rows,
            )

            dequant_engram_rows(
                self._host_values[:count], self._host_scales[:count], rows_out
            )
            self._count_lookup(host_ns=time.perf_counter_ns() - start)
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "sm_80 Engram lookups read the host and cannot be captured; "
                "run them eagerly or from an eager break of a breakable CUDA "
                "graph"
            )
        from vllm.models.deepseek_v41.nvidia.ops.sm80_engram_dequant import (
            dequant_engram_rows,
        )

        stream = torch.cuda.current_stream()
        start = time.perf_counter_ns()
        # The device staging is free once the previous dequant has read it.
        stream.wait_event(self._consumed)
        self._stage_ids(indices, num_tokens)
        self._host_ids[:count].copy_(self._dev_ids[:count], non_blocking=True)
        self._ids_ready.record(stream)
        # The host staging is free once the previous copy has read it.
        wait_start = time.perf_counter_ns()
        self._staged.synchronize()
        self._ids_ready.synchronize()
        gpu_wait = time.perf_counter_ns() - wait_start
        self.store.fetch(ids, values, scales)
        dev_values = self._dev_values[:count]
        dev_scales = self._dev_scales[:count]
        dev_values.copy_(self._host_values[:count], non_blocking=True)
        dev_scales.copy_(self._host_scales[:count], non_blocking=True)
        self._staged.record(stream)
        dequant_engram_rows(dev_values, dev_scales, rows_out)
        self._consumed.record(stream)
        self._count_lookup(gpu_wait_ns=gpu_wait, host_ns=time.perf_counter_ns() - start)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("sm_80 Engram tables are read through Engram.lookup")

    def mask_failed_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Zero the tokens of gathered rows [T, n_hash_cols, dim] that hold a
        poisoned row and remember them for :meth:`take_failed_tokens`.

        Runs on every rank after the TP gather, so each rank sees the rows
        any rank failed to read. Graph-capturable.
        """
        failed = rows[:, :, 0].isnan().any(-1)
        self._failed_tokens[: rows.shape[0]].copy_(failed)
        return rows.masked_fill(failed[:, None, None], 0)

    def take_failed_tokens(self) -> torch.Tensor:
        """The failed-token mask over the last forward's tokens (padded to
        the staging capacity), cleared once taken (vllm/sm80/failed_inputs.py)."""
        failed = self._failed_tokens.clone()
        self._failed_tokens.zero_()
        return failed

    def close(self) -> None:
        """Wait for queued staging copies, then close the table file."""
        if self.device.type == "cuda":
            self._staged.synchronize()
        if self.log_every:
            self.log_telemetry("at close")
        self.store.close()


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def engram_table_dir(model_config) -> str:
    """The base-checkpoint directory holding the Engram tables."""
    for config in (model_config.hf_config, model_config.hf_text_config):
        table_dir = getattr(config, "engram_table_dir", None)
        if table_dir:
            return str(table_dir)
    raise ValueError(
        "sm_80 Engram tables are read from the base checkpoint: set the model "
        'config field engram_table_dir (--hf-overrides \'{"engram_table_dir": '
        '"<dir>"}\') to the directory holding its model.safetensors.index.json '
        "and Engram shards"
    )


def build_sm80_engram_embedding(layout, layer_hash_index: int) -> Sm80EngramEmbedding:
    """The sm_80 Engram table of one Engram layer of the current model."""
    from vllm import envs
    from vllm.config import get_current_vllm_config
    from vllm.distributed import (
        get_engram_dp_size,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    vllm_config = get_current_vllm_config()
    engram_config = vllm_config.engram_config
    if engram_config is not None and (
        engram_config.dp_shared_memory or engram_config.use_thp
    ):
        raise ValueError(
            "sm_80 Engram tables support neither dp_shared_memory nor use_thp"
        )
    if get_engram_dp_size() > 1:
        raise NotImplementedError("sm_80 Engram tables are sharded over TP only")
    layer_id = layout.layer_ids[layer_hash_index]
    rows = layout.num_embeddings[layer_hash_index]
    files = locate_engram_table(
        engram_table_dir(vllm_config.model_config), layer_id, rows, layout.head_dim
    )
    placement = envs.VLLM_ENGRAM_PLACEMENT
    cache_gib = envs.VLLM_ENGRAM_ROW_CACHE_GIB
    if cache_gib < 0:
        raise ValueError("VLLM_ENGRAM_ROW_CACHE_GIB must be nonnegative")
    log_every = envs.VLLM_ENGRAM_TELEMETRY_STEPS
    if log_every < 0:
        raise ValueError("VLLM_ENGRAM_TELEMETRY_STEPS must be nonnegative")
    table = Sm80EngramEmbedding(
        rows,
        layout.head_dim,
        tuple(size for order in layout.primes[layer_hash_index] for size in order),
        files,
        tp_rank=get_tensor_model_parallel_rank(),
        tp_size=get_tensor_model_parallel_world_size(),
        max_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        placement=placement,
        cache_bytes=int(cache_gib * GiB) // len(layout.layer_ids),
        native_read=envs.VLLM_ENGRAM_NATIVE_READ,
        device=torch.device("cuda", torch.cuda.current_device()),
        label=f"layer {layer_id}",
        log_every=log_every,
    )
    _TABLES.add(table)
    if log_every:
        # Engine shutdown does not close modules: log the totals at exit.
        atexit.register(table.log_telemetry, "at exit")
    resident, total = table.page_cache_residency()
    if placement == "ram":
        values = f"resident {table.store.table.nbytes / GiB:.2f} GiB"
    else:
        values = f"row cache {table.store.cache.nbytes / GiB:.2f} GiB"
    logger.info(
        "sm_80 Engram layer %d (%s): placement=%s rows [%d, %d) heads [%d, %d); "
        "host %.2f GiB (scales %.2f GiB pinned, values %s); device staging "
        "%.1f MiB; native read %s; page cache %d/%d value pages resident",
        layer_id,
        files.values_path,
        placement,
        table.vocab_start_idx,
        table.vocab_end_idx,
        table.head_start,
        table.head_start + table.part_n_hash_cols,
        table.host_bytes() / GiB,
        table.store.scales.nbytes / GiB,
        values,
        table.device_bytes() / (1 << 20),
        "on" if envs.VLLM_ENGRAM_NATIVE_READ else "off",
        resident,
        total,
    )
    return table


def sm80_engram_embedding(create: Callable) -> Callable:
    """Decorates upstream ``Engram._create_embedding``: on sm_80 the layer's
    table is a :class:`Sm80EngramEmbedding`; elsewhere upstream's."""

    @functools.wraps(create)
    def build(engram, layout, layer_hash_index: int):
        from vllm.sm80.kernels._platform import _is_sm80

        if not _is_sm80():
            return create(engram, layout, layer_hash_index)
        table = build_sm80_engram_embedding(layout, layer_hash_index)
        mask_failed_after_gather(engram, table)
        return table

    return build


def mask_failed_after_gather(engram, table: Sm80EngramEmbedding) -> None:
    """Route ``engram.embed`` (the lookup's rows after the TP gather) through
    :meth:`Sm80EngramEmbedding.mask_failed_rows`."""
    gather = engram.embed

    def embed(hash_ids: torch.Tensor) -> torch.Tensor:
        return table.mask_failed_rows(gather(hash_ids))

    engram.embed = embed
