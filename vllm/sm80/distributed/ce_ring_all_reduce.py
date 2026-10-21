# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy-engine ring all-reduce for large tensor-parallel messages.

The design follows the PCIe DMA all-reduce of b12x (Apache-2.0,
``b12x/comm/pcie/pcie_dma.py``, local-inference-lab): a reduce-scatter +
all-gather ring whose data moves by copy-engine copies
(``cudaMemcpyAsync`` into the next rank's IPC scratch), while the SMs only
signal, wait and add. An NCCL ring moves the same bytes with SM kernels that
stay resident for the whole transfer; the compute beside them then runs
slower. Here the SM work per step is a one-block flag store, a one-block flag
wait and one elementwise add (reduce-scatter) or nothing (all-gather).

Per call, on the caller's stream, with ``epoch`` the call count (the same on
every rank: every rank issues the same calls in the same order):

1. ready: store ``epoch`` into the previous rank's ready flag (this rank's
   scratch is free: its last reader is earlier on this stream), wait until the
   next rank's store reaches this rank's flag;
2. ``2 (world - 1)`` steps ``k``: copy the send shard into the next rank's
   scratch slot ``k``, store ``epoch`` into its flag ``k``, wait for this
   rank's flag ``k`` (the previous rank's copy into this rank's slot ``k`` has
   landed), then add the slot into the received shard (reduce-scatter:
   ``out = input + slot``, the first and only touch of that shard) or copy it
   there (all-gather).

Each element is summed in the order its owner's reduce-scatter visits the
ranks, in the message dtype (FP32 adds for FP32, BF16 rounding per add for
BF16, as NCCL's ring); every rank ends with the same bits.
:func:`emulate` computes those bits on the host. Flags are monotonic epochs
(no reset); scratch slots are per step, so no slot is rewritten within a call.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch

FLAG_STRIDE_BYTES = 128
SCRATCH_ALIGN = 256
DTYPES = (torch.float32, torch.bfloat16)


@dataclass(frozen=True)
class Step:
    k: int
    reduce: bool
    send: int
    recv: int


def ring_steps(world: int, rank: int) -> list[Step]:
    """The ring's steps for ``rank``: shard indices it sends to ``rank + 1``
    and receives from ``rank - 1``; after the reduce-scatter it owns shard
    ``rank + 1``."""
    steps = []
    for k in range(2 * (world - 1)):
        if k < world - 1:
            steps.append(Step(k, True, (rank - k) % world, (rank - k - 1) % world))
        else:
            j = k - (world - 1)
            steps.append(Step(k, False, (rank + 1 - j) % world, (rank - j) % world))
    return steps


def emulate(inputs: list[torch.Tensor]) -> torch.Tensor:
    """The ring's result (every rank's) for per-rank ``inputs``, on the host,
    with the device's per-add rounding."""
    world = len(inputs)
    shards = [list(t.reshape(world, -1).unbind(0)) for t in inputs]
    out = [[None] * world for _ in range(world)]
    for k in range(world - 1):
        sent = []
        for r in range(world):
            s = ring_steps(world, r)[k]
            src = shards[r][s.send] if k == 0 else out[r][s.send]
            sent.append(src)
        for r in range(world):
            s = ring_steps(world, r)[k]
            out[r][s.recv] = shards[r][s.recv] + sent[(r - 1) % world]
    return torch.cat([out[(c - 1) % world][c] for c in range(world)])


def _ready_slot(world: int) -> int:
    return 2 * (world - 1)


def flag_bytes(world: int) -> int:
    return (_ready_slot(world) + 1) * FLAG_STRIDE_BYTES


def ring_ops(backend, world: int, rank: int, epoch: int, inp: int, out: int,
             shard_bytes: int, shard_numel: int, dtype: torch.dtype,
             scratch: list[int], flags: list[int], slot_bytes: int) -> Iterator[None]:
    """One call's operations on ``backend`` (pointers are integers). A
    generator: a backend's ``wait_flag`` may yield until the flag is set (the
    host simulation); a device backend queues the wait and never yields."""
    nxt, prv = (rank + 1) % world, (rank - 1) % world
    ready = _ready_slot(world) * FLAG_STRIDE_BYTES
    backend.set_flag(flags[prv] + ready, epoch)
    yield from backend.wait_flag(flags[rank] + ready, epoch)
    for s in ring_steps(world, rank):
        src = (inp if s.k == 0 else out) + s.send * shard_bytes
        backend.copy(scratch[nxt] + s.k * slot_bytes, src, shard_bytes)
        backend.set_flag(flags[nxt] + s.k * FLAG_STRIDE_BYTES, epoch)
        yield from backend.wait_flag(flags[rank] + s.k * FLAG_STRIDE_BYTES, epoch)
        slot = scratch[rank] + s.k * slot_bytes
        dst = out + s.recv * shard_bytes
        if s.reduce:
            backend.add(dst, inp + s.recv * shard_bytes, slot, shard_numel, dtype)
        else:
            backend.copy(dst, slot, shard_bytes)


# --- device backend -------------------------------------------------------------

class _Ptr:
    """A raw device pointer with a dtype, as Triton takes a tensor argument."""

    __slots__ = ("_ptr", "dtype")

    def __init__(self, ptr: int, dtype: torch.dtype):
        self._ptr, self.dtype = ptr, dtype

    def data_ptr(self) -> int:
        return self._ptr


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def set_flag(flag_ptr, value):
        # Queued after the copy on the same stream: the copy has completed.
        tl.store(flag_ptr, value)

    @triton.jit
    def wait_flag(flag_ptr, value):
        cur = tl.load(flag_ptr, volatile=True)
        while cur < value:
            cur = tl.load(flag_ptr, volatile=True)

    @triton.jit
    def add(dst_ptr, a_ptr, b_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        tl.store(dst_ptr + offs, (a.to(tl.float32) + b.to(tl.float32)).to(a.dtype), mask=mask)

    return set_flag, wait_flag, add


class DeviceBackend:
    """Queues a call's operations on the current CUDA stream."""

    ADD_BLOCK = 2048

    def __init__(self, memcpy_async: Callable[[int, int, int, int], None], kernels):
        self._memcpy, (self._set, self._wait, self._add) = memcpy_async, kernels
        self.stream = torch.cuda.current_stream().cuda_stream

    def copy(self, dst: int, src: int, nbytes: int) -> None:
        self._memcpy(dst, src, nbytes, self.stream)

    def set_flag(self, ptr: int, value: int) -> None:
        self._set[(1,)](_Ptr(ptr, torch.int32), value)

    def wait_flag(self, ptr: int, value: int) -> Iterator[None]:
        self._wait[(1,)](_Ptr(ptr, torch.int32), value)
        return iter(())

    def add(self, dst: int, a: int, b: int, n: int, dtype: torch.dtype) -> None:
        grid = (triton_cdiv(n, self.ADD_BLOCK),)
        self._add[grid](_Ptr(dst, dtype), _Ptr(a, dtype), _Ptr(b, dtype), n, BLOCK=self.ADD_BLOCK)


def triton_cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _memcpy_async_fn() -> Callable[[int, int, int, int], None]:
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib = CudaRTLibrary().lib
    fn = lib.cudaMemcpyAsync
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]

    def memcpy_async(dst: int, src: int, nbytes: int, stream: int) -> None:
        rc = fn(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, 4, ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync failed: cudaError {rc}")

    return memcpy_async


class CeRingAllReduce:
    """The TP group's copy-engine ring: per rank one scratch slot per ring step
    for shards of messages up to ``capacity_bytes`` and one flag block, shared
    across ranks through CUDA IPC (collective: every rank constructs it)."""

    def __init__(self, group, capacity_bytes: int):
        import torch.distributed as dist

        from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
        from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

        self.world, self.rank = group.world_size, group.rank_in_group
        if self.world < 2:
            raise ValueError("the copy-engine ring needs at least two ranks")
        shard = -(-capacity_bytes // self.world)
        self.slot_bytes = -(-shard // SCRATCH_ALIGN) * SCRATCH_ALIGN
        self.capacity_bytes = self.slot_bytes * self.world
        steps = 2 * (self.world - 1)
        cpu = group.cpu_group
        self.scratch = CustomAllreduce.create_shared_buffer(self.slot_bytes * steps, group=cpu)
        self.flags = CustomAllreduce.create_shared_buffer(flag_bytes(self.world), group=cpu)
        CudaRTLibrary().cudaMemset(ctypes.c_void_p(self.flags[self.rank]), 0, flag_bytes(self.world))
        torch.cuda.synchronize()
        dist.barrier(group=cpu)
        self.device_bytes = self.slot_bytes * steps + flag_bytes(self.world)
        self.epoch = 0
        self._memcpy = _memcpy_async_fn()
        self._kernels = _kernels()

    def takes(self, t: torch.Tensor) -> bool:
        """Whether this ring reduces ``t``: an FP32 or BF16 contiguous CUDA
        tensor within the capacity whose shards are 16-byte multiples, outside
        CUDA graph capture."""
        if not t.is_cuda or t.dtype not in DTYPES or not t.is_contiguous():
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        nbytes = t.numel() * t.element_size()
        return 0 < nbytes <= self.capacity_bytes and nbytes % (16 * self.world) == 0

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        """The sum of every rank's ``t``, queued on the current stream."""
        out = torch.empty_like(t)
        self.epoch += 1
        shard_numel = t.numel() // self.world
        backend = DeviceBackend(self._memcpy, self._kernels)
        for _ in ring_ops(backend, self.world, self.rank, self.epoch, t.data_ptr(), out.data_ptr(),
                          shard_numel * t.element_size(), shard_numel, t.dtype,
                          self.scratch, self.flags, self.slot_bytes):
            raise RuntimeError("a device wait yielded")
        return out


_RING: CeRingAllReduce | None = None


def init_ce_ring(group, capacity_bytes: int) -> CeRingAllReduce:
    """Build the process's ring once (collective over ``group``)."""
    global _RING
    if _RING is None:
        _RING = CeRingAllReduce(group, capacity_bytes)
    return _RING


def ce_ring() -> CeRingAllReduce | None:
    return _RING
