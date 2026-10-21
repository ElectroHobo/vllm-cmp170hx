# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the copy-engine ring all-reduce (vllm/sm80/distributed/ce_ring_all_reduce.py):
the ring schedule, the host emulation of its bits, and ``ring_ops`` run for every rank against a
host model of the shared memory, with every copy's content checked where it lands.

    pytest -q tests/sm80/test_ce_ring_all_reduce.py
"""

import pytest
import torch

from vllm.sm80.distributed import ce_ring_all_reduce as ce

DT = {torch.float32: 4, torch.bfloat16: 2}


@pytest.mark.parametrize("world", [2, 3, 4, 8])
def test_ring_schedule_reduces_every_shard_once_and_gathers_all(world) -> None:
    for rank in range(world):
        steps = ce.ring_steps(world, rank)
        assert len(steps) == 2 * (world - 1)
        rs = [s for s in steps if s.reduce]
        ag = [s for s in steps if not s.reduce]
        # Reduce-scatter: every shard but this rank's own is received once.
        assert sorted(s.recv for s in rs) == sorted(set(range(world)) - {rank})
        # The shard sent at step k+1 is the one received at step k.
        for a, b in zip(steps, steps[1:]):
            assert b.send == a.recv
        assert steps[0].send == rank
        # All-gather: the owned shard (rank + 1) goes out first; every other shard comes in once.
        assert ag[0].send == (rank + 1) % world
        assert sorted(s.recv for s in ag) == sorted(set(range(world)) - {(rank + 1) % world})
        # What this rank receives at step k is what the previous rank sends then.
        prev = ce.ring_steps(world, (rank - 1) % world)
        assert [s.recv for s in steps] == [p.send for p in prev]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_emulation_is_the_sum(dtype) -> None:
    g = torch.Generator().manual_seed(1)
    parts = [torch.randn(4 * 96, generator=g).to(dtype) for _ in range(4)]
    want = torch.stack([p.double() for p in parts]).sum(0)
    got = ce.emulate(parts).double()
    tol = 1e-5 if dtype == torch.float32 else 0.06
    assert (got - want).abs().max().item() < tol


class _Memory:
    """Host model of the ranks' device memory: regions at disjoint fake addresses."""

    def __init__(self):
        self.regions = []

    def alloc(self, nbytes: int) -> int:
        base = (len(self.regions) + 1) << 40
        self.regions.append((base, torch.zeros(nbytes, dtype=torch.uint8)))
        return base

    def put(self, t: torch.Tensor) -> int:
        b = t.contiguous().view(torch.uint8).flatten()
        base = self.alloc(b.numel())
        self.at(base, b.numel()).copy_(b)
        return base

    def at(self, addr: int, nbytes: int) -> torch.Tensor:
        for base, buf in self.regions:
            if base <= addr and addr + nbytes <= base + buf.numel():
                return buf[addr - base:addr - base + nbytes]
        raise AssertionError(f"address {addr:#x}+{nbytes} outside every region")


class _HostBackend:
    """One rank's operations on the host model. Copies are checked byte for byte where they land."""

    def __init__(self, mem: _Memory, log: list):
        self.mem, self.log = mem, log

    def copy(self, dst, src, nbytes):
        data = self.mem.at(src, nbytes).clone()
        self.mem.at(dst, nbytes).copy_(data)
        assert torch.equal(self.mem.at(dst, nbytes), data)
        self.log.append(("copy", dst, data))

    def set_flag(self, ptr, value):
        self.mem.at(ptr, 4).view(torch.int32)[0] = value

    def wait_flag(self, ptr, value):
        while int(self.mem.at(ptr, 4).view(torch.int32)[0]) < value:
            yield

    def add(self, dst, a, b, n, dtype):
        size = DT[dtype] * n
        x = self.mem.at(a, size).view(dtype).float()
        y = self.mem.at(b, size).view(dtype).float()
        self.mem.at(dst, size).view(dtype).copy_((x + y).to(dtype))


def _run_ring(world, inputs_per_call, slot_bytes):
    mem = _Memory()
    steps = 2 * (world - 1)
    scratch = [mem.alloc(slot_bytes * steps) for _ in range(world)]
    flags = [mem.alloc(ce.flag_bytes(world)) for _ in range(world)]
    results = []
    logs = [[] for _ in range(world)]
    for epoch, inputs in enumerate(inputs_per_call, start=1):
        dtype = inputs[0].dtype
        shard_numel = inputs[0].numel() // world
        inp = [mem.put(t) for t in inputs]
        out = [mem.alloc(t.numel() * DT[dtype]) for t in inputs]
        gens = [ce.ring_ops(_HostBackend(mem, logs[r]), world, r, epoch, inp[r], out[r],
                            shard_numel * DT[dtype], shard_numel, dtype, scratch, flags, slot_bytes)
                for r in range(world)]
        live, spins = set(range(world)), 0
        while live:
            for r in sorted(live):
                try:
                    next(gens[r])
                except StopIteration:
                    live.discard(r)
            spins += 1
            assert spins < 10_000, "ring deadlocked"
        results.append([mem.at(out[r], inputs[r].numel() * DT[dtype]).view(dtype).clone()
                        for r in range(world)])
    return results, logs


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ring_ops_give_the_emulated_bits_on_every_rank(world, dtype) -> None:
    """Three calls back to back (fresh data, monotonic epochs, reused scratch slots): every rank ends
    with the emulation's bits, and every copy put exactly its source bytes where it landed."""
    g = torch.Generator().manual_seed(world)
    calls = [[torch.randn(world * 40, generator=g).to(dtype) for _ in range(world)] for _ in range(3)]
    slot_bytes = 256 * ((40 * 4 + 255) // 256)
    results, logs = _run_ring(world, calls, slot_bytes)
    for inputs, outs in zip(calls, results):
        want = ce.emulate(inputs)
        for o in outs:
            assert torch.equal(o, want)
    # Each call: 2(world-1) peer copies plus (world-1) all-gather local copies per rank.
    for log in logs:
        assert len(log) == 3 * (2 * (world - 1) + (world - 1))


def test_ring_ops_waits_for_the_ready_flag_before_writing() -> None:
    """A rank never copies into the next rank's scratch before that rank's ready store."""
    world, mem, events = 2, _Memory(), []
    scratch = [mem.alloc(256 * 2) for _ in range(world)]
    flags = [mem.alloc(ce.flag_bytes(world)) for _ in range(world)]

    class Logged(_HostBackend):
        def copy(self, dst, src, nbytes):
            events.append(("copy", dst))
            super().copy(dst, src, nbytes)

        def set_flag(self, ptr, value):
            events.append(("flag", ptr))
            super().set_flag(ptr, value)

    x = torch.ones(world * 4)
    inp = [mem.put(x) for _ in range(world)]
    out = [mem.alloc(x.numel() * 4) for _ in range(world)]
    g0 = ce.ring_ops(Logged(mem, []), world, 0, 1, inp[0], out[0], 8, 2, torch.float32,
                     scratch, flags, 256)
    next(g0)  # rank 0 has published its ready flag and is waiting on rank 1's
    ready = (2 * (world - 1)) * ce.FLAG_STRIDE_BYTES
    assert events == [("flag", flags[1] + ready)]


def test_takes_refuses_host_tensors() -> None:
    ring = object.__new__(ce.CeRingAllReduce)
    ring.world, ring.capacity_bytes = 4, 1 << 20
    assert not ring.takes(torch.zeros(1024))
