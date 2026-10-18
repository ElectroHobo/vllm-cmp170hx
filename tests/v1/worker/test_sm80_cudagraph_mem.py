# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the CUDA-graph memory estimate split
(vllm/sm80/worker/cudagraph_mem.py and graph_mem_worker.py,
VLLM_CUDAGRAPH_ESTIMATE_SPLIT).

A fake device stands in for the allocator: free and reserved bytes, the
allocator cache that empty_cache() returns, and the segments a memory
snapshot lists. Each test plays one profiling run through the fork model
runner and vLLM's own ``profile_cudagraph_memory`` (bootstrap, capture and
teardown faked as in test_gpu_model_runner_v2_cudagraph_profiling.py) and
checks the estimate against what the real capture keeps.
"""

import contextlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.envs as envs
from vllm.config.compilation import CUDAGraphMode
from vllm.sm80.worker import cudagraph_mem, graph_mem_worker
from vllm.v1.worker.gpu import cudagraph_utils as cgu
from vllm.v1.worker.gpu import model_runner as mrv2
from vllm.v1.worker.gpu.eplb_utils import EPLBController

MiB = 1 << 20
POOL = (7, 0)
GLOBAL_POOL = (1, 0)
FLAG = "VLLM_CUDAGRAPH_ESTIMATE_SPLIT"


@dataclass
class FakeDevice:
    free: int = 10_000 * MiB
    reserved: int = 0
    cache: int = 0
    segments: list[dict[str, Any]] = field(default_factory=list)
    next_address: int = 1 << 32

    def alloc(self, size: int, pool: tuple[int, int] = (0, 0)) -> int:
        """A new segment with one active block of ``size``."""
        address = self.next_address
        self.next_address += size + MiB
        self.segments.append(
            {
                "device": 0,
                "address": address,
                "total_size": size,
                "segment_pool_id": pool,
                "blocks": [
                    {"address": address, "size": size, "state": "active_allocated"}
                ],
            }
        )
        self.free -= size
        self.reserved += size
        return address

    def release(self, address: int) -> None:
        seg = next(s for s in self.segments if s["address"] == address)
        self.segments.remove(seg)
        self.free += seg["total_size"]
        self.reserved -= seg["total_size"]

    def cached(self, size: int) -> None:
        """Freed blocks the allocator keeps (released by empty_cache)."""
        self.free -= size
        self.reserved += size
        self.cache += size

    def empty_cache(self) -> None:
        self.free += self.cache
        self.reserved -= self.cache
        self.cache = 0


@dataclass
class Capture:
    """What one capture does on the fake device."""

    pool: int = 150 * MiB
    objects: int = 10 * MiB  # graph objects outside the torch allocator
    cache: int = 900 * MiB  # warmup transients left in the allocator cache
    buffer: int = 0  # a torch buffer created by the capture
    buffer_outlives_teardown: bool = False
    held: int = 0  # one-time cost outside torch, never given back
    pool_leaks: bool = False  # the pool segment survives the teardown
    samples: list[int] = field(default_factory=list)  # FULL-graph samples
    num_full: int = 0


class _FakePlatform:
    _global_graph_pool: Any = GLOBAL_POOL

    @staticmethod
    def graph_pool_handle() -> Any:
        return POOL

    def get_global_graph_pool(self) -> Any:
        return type(self)._global_graph_pool


class _FakeManager:
    def __init__(self, cap: Capture) -> None:
        self.pool: Any = GLOBAL_POOL
        descs = [object() for _ in range(cap.num_full)]
        self._capture_descs = (
            {CUDAGraphMode.FULL: descs}
            if cap.num_full
            else {CUDAGraphMode.PIECEWISE: [object()]}
        )
        self._max_full_descs_to_capture: int | None = None
        self._capture_mem_samples: list[int] | None = None
        self.use_breakable_cg = False

    def needs_capture(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    monkeypatch.setattr(cudagraph_mem, "_window", None)
    monkeypatch.setattr(cudagraph_mem, "_estimate", None)


def _setup(monkeypatch, dev: FakeDevice, cap: Capture) -> tuple[Any, dict]:
    """The fork V2 runner whose profiling and real captures play ``cap`` on
    ``dev``."""
    monkeypatch.setattr(cudagraph_mem, "_free", lambda: dev.free)
    monkeypatch.setattr(cudagraph_mem, "_reserved", lambda: dev.reserved)
    monkeypatch.setattr(cudagraph_mem, "_segments", lambda: list(dev.segments))
    monkeypatch.setattr(
        cudagraph_mem, "_graph_pool", lambda: _FakePlatform._global_graph_pool
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", dev.empty_cache)

    @contextlib.contextmanager
    def _no_config(_cfg):
        yield

    _FakePlatform._global_graph_pool = GLOBAL_POOL
    monkeypatch.setattr(cgu, "set_current_vllm_config", _no_config)
    monkeypatch.setattr(cgu, "current_platform", _FakePlatform())

    cls = graph_mem_worker.runner_cls()
    runner: Any = cls.__new__(cls)
    runner.eplb = EPLBController(
        SimpleNamespace(enable_eplb=False), torch.device("cpu")
    )
    runner.compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.PIECEWISE
    )
    runner.cudagraph_manager = _FakeManager(cap)
    runner.vllm_config = SimpleNamespace()
    state: dict[str, Any] = {}

    def init(_r) -> None:
        state["kv"] = dev.alloc(64 * MiB)  # the minimal KV cache

    def capture_model(self, *, profile_only: bool = False) -> int:
        """vLLM's capture_model: empties the cache, then the free drop."""
        dev.empty_cache()
        start = dev.free
        state["pool_seen"] = _FakePlatform._global_graph_pool
        pool = POOL if profile_only else (9, 0)
        state["pool"] = dev.alloc(cap.pool, pool)
        dev.free -= cap.objects
        if profile_only:
            dev.free -= cap.held  # one-time: the real capture is warm
        dev.cached(cap.cache)
        if cap.buffer:
            state["buffer"] = dev.alloc(cap.buffer)
        samples = runner.cudagraph_manager._capture_mem_samples
        if profile_only and samples is not None:
            samples.extend(cap.samples)
        return start - dev.free

    def teardown(_r) -> None:
        dev.free += cap.objects
        if not cap.pool_leaks:
            dev.release(state["pool"])
        dev.release(state["kv"])
        if cap.buffer and not cap.buffer_outlives_teardown:
            dev.release(state["buffer"])
        dev.empty_cache()

    monkeypatch.setattr(mrv2.GPUModelRunner, "capture_model", capture_model)
    monkeypatch.setattr(cgu, "_init_minimal_kv_cache_for_profiling", init)
    monkeypatch.setattr(cgu, "_teardown_profiling_state", teardown)
    return runner, state


def _vllm_estimate(cap: Capture) -> int:
    """vLLM's estimate: the capture's whole drop, cache included."""
    total = cap.pool + cap.objects + cap.cache + cap.buffer + cap.held
    if cap.samples:
        total += cgu._extrapolate_full_graph_memory(cap.samples, cap.num_full)
        total -= sum(cap.samples)
    return total


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert envs.VLLM_CUDAGRAPH_ESTIMATE_SPLIT is False
    assert not cudagraph_mem.enabled()


@pytest.mark.parametrize("flag", ["0", "1"])
def test_select_worker(monkeypatch, flag):
    monkeypatch.setenv(FLAG, flag)
    parallel = SimpleNamespace(worker_cls=cudagraph_mem.UPSTREAM_WORKER)
    cudagraph_mem.select_worker(parallel)
    expected = cudagraph_mem.WORKER if flag == "1" else cudagraph_mem.UPSTREAM_WORKER
    assert parallel.worker_cls == expected
    # The qualname resolves to the fork worker.
    assert (
        f"{graph_mem_worker.GraphMemWorker.__module__}."
        f"{graph_mem_worker.GraphMemWorker.__qualname__}" == cudagraph_mem.WORKER
    )


def test_select_worker_keeps_a_custom_worker(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    parallel = SimpleNamespace(worker_cls="my.Worker")
    cudagraph_mem.select_worker(parallel)
    assert parallel.worker_cls == "my.Worker"


@pytest.mark.parametrize(
    ("v2", "encoder_only", "fork_runner"),
    [(True, False, True), (False, False, False), (True, True, False)],
)
def test_worker_builds_fork_runner_for_v2_only(monkeypatch, v2, encoder_only, fork_runner):
    class _Runner:
        def __init__(self, *args) -> None:
            pass

    monkeypatch.setattr(graph_mem_worker, "runner_cls", lambda: _Runner)
    monkeypatch.setattr(
        graph_mem_worker.Worker, "_make_model_runner", lambda self: "upstream"
    )
    worker: Any = graph_mem_worker.GraphMemWorker.__new__(
        graph_mem_worker.GraphMemWorker
    )
    worker.use_v2_model_runner = v2
    worker.vllm_config = SimpleNamespace(is_mm_encoder_only=encoder_only)
    worker.device = torch.device("cpu")
    runner = worker._make_model_runner()
    assert isinstance(runner, _Runner) == fork_runner
    assert (runner == "upstream") == (not fork_runner)


def test_cold_costs_given_back(monkeypatch):
    """The cache of the first capture is not reserved: pool + objects."""
    dev = FakeDevice()
    cap = Capture()
    runner, state = _setup(monkeypatch, dev, cap)
    free_before = dev.free

    estimate = runner.profile_cudagraph_memory()

    assert estimate == cap.pool + cap.objects
    assert _vllm_estimate(cap) - estimate == cap.cache
    # The capture saw the throwaway pool; profiling left nothing behind.
    assert state["pool_seen"] == POOL
    assert dev.free == free_before and dev.cache == 0


def test_cold_cost_held_after_teardown_is_reserved(monkeypatch):
    dev = FakeDevice()
    cap = Capture(held=700 * MiB)
    runner, _ = _setup(monkeypatch, dev, cap)

    assert runner.profile_cudagraph_memory() == cap.pool + cap.objects + cap.held


def test_capture_buffer_is_reserved(monkeypatch):
    """A buffer the capture creates and the teardown frees may be created
    again by the real capture: it is reserved."""
    dev = FakeDevice()
    cap = Capture(buffer=32 * MiB)
    runner, _ = _setup(monkeypatch, dev, cap)

    assert runner.profile_cudagraph_memory() == cap.pool + cap.objects + cap.buffer


def test_buffer_outliving_teardown_counted_twice(monkeypatch):
    dev = FakeDevice()
    cap = Capture(buffer=32 * MiB, buffer_outlives_teardown=True)
    runner, _ = _setup(monkeypatch, dev, cap)

    # Once as a capture buffer, once as held after the teardown.
    assert (
        runner.profile_cudagraph_memory() == cap.pool + cap.objects + 2 * cap.buffer
    )


def test_leaked_pool_counted_on_top_of_new_pool(monkeypatch):
    """A throwaway pool the teardown cannot free stays allocated next to the
    real capture's pool: both are reserved."""
    dev = FakeDevice()
    cap = Capture(pool_leaks=True)
    runner, _ = _setup(monkeypatch, dev, cap)

    assert runner.profile_cudagraph_memory() == 2 * cap.pool + cap.objects


def test_full_extrapolation_kept(monkeypatch):
    """vLLM's FULL-graph extrapolation term is added unchanged."""
    dev = FakeDevice()
    cap = Capture(samples=[100 * MiB, 20 * MiB], num_full=5)
    runner, _ = _setup(monkeypatch, dev, cap)

    full_extra = cgu._extrapolate_full_graph_memory(cap.samples, cap.num_full) - sum(
        cap.samples
    )
    assert full_extra == 3 * 20 * MiB
    assert runner.profile_cudagraph_memory() == cap.pool + cap.objects + full_extra


@pytest.mark.parametrize("buffer_outlives_teardown", [False, True])
def test_real_capture_within_reserve(
    monkeypatch, caplog_vllm, buffer_outlives_teardown
):
    """The real capture after profiling keeps no more than the estimate, and
    says so; its allocator cache is released."""
    dev = FakeDevice()
    cap = Capture(
        buffer=8 * MiB,
        held=300 * MiB,
        buffer_outlives_teardown=buffer_outlives_teardown,
    )
    runner, _ = _setup(monkeypatch, dev, cap)
    free_before = dev.free

    estimate = runner.profile_cudagraph_memory()
    reported = runner.capture_model()

    assert reported == cap.pool + cap.objects + cap.cache + cap.buffer
    assert dev.cache == 0
    assert free_before - dev.free <= estimate
    assert "within the reserve" in caplog_vllm.text
    assert "OVER the reserve" not in caplog_vllm.text


def test_real_capture_over_reserve_warns(monkeypatch, caplog_vllm):
    dev = FakeDevice()
    cap = Capture()
    runner, _ = _setup(monkeypatch, dev, cap)
    runner.profile_cudagraph_memory()
    cudagraph_mem._estimate = cap.pool  # pretend the estimate missed the objects

    runner.capture_model()

    assert "OVER the reserve" in caplog_vllm.text


def test_no_capture_returns_vllm_value(monkeypatch):
    dev = FakeDevice()
    runner, _ = _setup(monkeypatch, dev, Capture())
    monkeypatch.setattr(mrv2, "_profile_cudagraph_memory", lambda r: 0)

    assert runner.profile_cudagraph_memory() == 0
    assert cudagraph_mem._window is None


def test_window_cleared_on_error(monkeypatch):
    dev = FakeDevice()
    _setup(monkeypatch, dev, Capture())

    def boom(_r):
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError):
        cudagraph_mem.profile(boom, None)
    assert cudagraph_mem._window is None


def test_split_logged(monkeypatch, caplog_vllm):
    dev = FakeDevice()
    runner, _ = _setup(monkeypatch, dev, Capture())

    runner.profile_cudagraph_memory()

    assert (
        "CUDA graph memory estimate split: graph pool 150.0 MiB, graph objects "
        "10.0 MiB, capture buffers 0.0 MiB, held after teardown (cold one-time) "
        "0.0 MiB, FULL extrapolation 0.0 MiB; estimate 160.0 MiB; vLLM's "
        "estimate 1060.0 MiB, given back 900.0 MiB (allocator cache 900.0 MiB)"
    ) in caplog_vllm.text
