# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph memory estimate as the bytes the real capture keeps
(VLLM_CUDAGRAPH_ESTIMATE_SPLIT).

vLLM's estimate (``cudagraph_utils.profile_cudagraph_memory``) is the drop in
free device memory over one profiling ``capture_model()``. That capture is the
first time attention and the capture warmups run at all, so the drop also holds
one-time costs (allocator cache of the warmup transients, lazily created
buffers and driver state) that the real capture, after the KV cache is sized,
does not pay again. With the flag set, the profiling run is split:

* ``pool``: bytes of the segments of the throwaway graph pool at the end of the
  capture -- what the real capture's pool holds for the same capture sizes;
* ``objects``: device memory outside the torch allocator that the teardown of
  the profiling graphs gives back (the instantiated graphs);
* ``buffers``: torch blocks outside the pool created during the capture and
  live at its end (persistent buffers a capture creates, per-stream
  workspaces); a real capture may create them again;
* ``held``: the drop in free memory from before the minimal KV cache is built
  to after the teardown -- whatever profiling leaves allocated for good
  (one-time driver state, buffers that outlive the teardown);
* ``cache``: the allocator cache the capture warmups left, released after the
  capture (vLLM counts it; it is free memory).

The estimate is ``pool + objects + buffers + held`` plus vLLM's own FULL-graph
extrapolation term, unchanged. ``buffers`` that outlive the teardown are
counted twice (in ``buffers`` and ``held``), so a buffer the real capture
creates again is covered either way. After the real capture the allocator
cache is released and what the capture keeps is logged against the estimate.

The worker and model runner subclasses that call in here are in
``graph_mem_worker``; the fork platform selects that worker only with the flag
set, so without it nothing here runs.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_MiB = float(1 << 20)

UPSTREAM_WORKER = "vllm.v1.worker.gpu_worker.Worker"
WORKER = "vllm.sm80.worker.graph_mem_worker.GraphMemWorker"


def enabled() -> bool:
    return bool(envs.VLLM_CUDAGRAPH_ESTIMATE_SPLIT)


def select_worker(parallel_config: Any) -> None:
    """With the flag, run the fork worker in place of upstream's GPU worker
    (called by the fork platform's ``check_and_update_config``)."""
    if not enabled():
        return
    if parallel_config.worker_cls == UPSTREAM_WORKER:
        parallel_config.worker_cls = WORKER
    elif parallel_config.worker_cls != WORKER:
        logger.warning(
            "VLLM_CUDAGRAPH_ESTIMATE_SPLIT=1 is set but the worker is %s, not "
            "vLLM's GPU worker; the CUDA-graph memory estimate stays vLLM's.",
            parallel_config.worker_cls,
        )


@dataclass
class _Window:
    """Readings of one profiling capture."""

    start_blocks: frozenset[tuple[int, int]] | None = None
    measured: int | None = None
    pool: int = 0
    buffers: int = 0
    cache: int = 0
    free_end: int = 0
    reserved_end: int = 0


_window: _Window | None = None
_estimate: int | None = None


def _free() -> int:
    return int(torch.accelerator.get_memory_info()[0])


def _reserved() -> int:
    return int(torch.accelerator.memory_reserved())


def _release() -> int:
    """Synchronize, empty the allocator cache; return the bytes it freed."""
    torch.accelerator.synchronize()
    before = _free()
    torch.accelerator.empty_cache()
    return max(_free() - before, 0)


def _segments() -> list[dict[str, Any]]:
    device = torch.accelerator.current_device_index()
    return [
        seg
        for seg in torch.cuda.memory_snapshot(include_traces=False)
        if seg.get("device", device) == device
    ]


def _graph_pool() -> Any:
    """The global graph pool; the profiling run points it at its throwaway
    pool for the length of its capture."""
    from vllm.platforms import current_platform

    return current_platform.get_global_graph_pool()


def _in_pool(segment: dict[str, Any], pool: Any) -> bool:
    return tuple(segment.get("segment_pool_id", ())) == tuple(pool)


def _active_blocks(
    segments: list[dict[str, Any]], pool: Any = None
) -> dict[tuple[int, int], int]:
    """Active blocks ``(address, size) -> size`` outside ``pool``."""
    out: dict[tuple[int, int], int] = {}
    for seg in segments:
        if pool is not None and _in_pool(seg, pool):
            continue
        for block in seg.get("blocks", ()):
            if block.get("state") != "inactive":
                size = int(block["size"])
                out[(int(block["address"]), size)] = size
    return out


def _window_start(w: _Window) -> None:
    torch.accelerator.synchronize()
    w.start_blocks = frozenset(_active_blocks(_segments()))


def _window_end(w: _Window, pool: Any, measured: int) -> None:
    assert w.start_blocks is not None
    w.cache += _release()
    w.free_end = _free()
    w.reserved_end = _reserved()
    segments = _segments()
    w.pool = sum(int(seg["total_size"]) for seg in segments if _in_pool(seg, pool))
    w.buffers = sum(
        size
        for key, size in _active_blocks(segments, pool).items()
        if key not in w.start_blocks
    )
    w.measured = int(measured)


def capture(capture_fn: Callable[..., int], profile_only: bool) -> int:
    """``capture_fn(profile_only=...)`` (vLLM's ``capture_model``). Inside the
    profiling run, take the readings around it; for the real capture, release
    the allocator cache after it and log what it keeps against the estimate.
    Returns vLLM's value unchanged."""
    w = _window
    if profile_only:
        if w is None:
            return capture_fn(profile_only=True)
        _window_start(w)
        measured = int(capture_fn(profile_only=True))
        _window_end(w, _graph_pool(), measured)
        return measured
    _release()
    before = _free()
    result = capture_fn(profile_only=False)
    cache = _release()
    kept = before - _free()
    if _estimate is None:
        logger.info(
            "CUDA graph capture keeps %.1f MiB after releasing %.1f MiB of "
            "allocator cache (vLLM reports %.1f MiB); no estimate was made",
            kept / _MiB,
            cache / _MiB,
            result / _MiB,
        )
        return result
    log = logger.info if kept <= _estimate else logger.warning
    log(
        "CUDA graph capture keeps %.1f MiB after releasing %.1f MiB of allocator "
        "cache (vLLM reports %.1f MiB); estimate %.1f MiB: %s",
        kept / _MiB,
        cache / _MiB,
        result / _MiB,
        _estimate / _MiB,
        "within the reserve" if kept <= _estimate else "OVER the reserve",
    )
    return result


def profile(profile_fn: Callable[[Any], int], runner: Any) -> int:
    """``profile_fn(runner)`` (vLLM's profiling run, which calls
    ``runner.capture_model(profile_only=True)`` and so :func:`capture`); the
    bytes the real capture keeps instead of the profiling capture's drop."""
    global _window, _estimate
    gc.collect()
    _release()
    free0 = _free()
    _window = window = _Window()
    try:
        upstream = int(profile_fn(runner))
    finally:
        _window = None
    if window.measured is None:
        # No capture ran (no graphs to capture): vLLM's value stands.
        _estimate = upstream
        return upstream
    _release()
    free1 = _free()
    reserved1 = _reserved()
    held = max(free0 - free1, 0)
    teardown_freed = free1 - window.free_end
    objects = max(teardown_freed - (window.reserved_end - reserved1), 0)
    # vLLM's FULL-graph extrapolation (0 without sampled FULL graphs).
    full_extra = max(upstream - window.measured, 0)
    estimate = window.pool + objects + window.buffers + held + full_extra
    logger.info(
        "CUDA graph memory estimate split: graph pool %.1f MiB, graph objects "
        "%.1f MiB, capture buffers %.1f MiB, held after teardown (cold "
        "one-time) %.1f MiB, FULL extrapolation %.1f MiB; estimate %.1f MiB; "
        "vLLM's estimate %.1f MiB, given back %.1f MiB (allocator cache "
        "%.1f MiB)",
        window.pool / _MiB,
        objects / _MiB,
        window.buffers / _MiB,
        held / _MiB,
        full_extra / _MiB,
        estimate / _MiB,
        upstream / _MiB,
        (upstream - estimate) / _MiB,
        window.cache / _MiB,
    )
    _estimate = estimate
    return estimate
