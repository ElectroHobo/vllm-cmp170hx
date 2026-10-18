# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's GPU worker with the CUDA-graph memory estimate split
(VLLM_CUDAGRAPH_ESTIMATE_SPLIT, ``cudagraph_mem``).

The fork platform names :class:`GraphMemWorker` as the worker only with the
flag set. Its model runner is vLLM's V2 runner with
``profile_cudagraph_memory`` and ``capture_model`` routed through
``cudagraph_mem`` (which also has the profiling capture capture every FULL
graph); everything else is upstream's. Other runners (V1, the
encoder-only runner) are built unchanged and keep vLLM's estimate.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.sm80.worker import cudagraph_mem
from vllm.v1.worker.gpu_worker import Worker

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


@functools.cache
def runner_cls() -> type:
    """vLLM's V2 model runner with the estimate split (imported on first use,
    as ``Worker._make_model_runner`` imports the runner)."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2

    class GraphMemModelRunner(GPUModelRunnerV2):
        def profile_cudagraph_memory(self) -> int:
            return cudagraph_mem.profile(
                GPUModelRunnerV2.profile_cudagraph_memory, self
            )

        def capture_model(self, *, profile_only: bool = False) -> int:
            if profile_only:
                cudagraph_mem.measure_full_graphs(self.cudagraph_manager)
            return cudagraph_mem.capture(super().capture_model, profile_only)

    return GraphMemModelRunner


class GraphMemWorker(Worker):
    def _make_model_runner(self) -> GPUModelRunner:
        if self.use_v2_model_runner and not self.vllm_config.is_mm_encoder_only:
            return runner_cls()(self.vllm_config, self.device)  # type: ignore[return-value]
        logger.warning_once(
            "VLLM_CUDAGRAPH_ESTIMATE_SPLIT=1 applies to the V2 model runner "
            "only; this worker keeps vLLM's CUDA-graph memory estimate."
        )
        return super()._make_model_runner()
