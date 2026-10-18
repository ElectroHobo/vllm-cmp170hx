# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculation off under load: DSpark drafts 5 below 4 scheduled requests and
nothing from 4 on (``num_speculative_tokens_per_batch_size``).

Covers the three pieces that make a zero-draft step plain decode: the
scheduler's choice of k, the decode CUDA graphs both regimes need, and the
drafter writing only its context K/V (no draft block) when k is 0.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import CompilationConfig, CUDAGraphMode, ParallelConfig
from vllm.config import SchedulerConfig, VllmConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_module
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

pytestmark = pytest.mark.cpu_test

NUM_SPEC = 5
MAX_NUM_SEQS = 8
SCHEDULE = [(1, 3, NUM_SPEC), (4, MAX_NUM_SEQS, 0)]
CAPTURE_SIZES = [4, 5, 6, 7, 8, 12, 18]


def _scheduler() -> Scheduler:
    base = create_scheduler(
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_SEQS * 10,
        num_speculative_tokens=NUM_SPEC,
    )
    base.vllm_config.speculative_config.num_speculative_tokens_per_batch_size = (
        SCHEDULE
    )
    return Scheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=base.kv_cache_config,
        block_size=base.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base.vllm_config),
    )


@pytest.mark.parametrize("num_requests", range(1, MAX_NUM_SEQS + 1))
def test_drafts_five_below_four_requests_and_none_from_four(num_requests):
    scheduler = _scheduler()
    for request in create_requests(num_requests=num_requests, num_tokens=10):
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert len(output.num_scheduled_tokens) == num_requests
    expected = NUM_SPEC if num_requests < 4 else 0
    assert output.num_spec_tokens_to_schedule == expected


def _graph_manager(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=CAPTURE_SIZES,
    )
    compilation_config.max_cudagraph_capture_size = CAPTURE_SIZES[-1]
    compilation_config.post_init_cudagraph_sizes()
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=MAX_NUM_SEQS
    )
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.num_speculative_tokens = NUM_SPEC
    speculative_config = MagicMock()
    speculative_config.uses_dynamic_speculative_decoding.return_value = True
    speculative_config.num_speculative_tokens_per_batch_size = SCHEDULE
    speculative_config.uses_adaptive_k.return_value = False
    vllm_config.speculative_config = speculative_config
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_len=NUM_SPEC + 1,
    )
    manager._graphs_captured = True
    return manager


def test_capture_sizes_cover_both_regimes(monkeypatch):
    manager = _graph_manager(monkeypatch)
    # Drafting (1-3 requests): 6 tokens per request, graphs 6, 12, 18.
    for num_reqs in (1, 2, 3):
        desc = manager.dispatch(
            num_reqs=num_reqs,
            num_tokens=num_reqs * (NUM_SPEC + 1),
            uniform_token_count=NUM_SPEC + 1,
            num_active_loras=0,
        )
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert desc.num_tokens == num_reqs * (NUM_SPEC + 1)
    # Plain decode (4-8 requests): 1 token per request, graphs 4..8.
    for num_reqs in range(4, MAX_NUM_SEQS + 1):
        desc = manager.dispatch(
            num_reqs=num_reqs,
            num_tokens=num_reqs,
            uniform_token_count=1,
            num_active_loras=0,
        )
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert desc.num_tokens == num_reqs


class _Stop(Exception):
    pass


def _dspark(monkeypatch, calls):
    """A DSpark drafter with host state only; every device step is recorded."""
    s = object.__new__(DSparkSpeculator)
    max_reqs, max_tokens = MAX_NUM_SEQS, 64
    s.max_width = s.num_speculative_steps = s.width = NUM_SPEC
    s.widths = (NUM_SPEC,)
    s.draft_tokens = torch.full((max_reqs, NUM_SPEC), 7, dtype=torch.int64)
    s._draft_token_views = {NUM_SPEC: s.draft_tokens}
    s.sample_col = torch.zeros(max_reqs * NUM_SPEC, dtype=torch.int32)
    s._sample_cols = {NUM_SPEC: s.sample_col}
    s._cg_managers = {}
    s.query_cudagraph_manager = None
    s.model = SimpleNamespace()
    s.num_query_per_req = NUM_SPEC
    s.max_model_len = 4096
    s.max_num_reqs = max_reqs
    s.max_num_tokens = max_tokens
    s.hidden_states = torch.zeros(max_tokens, 4)
    s.pcp_manager = None
    s.draft_kv_cache_group_id = 0
    s.draft_kv_cache_group_ids = [0]
    s.block_tables = SimpleNamespace(
        slot_mappings=[None],
        input_block_tables=[None],
        block_sizes=[16],
        cp_rank=0,
        cp_interleave=1,
    )
    s.input_buffers = s.context_positions = s._context_slot_mappings = [None]
    s.sample_indices = s.sample_pos = s.sample_idx_mapping = None
    s.temperature = s.seeds = None
    s.dcp_size = 1
    s.dp_size = 1
    s.dp_rank = 0
    s.parallel_drafting_token_id = 0
    s.sample_from_anchor = True

    monkeypatch.setattr(
        dflash_module, "prepare_dflash_inputs", lambda *a: calls.append("prepare")
    )
    s._precompute_context_kv = lambda start, end, dummy_run=False: calls.append(
        ("context_kv", start, end)
    )

    def dispatch(*args, **kwargs):
        calls.append("draft_block")
        raise _Stop

    monkeypatch.setattr(dflash_module, "dispatch_cg_and_sync_dp", dispatch)
    s._generate_draft = lambda *a, **k: calls.append("draft_block")
    return s


def _propose(s, num_reqs, num_speculative_tokens):
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_tokens=num_reqs,
        seq_lens_cpu_upper_bound=np.full(num_reqs, 100),
    )
    zeros = torch.zeros(num_reqs, dtype=torch.int32)
    return s.propose(
        batch,
        {},
        {},
        torch.zeros(num_reqs, 4),
        None,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        num_speculative_tokens=num_speculative_tokens,
    )


def test_zero_drafts_store_context_kv_and_skip_the_draft_block(monkeypatch):
    calls: list = []
    s = _dspark(monkeypatch, calls)
    out = _propose(s, num_reqs=6, num_speculative_tokens=0)
    assert calls == ["prepare", ("context_kv", 0, 6)]
    assert out.shape == (6, NUM_SPEC)


def test_drafting_steps_run_the_draft_block(monkeypatch):
    calls: list = []
    s = _dspark(monkeypatch, calls)
    with pytest.raises(_Stop):
        _propose(s, num_reqs=3, num_speculative_tokens=NUM_SPEC)
    assert calls == ["prepare", "draft_block"]


def test_data_parallel_keeps_the_full_step(monkeypatch):
    calls: list = []
    s = _dspark(monkeypatch, calls)
    s.dp_size = 2
    with pytest.raises(_Stop):
        _propose(s, num_reqs=6, num_speculative_tokens=0)
    assert calls == ["prepare", "draft_block"]
