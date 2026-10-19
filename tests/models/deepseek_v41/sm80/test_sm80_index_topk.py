# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses

import pytest
import torch

from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core
from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_pipeline as pipe
from vllm.models.deepseek_v41.nvidia.sm80_pipeline_capture import snapshot_cache


def config(**kwargs):
    cfg = core.Sm80AttnConfig(16, 512, 64, 2, 128, 2, True, True, True, False,
                            32, 128, 512, 2048, 8, 4, 1e-6)
    return dataclasses.replace(cfg, **kwargs)


def test_complete_workspace_bound_and_static_shape_gate():
    cfg = config()
    need = pipe.PipelineWorkspace.bytes(48, 262144, 1280)
    assert pipe.supports(cfg, 48, 262144, 1280, need) is None
    assert pipe.supports(cfg, 48, 262144, 1280, need - 1)
    assert pipe.supports(cfg, 49, 262144, 1280, 1 << 30)
    assert pipe.supports(config(compress_ratio=4), 48, 262144, 1280, 1 << 30)
    assert pipe.supports(config(candidate_block_size=16), 48, 262144, 1280, 1 << 30)


def test_pipeline_and_pre_rope_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match='excludes pre-RoPE'):
        core.prepare_layer(None, None, None, None, None, None, None,
                           fused=True, pipeline=True, workspace=object())
    batch = type('Batch', (), {'num_tokens': 0})()
    with pytest.raises(ValueError, match='excludes pre-RoPE'):
        core.attend_layer(None, None, None, batch, None, None,
                          fused=True, pipeline=True, workspace=object())


def test_cache_snapshot_preserves_aliases_and_negative_slots():
    cache = torch.arange(6 * 4 * 8).view(6, 4, 8)
    table = torch.tensor([[2, 4], [2, 5]], dtype=torch.int32)
    slots = torch.tensor([8, 9, 21, -1])
    indices = torch.tensor([[8, 9, -1], [8, 21, -1]], dtype=torch.int32)
    copied, ids, bt, ss, ii = snapshot_cache(cache, table=table, slots=slots, indices=indices)
    assert bt[0, 0] == bt[1, 0]
    assert ss[0] == ii[0, 0] == ii[1, 0]
    assert ss[-1] == -1 and bool((ii[:, -1] == -1).all())
    for s, mapped in zip(slots[:-1], ss[:-1]):
        assert torch.equal(cache[s // 4, s % 4], copied[mapped // 4, mapped % 4])
    assert bool((ids == 0).any())


def test_index_select_family_rejects_nonfinite_visible_arithmetic():
    from .attn_pipeline_cases import index_select_case
    from .fp64_cases import Source
    a = dict(scores=torch.tensor([[float('nan'), 0.0]]), n_vis=torch.tensor([1]), k=512)
    with pytest.raises(ValueError, match='nonfinite arithmetic'):
        index_select_case(a, torch.device('cpu'), Source({}, {}))


def test_index_select_family_exact_ties_and_padding():
    from .attn_pipeline_cases import index_select_case
    from .fp64_cases import Source
    a = dict(scores=torch.tensor([[0., -0., -core.FLT_MAX, 1.], [9., 9., 9., 9.]]),
             n_vis=torch.tensor([4, 0]), k=3)
    call = index_select_case(a, torch.device('cpu'), Source({}, {}))
    assert call.row_invariant
    assert torch.equal(call.cand, torch.tensor([[0, 1, 3], [-1, -1, -1]], dtype=torch.int32))
    assert torch.equal(call.cand.double(), call.exact)
