# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU checks of the sm_80 DeepSeek-V4.1 attention kernels (skip without CUDA).

* index scores (Triton) equal the torch definition bit for bit: FP4-qdq'd
  queries and keys make every dot product exact in fp32, so only the
  rounding points matter and those are the same;
* the top-k on CUDA (``torch.topk`` + the shared tie fix) equals the stable
  CPU definition on tie-heavy scores;
* sparse attention (Triton) matches the torch definition to BF16 rounding;
* both kernels address rows past 2**31 elements correctly (int64 offsets);
* the whole layer on CUDA passes the same reference comparison as on CPU.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_kernels_gpu.py
"""

import pytest
import torch

from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core
from vllm.models.deepseek_v41.nvidia.ops.sm80_act_quant import fp4_qdq

from . import test_attn_layers as layers
from .dsv41_ref import LayerSpec, RefArgs, make_weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

BF16 = torch.bfloat16


def _index_case(g: torch.Generator, n: int = 37, blocks: int = 60, entries: int = 64):
    idxk = fp4_qdq(torch.randn(blocks, entries, 128, generator=g), 32).to(BF16)
    q = fp4_qdq(torch.randn(n, 32, 128, generator=g) * 2, 32).to(BF16)
    w = (torch.randn(n, 32, generator=g) / 8).to(BF16)
    reqs = 3
    table = torch.stack([torch.randperm(blocks, generator=g) for _ in range(reqs)])
    req = torch.randint(0, reqs, (n,), generator=g)
    n_vis = torch.randint(0, blocks * entries, (n,), generator=g)
    n_vis[:3] = torch.tensor([0, 1, blocks * entries])
    return q, w, n_vis, req, idxk, table.to(torch.int32)


@pytest.mark.parametrize("heads_per_rank", [8, 32])
def test_index_scores_triton_bitwise(heads_per_rank: int) -> None:
    from vllm.models.deepseek_v41.nvidia.ops.sm80_index_scores import (
        index_scores_triton,
    )

    g = torch.Generator().manual_seed(1)
    q, w, n_vis, req, idxk, table = _index_case(g)
    n_cols = int(n_vis.max())
    want = core.index_scores_torch(q, w, n_vis, req, idxk, table, n_cols, heads_per_rank)
    got = index_scores_triton(
        q.cuda(), w.cuda(), n_vis.cuda(), req.cuda(), idxk.cuda(), table.cuda(),
        n_cols, heads_per_rank,
    ).cpu()
    assert torch.equal(got, want)


def test_topk_cuda_canonical_on_ties() -> None:
    g = torch.Generator().manual_seed(2)
    n, cols = 64, 3000
    # Few distinct values: the k-th score is tied across many positions.
    scores = torch.randint(-4, 5, (n, cols), generator=g).float() / 4
    n_vis = torch.randint(0, cols + 1, (n,), generator=g)
    n_vis[:2] = torch.tensor([cols, 5])
    scores[3, :100] = -core.FLT_MAX  # masked-but-visible positions
    for k in (512, 24):
        want = core.topk_canonical(scores, n_vis, k)
        got = core.topk_canonical_cuda(scores.cuda(), n_vis.cuda(), k).cpu()
        assert torch.equal(got, want), k


def _attn_case(g: torch.Generator, n: int = 29, h: int = 16):
    swa = torch.randn(40, 32, 512, generator=g).to(BF16)
    comp = torch.randn(50, 64, 512, generator=g).to(BF16)
    q = (torch.randn(n, h, 512, generator=g) * 0.3).to(BF16)
    swa_idx = torch.randint(0, 40 * 32, (n, 128), generator=g).int()
    swa_idx[:, 100:] = -1
    swa_idx[0] = -1  # no valid row at all
    table = torch.stack([torch.randperm(50, generator=g) for _ in range(2)]).int()
    req = torch.randint(0, 2, (n,), generator=g)
    local = torch.randint(0, 50 * 64, (n, 512), generator=g).int()
    local[:, 400:] = -1
    local[0] = -1
    sink = torch.randn(h, generator=g)
    return q, swa, swa_idx, comp, table, local, req, sink


@pytest.mark.parametrize("with_comp", [True, False])
def test_sparse_attention_triton(with_comp: bool) -> None:
    from vllm.models.deepseek_v41.nvidia.ops.sm80_sparse_attn import (
        sparse_attention_triton,
    )

    g = torch.Generator().manual_seed(3)
    q, swa, swa_idx, comp, table, local, req, sink = _attn_case(g)
    if not with_comp:
        local = None
    want = core.sparse_attention_torch(
        q, swa, swa_idx, comp, table, local, req, sink, 512**-0.5
    )
    cu = lambda t: None if t is None else t.cuda()  # noqa: E731
    got = sparse_attention_triton(
        cu(q), cu(swa), cu(swa_idx), cu(comp), cu(table), cu(local), cu(req),
        cu(sink), 512**-0.5,
    ).cpu()
    assert not got[0].any(), "a query without rows gives zeros"
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-3)


def _huge_view(rows: int, entries: int, dim: int, block_stride: int):
    """A [3, entries, dim] cache view whose block 2 starts past 2**31 elements."""
    need = 2 * block_stride + entries * dim
    free, _ = torch.cuda.mem_get_info()
    if free < need * 2 + (8 << 30):
        pytest.skip("not enough free GPU memory for a >2**31-element cache")
    storage = torch.zeros(need, dtype=BF16, device="cuda")
    return torch.as_strided(storage, (3, entries, dim), (block_stride, dim, 1))


def test_kernels_int64_offsets() -> None:
    from vllm.models.deepseek_v41.nvidia.ops.sm80_index_scores import (
        index_scores_triton,
    )
    from vllm.models.deepseek_v41.nvidia.ops.sm80_sparse_attn import (
        sparse_attention_triton,
    )

    g = torch.Generator().manual_seed(4)
    stride = 2**30 + 512
    # Index keys: block table points at physical block 2 (offset > 2**31).
    idxk = _huge_view(3, 4, 128, stride)
    keys = fp4_qdq(torch.randn(3, 4, 128, generator=g), 32).to(BF16).cuda()
    idxk[:] = keys
    q = fp4_qdq(torch.randn(2, 32, 128, generator=g), 32).to(BF16).cuda()
    w = (torch.randn(2, 32, generator=g) / 8).to(BF16).cuda()
    table = torch.tensor([[2, 0, 1]], dtype=torch.int32, device="cuda")
    n_vis = torch.tensor([12, 5], device="cuda")
    req = torch.zeros(2, dtype=torch.int64, device="cuda")
    got = index_scores_triton(q, w, n_vis, req, idxk, table, 12, 8)
    want = core.index_scores_torch(q, w, n_vis, req, idxk, table, 12, 8)
    assert torch.equal(got, want)
    del idxk
    torch.cuda.empty_cache()

    swa = _huge_view(3, 4, 512, stride)
    swa[:] = torch.randn(3, 4, 512, generator=g).to(BF16).cuda()
    qa = (torch.randn(2, 16, 512, generator=g) * 0.3).to(BF16).cuda()
    swa_idx = torch.tensor([[8, 9, 10, 11], [8, 1, -1, -1]], dtype=torch.int32).cuda()
    sink = torch.randn(16, generator=g).cuda()
    got = sparse_attention_triton(
        qa, swa, swa_idx, None, None, None, req, sink, 512**-0.5
    )
    want = core.sparse_attention_torch(
        qa, swa, swa_idx, None, None, None, req, sink, 512**-0.5
    )
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize(
    "specs",
    [
        [LayerSpec(0)],
        [LayerSpec(2, kv_source=True, index_source=True)],
        [
            LayerSpec(1, kv_source=True, index_source=True, candidate_source=True),
            LayerSpec(1),
            LayerSpec(1, index_source=True, uses_candidates=True),
        ],
    ],
)
def test_layer_on_cuda_matches_reference(specs) -> None:
    args = RefArgs()
    weights = [make_weights(args, s, 100 + i) for i, s in enumerate(specs)]
    xs = layers.make_inputs(args, len(specs), 5)
    schedule = layers.make_schedule(11)
    run, outs, topks = layers.run_sm80(args, specs, weights, xs, schedule, "cuda")
    ref_layers, ref_outs, ref_topks = layers.run_reference(args, specs, weights, xs)
    layers.check(args, specs, run, outs, topks, ref_layers, ref_outs, ref_topks)
