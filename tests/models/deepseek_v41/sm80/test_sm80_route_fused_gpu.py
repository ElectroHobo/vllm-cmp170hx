# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused sm_80 DeepSeek-V4.1 MoE routing (``ops/sm80_route.py``) on one
sm_80 card.

- Bytes: weights and int32 ids equal to ``sm80_moe.route`` + ``.to(int32)``
  on CUDA, for the target's router (384 experts, top 6) and the drafter's
  (128, top 3), at every decode graph size and two prefill sizes, with and
  without image tokens, renormalised or not, on logits spanning both
  softplus branches and exp's subnormal range.
- The CPU half's model of torch's CUDA row reduction
  (``test_sm80_route_fused.torch_cuda_row_sum``) is ``torch.sum`` on CUDA,
  bitwise, for rows of 1 to 16 values.
- CUDA graph replay equals eager; two runs are bitwise equal.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_route_fused_gpu.py
"""

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.nvidia.ops.sm80_route import route_fused
from vllm.models.deepseek_v41.nvidia.sm80_moe import image_token_mask, route

from .test_sm80_route_fused import IMAGE_IDS, ROUTERS, SENTINEL, torch_cuda_row_sum

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="needs an sm_80 card",
)

TOKENS = (1, 4, 5, 6, 7, 8, 12, 16, 18, 24, 28, 32, 40, 48, 2048, 2044)


def cuda_case(tokens, experts, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(tokens, experts, generator=g) * 4.0
    # Both softplus branches and exp's subnormal range.
    logits[:, :6] = torch.tensor([25.0, 20.0, 19.999, -88.0, -95.0, -103.0])
    bias = torch.randn(experts, generator=g) * 0.1
    bias_vl = torch.randn(experts, generator=g) * 0.1
    input_ids = torch.randint(0, 2000, (tokens,), generator=g, dtype=torch.int32)
    input_ids[::3] = SENTINEL + (torch.arange(0, tokens, 3) % IMAGE_IDS).int()
    return [t.cuda() for t in (logits, bias, bias_vl, input_ids)]


def both(logits, bias, bias_vl, input_ids, top_k, images, renorm):
    lo = SENTINEL if images else 0
    got = route_fused(logits, bias, top_k, bias_vl=bias_vl, input_ids=input_ids,
                      sentinel_lo=lo, sentinel_hi=lo + IMAGE_IDS, renormalize=renorm,
                      routed_scaling_factor=1.5)
    w, i = route(logits, bias, top_k, bias_vl=bias_vl,
                 image_mask=image_token_mask(input_ids, lo), renormalize=renorm,
                 routed_scaling_factor=1.5)
    return got, (w, i.to(torch.int32))


@pytest.mark.parametrize("experts,top_k", ROUTERS)
@pytest.mark.parametrize("images", (False, True))
@pytest.mark.parametrize("renorm", (True, False))
def test_bytes_equal_the_torch_path(experts, top_k, images, renorm):
    for tokens in TOKENS:
        args = cuda_case(tokens, experts, tokens * 7 + experts)
        (gw, gi), (ww, wi) = both(*args, top_k, images, renorm)
        assert gi.dtype == torch.int32 and gw.dtype == torch.float32
        assert torch.equal(gi, wi), tokens
        assert torch.equal(gw.view(torch.int32), ww.view(torch.int32)), tokens


def test_row_sum_model_is_torch_sum_on_cuda():
    g = torch.Generator().manual_seed(5)
    for k in range(1, 17):
        rows = (torch.randn(4096, k, generator=g) ** 2
                * 10.0 ** torch.randint(-3, 3, (4096, k), generator=g)).float()
        got = rows.cuda().sum(dim=-1).cpu().numpy()
        want = np.array([torch_cuda_row_sum(r) for r in rows.numpy()], dtype=np.float32)
        assert np.array_equal(got.view(np.int32), want.view(np.int32)), k


def test_graph_replay_and_determinism():
    for experts, top_k in ROUTERS:
        logits, bias, bias_vl, ids = cuda_case(6, experts, 11)
        call = lambda: route_fused(logits, bias, top_k, bias_vl=bias_vl, input_ids=ids,  # noqa: E731
                                   sentinel_lo=SENTINEL, sentinel_hi=SENTINEL + IMAGE_IDS,
                                   routed_scaling_factor=1.5)
        eager = call()
        again = call()
        assert all(torch.equal(a, b) for a, b in zip(eager, again))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call()
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(eager, captured))
