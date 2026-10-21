# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark verification row mask in the DeepSeek-V4.1 target MoE routing
(``sm80_moe.Sm80DeepseekV41Router``, ``sm80_moe.adopt_row_mask``,
``ops/sm80_route.py``; the shared state is
``vllm/sm80/spec_decode/dspark_row_mask.py``).

CPU:

- ``adopt_row_mask`` marks the routers of the layers it is given (the
  target's) only with ``VLLM_DSPARK_ROW_MASK``; routers it is not given (the
  DSpark drafter's) stay unmarked.
- torch path: a marked router gives every row whose ``row_live`` entry is 0
  expert id -1 and weight +0.0, and every other row the bytes of the unmasked
  routing; with every row live, and for an unmarked router whatever
  ``row_live`` holds, the bytes are the unmasked routing's.
- switch off: the routing runs exactly the ops of ``route`` and its cast.
- fused path: the router hands the state's ``row_live`` to ``route_fused``
  only when marked; the kernel in the Triton interpreter masks those rows and
  leaves every other row's bytes as without ``row_live``.
- a marked router refuses CUDA graph capture before the state exists (the
  graph would never read the mask).

GPU (sm_80; listed for the GPU run):

- the router on CUDA, fused and torch paths: equal bytes, masked rows -1/0;
  a captured routing reads ``row_live`` in place at replay;
- moe_out and moe_out_v2 at hidden 5120 (the served routed experts, rank 0
  of 4): every live row's output is the same bits when other rows of the
  call are masked (ids -1), and a masked row's routed output is exact zero.

    pytest -q tests/models/deepseek_v41/sm80/test_row_mask_moe.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[4]
ROUTERS = ((384, 6), (128, 3))
COSTS = {
    "schema": "dspark-row-mask-costs/1",
    "max_draft": 5,
    "by_batch": {"1": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]},
}
MAX_TOKENS = 64
K = 5
# Four requests of a K-draft verify step keeping 5, 0, 2 and 3 drafts, then
# two padding rows.
LIVE_LEN = (5, 0, 2, 3)
TOKENS = len(LIVE_LEN) * (K + 1) + 2

needs_sm80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="needs an sm_80 card",
)


def row_pattern(tokens: int = TOKENS) -> torch.Tensor:
    """int8 [MAX_TOKENS]: rows of LIVE_LEN's requests past their live length
    are 0; padding rows and rows past the call are 1."""
    live = torch.ones(MAX_TOKENS, dtype=torch.int8)
    for r, n in enumerate(LIVE_LEN):
        live[r * (K + 1) + n + 1 : (r + 1) * (K + 1)] = 0
    live[tokens:] = 1
    return live


def make_router(experts: int, top_k: int, seed: int):
    from vllm.models.deepseek_v41.nvidia.sm80_moe import Sm80DeepseekV41Router

    g = torch.Generator().manual_seed(seed)
    router = object.__new__(Sm80DeepseekV41Router)
    router.top_k = top_k
    router.e_score_correction_bias = torch.randn(experts, generator=g) * 0.1
    router.bias_vl = None
    router.image_sentinel_lo = 0
    router.renormalize = True
    router.routed_scaling_factor = 1.5
    router.gate_temp = 1.0
    return router


def layer_of(router):
    return SimpleNamespace(ffn=SimpleNamespace(experts=SimpleNamespace(router=router)))


def make_logits(tokens: int, experts: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(tokens, experts, generator=g) * 3.0
    logits[:, :4] = torch.tensor([25.0, -30.0, 19.99, 20.01])
    return logits.float().contiguous()


def unmasked(router, logits, indices_type=torch.int32):
    from vllm.models.deepseek_v41.nvidia.sm80_moe import route

    w, i = route(
        logits,
        router.e_score_correction_bias,
        router.top_k,
        renormalize=router.renormalize,
        routed_scaling_factor=router.routed_scaling_factor,
    )
    return w, i.to(indices_type)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype == torch.float32:
        return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))
    return a.dtype == b.dtype and torch.equal(a, b)


def _switch(monkeypatch, value: str) -> None:
    """Set VLLM_DSPARK_ROW_MASK in the environment and on ``vllm.envs``: a test
    elsewhere that monkeypatched the module attribute leaves a plain attribute
    behind, which then shadows the environment."""
    import vllm.envs as envs

    monkeypatch.setenv("VLLM_DSPARK_ROW_MASK", value)
    monkeypatch.setattr(envs, "VLLM_DSPARK_ROW_MASK", value, raising=False)


@pytest.fixture
def state(monkeypatch, tmp_path):
    from vllm.sm80.spec_decode import dspark_row_mask

    path = tmp_path / "costs.json"
    path.write_text(json.dumps(COSTS))
    _switch(monkeypatch, str(path))
    dspark_row_mask._reset_for_tests()
    s = dspark_row_mask.init_row_mask_state(
        max_num_reqs=8, max_num_tokens=MAX_TOKENS, max_draft=K, device=torch.device("cpu")
    )
    yield s
    dspark_row_mask._reset_for_tests()


@pytest.fixture
def switch_off(monkeypatch):
    from vllm.sm80.spec_decode import dspark_row_mask

    _switch(monkeypatch, "")
    dspark_row_mask._reset_for_tests()
    yield
    dspark_row_mask._reset_for_tests()


class _Ops(TorchDispatchMode):
    """The aten ops a call runs."""

    def __init__(self):
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


# ----------------------------------------------------------------- marking


def test_adopt_row_mask_marks_the_given_routers_only(state):
    from vllm.models.deepseek_v41.nvidia.sm80_moe import (
        Sm80DeepseekV41Router,
        adopt_row_mask,
    )

    assert Sm80DeepseekV41Router.row_mask is False
    target = [make_router(384, 6, s) for s in range(3)]
    drafter = [make_router(128, 3, s) for s in range(3)]
    adopt_row_mask(layer_of(r) for r in target)
    assert all(r.row_mask is True for r in target)
    assert all(r.row_mask is False for r in drafter)


def test_adopt_row_mask_refuses_another_router(state):
    from vllm.models.deepseek_v41.nvidia.sm80_moe import adopt_row_mask

    with pytest.raises(TypeError, match="router"):
        adopt_row_mask([layer_of(SimpleNamespace())])


def test_switch_off_marks_nothing(switch_off):
    from vllm.models.deepseek_v41.nvidia.sm80_moe import adopt_row_mask

    router = make_router(384, 6, 0)
    adopt_row_mask([layer_of(router)])
    assert router.row_mask is False
    assert "row_mask" not in vars(router)


# --------------------------------------------------------------- torch path


@pytest.mark.parametrize("experts,top_k", ROUTERS)
def test_target_router_masks_rows_and_keeps_live_rows_bitwise(state, experts, top_k):
    from vllm.models.deepseek_v41.nvidia.sm80_moe import adopt_row_mask

    router = make_router(experts, top_k, experts)
    adopt_row_mask([layer_of(router)])
    logits = make_logits(TOKENS, experts, experts + 1)
    hidden = torch.zeros(TOKENS, 8)
    want_w, want_i = unmasked(router, logits)

    # Every row live (graph capture, steps without a mask): unmasked bytes.
    w, i = router._compute_routing(hidden, logits, None)
    assert same_bits(w, want_w) and same_bits(i, want_i)

    state.row_live.copy_(row_pattern())
    state.row_live[TOKENS : TOKENS + 4] = 0  # past the call: never read
    live = state.row_live[:TOKENS] != 0
    assert int((~live).sum()) == sum(K - n for n in LIVE_LEN)
    for indices_type in (None, torch.int32, torch.int64):
        w, i = router._compute_routing(hidden, logits, indices_type)
        want_w, want_i = unmasked(router, logits, indices_type or torch.int32)
        assert i.dtype == want_i.dtype and w.dtype == torch.float32
        assert same_bits(w[live], want_w[live]) and same_bits(i[live], want_i[live])
        assert bool((i[~live] == -1).all())
        assert bool((w[~live].view(torch.int32) == 0).all())  # +0.0 exactly


@pytest.mark.parametrize("experts,top_k", ROUTERS)
def test_unmarked_router_ignores_the_mask(state, experts, top_k):
    router = make_router(experts, top_k, experts)  # a drafter layer's router
    logits = make_logits(TOKENS, experts, experts + 2)
    state.row_live.copy_(row_pattern())
    w, i = router._compute_routing(torch.zeros(TOKENS, 8), logits, None)
    want_w, want_i = unmasked(router, logits)
    assert same_bits(w, want_w) and same_bits(i, want_i)


def test_switch_off_runs_exactly_the_routing_ops(switch_off):
    from vllm.models.deepseek_v41.nvidia.sm80_moe import adopt_row_mask

    router = make_router(384, 6, 3)
    adopt_row_mask([layer_of(router)])
    logits = make_logits(TOKENS, 384, 4)
    hidden = torch.zeros(TOKENS, 8)
    with _Ops() as plain:
        unmasked(router, logits)
    with _Ops() as routed:
        router._compute_routing(hidden, logits, None)
    assert routed.ops == plain.ops


def test_marked_router_adds_the_mask_ops(state):
    # The op log above sees the mask: a marked router runs more ops.
    from vllm.models.deepseek_v41.nvidia.sm80_moe import adopt_row_mask

    router = make_router(384, 6, 3)
    adopt_row_mask([layer_of(router)])
    logits = make_logits(TOKENS, 384, 4)
    hidden = torch.zeros(TOKENS, 8)
    with _Ops() as plain:
        unmasked(router, logits)
    with _Ops() as routed:
        router._compute_routing(hidden, logits, None)
    assert routed.ops[: len(plain.ops)] == plain.ops
    assert any("masked_fill" in op for op in routed.ops[len(plain.ops) :])


def test_too_many_rows_for_the_mask_are_refused(state):
    from vllm.models.deepseek_v41.nvidia.sm80_moe import adopt_row_mask

    router = make_router(128, 3, 5)
    adopt_row_mask([layer_of(router)])
    logits = make_logits(MAX_TOKENS + 1, 128, 6)
    with pytest.raises(ValueError, match="row mask"):
        router._compute_routing(torch.zeros(MAX_TOKENS + 1, 8), logits, None)


def test_capture_without_the_state_is_refused(switch_off, monkeypatch):
    router = make_router(128, 3, 5)
    router.row_mask = True  # marked, but the state was never created
    logits = make_logits(4, 128, 6)
    hidden = torch.zeros(4, 8)
    # Eager: nothing to read, the routing is the unmasked one.
    w, i = router._compute_routing(hidden, logits, None)
    want_w, want_i = unmasked(router, logits)
    assert same_bits(w, want_w) and same_bits(i, want_i)
    # A CUDA graph captured now would never read the mask.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        router._row_live(4, is_cuda=True)


# --------------------------------------------------------------- fused path


def test_fused_path_gets_row_live_only_when_marked(state, monkeypatch):
    from vllm.models.deepseek_v41.nvidia.ops import sm80_route
    from vllm.models.deepseek_v41.nvidia.sm80_moe import adopt_row_mask

    seen = []

    def fake_route_fused(logits, bias, top_k, **kwargs):
        seen.append(kwargs.get("row_live", "absent"))
        n = logits.shape[0]
        return torch.zeros(n, top_k), torch.zeros(n, top_k, dtype=torch.int32)

    monkeypatch.setattr(sm80_route, "route_fused", fake_route_fused)
    router = make_router(384, 6, 7)
    router.fused_route = True
    # The fused path runs on CUDA logits only; the stand-in carries what the
    # router reads before calling route_fused.
    logits = SimpleNamespace(dtype=torch.float32, is_cuda=True, shape=(TOKENS, 384))
    router._compute_routing(None, logits, None)
    adopt_row_mask([layer_of(router)])
    router._compute_routing(None, logits, None)
    assert seen[0] is None
    assert seen[1] is state.row_live


def interpreted_row_mask_results() -> dict:
    """In the Triton interpreter: per case, whether the masked rows are -1/0
    and every other row has the bytes of the call without ``row_live``."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    from tests.models.deepseek_v41.sm80.test_sm80_route_fused import (
        IMAGE_IDS,
        SENTINEL,
        make_case,
    )
    from vllm.models.deepseek_v41.nvidia.ops.sm80_route import route_fused

    out = {}
    for experts, top_k in ROUTERS:
        for tokens in (1, 6, TOKENS):
            for images in (False, True):
                logits, bias, bias_vl, ids_in = make_case(tokens, experts, tokens * 3 + experts)
                lo = SENTINEL if images else 0
                if tokens == 1:
                    row_live = torch.ones(MAX_TOKENS, dtype=torch.int8)
                    row_live[0] = 0
                elif tokens == 6:
                    row_live = torch.ones(MAX_TOKENS, dtype=torch.int8)
                    row_live[3:6] = 0
                else:
                    row_live = row_pattern(tokens)
                row_live[tokens : tokens + 4] = 0  # past the call: never read

                def call(rows):
                    return route_fused(
                        logits, bias, top_k, bias_vl=bias_vl, input_ids=ids_in,
                        sentinel_lo=lo, sentinel_hi=lo + IMAGE_IDS,
                        routed_scaling_factor=1.5, row_live=rows,
                    )

                base_w, base_i = call(None)
                got_w, got_i = call(row_live)
                ones_w, ones_i = call(torch.ones(MAX_TOKENS, dtype=torch.int8))
                live = row_live[:tokens] != 0
                out[f"{experts}x{top_k}/T{tokens}/img{int(images)}"] = dict(
                    live_bitwise=same_bits(got_w[live], base_w[live])
                    and same_bits(got_i[live], base_i[live]),
                    masked_ids=bool((got_i[~live] == -1).all()),
                    masked_weights_zero=bool((got_w[~live].view(torch.int32) == 0).all()),
                    masked_rows=int((~live).sum()),
                    all_live_bitwise=same_bits(ones_w, base_w) and same_bits(ones_i, base_i),
                    dtypes=[str(got_w.dtype), str(got_i.dtype)],
                )
    return out


def test_interpreted_kernel_masks_rows_and_keeps_live_rows_bitwise():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = (
        "import json; from tests.models.deepseek_v41.sm80.test_row_mask_moe import "
        "interpreted_row_mask_results as f; print('RESULT ' + json.dumps(f()))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
        timeout=1800,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    results = json.loads(line[len("RESULT ") :])
    assert len(results) == len(ROUTERS) * 3 * 2
    for key, c in results.items():
        tokens = int(key.split("/T")[1].split("/")[0])
        want_masked = {1: 1, 6: 3}.get(tokens, sum(K - n for n in LIVE_LEN))
        assert c["masked_rows"] == want_masked, (key, c)
        assert c["live_bitwise"] and c["all_live_bitwise"], (key, c)
        assert c["masked_ids"] and c["masked_weights_zero"], (key, c)
        assert c["dtypes"] == ["torch.float32", "torch.int32"], (key, c)


# ---------------------------------------------------------------------- GPU


@pytest.fixture
def cuda_state(monkeypatch, tmp_path):
    from vllm.sm80.spec_decode import dspark_row_mask

    path = tmp_path / "costs.json"
    path.write_text(json.dumps(COSTS))
    _switch(monkeypatch, str(path))
    dspark_row_mask._reset_for_tests()
    s = dspark_row_mask.init_row_mask_state(
        max_num_reqs=8, max_num_tokens=MAX_TOKENS, max_draft=K, device=torch.device("cuda")
    )
    yield s
    dspark_row_mask._reset_for_tests()


@needs_sm80
@pytest.mark.parametrize("experts,top_k", ROUTERS)
def test_cuda_router_fused_and_torch_paths_mask_alike(cuda_state, experts, top_k):
    from vllm.models.deepseek_v41.nvidia.sm80_moe import adopt_row_mask

    router = make_router(experts, top_k, experts)
    router.e_score_correction_bias = router.e_score_correction_bias.cuda()
    adopt_row_mask([layer_of(router)])
    logits = make_logits(TOKENS, experts, experts + 9).cuda()
    hidden = torch.zeros(TOKENS, 8, device="cuda")
    want_w, want_i = unmasked(router, logits)
    cuda_state.row_live.copy_(row_pattern())
    live = cuda_state.row_live[:TOKENS] != 0
    results = []
    for fused in (False, True):
        router.fused_route = fused
        results.append(router._compute_routing(hidden, logits, None))
    (tw, ti), (fw, fi) = results
    assert same_bits(tw, fw) and same_bits(ti, fi)
    assert same_bits(fw[live], want_w[live]) and same_bits(fi[live], want_i[live])
    assert bool((fi[~live] == -1).all()) and bool((fw[~live].view(torch.int32) == 0).all())


@needs_sm80
@pytest.mark.parametrize("fused", (False, True))
def test_captured_routing_reads_row_live_in_place(cuda_state, fused):
    from vllm.models.deepseek_v41.nvidia.sm80_moe import adopt_row_mask

    router = make_router(384, 6, 11)
    router.e_score_correction_bias = router.e_score_correction_bias.cuda()
    router.fused_route = fused
    adopt_row_mask([layer_of(router)])
    logits = make_logits(TOKENS, 384, 12).cuda()
    hidden = torch.zeros(TOKENS, 8, device="cuda")
    want_w, want_i = unmasked(router, logits)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        router._compute_routing(hidden, logits, None)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w, i = router._compute_routing(hidden, logits, None)
    graph.replay()
    torch.cuda.synchronize()
    assert same_bits(w, want_w) and same_bits(i, want_i)
    cuda_state.row_live.copy_(row_pattern())
    graph.replay()
    torch.cuda.synchronize()
    live = cuda_state.row_live[:TOKENS] != 0
    assert same_bits(w[live], want_w[live]) and same_bits(i[live], want_i[live])
    assert bool((i[~live] == -1).all()) and bool((w[~live].view(torch.int32) == 0).all())
    cuda_state.row_live.fill_(1)
    graph.replay()
    torch.cuda.synchronize()
    assert same_bits(w, want_w) and same_bits(i, want_i)


@pytest.fixture(scope="module", params=(6, 4), ids=lambda b: f"bits_x2={b}")
def served_experts(request):
    from tests.kernels.quantization.test_exl3_moe_v2_hidden5120 import _synthetic

    experts = _synthetic(request.param)
    yield experts
    experts.release()


@needs_sm80
@pytest.mark.parametrize("op", ("moe_out", "moe_out_v2"))
@pytest.mark.parametrize("rows", (6, TOKENS - 2, 48))
def test_live_rows_do_not_depend_on_masked_rows(served_experts, op, rows):
    from tests.kernels.quantization.test_exl3_moe_v2_hidden5120 import _case, _v1, _v2

    run = _v1 if op == "moe_out" else _v2
    x, ids, weights = _case("uniform", rows, 1021 + rows)
    live = torch.ones(rows, dtype=torch.bool)
    for r in range(0, rows, K + 1):  # every request keeps (r // 6) % 6 drafts
        live[r + (r // (K + 1)) % (K + 1) + 1 : r + K + 1] = False
    live = live.cuda()
    assert bool((~live).any())
    whole = run(served_experts, x, ids, weights)
    m_ids = ids.masked_fill(~live.unsqueeze(1), -1)
    m_weights = weights.masked_fill(~live.unsqueeze(1), 0.0)
    masked = run(served_experts, x, m_ids, m_weights)
    assert same_bits(masked[live].view(torch.int16), whole[live].view(torch.int16))
    assert bool((masked[~live].view(torch.int16) == 0).all())
