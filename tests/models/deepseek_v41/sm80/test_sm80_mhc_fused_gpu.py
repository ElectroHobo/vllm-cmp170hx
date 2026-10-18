# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused sm_80 DeepSeek-V4.1 delayed mHC (``ops/sm80_mhc_fused.py``) on one
sm_80 card, at the model's shapes (hidden 5120, hc 4, 20 Sinkhorn iterations,
eps 1e-6, RMS eps 1e-20).

- FP64 rule (V2): every output of the fused call and of the pre-only call,
  with and without a carried pre-mix, against FP64 without intermediate
  rounding and at the reference's rounding points
  (``fp64_cases.mhc_delayed_fp64``): mean error at most 1.10x and max error
  at most 1.25x the torch formulas' (``ops/sm80_mhc.py``, on the same card),
  at every graph capture size and two prefill sizes, each case pooling
  independent calls up to 64 tokens (as a captured case pools the calls of
  its shape); rows run alone give the bits of the whole call within one
  kernel A form (up to 8 rows).
- Bitwise run-to-run determinism, CUDA graph replay equal to eager.
- Two kernels per call (three with ``capture_aux``).

Synthetic inputs; the real-input check is the ``mhc_delayed`` /
``mhc_pre_delayed`` families of ``fp64_cases.py`` on captured calls.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_mhc_fused_gpu.py
"""

import pytest
import torch

from . import fp64_cases as F
from .test_sm80_mhc_fused import call, variants

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="needs an sm_80 card",
)

HIDDEN = 5120
CUDA = torch.device("cuda")
# the served graph capture sizes (verify steps of DSpark up to 48 rows) and two
# prefill sizes
ROWS = (1, 2, 3, 4, 6, 8, 12, 16, 18, 24, 30, 36, 42, 48, 256, 2048)
VARIANTS = ("fused", "fused_identity", "pre_only", "pre_only_identity")


def _outputs(variant: str) -> list[str]:
    names = ["post", "comb", "layer_input", "pre"]
    out = names + [n + ".rp" for n in names]
    return (["residual"] if variant.startswith("fused") else []) + out


@pytest.fixture(scope="module")
def fused():
    from vllm.models.deepseek_v41.nvidia.ops import sm80_mhc_fused

    return sm80_mhc_fused


def _cuda(a: dict) -> dict:
    return {k: (v.to(CUDA) if isinstance(v, torch.Tensor) else v) for k, v in a.items()}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("T", ROWS)
def test_fp64_within_v2_of_the_torch_formulas(T, variant):
    family = F.mhc_delayed if variant.startswith("fused") else F.mhc_pre_delayed
    calls = [
        {k: v for k, v in variants(T, HIDDEN, seed)[variant].items() if v is not None}
        for seed in range(max(1, 64 // T))
    ]
    for output in _outputs(variant):
        cs = cm = rs = rm = 0.0
        n = 0
        for a in calls:
            c = family(a, CUDA, output)
            s, mx, k = F.abs_errors(c.cand, c.exact)
            cs, cm, n = cs + s, max(cm, mx), n + k
            s, mx, _ = F.abs_errors(c.ref, c.exact)
            rs, rm = rs + s, max(rm, mx)
            assert k == c.cand.numel() > 0, output
            if T <= 8:
                assert c.row_invariant, output
        assert cs / n <= 1.10 * (rs / n), (output, cs / n, rs / n)
        assert cm <= 1.25 * rm, (output, cm, rm)


@pytest.mark.parametrize("T", (1, 6, 48))
def test_deterministic_and_graph_replay_equal(T, fused):
    for variant in ("fused", "pre_only"):
        a = _cuda(variants(T, HIDDEN)[variant])
        first = call(fused, a)
        torch.cuda.synchronize()
        again = call(fused, a)
        for k in first:
            assert F.bits_equal(first[k], again[k]), (variant, k)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            captured = call(fused, a)
        g.replay()
        torch.cuda.synchronize()
        for k in first:
            assert F.bits_equal(first[k], captured[k]), (variant, k)


def _kernels(fn) -> int:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(
        1
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA and "memcpy" not in e.name.lower()
        and "memset" not in e.name.lower()
    )


@pytest.mark.parametrize("T", (1, 48))
def test_two_kernels_per_call(T, fused):
    a = _cuda(variants(T, HIDDEN)["fused"])
    common = [a[k] for k in ("fn", "hc_scale", "hc_base", "rms_eps", "hc_pre_eps",
                             "hc_sinkhorn_eps", "hc_post_mult_value", "sinkhorn_repeat")]
    kw = dict(pre_mix=a["pre_mix"], norm_weight=a["norm_weight"], norm_eps=a["norm_eps"])

    def post_pre(aux=False):
        return fused.mhc_fused_post_pre_delayed(
            a["x"], a["residual"], a["post_layer_mix"], a["comb_res_mix"], *common,
            capture_aux=aux, **kw)

    assert _kernels(post_pre) == 2
    assert _kernels(lambda: post_pre(aux=True)) == 3
    assert _kernels(lambda: fused.mhc_pre_delayed(a["residual"], *common, **kw)) == 2
