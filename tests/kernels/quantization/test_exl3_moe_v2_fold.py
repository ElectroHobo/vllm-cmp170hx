# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 MoE v2 folded (VLLM_EXL3_MOE_V2_FOLD, csrc/exl3/moe_v2.cu FoldArgs):
moe_out_v2_fold runs the activation at the end of the gate/up lane launch and
the combine at the end of the down lane launch (completion counters), three
launches instead of five, and must return moe_out_v2's output bits.

CPU:
- the switch defaults off; Exl3Experts.apply takes moe_out_v2_fold for the
  calls moe_out_v2 serves only when the layer's experts are folded;
  Exl3MoEMethod._moe_v2 requires the operator and warms both arms;
- an exact emulation of the completion counting (groups_over_block and the
  expected counts) over the route tiling: every (row tile, H128 block),
  row tile and (token, H128 block) counter is completed by exactly one item,
  under random completion orders, for the served shapes and the
  tensor-split intermediate sizes 640 and 512;
- the extension registers moe_out_v2_fold;
- moe_v2.cu compiled for sm_80 (nvcc, no GPU): the fold instantiations of
  every served shape and bit width exist, and no lane kernel's GEMM loop
  touches local memory.

GPU (SM 8.0): moe_out_v2_fold's output bytes equal moe_out_v2's at hidden
5120 (bits_x2 6 and 4: the DeepSeek-V4.1 target layers; its drafter's
bits_x2 8 experts take moe_out, which the switch does not change) and
hidden 2560 (bits_x2 10, short and long K slices), for every row count 1..48
and 64 over routings with rows without local routes, hot experts (several
8-row tiles), masked and int64 ids and the reached SwiGLU clamp, with NaN in
the output and workspace before each call; calls repeat bitwise; a captured
call replays bitwise without allocator events; and the real experts of
layers 10 and 20 when VLLM_EXL3_TEST_CHECKPOINT names the checkpoint.
"""

import os
import random
import re
import shutil
import subprocess
from collections import Counter
from functools import partial
from pathlib import Path

import pytest
import torch

from tests.kernels.quantization import test_exl3_moe_v2 as h2560
from tests.kernels.quantization import test_exl3_moe_v2_hidden5120 as h5120
from vllm.sm80.kernels.exl3 import ops as exl3_ops

needs_sm80 = h2560.needs_sm80
ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------- CPU tests


def test_switch_defaults_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_EXL3_MOE_V2_FOLD", raising=False)
    assert envs.VLLM_EXL3_MOE_V2_FOLD is False
    monkeypatch.setenv("VLLM_EXL3_MOE_V2_FOLD", "1")
    assert envs.VLLM_EXL3_MOE_V2_FOLD is True


@pytest.mark.parametrize("fold", [False, True])
def test_apply_takes_the_folded_operator_for_the_v2_calls(monkeypatch, fold):
    calls = []
    for name in ("moe_out", "moe_out_v2", "moe_out_v2_fold"):
        monkeypatch.setattr(
            exl3_ops,
            name,
            lambda *args, _name=name, **kwargs: calls.append(
                (_name, args[0].shape[0], kwargs)
            ),
        )
    experts = h2560._experts(64)
    experts.moe_v2_fold = fold
    workspace2 = torch.zeros(8, dtype=torch.bfloat16)
    for rows in (1, 48, 64, 65):
        x = torch.zeros(rows, h2560.HIDDEN, dtype=torch.bfloat16)
        h2560._apply(experts, x, "ids", "weights", "ptrs", "meta", "out", workspace2)
    v2 = "moe_out_v2_fold" if fold else "moe_out_v2"
    assert [(name, rows) for name, rows, _ in calls] == [
        (v2, 1),
        (v2, 48),
        (v2, 64),
        ("moe_out", 65),
    ]
    for name, _, kwargs in calls[:3]:
        assert kwargs == {"swiglu_limit": 0.0, "bits_x2": h2560.BITS_X2}


def test_method_requires_and_warms_the_folded_operator(monkeypatch):
    warmed, required = [], []
    method, layer = h2560._method(monkeypatch, warmed)
    monkeypatch.setattr(
        exl3_ops,
        "warmup_moe_v2",
        lambda *args, **kwargs: warmed.append((args, kwargs)),
    )
    monkeypatch.setattr(exl3_ops, "require_moe_v2_fold", lambda: required.append(1))
    monkeypatch.delenv("VLLM_EXL3_MOE_V2_FOLD", raising=False)
    assert method._moe_v2(layer, h2560.SERVED_META, 3200, 7) == (64, 10, 64000)
    assert [kwargs for _, kwargs in warmed] == [{}] and not required
    warmed.clear()
    monkeypatch.setenv("VLLM_EXL3_MOE_V2_FOLD", "1")
    assert method._moe_v2(layer, h2560.SERVED_META, 3200, 7) == (64, 10, 64000)
    # Both arms warm (a dev A/B server switches between them), fold last.
    assert [kwargs for _, kwargs in warmed] == [{}, {"fold": True}]
    assert warmed[0][0] == warmed[1][0] and required == [1]


# Completion counting, mirrored from moe_v2.cu (FoldGateUp, FoldDown,
# fold_activate, groups_over_block): (hidden, intermediate, column tiles,
# gate/up slices, down slices) per launch configuration.
SHAPES = [
    (5120, 2304, 4, 16, 8),
    (2560, 640, 5, 8, 2),
    (2560, 640, 5, 4, 1),
    (5120, 640, 4, 16, 8),
    (5120, 512, 4, 16, 8),
]


def _groups_over_block(block, group_tiles, n_tiles):
    return min(8 * block + 7, n_tiles - 1) // group_tiles - 8 * block // group_tiles + 1


def _items(n_tiles, group_tiles, projections, slices):
    """Per item of one row tile: the H128 blocks it counts in."""
    for _ in range(projections * slices):
        for n0 in range(0, n_tiles, group_tiles):
            yield list(range(n0 // 8, (n0 + group_tiles - 1) // 8 + 1))


def _last_once(counts, expected, order):
    """Count `order` (counter keys) atomically: the arrivals that see the
    expected count minus one, per counter."""
    seen, last = Counter(), Counter()
    for key in order:
        if seen[key] == expected[key] - 1:
            last[key] += 1
        seen[key] += 1
    assert seen == counts
    return last


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_completion_counters_fire_once_per_block(shape):
    hidden, intermediate, group_tiles, gate_up_slices, down_slices = shape
    topk, local, rows = 6, 96, 48
    rng = random.Random(hash(shape) & 0xFFFF)
    # Gate/up: one row tile's items over the intermediate's blocks.
    n_tiles = intermediate // 16
    row_blocks = intermediate // 128
    order = [b for item in _items(n_tiles, group_tiles, 2, gate_up_slices) for b in item]
    counts = Counter(order)
    expected = {
        b: 2 * gate_up_slices * _groups_over_block(b, group_tiles, n_tiles)
        for b in range(row_blocks)
    }
    assert counts == Counter(expected)
    rng.shuffle(order)
    assert _last_once(counts, expected, order) == Counter(range(row_blocks))
    # Down: route tiles of up to 8 rows per local expert (route_block), items
    # counting (token, block) per row.
    ids = [[rng.randrange(-1, 4 * local) for _ in range(topk)] for _ in range(rows)]
    ids[0] = [local + 1] * topk  # a token without a local route
    by_expert = {}
    for a in range(rows * topk):
        e = ids[a // topk][a % topk]
        if 0 <= e < local:
            by_expert.setdefault(e, []).append(a)
    tiles = [
        members[i : i + 8]
        for _, members in sorted(by_expert.items())
        for i in range(0, len(members), 8)
    ]
    n_tiles = hidden // 16
    routes = Counter(a // topk for tile in tiles for a in tile)
    order = [
        (a // topk, b)
        for tile in tiles
        for item in _items(n_tiles, group_tiles, 1, down_slices)
        for a in tile
        for b in item
    ]
    counts = Counter(order)
    expected = {
        (t, b): routes[t] * down_slices * _groups_over_block(b, group_tiles, n_tiles)
        for t in routes
        for b in range(hidden // 128)
    }
    assert counts == Counter(expected)
    rng.shuffle(order)
    assert _last_once(counts, expected, order) == Counter(expected.keys())
    assert 0 not in routes  # its row is the route launch's exact zero


def test_extension_registers_the_folded_operator():
    h2560._extension_or_skip()
    exl3_ops.require_moe_v2_fold()
    torch._C._dispatch_find_schema_or_throw("_exl3_C::moe_out_v2_fold", "")


# The lane kernels (fold and not) of every served shape and bit width.
LANE_KERNELS = {
    f"moe_lane_kernel<{nt}, {kt}, {bits}{fold}>"
    for nt, kts, bits_list, folds in (
        (4, (20,), (6, 4), (", NoFold", ", FoldGateUp<16, 18>")),
        (4, (18,), (6, 4), (", NoFold", ", FoldDown<8>")),
        (5, (20,), (10,), (", NoFold", ", FoldGateUp<8, 5>", ", FoldDown<2>")),
        (5, (40,), (10,), (", NoFold", ", FoldGateUp<4, 5>", ", FoldDown<1>")),
    )
    for kt in kts
    for bits in bits_list
    for fold in folds
}


@pytest.fixture(scope="module")
def sm80_sass():
    """moe_v2.cu as an sm_80 cubin (build_standalone.py's CUDA flags), SASS
    per demangled kernel name."""
    from torch.utils import cpp_extension

    home = cpp_extension.CUDA_HOME
    nvcc = home and Path(home, "bin", "nvcc")
    cuobjdump = home and Path(home, "bin", "cuobjdump")
    if not (nvcc and nvcc.exists() and cuobjdump.exists() and shutil.which("c++filt")):
        pytest.skip("needs the CUDA toolkit (nvcc, cuobjdump) and c++filt")
    tmp = Path(os.environ.get("TMPDIR") or "/tmp")
    cubin = tmp / f"moe_v2_sm80_{os.getpid()}.cubin"
    includes = []
    for path in cpp_extension.include_paths(device_type="cuda"):
        includes += ["-isystem", path]
    import sysconfig

    command = [
        str(nvcc), "-cubin", "-arch=sm_80", "-O3", "-std=c++20", "-DNDEBUG",
        "-DUSE_CUDA", "-DPy_LIMITED_API=3", "-DTORCH_EXTENSION_NAME=_exl3_C",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        "-I", str(ROOT / "csrc"), "-I", str(ROOT / "csrc" / "exl3"), *includes,
        "-isystem", sysconfig.get_paths()["include"],
        "-o", str(cubin), str(ROOT / "csrc" / "exl3" / "moe_v2.cu"),
    ]  # fmt: skip
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        sass = subprocess.run(
            [str(cuobjdump), "-sass", str(cubin)],
            check=True, capture_output=True, text=True,
        ).stdout  # fmt: skip
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"{exc.cmd[0]} failed:\n{exc.stderr[-4000:]}")
    finally:
        cubin.unlink(missing_ok=True)
    functions, name = {}, None
    mangled = re.findall(r"Function : (\S+)", sass)
    demangled = subprocess.run(
        ["c++filt"], input="\n".join(mangled), capture_output=True, text=True,
        check=True,
    ).stdout.split("\n")  # fmt: skip
    names = dict(zip(mangled, demangled))
    for line in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            name = names[m.group(1)].replace("exl3::moe_v2::(anonymous namespace)::", "")
            functions[name] = []
            continue
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;", line)
        if name and m:
            functions[name].append((int(m.group(1), 16), m.group(2)))
    return functions


def _gemm_loops(instructions):
    """The innermost loops (backward branches) that hold an HMMA."""
    loops = []
    for address, text in instructions:
        m = re.search(r"\bBRA\b.*?0x([0-9a-f]+)\s*$", text)
        if m and int(m.group(1), 16) < address:
            begin = int(m.group(1), 16)
            body = [t for a, t in instructions if begin <= a <= address]
            if any(t.startswith("HMMA") or " HMMA" in t for t in body):
                loops.append((begin, address, body))
    return [
        (b, e, body)
        for b, e, body in loops
        if not any(b <= b2 and e2 <= e and (b2, e2) != (b, e) for b2, e2, _ in loops)
    ]


def test_sm80_build_has_every_fold_kernel_and_spill_free_gemm_loops(sm80_sass):
    lane = {}
    for name, instructions in sm80_sass.items():
        if "moe_lane_kernel<" in name:
            key = re.sub(r"\(.*", "", name).replace("void ", "")
            lane[re.sub(r"\s+>", ">", key)] = instructions
    assert LANE_KERNELS <= set(lane), sorted(LANE_KERNELS - set(lane))
    routes = [n for n in sm80_sass if "route_and_gate_up_input<" in n and ", true>" in n]
    assert len(routes) == 4, routes  # (bits_x2, scale group) of each shape
    # The kernel body's GEMM loops (lane_tiles inlined: aligned trellis
    # bases, every served layer) stay free of local memory with the fold
    # epilogue; out-of-line callees (lane_tiles_unaligned, the fold stages:
    # from a CALL target to its RET) are not the body.
    for name in LANE_KERNELS:
        instructions = lane[name]
        # (A call to the next instruction is a convergence idiom, not a callee.)
        targets = {
            int(m.group(1), 16)
            for address, text in instructions
            if (m := re.search(r"\bCALL\.\S*\s+0x([0-9a-f]+)", text))
            and int(m.group(1), 16) > address + 0x20
        }
        callees = [
            (t, next(a for a, x in instructions if a >= t and x.startswith("RET")))
            for t in targets
        ]
        loops = [
            (b, e, body)
            for b, e, body in _gemm_loops(instructions)
            if not any(t <= b and e <= r for t, r in callees)
        ]
        assert len(loops) >= 2, name  # one per row-group count (1 and 2)
        for begin, end, body in loops:
            local = [t for t in body if re.search(r"\b(LDL|STL)\b", t)]
            assert not local, (name, hex(begin), hex(end), local[:4])

# ---------------------------------------------------------------- GPU tests

ROWS = (*range(1, 49), 64)


def _fold_5120(experts, x, ids, weights):
    return h5120._run(
        partial(exl3_ops.moe_out_v2_fold, bits_x2=experts.bits_x2),
        experts,
        x,
        ids,
        weights,
        exl3_ops.moe_workspace_size_v2,
    )


def _fold_2560(served, x, ids, weights):
    return h2560._run(
        partial(exl3_ops.moe_out_v2_fold, bits_x2=h2560.BITS_X2),
        served,
        x,
        ids,
        weights,
        exl3_ops.moe_workspace_size_v2,
    )


@pytest.fixture(scope="module", params=h5120.BITS, ids=lambda b: f"bits_x2={b}")
def synthetic(request):
    experts = h5120._synthetic(request.param)
    exl3_ops.require_moe_v2_fold()
    yield experts
    experts.release()


@pytest.fixture(scope="module")
def served():
    served = h2560.served_experts(h2560.UNALIGNED_SET)
    exl3_ops.require_moe_v2_fold()
    return served


SCENARIOS_5120 = ["uniform", "local", "masked", "nonlocal", "hot", "single", "int64"]
SCENARIOS_2560 = ["uniform", "masked", "nonlocal", "hot", "single", "int64"]


@needs_sm80
@pytest.mark.parametrize("scenario", SCENARIOS_5120)
def test_fold_is_bitwise_v2_hidden5120(synthetic, scenario):
    for rows in ROWS:
        for scale in (1.0, 16.0) if scenario in ("uniform", "local") else (1.0,):
            x, ids, w = h5120._case(scenario, rows, 7 * rows + 1, scale)
            v2 = h5120._v2(synthetic, x, ids, w)
            fold = _fold_5120(synthetic, x, ids, w)
            h2560._bitwise(fold, v2, f"{scenario} rows={rows} scale={scale}")
            assert not torch.isnan(fold).any()


@needs_sm80
@pytest.mark.parametrize("scenario", SCENARIOS_2560)
def test_fold_is_bitwise_v2_hidden2560(served, scenario):
    # 1..15 rows: short K slices (< 160 routed assignments); 16..64: long.
    for rows in ROWS:
        x, ids, w = h2560._case(served, scenario, rows, 7 * rows + 2)
        v2 = h2560._v2(served, x, ids, w)
        fold = _fold_2560(served, x, ids, w)
        h2560._bitwise(fold, v2, f"{scenario} rows={rows}")
        assert not torch.isnan(fold).any()


@needs_sm80
def test_fold_repeats_bitwise_with_a_reused_workspace(synthetic):
    """The counters are zeroed per call: a workspace left by calls of other
    sizes (and garbage) gives the same bits."""
    rows = 48
    workspace = torch.full(
        (exl3_ops.moe_workspace_size_v2(64, h5120.HIDDEN, h5120.INTERMEDIATE,
                                        h5120.TOPK, h5120.LOCAL),),
        float("nan"),
        device="cuda",
    )  # fmt: skip
    first = None
    for count in (rows, 64, 1, rows, rows):
        x, ids, w = h5120._case("uniform", count, 99 if count == rows else count)
        out = torch.full_like(x, float("nan"))
        exl3_ops.moe_out_v2_fold(
            x, ids, w, synthetic.pointers, synthetic.meta, synthetic.expert_map,
            h5120.INTERMEDIATE, out, workspace, swiglu_limit=h5120.SWIGLU_LIMIT,
            bits_x2=synthetic.bits_x2,
        )  # fmt: skip
        if count == rows:
            if first is None:
                first = out.clone()
                h2560._bitwise(out, h5120._v2(synthetic, x, ids, w), "fold vs v2")
            h2560._bitwise(out, first, "repeat")
        workspace.view(torch.int32)[::7] = -12345  # garbage between calls


@needs_sm80
@pytest.mark.parametrize("rows", [6, 48])
def test_fold_graph_replay_is_bitwise_without_allocations(synthetic, rows):
    exl3_ops.warmup_moe_v2(
        synthetic.pointers,
        synthetic.meta,
        synthetic.expert_map,
        list(range(h5120.LOCAL)),
        h5120.HIDDEN,
        h5120.INTERMEDIATE,
        h5120.TOPK,
        synthetic.bits_x2,
        h5120.SWIGLU_LIMIT,
        fold=True,
    )
    x, ids, weights = h5120._case("uniform", rows, 11)
    out = torch.empty_like(x)
    workspace = torch.empty(
        exl3_ops.moe_workspace_size_v2(
            rows, h5120.HIDDEN, h5120.INTERMEDIATE, h5120.TOPK, h5120.LOCAL
        ),
        device=x.device,
    )

    def call():
        exl3_ops.moe_out_v2_fold(
            x, ids, weights, synthetic.pointers, synthetic.meta, synthetic.expert_map,
            h5120.INTERMEDIATE, out, workspace, swiglu_limit=h5120.SWIGLU_LIMIT,
            bits_x2=synthetic.bits_x2,
        )  # fmt: skip

    torch.cuda.synchronize()
    before = h2560._allocations()
    call()
    torch.cuda.synchronize()
    assert h2560._allocations() == before, "eager call allocated"
    h2560._bitwise(out, h5120._v2(synthetic, x, ids, weights), "eager fold vs v2")
    eager = out.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    empty = torch.cuda.CUDAGraph()
    before = h2560._allocations()
    with torch.cuda.graph(empty, stream=stream):
        pass
    calibration = h2560._allocations() - before
    graph = torch.cuda.CUDAGraph()
    before = h2560._allocations()
    with torch.cuda.graph(graph, stream=stream):
        call()
    assert h2560._allocations() - before <= calibration, "capture allocated"
    out.fill_(float("nan"))
    workspace.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    h2560._bitwise(out, eager, "replay vs eager")
    x2, ids2, weights2 = h5120._case("masked", rows, 12)
    x.copy_(x2)
    ids.copy_(ids2)
    weights.copy_(weights2)
    graph.replay()
    torch.cuda.synchronize()
    h2560._bitwise(out, h5120._v2(synthetic, x2, ids2, weights2), "replay new inputs")


@needs_sm80
@pytest.mark.parametrize("layer", sorted(h5120.REAL_LAYERS))
def test_fold_is_bitwise_v2_on_real_layer_weights(layer):
    experts, norm_weight = h5120._real_layer(layer, h5120.REAL_LAYERS[layer])
    exl3_ops.require_moe_v2_fold()
    for rows in ROWS:
        generator = torch.Generator().manual_seed(1000 * layer + rows)
        ids, weights = h5120._routes("uniform", rows, generator)
        h = torch.randn(rows, h5120.HIDDEN, generator=generator, dtype=torch.float64)
        h = h / h.pow(2).mean(-1, keepdim=True).sqrt()
        x = (h.cuda().float() * norm_weight).to(torch.bfloat16)
        ids, weights = ids.cuda(), weights.cuda()
        h2560._bitwise(
            _fold_5120(experts, x, ids, weights),
            h5120._v2(experts, x, ids, weights),
            f"layer {layer} rows={rows}",
        )
    experts.release()
