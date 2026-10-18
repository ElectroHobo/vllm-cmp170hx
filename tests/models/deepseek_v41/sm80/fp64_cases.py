# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP64 errors of the DeepSeek-V4.1 sm_80 kernels on captured served calls.

    python tests/models/deepseek_v41/sm80/fp64_cases.py --cases <file> --out <jsonl>

Run from the tree root on one CUDA card (it refuses to run without CUDA).

Cases: a JSON object with schema ``served_cases/1`` (an export of captured
calls) or ``generated_cases/1``; each case has an ``id`` and
``captured_sources``, the files of the calls it pools. A source is a
``torch.save`` dict, read with ``weights_only=True``:

    {"family": <family>, "meta": {...}, "tensors": {argument: value}}

``tensors`` holds the arguments of one call of the family's capture point by
parameter name (tensors, numbers, strings, lists of strings). Paged buffers
may be compacted to the rows the call reads, with their slot or block-table
values renumbered: every family below reads them only through those values.

Output: a first line ``{"header": {torch, triton, cuda, gpu, sm_count}}``,
then one line per case, pooled over its sources:

    {"case": id, "family": family,
     "fp64": {"cand_mean", "cand_max", "ref_mean", "ref_max"},
     "bitwise": bool, "row_invariance": bool, "numel": int}

``cand`` is the sm_80 path as served (the capture point's function on CUDA),
``ref`` the reference path on the same inputs; each is measured as
``|x - FP64|`` over the elements whose FP64 value is finite, and a
non-finite FP64 value must be matched exactly (else the error is infinite).
The check: candidate mean error at most 1.10x and max error at most 1.25x
the reference path's, exact where the reference path is exact.
``bitwise``: the candidate equals the reference path bit for bit.
``row_invariance``: the first, middle and last rows (tokens; the first and
last request for the compressor; for moe_out_prefill, which takes no fewer
than MOE_PREFILL_MIN_ROWS rows, the first and last calls of that size) give
the same bits run alone as in the whole call. ``numel``: output elements.
A case whose sources cannot be run gets no line; the reason goes to stderr.

Families: capture point (arguments) -> candidate / reference path / FP64.

- ``sparse_attention``: ``sm80_attn_core.sparse_attention`` (q, swa_cache,
  swa_indices, comp_cache, comp_block_table, comp_local, req, sink, scale)
  -> the Triton kernel / the reference ``sparse_attn`` arithmetic
  (``ref_sparse_attn``) over the same rows / sink attention in FP64.
- ``index_scores``: ``sm80_attn_core.index_scores`` (q, w, n_vis, req, idxk,
  block_table, n_cols, heads_per_rank) -> the Triton kernel /
  ``index_scores_torch`` / ReLU'd per-head dot products times the head
  weights, summed over every head in FP64 (invisible columns are -inf).
- ``qdq_fp8``: ``sm80_attn_core.qdq_fp8`` (x, block) -> Triton /
  ``act_quant_qdq_fp8`` on CPU / the independent definition
  ``test_act_quant.ref_fp8``. Exact: both errors are zero.
- ``qdq_fp4``: ``sm80_attn_core.qdq_fp4`` (x, block, scale) -> Triton /
  ``fp4_qdq`` on CPU / ``test_act_quant.ref_fp4``. Exact.
- ``compress``: ``sm80_attn_core.compress``; tensors kv_score, norm_weight,
  ring (as before the call; absent at ratio 1), positions, token_to_req,
  query_start_loc, comp_slots, ring_slots (absent at ratio 1); numbers
  compress_ratio, head_dim, eps -> torch on CUDA / the same on CPU /
  softmax pooling and RMS norm in FP64. Both must close the same groups.
- ``index_keys``: ``sm80_attn_core.index_keys``; tensors wk_out (the
  ``weights.wk(latent)`` the path starts from), k_norm, freqs, group_pos;
  number eps -> CUDA (torch norm and RoPE, Triton FP4) / CPU / RMS norm and
  RoPE in FP64, then FP4 by ``test_act_quant.ref_fp4``.
- ``engram_dequant``: ``sm80_engram_dequant.dequant_engram_rows`` (values,
  scales) -> Triton / ``dequant_engram_rows_reference`` on CPU / the E4M3 x
  E8M0 product in FP64. Exact: bitwise.
- ``exl3_moe``: ``Exl3Experts.apply``; tensors x, topk_ids, topk_weights,
  expert_map, packed_meta; numbers intermediate_size, swiglu_limit; strings
  operator (the one the dispatch chose: moe_out, moe_out_v2 or
  moe_out_prefill), checkpoint (the EXL3 checkpoint directory) and matrices
  (per local expert its gate, up and down matrix names) -> the operator /
  moe_out / the routed experts in FP64 from the decoded checkpoint weights
  (``exl3_real_cases.moe_fp64`` with the SwiGLU clamp).
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]
SCHEMAS = ("served_cases/1", "generated_cases/1")
# Bound of the intermediate tensors of one FP64 or reference chunk.
CHUNK_BYTES = 1 << 30
CPU = torch.device("cpu")
BF16 = torch.bfloat16


@dataclass
class Call:
    """One source replayed: both paths' outputs, the FP64 value, and whether
    the candidate's rows run alone gave the bits of the whole call."""

    cand: torch.Tensor
    ref: torch.Tensor
    exact: torch.Tensor
    row_invariant: bool


def picks(n: int) -> list[int]:
    return sorted({0, n // 2, n - 1}) if n else []


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def abs_errors(out: torch.Tensor, exact: torch.Tensor) -> tuple[float, float, int]:
    """(sum, max, count) of ``|out - exact|`` over the finite FP64 values;
    infinite when a non-finite FP64 value is not matched exactly."""
    if out.shape != exact.shape:
        raise ValueError(f"output shape {tuple(out.shape)} != FP64 {tuple(exact.shape)}")
    out = out.double()
    exact = exact.to(out.device)
    finite = torch.isfinite(exact)
    o, e = out[~finite], exact[~finite]
    if not bool(((o == e) | (o.isnan() & e.isnan())).all()):
        return math.inf, math.inf, int(finite.sum())
    err = (out[finite] - exact[finite]).abs()
    if err.numel() == 0:
        return 0.0, 0.0, 0
    return float(err.sum()), float(err.max()), err.numel()


def _to(value, device):
    return value.to(device) if isinstance(value, torch.Tensor) else value


# ---------------------------------------------------------------- attention


def sparse_attention(a: dict, device: torch.device) -> Call:
    from tests.models.deepseek_v41.sm80.dsv41_ref import ref_sparse_attn
    from tests.models.deepseek_v41.sm80.test_sm80_kernels_gpu import (
        _fp64_attention,
        _gather_rows,
    )
    from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core

    names = ("q", "swa_cache", "swa_indices", "comp_cache", "comp_block_table",
             "comp_local", "req", "sink")
    t = {k: _to(a.get(k), device) for k in names}
    q, local, scale = t["q"], t["comp_local"], float(a["scale"])

    def cand(rows):
        return core.sparse_attention(
            q[rows], t["swa_cache"], t["swa_indices"][rows], t["comp_cache"],
            t["comp_block_table"], None if local is None else local[rows],
            t["req"][rows], t["sink"], scale,
        )

    whole = cand(slice(None))
    n, h, d = q.shape
    k = t["swa_indices"].shape[1] + (0 if local is None else local.shape[1])
    step = max(1, CHUNK_BYTES // max(1, k * (d + h) * 8))
    refs, exacts = [], []
    for start in range(0, n, step):
        rows = slice(start, min(n, start + step))
        gathered, valid = _gather_rows(
            SimpleNamespace(compress_ratio=int(local is not None)),
            SimpleNamespace(
                swa=t["swa_cache"], comp=t["comp_cache"],
                topk=None if local is None else local[rows],
            ),
            SimpleNamespace(
                num_tokens=rows.stop - rows.start,
                swa_indices=t["swa_indices"][rows],
                token_to_req=t["req"][rows],
                comp_block_table=t["comp_block_table"],
            ),
        )
        qc = q[rows]
        exacts.append(_fp64_attention(qc, gathered, valid, t["sink"], scale))
        idx = torch.where(
            valid, torch.arange(valid.shape[1], device=device), -1
        ).int()[:, None, :]
        with torch.device(device):
            refs.append(
                ref_sparse_attn(qc[:, None], gathered, t["sink"][:h], idx, scale)[:, 0]
            )
    alone = all(bits_equal(whole[i : i + 1], cand(slice(i, i + 1))) for i in picks(n))
    return Call(whole, torch.cat(refs), torch.cat(exacts), alone)


def index_scores(a: dict, device: torch.device) -> Call:
    from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core

    q, w, n_vis, req, idxk, table = (
        _to(a[k], device) for k in ("q", "w", "n_vis", "req", "idxk", "block_table")
    )
    n_cols, per_rank = int(a["n_cols"]), int(a["heads_per_rank"])

    def cand(rows):
        return core.index_scores(
            q[rows], w[rows], n_vis[rows], req[rows], idxk, table, n_cols, per_rank
        )

    whole = cand(slice(None))
    ref = core.index_scores_torch(q, w, n_vis, req, idxk, table, n_cols, per_rank)
    exact = torch.full(
        (q.shape[0], n_cols), float("-inf"), dtype=torch.float64, device=device
    )
    heads = q.shape[1]
    for r in torch.unique(req).tolist():
        sel = torch.nonzero(req == r).flatten()
        width = int(n_vis[sel].max())
        if width == 0:
            continue
        cols = torch.arange(width, device=device)
        keys = core.gather_compressed_rows(
            idxk, table, torch.full_like(cols, r), cols
        ).double()
        step = max(1, CHUNK_BYTES // (heads * width * 8))
        for start in range(0, sel.numel(), step):
            part = sel[start : start + step]
            dots = torch.einsum("thd,nd->thn", q[part].double(), keys)
            s = (dots.clamp_min(0) * w[part].double().unsqueeze(-1)).sum(1)
            exact[part, :width] = torch.where(
                cols[None, :] < n_vis[part, None], s, float("-inf")
            )
    alone = all(
        bits_equal(whole[i : i + 1], cand(slice(i, i + 1))) for i in picks(q.shape[0])
    )
    return Call(whole, ref, exact, alone)


# ---------------------------------------------------------------- FP8 / FP4


def _qdq(a: dict, device: torch.device, fp4: bool) -> Call:
    from tests.models.deepseek_v41.sm80.test_act_quant import ref_fp4, ref_fp8
    from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core
    from vllm.models.deepseek_v41.nvidia.ops.sm80_act_quant import (
        act_quant_qdq_fp8,
        fp4_qdq,
    )

    x, block = a["x"], int(a["block"])
    xd = x.to(device)
    if fp4:
        scale = a.get("scale", "e8m0")
        cand = lambda rows: core.qdq_fp4(xd[rows], block, scale)  # noqa: E731
        ref, exact = fp4_qdq(x, block, scale), ref_fp4(x, block, scale)
    else:
        cand = lambda rows: core.qdq_fp8(xd[rows], block)  # noqa: E731
        ref, exact = act_quant_qdq_fp8(x, block), ref_fp8(x, block)
    whole = cand(slice(None))
    alone = all(
        bits_equal(whole[i : i + 1], cand(slice(i, i + 1))) for i in picks(x.shape[0])
    )
    return Call(whole, ref, exact.double(), alone)


def qdq_fp8(a: dict, device: torch.device) -> Call:
    return _qdq(a, device, fp4=False)


def qdq_fp4(a: dict, device: torch.device) -> Call:
    return _qdq(a, device, fp4=True)


# ---------------------------------------------------------------- KV production


def _compress_run(a: dict, device: torch.device, tokens: slice):
    from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core

    def get(name):
        value = a.get(name)
        return None if value is None else value[tokens].to(device)

    pos = get("positions")
    qsl = a["query_start_loc"]
    if tokens == slice(None):
        req, qsl = a["token_to_req"].to(device), qsl.to(device)
    else:
        req = torch.zeros_like(pos)
        qsl = torch.tensor([0, pos.shape[0]], dtype=qsl.dtype, device=device)
    batch = core.Sm80AttnBatch(
        positions=pos,
        token_to_req=req,
        query_start_loc=qsl,
        swa_slots=torch.full_like(pos, -1),
        swa_indices=pos.new_empty((pos.shape[0], 0)),
        comp_slots=get("comp_slots"),
        ring_slots=get("ring_slots"),
    )
    cfg = SimpleNamespace(
        compress_ratio=int(a["compress_ratio"]),
        head_dim=int(a["head_dim"]),
        eps=float(a["eps"]),
    )
    ring = a.get("ring")
    # compress writes the ring's tail: every run starts from the captured ring.
    ring = None if ring is None else ring.to(device, copy=True)
    return core.compress(
        cfg, batch, get("kv_score"), a["norm_weight"].to(device), ring
    )


def _compress_fp64(a: dict, rows: torch.Tensor) -> torch.Tensor:
    """FP64 latents of the closing tokens ``rows``: the pooled pair (ratio 2,
    the predecessor from the ring when the group started before the chunk)
    or the token's kv (ratio 1), RMS-normalized; no intermediate rounding."""
    d, eps = int(a["head_dim"]), float(a["eps"])
    kv = a["kv_score"]
    if int(a["compress_ratio"]) == 1:
        x = kv[rows, :d].to(BF16).double()
    else:
        pos = a["positions"].long()
        req = a["token_to_req"].long()
        qsl = a["query_start_loc"].long()
        first = rows == qsl[req[rows]]
        prev = kv[(rows - 1).clamp_min(0)].double()
        if bool(first.any()):
            ring = a["ring"]
            capacity = ring.shape[1]
            slot = a["ring_slots"][rows[first]].long()
            prev[first] = ring[slot // capacity, (pos[rows[first]] - 1) % capacity].double()
        cur = kv[rows].double()
        pair = torch.stack((prev[:, :d], cur[:, :d]), dim=1)
        gate = torch.stack((prev[:, d:], cur[:, d:]), dim=1).softmax(dim=1)
        x = (pair * gate).sum(dim=1)
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return x * a["norm_weight"].double()


def compress(a: dict, device: torch.device) -> Call:
    rows, whole = _compress_run(a, device, slice(None))
    ref_rows, ref = _compress_run(a, CPU, slice(None))
    if not torch.equal(rows.cpu(), ref_rows):
        raise ValueError("the candidate closes other groups than the reference path")
    qsl = a["query_start_loc"].tolist()
    alone = True
    for r in sorted({0, len(qsl) - 2}):
        start, end = qsl[r], qsl[r + 1]
        sub_rows, sub = _compress_run(a, device, slice(start, end))
        mine = (rows >= start) & (rows < end)
        alone &= torch.equal((rows[mine] - start).cpu(), sub_rows.cpu())
        alone &= bits_equal(whole[mine], sub)
    return Call(whole, ref, _compress_fp64(a, ref_rows), alone)


def index_keys(a: dict, device: torch.device) -> Call:
    from tests.models.deepseek_v41.sm80.test_act_quant import ref_fp4
    from tests.models.deepseek_v41.sm80.test_sm80_kernels_gpu import (
        _inverse_rope_fp64,
    )
    from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core

    eps = float(a["eps"])

    def run(dev, rows):
        k = a["wk_out"][rows].to(dev)
        weights = SimpleNamespace(
            wk=lambda _latent: k, k_norm=a["k_norm"].to(dev), freqs=a["freqs"].to(dev)
        )
        return core.index_keys(
            SimpleNamespace(eps=eps), weights, k, a["group_pos"][rows].to(dev)
        )

    whole = run(device, slice(None))
    x = a["wk_out"].double()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * a["k_norm"].double()
    # Forward RoPE with the layer's fp32 angles: the inverse rotation by the
    # negated frequencies (the sign of every fp32 angle flips exactly).
    x = _inverse_rope_fp64(x[:, None, :], a["group_pos"], -a["freqs"])[:, 0]
    exact = ref_fp4(x.float(), 32, "e8m0").double()
    n = a["wk_out"].shape[0]
    alone = all(bits_equal(whole[i : i + 1], run(device, slice(i, i + 1))) for i in picks(n))
    return Call(whole, run(CPU, slice(None)), exact, alone)


# ---------------------------------------------------------------- Engram


def engram_dequant(a: dict, device: torch.device) -> Call:
    from vllm.models.deepseek_v41.nvidia.ops.sm80_engram_dequant import (
        ENGRAM_QUANT_BLOCK,
        dequant_engram_rows,
        dequant_engram_rows_reference,
    )

    values, scales = a["values"], a["scales"]

    def cand(rows):
        v = values[rows].to(device).contiguous()
        out = torch.empty(v.shape, dtype=BF16, device=device)
        dequant_engram_rows(v, scales[rows].to(device).contiguous(), out)
        return out

    whole = cand(slice(None))
    n, dim = values.shape
    exact = (
        values.view(torch.float8_e4m3fn).float().double().view(n, -1, ENGRAM_QUANT_BLOCK)
        * scales.view(torch.float8_e8m0fnu).float().double().unsqueeze(-1)
    ).view(n, dim)
    alone = all(bits_equal(whole[i : i + 1], cand(slice(i, i + 1))) for i in picks(n))
    return Call(whole, dequant_engram_rows_reference(values, scales), exact, alone)


# ---------------------------------------------------------------- EXL3 MoE

_WORKSPACE = {
    "moe_out": "moe_workspace_size",
    "moe_out_v2": "moe_workspace_size_v2",
    "moe_out_prefill": "moe_workspace_size_prefill",
}


def expert_set(checkpoint: str, matrices, packed_meta: torch.Tensor, device):
    """The local experts as Exl3MoEMethod lays them out, read from the
    checkpoint: (per projection (trellis, suh, svh), pointer table, meta).
    The meta must be the captured ``packed_meta``."""
    from tests.kernels.quantization import exl3_real_cases as real

    ckpt = real.Checkpoint(Path(checkpoint))
    weights, pointers, meta = [], [], []
    for names in matrices:
        ptrs, rows = [], []
        for name in names:
            matrix = ckpt.matrix(name)
            parts = [t.to(device) for t in ckpt.whole(name)]
            weights.append(parts)
            ptrs.append([p.data_ptr() for p in parts])
            rows.append(
                [matrix.padded_in, matrix.padded_out, matrix.bits_x2, matrix.codebook]
            )
        pointers.append(ptrs)
        meta.append(rows)
    meta = torch.tensor(meta, dtype=torch.int64)
    if not torch.equal(meta, packed_meta.cpu()):
        raise ValueError("the checkpoint's expert layout differs from the captured one")
    pointers = torch.tensor(pointers, dtype=torch.int64, device=device)
    return weights, pointers, meta.to(device)


def moe_exact(x, ids, weights_t, expert_map, weights, meta, intermediate, limit):
    """FP64 routed-expert output of one call (exl3_real_cases.moe_fp64)."""
    from tests.kernels.quantization import exl3_real_cases as real

    def decode(local):
        parts = weights[3 * local : 3 * local + 3]
        return [
            (real.core_weight(t, int(m[2]), int(m[3])), s, v)
            for (t, s, v), m in zip(parts, meta[local].tolist())
        ]

    calls = [(x, ids, weights_t)]
    return real.moe_fp64(calls, expert_map, decode, intermediate, limit)[0]


def exl3_moe(a: dict, device: torch.device) -> Call:
    from vllm.sm80.kernels.exl3 import ops

    name = a["operator"]
    if name not in _WORKSPACE:
        raise ValueError(f"operator {name!r} is not one of {sorted(_WORKSPACE)}")
    x, ids, tw, emap = (
        a[k].to(device) for k in ("x", "topk_ids", "topk_weights", "expert_map")
    )
    inter, limit = int(a["intermediate_size"]), float(a["swiglu_limit"])
    weights, pointers, meta = expert_set(
        a["checkpoint"], a["matrices"], a["packed_meta"], device
    )

    def run(op, rows):
        xr = x[rows]
        out = torch.full_like(xr, float("nan"))
        size = getattr(ops, _WORKSPACE[op])(
            xr.shape[0], x.shape[1], inter, ids.shape[1], meta.shape[0]
        )
        workspace = torch.full((size,), float("nan"), device=device)
        getattr(ops, op)(
            xr, ids[rows], tw[rows], pointers, meta, emap, inter, out, workspace,
            swiglu_limit=limit,
        )
        return out

    whole = run(name, slice(None))
    ref = run("moe_out", slice(None))
    exact = moe_exact(x, ids, tw, emap, weights, meta, inter, limit)
    n = x.shape[0]
    if name == "moe_out_prefill":
        lo = ops.MOE_PREFILL_MIN_ROWS
        subs = [slice(0, lo), slice(n - lo, n)] if n > lo else []
    else:
        subs = [slice(i, i + 1) for i in picks(n)]
    alone = all(bits_equal(whole[s], run(name, s)) for s in subs)
    return Call(whole, ref, exact, alone)


FAMILIES = {
    "sparse_attention": sparse_attention,
    "index_scores": index_scores,
    "qdq_fp8": qdq_fp8,
    "qdq_fp4": qdq_fp4,
    "compress": compress,
    "index_keys": index_keys,
    "engram_dequant": engram_dequant,
    "exl3_moe": exl3_moe,
}


# ---------------------------------------------------------------- driver


def case_row(case: dict, device: torch.device) -> dict:
    sources = case.get("captured_sources")
    if not sources:
        raise ValueError("no captured_sources: the case cannot be replayed")
    cand_sum = cand_max = ref_sum = ref_max = 0.0
    count = numel = 0
    bitwise = invariant = True
    families = set()
    for path in sources:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        family = obj.get("family")
        if family not in FAMILIES:
            raise ValueError(f"{path}: family {family!r} is not one of {sorted(FAMILIES)}")
        families.add(family)
        call = FAMILIES[family](obj["tensors"], device)
        cs, cm, n = abs_errors(call.cand, call.exact)
        rs, rm, _ = abs_errors(call.ref, call.exact)
        cand_sum, cand_max = cand_sum + cs, max(cand_max, cm)
        ref_sum, ref_max = ref_sum + rs, max(ref_max, rm)
        count += n
        numel += call.cand.numel()
        bitwise &= bits_equal(call.cand, call.ref)
        invariant &= call.row_invariant
    count = max(count, 1)
    return {
        "case": case["id"],
        "family": ",".join(sorted(families)),
        "fp64": {
            "cand_mean": cand_sum / count,
            "cand_max": cand_max,
            "ref_mean": ref_sum / count,
            "ref_max": ref_max,
        },
        "bitwise": bool(bitwise),
        "row_invariance": bool(invariant),
        "numel": int(numel),
    }


def header(device: torch.device) -> dict:
    import triton

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu, sms = props.name, props.multi_processor_count
    else:
        gpu, sms = "cpu", 0
    return {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "sm_count": sms,
    }


def run(cases: Path, out: Path, device: torch.device) -> int:
    """Write the header and one line per case that runs; 2 when the case
    file is not one of SCHEMAS."""
    doc = json.loads(Path(cases).read_text())
    if not isinstance(doc, dict) or doc.get("schema") not in SCHEMAS:
        print(f"{cases}: schema is not one of {SCHEMAS}", file=sys.stderr)
        return 2
    with Path(out).open("w") as f, torch.inference_mode():
        f.write(json.dumps({"header": header(device)}) + "\n")
        for case in doc.get("cases") or []:
            try:
                row = case_row(case, device)
            except Exception as exc:  # noqa: BLE001 - reported, the other cases still run
                print(f"case {case.get('id')}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("fp64_cases: the candidate path runs on CUDA; no CUDA device", file=sys.stderr)
        return 2
    return run(a.cases, a.out, torch.device("cuda"))


if __name__ == "__main__":
    # The tree this file is in is the one under test.
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
