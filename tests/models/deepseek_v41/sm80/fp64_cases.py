# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP64 errors of the DeepSeek-V4.1 sm_80 kernels on captured served calls.

    python tests/models/deepseek_v41/sm80/fp64_cases.py --cases <file> --out <jsonl>

Run from the tree root on one CUDA card (it refuses to run without CUDA).

Cases: a JSON object with schema ``served_cases/1`` (the export of captured
calls) or ``generated_cases/1``. Each case has an ``id``, a ``family`` and
``captured_sources``, the files of the calls it pools; the file's top-level
``checkpoint`` ({role: directory} of the served launch) names the weights
the EXL3 MoE families replay. A source is a ``torch.save`` dict
(``lab-point-capture/1``), read with ``weights_only=True``:

    {"schema": "lab-point-capture/1", "family": <family>, "meta": {...},
     "tensors": {argument: value}}

``tensors`` holds the arguments of one call of the family's capture point
(below) by parameter name, as the call received them (defaults applied,
buffers the call writes as they were before it). ``meta.module_prefix`` is
the vLLM prefix of the module that made the call.

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
last request for the compressor; for moe_out_prefill and moe_out_prefill_v2,
which take no fewer than MOE_PREFILL_MIN_ROWS rows, the first and last calls
of that size) give
the same bits run alone as in the whole call. ``numel``: output elements.
A case whose sources cannot be run gets no line and its reason goes to
stderr; the driver then exits 1 once every other case is written, so the
run is refused as a whole and no planned case goes unjudged.

Families: capture point (recorded arguments) -> candidate / reference path /
FP64. Module paths below are under ``vllm.models.deepseek_v41.nvidia.ops``
unless named in full.

- ``sparse_attention``: ``sm80_attn_core:sparse_attention`` (q, swa_cache,
  swa_indices, comp_cache, comp_block_table, comp_local, req, sink, scale)
  -> the Triton kernel / the reference ``sparse_attn`` arithmetic
  (``ref_sparse_attn``) over the same rows / sink attention in FP64.
- ``index_scores``: ``sm80_attn_core:index_scores`` (q, w, n_vis, req,
  idxk, block_table, n_cols, heads_per_rank) -> the Triton kernel /
  ``index_scores_torch`` / ReLU'd per-head dot products times the head
  weights, summed over every head in FP64 (invisible columns are -inf).
- ``qdq_fp8``: ``sm80_attn_core:qdq_fp8`` (x, block) -> Triton /
  ``act_quant_qdq_fp8`` on CPU / the independent definition
  ``test_act_quant.ref_fp8``. Exact: both errors are zero.
- ``qdq_fp4``: ``sm80_attn_core:qdq_fp4`` (x, block, scale) -> Triton /
  ``fp4_qdq`` on CPU / ``test_act_quant.ref_fp4``. Exact.
- ``compress``: ``sm80_attn_core:compress_rows`` (kv_score, norm_weight,
  ring, positions, token_to_req, query_start_loc, comp_slots, ring_slots,
  compress_ratio, head_dim, eps) -> torch on CUDA / the same on CPU /
  softmax pooling and RMS norm in FP64. Both must close the same groups.
- ``index_keys``: ``sm80_attn_core:index_key_rows`` (k, k_norm, freqs,
  group_pos, eps) -> CUDA (torch norm and RoPE, Triton FP4) / CPU / RMS
  norm and RoPE in FP64, then FP4 by ``test_act_quant.ref_fp4``.
- ``engram_dequant``: ``sm80_engram_dequant:dequant_engram_rows`` (values,
  scales) -> Triton / ``dequant_engram_rows_reference`` on CPU / the E4M3 x
  E8M0 product in FP64. Exact: bitwise.
- ``exl3_moe_out``, ``exl3_moe_out_v2``, ``exl3_moe_out_prefill``,
  ``exl3_moe_out_prefill_v2``: ``vllm.sm80.kernels.exl3.ops:moe_out`` /
  ``moe_out_v2`` / ``moe_out_prefill`` / ``moe_out_prefill_v2`` (x,
  topk_ids, topk_weights, packed_meta, expert_map,
  intermediate_size, swiglu_limit) -> that operator / ``moe_out`` / the
  routed experts in FP64 (``exl3_real_cases.moe_fp64`` with the SwiGLU
  clamp). The local experts are read from the ``target`` checkpoint by the
  layer's expert matrix names (the model's EXL3 layout applied to
  ``meta.module_prefix`` and the expert map); their layout must equal the
  recorded ``packed_meta``. moe_out_v2 and moe_out_prefill_v2 also record
  ``bits_x2``, the layer's one routed bit width, which must be the one in
  ``packed_meta``.
  ``moe_out`` is its own reference path.
- ``mhc_delayed``: ``sm80_mhc:mhc_fused_post_pre_delayed`` (x, residual,
  post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
  hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix, norm_weight,
  norm_eps) and ``mhc_pre_delayed``: ``sm80_mhc:mhc_pre_delayed`` (the same
  without x, post_layer_mix, comb_res_mix) -> the fused kernels
  (``sm80_mhc_fused``; the torch formulas on CPU) / the torch formulas
  (``sm80_mhc``) / :func:`mhc_delayed_fp64`. Several outputs, judged one per
  case: the case's ``output``, else the suffix after ``:`` of the case's
  or the file's ``family`` (e.g. ``mhc_delayed:comb``); one of residual (fused only),
  post, comb, layer_input, pre against the FP64 op with no intermediate
  rounding, or post.rp, comb.rp, layer_input.rp, pre.rp against the FP64 op
  at each path's OWN BF16 rounding points (its own BF16 streams feed the
  coefficients and the collapse, the collapse is rounded before the norm):
  each path is judged against FP64 from the streams it produced, which
  isolates the arithmetic after the streams from which stream elements the
  path's BF16 rounding happened to round the other way (a single such
  element moves the coefficients by more than their FP32 arithmetic does).
  Row invariance holds within a kernel variant (up to 8 rows and from 9
  rows, the two kernel A forms reduce in different orders).
- ``attn_prologue``: ``sm80_attn_fused:prologue`` (qr_kv, q_norm, kv_norm,
  positions, freqs, swa_cache, swa_slots, eps) -> the fused prologue
  (``sm80_attn_fused``; the reference path on CPU) / ``sm80_attn_core``'s
  ``rms_norm`` pair, ``window_kv`` and ``store_rows`` / FP64. Two outputs,
  judged one per case (as the mHC families): ``qr`` (the query RMSNorm,
  against FP64 unrounded) or ``window`` (the cache rows of the slots >= 0:
  RMSNorm and RoPE in FP64, each rounded to BF16 as the reference does, then
  the exact FP8 qdq).
- ``attn_attend``: ``sm80_attn_fused:attend`` (q, positions, freqs,
  swa_cache, swa_indices, comp_cache, comp_block_table, comp_local, req,
  attn_sink, scale, n_groups, out) -> the fused attend (the reference path
  on CPU) / ``sm80_attn_core``'s ``rope``, ``sparse_attention`` and
  ``output_qdq`` / FP64: ``out[:T]`` with the query RoPE (rounded to BF16),
  the sink attention (rounded to BF16) and the inverse RoPE (rounded to
  BF16) in FP64, then the exact per-group FP8 qdq.
"""

import argparse
import functools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]
SCHEMAS = ("served_cases/1", "generated_cases/1")
SOURCE_SCHEMA = "lab-point-capture/1"
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
    # the FP64 value the reference path is judged against when it is not
    # ``exact`` (per-path references), and the reference path's own row
    # invariance where a family reports it
    ref_exact: torch.Tensor | None = None
    ref_row_invariant: bool | None = None


@dataclass
class Source:
    """What a family reads besides the arguments: the call's metadata,
    the served launch's checkpoints ({role: directory}) and, for the
    families with several outputs, the output the case judges."""

    meta: dict
    checkpoint: dict
    output: str | None = None


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


def sparse_attention(a: dict, device: torch.device, _src: Source) -> Call:
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


def index_scores(a: dict, device: torch.device, _src: Source) -> Call:
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


def qdq_fp8(a: dict, device: torch.device, _src: Source) -> Call:
    return _qdq(a, device, fp4=False)


def qdq_fp4(a: dict, device: torch.device, _src: Source) -> Call:
    return _qdq(a, device, fp4=True)


# ---------------------------------------------------------------- KV production


def _compress_run(a: dict, device: torch.device, tokens: slice):
    """``compress_rows`` on the recorded arguments, for all tokens or for the
    tokens of one request run alone (as its own one-request call)."""
    from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core

    def get(name):
        value = a.get(name)
        return None if value is None else value[tokens].to(device)

    pos = get("positions")
    qsl = a["query_start_loc"]
    if tokens == slice(None):
        req, qsl = a["token_to_req"].to(device), qsl.to(device)
    else:
        req = torch.zeros_like(a["token_to_req"][tokens]).to(device)
        qsl = torch.tensor([0, pos.shape[0]], dtype=qsl.dtype, device=device)
    ring = a.get("ring")
    # compress_rows writes the ring's tail: every run starts from the recorded ring.
    ring = None if ring is None else ring.to(device, copy=True)
    closes, latent = core.compress_rows(
        get("kv_score"), a["norm_weight"].to(device), ring, pos, req, qsl,
        get("comp_slots"), get("ring_slots"), int(a["compress_ratio"]),
        int(a["head_dim"]), float(a["eps"]),
    )
    # compress_rows returns every token; the closing ones are the latents stored.
    rows = torch.nonzero(closes).flatten()
    return rows, latent[rows]


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


def compress(a: dict, device: torch.device, _src: Source) -> Call:
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


def index_keys(a: dict, device: torch.device, _src: Source) -> Call:
    from tests.models.deepseek_v41.sm80.test_act_quant import ref_fp4
    from tests.models.deepseek_v41.sm80.test_sm80_kernels_gpu import (
        _inverse_rope_fp64,
    )
    from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core

    eps = float(a["eps"])

    def run(dev, rows):
        return core.index_key_rows(
            a["k"][rows].to(dev), a["k_norm"].to(dev), a["freqs"].to(dev),
            a["group_pos"][rows].to(dev), eps,
        )

    whole = run(device, slice(None))
    x = a["k"].double()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * a["k_norm"].double()
    # Forward RoPE with the layer's fp32 angles: the inverse rotation by the
    # negated frequencies (the sign of every fp32 angle flips exactly).
    x = _inverse_rope_fp64(x[:, None, :], a["group_pos"], -a["freqs"])[:, 0]
    exact = ref_fp4(x.float(), 32, "e8m0").double()
    n = a["k"].shape[0]
    alone = all(bits_equal(whole[i : i + 1], run(device, slice(i, i + 1))) for i in picks(n))
    return Call(whole, run(CPU, slice(None)), exact, alone)


# ---------------------------------------------------------------- Engram


def engram_dequant(a: dict, device: torch.device, _src: Source) -> Call:
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
    "moe_out_prefill_v2": "moe_workspace_size_prefill_v2",
}


def expert_set(checkpoint: str, module_prefix: str, expert_map, packed_meta, device):
    """The layer's local experts as Exl3MoEMethod lays them out, read from
    the checkpoint: (per projection (trellis, suh, svh), pointer table,
    meta). Local expert l is the global g with ``expert_map[g] == l``; its
    matrices are the model layout's expert matrices of ``module_prefix``.
    The meta must be the recorded ``packed_meta``."""
    from tests.kernels.quantization import exl3_real_cases as real
    from vllm.models.deepseek_v41.nvidia.sm80_exl3 import DeepseekV41Exl3Layout

    config = json.loads((Path(checkpoint) / "config.json").read_text())
    text = SimpleNamespace(**config.get("text_config", config))
    layout = DeepseekV41Exl3Layout(SimpleNamespace(text_config=text))
    resolver = SimpleNamespace()
    resolver.source_prefix = lambda prefix: layout.source_prefix(resolver, prefix)
    owner = {}
    for g, local in enumerate(expert_map.tolist()):
        if local >= 0:
            if local in owner:
                raise ValueError("the expert map assigns a local expert twice")
            owner[local] = g
    if sorted(owner) != list(range(packed_meta.shape[0])):
        raise ValueError("the expert map does not cover the local experts")
    ckpt = real.Checkpoint(Path(checkpoint))
    weights, pointers, meta = [], [], []
    for local in range(packed_meta.shape[0]):
        ptrs, rows = [], []
        for role in ("gate_proj", "up_proj", "down_proj"):
            name = layout.expert_matrix(resolver, module_prefix, owner[local], role)
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
        raise ValueError("the checkpoint's expert layout differs from the recorded one")
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


def _exl3_moe(name: str, a: dict, device: torch.device, src: Source) -> Call:
    from vllm.sm80.kernels.exl3 import ops

    checkpoint = src.checkpoint.get("target")
    prefix = src.meta.get("module_prefix")
    if not checkpoint or not prefix:
        raise ValueError("an EXL3 MoE call needs the cases file's checkpoint.target "
                         "and the source's meta.module_prefix")
    x, ids, tw, emap = (
        a[k].to(device) for k in ("x", "topk_ids", "topk_weights", "expert_map")
    )
    inter, limit = int(a["intermediate_size"]), float(a["swiglu_limit"])
    weights, pointers, meta = expert_set(
        checkpoint, prefix, a["expert_map"], a["packed_meta"], device
    )
    extra = {}
    if name in ("moe_out_v2", "moe_out_prefill_v2"):
        # The layer's one routed bit width (the dispatch's gate checks it).
        bits = sorted({int(b) for b in a["packed_meta"][..., 2].flatten()})
        if len(bits) != 1 or int(a.get("bits_x2", bits[0])) != bits[0]:
            raise ValueError(f"{name} bits_x2 {a.get('bits_x2')} for routed "
                             f"matrices of bits_x2 {bits}")
        extra = {"bits_x2": bits[0]}

    def run(op, rows):
        xr = x[rows]
        out = torch.full_like(xr, float("nan"))
        size = getattr(ops, _WORKSPACE[op])(
            xr.shape[0], x.shape[1], inter, ids.shape[1], meta.shape[0]
        )
        workspace = torch.full((size,), float("nan"), device=device)
        getattr(ops, op)(
            xr, ids[rows], tw[rows], pointers, meta, emap, inter, out, workspace,
            swiglu_limit=limit, **(extra if op == name else {}),
        )
        return out

    whole = run(name, slice(None))
    ref = run("moe_out", slice(None))
    exact = moe_exact(x, ids, tw, emap, weights, meta, inter, limit)
    n = x.shape[0]
    if name in ("moe_out_prefill", "moe_out_prefill_v2"):
        lo = ops.MOE_PREFILL_MIN_ROWS
        subs = [slice(0, lo), slice(n - lo, n)] if n > lo else []
    else:
        subs = [slice(i, i + 1) for i in picks(n)]
    alone = all(bits_equal(whole[s], run(name, s)) for s in subs)
    return Call(whole, ref, exact, alone)


# ---------------------------------------------------------------- mHC

MHC_TOKEN_ARGS = ("x", "residual", "post_layer_mix", "comb_res_mix", "pre_mix")
MHC_OUTPUTS = ("residual", "post", "comb", "layer_input", "pre")


def mhc_delayed_fp64(
    t: dict, rounding: bool = False, own: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """The delayed mHC of one call in FP64, from its inputs ``t`` (the
    capture point's arguments; no ``x``: the pre-only call). ``rounding``
    keeps the BF16 rounding points: the streams are rounded before the
    coefficients and the collapse (to ``own``, a path's own BF16 streams,
    when given), the collapse before the norm.
    Returns every output of the call by name (residual only with ``x``)."""
    res = t["residual"].double()
    T, hc, hidden = res.shape
    iters, eps = int(t["sinkhorn_repeat"]), float(t["hc_pre_eps"])
    if t.get("x") is not None:
        post = t["post_layer_mix"].double().reshape(T, hc)
        comb = t["comb_res_mix"].double().reshape(T, hc, hc)
        streams = post[..., None] * t["x"].double()[:, None, :] + torch.einsum(
            "tjk,tjh->tkh", comb, res
        )
    else:
        streams = res
    if not rounding:
        used = streams
    elif own is None:
        used = streams.to(BF16).double()
    else:
        used = own.double()
    flat = used.reshape(T, hc * hidden)
    rs = torch.rsqrt(flat.square().mean(-1, keepdim=True) + float(t["rms_eps"]))
    mixes = (flat @ t["fn"].double().T) * rs
    scale, base = t["hc_scale"].double(), t["hc_base"].double()
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    new_post = float(t["hc_post_mult_value"]) * torch.sigmoid(
        mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc]
    )
    c = (mixes[:, 2 * hc :] * scale[2] + base[2 * hc :]).view(T, hc, hc)
    c = torch.softmax(c, dim=-1) + eps
    c = c / (c.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(dim=-1, keepdim=True) + eps)
        c = c / (c.sum(dim=-2, keepdim=True) + eps)
    pre_mix = t.get("pre_mix")
    if pre_mix is None:
        w = torch.zeros(T, hc, dtype=torch.float64)
        w[:, 0] = 1.0
    else:
        w = pre_mix.double().reshape(T, hc)
    li = (w[..., None] * used).sum(dim=1)
    if rounding:
        li = li.to(BF16).double()
    norm_weight = t.get("norm_weight")
    if norm_weight is not None:
        li = li * torch.rsqrt(li.square().mean(-1, keepdim=True) + float(t["norm_eps"]))
        li = li * norm_weight.double()
    out = {"post": new_post[..., None], "comb": c, "layer_input": li, "pre": pre}
    if t.get("x") is not None:
        out["residual"] = streams
    return out


def _mhc_call(a: dict, device: torch.device, rows: slice, fused: bool) -> dict:
    """One path's outputs by name on ``device`` for the tokens ``rows``."""
    from vllm.models.deepseek_v41.nvidia.ops import sm80_mhc

    if fused:
        from vllm.models.deepseek_v41.nvidia.ops import sm80_mhc_fused as mod
    else:
        mod = sm80_mhc
    t = {
        k: (v[rows] if k in MHC_TOKEN_ARGS and v is not None else v)
        for k, v in a.items()
    }
    t = {k: _to(v, device) for k, v in t.items()}
    common = (
        t["fn"],
        t["hc_scale"],
        t["hc_base"],
        float(t["rms_eps"]),
        float(t["hc_pre_eps"]),
        float(t["hc_sinkhorn_eps"]),
        float(t["hc_post_mult_value"]),
        int(t["sinkhorn_repeat"]),
    )
    kw = dict(
        pre_mix=t.get("pre_mix"),
        norm_weight=t.get("norm_weight"),
        norm_eps=float(t.get("norm_eps", 1e-6)),
    )
    if t.get("x") is None:
        post, comb, li, pre = mod.mhc_pre_delayed(t["residual"], *common, **kw)
        return {"post": post, "comb": comb, "layer_input": li, "pre": pre}
    streams, post, comb, li, pre, _ = mod.mhc_fused_post_pre_delayed(
        t["x"], t["residual"], t["post_layer_mix"], t["comb_res_mix"], *common, **kw
    )
    return {"residual": streams, "post": post, "comb": comb, "layer_input": li, "pre": pre}


def _mhc(a: dict, device: torch.device, output: str | None) -> Call:
    if output is None:
        raise ValueError(
            "mHC cases name their output (case 'output', or the case's or "
            "file's 'family' "
            f"'<family>:<output>'): one of {MHC_OUTPUTS} or <output>.rp"
        )
    name, _, mode = output.partition(".")
    if name not in MHC_OUTPUTS or mode not in ("", "rp") or (name, mode) == ("residual", "rp"):
        raise ValueError(f"mHC output {output!r} is not one of {MHC_OUTPUTS} or <output>.rp")
    if name == "residual" and a.get("x") is None:
        raise ValueError("the pre-only call has no residual output")
    fused = device.type == "cuda"
    n = a["residual"].shape[0]
    cand_all = _mhc_call(a, device, slice(None), fused)
    ref_all = _mhc_call(a, device, slice(None), False)
    whole, ref = cand_all[name], ref_all[name]
    step = max(1, CHUNK_BYTES // max(1, 4 * a["residual"][0].numel() * 8))

    def fp64(streams):
        """FP64 output; at the rounding points of ``streams`` (a path's
        own BF16 streams) in the .rp mode."""
        parts = []
        for s in range(0, n, step):
            t = {
                k: (v[s : s + step] if k in MHC_TOKEN_ARGS and v is not None else v)
                for k, v in a.items()
            }
            own = None if streams is None else streams[s : s + step].cpu()
            parts.append(mhc_delayed_fp64(t, rounding=mode == "rp", own=own)[name])
        return torch.cat(parts)

    if mode == "rp":
        # each path against FP64 from its own BF16 streams (the input
        # streams for the pre-only call, which both paths read unchanged)
        own = (lambda p: p["residual"]) if a.get("x") is not None else (lambda p: None)
        exact, ref_exact = fp64(own(cand_all)), fp64(own(ref_all))
    else:
        exact, ref_exact = fp64(None), None

    def alone(is_fused, value):
        return all(
            bits_equal(value[i : i + 1], _mhc_call(a, device, slice(i, i + 1), is_fused)[name])
            for i in picks(n)
        )

    return Call(
        whole, ref, exact, alone(fused, whole), ref_exact=ref_exact,
        ref_row_invariant=alone(False, ref),
    )


def mhc_delayed(a: dict, device: torch.device, src: Source) -> Call:
    if a.get("x") is None:
        raise ValueError("mhc_delayed: the call has no x (pre-only calls are mhc_pre_delayed)")
    return _mhc(a, device, src.output)


def mhc_pre_delayed(a: dict, device: torch.device, src: Source) -> Call:
    if a.get("x") is not None:
        raise ValueError("mhc_pre_delayed: the call has an x (fused calls are mhc_delayed)")
    return _mhc(a, device, src.output)


# ---------------------------------------------------------------- fused attention core

ATTN_PROLOGUE_OUTPUTS = ("qr", "window")


def _rms_fp64(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.double()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w.double()


def _rope_fp64(x, positions, freqs, inverse=False):
    """FP64 RoPE of the last ``2 * len(freqs)`` dims of ``x`` ([T, ..., D])
    at the layer's FP32 angles (``fp32(position) * fp32 frequency``)."""
    rd = 2 * freqs.numel()
    angles = (positions.to(torch.float32)[:, None] * freqs.to(torch.float32)).double()
    cis = torch.polar(torch.ones_like(angles), angles)
    if inverse:
        cis = cis.conj()
    cis = cis.view((positions.shape[0],) + (1,) * (x.dim() - 2) + (rd // 2,))
    out = x.double().clone()
    tail = torch.view_as_complex(out[..., -rd:].contiguous().unflatten(-1, (-1, 2)))
    out[..., -rd:] = torch.view_as_real(tail * cis).flatten(-2)
    return out


def _fused_attn(device: torch.device):
    """The fused module where it runs (CUDA); None on CPU, where the
    reference path stands in for the candidate."""
    if device.type != "cuda":
        return None
    from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_fused

    return sm80_attn_fused


def attn_prologue(a: dict, device: torch.device, src: Source) -> Call:
    from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core
    from vllm.models.deepseek_v41.nvidia.ops.sm80_act_quant import act_quant_qdq_fp8

    output = src.output
    if output not in ATTN_PROLOGUE_OUTPUTS:
        raise ValueError(
            f"attn_prologue output {output!r} is not one of {ATTN_PROLOGUE_OUTPUTS}"
        )
    names = ("qr_kv", "q_norm", "kv_norm", "positions", "freqs", "swa_cache", "swa_slots")
    t = {k: _to(a[k], device) for k in names}
    eps = float(a["eps"])
    fused = _fused_attn(device)
    cache0, slots, pos = t["swa_cache"], t["swa_slots"], t["positions"]
    d = cache0.shape[-1]
    entries = cache0.shape[1]

    def ref(rows):
        qr, kv = t["qr_kv"][rows].split([t["qr_kv"].shape[1] - d, d], dim=-1)
        cache = cache0.clone()
        qr = core.rms_norm(qr, t["q_norm"], eps)
        kv = core.rms_norm(kv, t["kv_norm"], eps)
        core.store_rows(cache, slots[rows], core.window_kv(kv, pos[rows], t["freqs"]))
        return qr, cache

    def cand(rows):
        if fused is None:
            return ref(rows)
        cache = cache0.clone()
        qr = fused.prologue(
            t["qr_kv"][rows], t["q_norm"], t["kv_norm"], pos[rows], t["freqs"],
            cache, slots[rows], eps,
        )
        return qr, cache

    keep = slots >= 0
    written = slots[keep]

    def judged(qr, cache):
        if output == "qr":
            return qr
        return cache[written // entries, written % entries]

    whole = cand(slice(None))
    T = pos.shape[0]
    qr_part, kv_part = t["qr_kv"][:T].split([t["qr_kv"].shape[1] - d, d], dim=-1)
    if output == "qr":
        exact = _rms_fp64(qr_part, t["q_norm"], eps)
    else:
        kvn = _rms_fp64(kv_part, t["kv_norm"], eps).to(BF16)
        rope = _rope_fp64(kvn, pos, t["freqs"]).to(BF16)
        exact = act_quant_qdq_fp8(rope, 32).double()[keep]

    def alone(i):
        qr, cache = cand(slice(i, i + 1))
        if output == "qr":
            return bits_equal(qr, whole[0][i : i + 1])
        s = int(slots[i])
        if s < 0:
            return bits_equal(cache, cache0)
        return bits_equal(cache[s // entries, s % entries], whole[1][s // entries, s % entries])

    return Call(
        judged(*whole), judged(*ref(slice(None))), exact, all(alone(i) for i in picks(T))
    )


def attn_attend(a: dict, device: torch.device, _src: Source) -> Call:
    from tests.models.deepseek_v41.sm80.test_sm80_kernels_gpu import (
        _fp64_attention,
        _gather_rows,
    )
    from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core
    from vllm.models.deepseek_v41.nvidia.ops.sm80_act_quant import act_quant_qdq_fp8

    names = ("q", "positions", "freqs", "swa_cache", "swa_indices", "comp_cache",
             "comp_block_table", "comp_local", "req", "attn_sink", "out")
    t = {k: _to(a.get(k), device) for k in names}
    scale, groups = float(a["scale"]), int(a["n_groups"])
    q, pos, local = t["q"], t["positions"], t["comp_local"]
    fused = _fused_attn(device)
    T = pos.shape[0]

    def ref(rows):
        o = core.sparse_attention(
            core.rope(q[rows], pos[rows], t["freqs"]), t["swa_cache"],
            t["swa_indices"][rows], t["comp_cache"], t["comp_block_table"],
            None if local is None else local[rows], t["req"][rows], t["attn_sink"], scale,
        )
        return core.output_qdq(o, pos[rows], t["freqs"], groups)

    def cand(rows):
        if fused is None:
            return ref(rows)
        n = pos[rows].shape[0]
        out = t["out"][:n].clone()
        fused.attend(
            q[rows], pos[rows], t["freqs"], t["swa_cache"], t["swa_indices"][rows],
            t["comp_cache"], t["comp_block_table"],
            None if local is None else local[rows], t["req"][rows], t["attn_sink"],
            scale, groups, out,
        )
        return out

    whole = cand(slice(None))
    n_rows = t["swa_indices"].shape[1] + (0 if local is None else local.shape[1])
    h, d = q.shape[1], q.shape[2]
    step = max(1, CHUNK_BYTES // max(1, n_rows * (d + h) * 8))
    exacts = []
    for start in range(0, T, step):
        rows = slice(start, min(T, start + step))
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
        qr = _rope_fp64(q[rows], pos[rows], t["freqs"]).to(BF16)
        o = _fp64_attention(qr, gathered, valid, t["attn_sink"], scale).to(BF16)
        o = _rope_fp64(o, pos[rows], t["freqs"], inverse=True).to(BF16)
        n = o.shape[0]
        exacts.append(
            act_quant_qdq_fp8(o.view(n, groups, (h // groups) * d), 32).view(n, h, d)
        )
    alone = all(bits_equal(whole[i : i + 1], cand(slice(i, i + 1))) for i in picks(T))
    return Call(whole[:T], ref(slice(None)), torch.cat(exacts).double(), alone)


# Families with several outputs: the case names the one it judges.
MULTI_OUTPUT = {"mhc_delayed", "mhc_pre_delayed", "attn_prologue"}


FAMILIES = {
    "sparse_attention": sparse_attention,
    "index_scores": index_scores,
    "qdq_fp8": qdq_fp8,
    "qdq_fp4": qdq_fp4,
    "compress": compress,
    "index_keys": index_keys,
    "engram_dequant": engram_dequant,
    **{
        f"exl3_{op}": functools.partial(_exl3_moe, op)
        for op in ("moe_out", "moe_out_v2", "moe_out_prefill", "moe_out_prefill_v2")
    },
    "mhc_delayed": mhc_delayed,
    "mhc_pre_delayed": mhc_pre_delayed,
    "attn_prologue": attn_prologue,
    "attn_attend": attn_attend,
}


# Capture point and recorded arguments of each family (the profile's
# fp64_kernel.families lists the same).
_CORE = "vllm.models.deepseek_v41.nvidia.ops.sm80_attn_core"
_MOE_ARGS = ["x", "topk_ids", "topk_weights", "packed_meta", "expert_map",
             "intermediate_size", "swiglu_limit"]
_MHC = "vllm.models.deepseek_v41.nvidia.ops.sm80_mhc"
_MHC_ARGS = ["fn", "hc_scale", "hc_base", "rms_eps", "hc_pre_eps", "hc_sinkhorn_eps",
             "hc_post_mult_value", "sinkhorn_repeat", "pre_mix", "norm_weight", "norm_eps"]
_FUSED = "vllm.models.deepseek_v41.nvidia.ops.sm80_attn_fused"
POINTS = {
    "sparse_attention": (f"{_CORE}:sparse_attention",
                         ["q", "swa_cache", "swa_indices", "comp_cache",
                          "comp_block_table", "comp_local", "req", "sink", "scale"]),
    "index_scores": (f"{_CORE}:index_scores",
                     ["q", "w", "n_vis", "req", "idxk", "block_table", "n_cols",
                      "heads_per_rank"]),
    "qdq_fp8": (f"{_CORE}:qdq_fp8", ["x", "block"]),
    "qdq_fp4": (f"{_CORE}:qdq_fp4", ["x", "block", "scale"]),
    "compress": (f"{_CORE}:compress_rows",
                 ["kv_score", "norm_weight", "ring", "positions", "token_to_req",
                  "query_start_loc", "comp_slots", "ring_slots", "compress_ratio",
                  "head_dim", "eps"]),
    "index_keys": (f"{_CORE}:index_key_rows", ["k", "k_norm", "freqs", "group_pos", "eps"]),
    "engram_dequant": (
        "vllm.models.deepseek_v41.nvidia.ops.sm80_engram_dequant:dequant_engram_rows",
        ["values", "scales"]),
    "exl3_moe_out": ("vllm.sm80.kernels.exl3.ops:moe_out", _MOE_ARGS),
    "exl3_moe_out_v2": ("vllm.sm80.kernels.exl3.ops:moe_out_v2", _MOE_ARGS + ["bits_x2"]),
    "exl3_moe_out_prefill": ("vllm.sm80.kernels.exl3.ops:moe_out_prefill", _MOE_ARGS),
    "exl3_moe_out_prefill_v2": ("vllm.sm80.kernels.exl3.ops:moe_out_prefill_v2",
                                _MOE_ARGS + ["bits_x2"]),
    "mhc_delayed": (f"{_MHC}:mhc_fused_post_pre_delayed",
                    ["x", "residual", "post_layer_mix", "comb_res_mix", *_MHC_ARGS]),
    "mhc_pre_delayed": (f"{_MHC}:mhc_pre_delayed", ["residual", *_MHC_ARGS]),
    "attn_prologue": (f"{_FUSED}:prologue",
                      ["qr_kv", "q_norm", "kv_norm", "positions", "freqs", "swa_cache",
                       "swa_slots", "eps"]),
    "attn_attend": (f"{_FUSED}:attend",
                    ["q", "positions", "freqs", "swa_cache", "swa_indices", "comp_cache",
                     "comp_block_table", "comp_local", "req", "attn_sink", "scale",
                     "n_groups", "out"]),
}


# ---------------------------------------------------------------- driver


def case_row(
    case: dict, device: torch.device, checkpoint: dict, family_output: str | None = None
) -> dict:
    """``family_output``: the output a multi-output family judges when
    neither the case's ``output`` nor its ``family`` (after ``:``) names one
    (the cases file's ``family`` suffix)."""
    sources = case.get("captured_sources")
    if not sources:
        raise ValueError("no captured_sources: the case cannot be replayed")
    case_family, _, case_output = str(case.get("family") or "").partition(":")
    output = case.get("output") or case_output or family_output
    cand_sum = cand_max = ref_sum = ref_max = 0.0
    count = numel = 0
    bitwise = invariant = True
    families = set()
    for path in sources:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if obj.get("schema") != SOURCE_SCHEMA:
            raise ValueError(f"{path}: schema {obj.get('schema')!r} is not {SOURCE_SCHEMA}")
        family = obj.get("family")
        if family not in FAMILIES:
            raise ValueError(f"{path}: family {family!r} is not one of {sorted(FAMILIES)}")
        if (case_family or family) != family:
            raise ValueError(f"{path}: family {family!r} in a case of {case['family']!r}")
        multi = family in MULTI_OUTPUT
        families.add(f"{family}:{output}" if multi else family)
        src = Source(obj.get("meta") or {}, checkpoint, output if multi else None)
        call = FAMILIES[family](obj["tensors"], device, src)
        cs, cm, n = abs_errors(call.cand, call.exact)
        ref_exact = call.exact if call.ref_exact is None else call.ref_exact
        rs, rm, _ = abs_errors(call.ref, ref_exact)
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
    """Write the header and one line per case that runs. 0 when every case
    has its line, 1 when one has none (each named on stderr), 2 when the
    case file is not one of SCHEMAS."""
    doc = json.loads(Path(cases).read_text())
    if not isinstance(doc, dict) or doc.get("schema") not in SCHEMAS:
        print(f"{cases}: schema is not one of {SCHEMAS}", file=sys.stderr)
        return 2
    with Path(out).open("w") as f, torch.inference_mode():
        f.write(json.dumps({"header": header(device)}) + "\n")
        checkpoint = doc.get("checkpoint") or {}
        family_output = str(doc.get("family") or "").partition(":")[2] or None
        failed = []
        for case in doc.get("cases") or []:
            try:
                row = case_row(case, device, checkpoint, family_output)
            except Exception as exc:  # noqa: BLE001 - reported, the other cases still run
                print(f"case {case.get('id')}: {type(exc).__name__}: {exc}", file=sys.stderr)
                failed.append(str(case.get("id")))
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
    if failed:
        print(f"fp64_cases: {len(failed)} case(s) without a result: {', '.join(failed)}",
              file=sys.stderr)
        return 1
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
