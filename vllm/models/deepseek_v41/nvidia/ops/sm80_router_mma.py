# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 MoE gate + routing on the BF16 tensor cores (sm_80), as two
launches that share their scratch: :func:`gate` (``sm80_gate_mma.py``: exact
BF16 products, FP32 sums per K chunk, FP64 across chunks, one FP32 rounding)
and :func:`route` (``sm80_route_mma.py``: FP32 selection as the served route,
weights in FP64 rounded once, from exactly recomputed logits up to
:data:`EXACT_ROWS` rows). For decode rows with split K the gate leaves its
FP64 partials and the route completes them, so :func:`route` must consume the
:class:`Pending` of the :func:`gate` call that produced its logits.

Served configuration: hidden 5120, 384 experts top 6 (target) or 128 experts
top 3 (drafter), routed scaling 1.5, renormalised, gate_temp 1, image sentinel
ids [129264, 129269), BF16 x of 1 to :data:`MAX_ROWS` rows (:func:`serves`).
"""

from dataclasses import dataclass

import torch

from vllm.models.deepseek_v41.nvidia.ops import sm80_gate_mma as G
from vllm.models.deepseek_v41.nvidia.ops import sm80_route_mma as R
from vllm.triton_utils import triton

HIDDEN = 5120
EXPERTS_TOP_K = {384: 6, 128: 3}
SCALE = 1.5
SENTINEL_LO = 129264
SENTINEL_HI = SENTINEL_LO + 5
MAX_ROWS = 2047
# Rows whose weights come from the exactly recomputed logits; above, from the
# gate's FP32 logits.
EXACT_ROWS = 64
_ROUTE_WARPS = 4
_KP_MAX = 8
_SMS: dict = {}
_LOCKS: dict = {}


def supports(num_experts: int, hidden: int, top_k: int, renormalize: bool,
             scale: float, gate_temp: float, sentinel_lo: int) -> str | None:
    """None when this gate + route serves the layer, else the reason."""
    if EXPERTS_TOP_K.get(num_experts) != top_k or hidden != HIDDEN:
        return f"{num_experts} experts top {top_k}, hidden {hidden}"
    if not renormalize or scale != SCALE or gate_temp != 1.0:
        return f"renormalize {renormalize}, scale {scale}, gate_temp {gate_temp}"
    if sentinel_lo not in (0, SENTINEL_LO):
        return f"image sentinel {sentinel_lo}"
    return None


def serves(x: torch.Tensor) -> bool:
    """A call this path takes: CUDA sm_80, BF16 [1..MAX_ROWS, 5120], unit
    column stride, even row stride."""
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and 0 < x.shape[0] <= MAX_ROWS
        and x.shape[1] == HIDDEN
        and x.stride(1) == 1
        and x.stride(0) % 2 == 0
        and torch.cuda.get_device_capability(x.device) == (8, 0)
    )


def _sms(device: torch.device) -> int:
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _SMS:
        _SMS[key] = torch.cuda.get_device_properties(key).multi_processor_count
    return _SMS[key]


def _locks(device: torch.device) -> torch.Tensor:
    """The per-device counters (zero between calls); allocated on the first,
    eager call, before any graph capture."""
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _LOCKS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MoE gate counters must be allocated before graph capture")
        _LOCKS[key] = torch.zeros(G.LOCK_SLOTS, dtype=torch.int32, device=device)
    return _LOCKS[key]


def _route_split(M: int, K: int, sms: int) -> tuple[int, int]:
    """(SPLIT, BLOCK_D) of the route's exact recompute: the row's K over SPLIT
    programs, up to two per SM."""
    split = 1
    if M <= EXACT_ROWS:
        split = max(s for s in (1, 2, 4, 5) if K % s == 0 and (s == 1 or M * s <= 2 * sms))
    ds = K // split
    return split, min(512, ds & -ds)


@dataclass
class Pending:
    """One gate call's state for its :func:`route`."""

    x: torch.Tensor
    logits: torch.Tensor
    gpart: torch.Tensor
    rpart: torch.Tensor
    locks: torch.Tensor
    exact: bool
    gsplit: int
    gate_rows: int
    gcols: int
    split: int
    block_d: int


def gate(x: torch.Tensor, weight_bf16: torch.Tensor) -> tuple[torch.Tensor, Pending]:
    """FP32 logits [M, E] of ``x @ weight^T`` (complete unless the returned
    :class:`Pending` defers them to :func:`route`)."""
    M, K = x.shape
    E = weight_bf16.shape[0]
    sms = _sms(x.device)
    split, block_d = _route_split(M, K, sms)
    ng = G.part_numel(M, E, K, sms)
    nr = M * split * _KP_MAX if split > 1 else 0
    scratch = torch.empty(2 * (ng + nr) + M * E, dtype=torch.float32, device=x.device)
    gpart = scratch[: 2 * ng].view(torch.float64)
    rpart = scratch[2 * ng : 2 * (ng + nr)].view(torch.float64)
    logits = scratch[2 * (ng + nr) :].view(M, E)
    exact = M <= EXACT_ROWS
    block_t, block_e, _, gsplit, tb, _, _ = G.config(M, E, K, sms)
    defer = exact and gsplit > 1
    locks = _locks(x.device)
    G.launch(x, weight_bf16, logits, gpart, locks, sms, defer)
    return logits, Pending(
        x=x, logits=logits, gpart=gpart, rpart=rpart, locks=locks, exact=exact,
        gsplit=gsplit if defer else 1, gate_rows=tb * block_t,
        gcols=triton.cdiv(E, block_e) * block_e, split=split, block_d=block_d,
    )


def route(
    p: Pending,
    weight_bf16: torch.Tensor,
    bias: torch.Tensor,
    bias_vl: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    top_k: int,
    sentinel_lo: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 weights and int32 expert ids [M, top_k] of the gate call ``p``
    (image tokens: ``input_ids`` in [sentinel_lo, sentinel_lo + 5) select with
    ``bias_vl``; none when ``sentinel_lo`` is 0)."""
    M, E = p.logits.shape
    weights = torch.empty(M, top_k, dtype=torch.float32, device=p.x.device)
    ids = torch.empty(M, top_k, dtype=torch.int32, device=p.x.device)
    if input_ids is None or bias_vl is None or sentinel_lo <= 0:
        # No image tokens: the sentinel range is empty and the token reads
        # (of ``ids``' bytes) select nothing.
        tokens, lo, hi, bias_vl = ids, 0, 0, bias
    else:
        assert input_ids.shape == (M,) and input_ids.is_contiguous(), input_ids.shape
        tokens, lo, hi = input_ids, sentinel_lo, sentinel_lo + (SENTINEL_HI - SENTINEL_LO)
    kp = triton.next_power_of_2(top_k)
    assert kp <= _KP_MAX, top_k
    R._route_kernel[(M, p.split)](
        p.logits, p.logits.stride(0), p.x, p.x.stride(0), weight_bf16, bias, bias_vl, tokens, weights, ids,
        p.rpart, p.locks, p.gpart, p.gate_rows, E, lo, hi, SCALE, 1e-20, HIDDEN=HIDDEN, K=top_k, KP=kp,
        BLOCK_E=triton.next_power_of_2(E), BLOCK_D=p.block_d, SPLIT=p.split, EXACT=p.exact,
        GSPLIT=p.gsplit, GCOLS=p.gcols, num_warps=_ROUTE_WARPS)
    return weights, ids


def gate_route(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    bias_vl: torch.Tensor | None,
    top_k: int,
    input_ids: torch.Tensor | None = None,
    sentinel_lo: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`gate` then :func:`route` of one call; ``weight`` [E, 5120] BF16
    (or FP32 holding BF16 values)."""
    w16 = (weight if weight.dtype == torch.bfloat16 else weight.to(torch.bfloat16)).contiguous()
    _, pending = gate(x, w16)
    return route(pending, w16, bias, bias_vl, input_ids, top_k, sentinel_lo)
