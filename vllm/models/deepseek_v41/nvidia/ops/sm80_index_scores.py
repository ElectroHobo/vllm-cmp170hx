# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 index scores on sm_80 (Triton), over a paged BF16 key cache.

Same contract and rounding points as ``sm80_attn_core.index_scores_torch``:
one program per (query, tile of compressed positions) computes the 32 heads'
fp32 dot products with one MMA, rounds each to BF16, rectifies, multiplies by
the head's BF16 weight and rounds again, then sums each rank's heads in head
order (fp32, rounded to BF16) and adds the ranks in order with BF16 rounding.
Positions at or past the query's visible count score ``-inf``. No atomics;
every output element is written by exactly one program.
"""

import torch

from vllm.triton_utils import tl, triton

_BLOCK_N = 64


@triton.jit
def _row(s, hidx, h: tl.constexpr):
    """Row ``h`` of ``s`` ([H, N]) as an [N] tensor (adds only zeros)."""
    return tl.sum(tl.where(hidx[:, None] == h, s, 0.0), axis=0)


@triton.jit(do_not_specialize=["n_cols"])
def _index_scores_kernel(
    q_ptr,
    w_ptr,
    nvis_ptr,
    req_ptr,
    k_ptr,
    bt_ptr,
    out_ptr,
    n_cols,
    stride_qt,
    stride_qh,
    stride_wt,
    stride_kb,
    stride_ke,
    stride_bt,
    stride_ot,
    ENTRIES: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    HPR: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    n_vis = tl.load(nvis_ptr + t).to(tl.int32)
    valid = cols < n_vis
    req = tl.load(req_ptr + t).to(tl.int64)
    blk = tl.load(bt_ptr + req * stride_bt + cols // ENTRIES, mask=valid, other=0)
    koff = blk.to(tl.int64) * stride_kb + (cols % ENTRIES).to(tl.int64) * stride_ke
    d = tl.arange(0, D)
    k = tl.load(k_ptr + koff[:, None] + d[None, :], mask=valid[:, None], other=0.0)
    hidx = tl.arange(0, H)
    q = tl.load(q_ptr + t * stride_qt + hidx[:, None] * stride_qh + d[None, :])
    s = tl.dot(q, tl.trans(k))  # [H, BLOCK_N] fp32
    s = tl.maximum(s.to(tl.bfloat16).to(tl.float32), 0.0)
    w = tl.load(w_ptr + t * stride_wt + hidx).to(tl.float32)
    s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
    total = tl.zeros([BLOCK_N], dtype=tl.float32)
    for r in tl.static_range(H // HPR):
        part = _row(s, hidx, r * HPR)
        for i in tl.static_range(1, HPR):
            part = part + _row(s, hidx, r * HPR + i)
        part = part.to(tl.bfloat16).to(tl.float32)
        if r == 0:
            total = part
        else:
            total = (total + part).to(tl.bfloat16).to(tl.float32)
    total = tl.where(valid, total, float("-inf"))
    tl.store(out_ptr + t * stride_ot + cols, total, mask=cols < n_cols)


def index_scores_triton(
    q: torch.Tensor,
    w: torch.Tensor,
    n_vis: torch.Tensor,
    req: torch.Tensor,
    idxk: torch.Tensor,
    block_table: torch.Tensor,
    n_cols: int,
    heads_per_rank: int,
) -> torch.Tensor:
    """``[n, n_cols]`` fp32 index scores; see ``index_scores_torch``.

    q: [n, H, D] BF16 (stride 1 on D); w: [n, H] BF16; n_vis, req: [n];
    idxk: [blocks, E, D] BF16 (any block stride); block_table: [R, nb] int32.
    """
    n, h, d = q.shape
    assert q.stride(2) == 1 and idxk.stride(2) == 1 and w.stride(1) == 1
    assert h % heads_per_rank == 0
    out = torch.empty((n, n_cols), dtype=torch.float32, device=q.device)
    if n == 0 or n_cols == 0:
        return out
    grid = (n, triton.cdiv(n_cols, _BLOCK_N))
    _index_scores_kernel[grid](
        q,
        w,
        n_vis.to(torch.int32).contiguous(),
        req.to(torch.int32).contiguous(),
        idxk,
        block_table,
        out,
        n_cols,
        q.stride(0),
        q.stride(1),
        w.stride(0),
        idxk.stride(0),
        idxk.stride(1),
        block_table.stride(0),
        out.stride(0),
        ENTRIES=idxk.shape[1],
        H=h,
        D=d,
        HPR=heads_per_rank,
        BLOCK_N=_BLOCK_N,
        num_warps=4,
    )
    return out
