# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, stable radix selection for paged DeepSeek index scores.

Four byte passes find the exact FP32 cutoff. Segment counts and a prefix
scatter emit ascending positions, taking the earliest equal scores. No heap,
full-context sort, tensor-driven host loop or inter-CTA waiting is used.
The score spool remains live until BOTH token and candidate selectors finish.
"""
import torch

from vllm.triton_utils import tl, triton

SEGMENT = 1024
RADIX = 256


@triton.jit
def float_key(x):
    # Numerical order, including -inf < -FLT_MAX; both zeros tie.
    bits = tl.where(x == 0.0, 0.0, x).to(tl.uint32, bitcast=True)
    return bits ^ tl.where((bits & 0x80000000) != 0, 0xffffffff, 0x80000000).to(tl.uint32)


@triton.jit
def _selection_init(out, state, nvis, k: tl.constexpr, os: tl.constexpr,
                    NS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1) * B + tl.arange(0, B)
    tl.store(out + row * os + j, -1, j < k)
    if tl.program_id(1) == 0:
        tl.store(state + row * 3, 0)
        tl.store(state + row * 3 + 1, tl.minimum(tl.load(nvis + row), k))
        tl.store(state + row * 3 + 2, 0)


@triton.jit
def _radix_hist(scores, nvis, state, hist, width, ss: tl.constexpr,
                NS: tl.constexpr, SHIFT: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    seg = tl.program_id(1)
    col = seg * B + tl.arange(0, B)
    valid = (col < width) & (col < tl.load(nvis + row))
    x = tl.load(scores + row * ss + col, col < width, float('-inf'))
    key = float_key(x)
    prefix = tl.load(state + row * 3).to(tl.uint32)
    if SHIFT < 24:
        valid = valid & ((key >> (SHIFT + 8)) == (prefix >> (SHIFT + 8)))
    byte = ((key >> SHIFT) & 255).to(tl.int32)
    counts = tl.histogram(tl.where(valid, byte, 256), 256)
    tl.store(hist + (row * NS + seg) * 256 + tl.arange(0, 256), counts)


@triton.jit
def _hist_reduce(hist, totals, NS: tl.constexpr, BS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    byte = tl.program_id(1)
    seg = tl.arange(0, BS)
    c = tl.load(hist + (row * NS + seg) * 256 + byte, seg < NS, 0)
    tl.store(totals + row * 256 + byte, tl.sum(c, 0))


@triton.jit
def _radix_cutoff(totals, state, SHIFT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    byte = tl.arange(0, 256)
    counts = tl.load(totals + row * 256 + byte)
    higher = tl.sum(counts, 0) - tl.cumsum(counts, 0)
    remaining = tl.load(state + row * 3 + 1)
    wins = (higher < remaining) & (higher + counts >= remaining)
    bucket = tl.max(tl.where(wins, byte, 0), 0).to(tl.uint32)
    above = tl.sum(tl.where(wins, higher, 0), 0)
    prefix = tl.load(state + row * 3).to(tl.uint32)
    tl.store(state + row * 3, prefix | (bucket << SHIFT))
    tl.store(state + row * 3 + 1, remaining - above)


@triton.jit
def _selection_counts(scores, nvis, state, counts, width, ss: tl.constexpr,
                      NS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    seg = tl.program_id(1)
    col = seg * B + tl.arange(0, B)
    valid = (col < width) & (col < tl.load(nvis + row))
    x = tl.load(scores + row * ss + col, col < width, float('-inf'))
    key = float_key(x)
    cutoff = tl.load(state + row * 3).to(tl.uint32)
    tl.store(counts + (row * NS + seg) * 2, tl.sum((valid & (key > cutoff)).to(tl.int32), 0))
    tl.store(counts + (row * NS + seg) * 2 + 1, tl.sum((valid & (key == cutoff)).to(tl.int32), 0))


@triton.jit
def _selection_prefix(counts, prefixes, state, NS: tl.constexpr, BS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    seg = tl.arange(0, BS)
    gt = tl.load(counts + (row * NS + seg) * 2, seg < NS, 0)
    eq = tl.load(counts + (row * NS + seg) * 2 + 1, seg < NS, 0)
    tl.store(prefixes + (row * NS + seg) * 2, tl.cumsum(gt, 0) - gt, seg < NS)
    tl.store(prefixes + (row * NS + seg) * 2 + 1, tl.cumsum(eq, 0) - eq, seg < NS)


@triton.jit
def _selection_scatter(scores, nvis, state, prefixes, out, width,
                       ss: tl.constexpr, os: tl.constexpr, NS: tl.constexpr,
                       K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    seg = tl.program_id(1)
    col = seg * B + tl.arange(0, B)
    visible = tl.load(nvis + row)
    valid = (col < width) & (col < visible)
    x = tl.load(scores + row * ss + col, col < width, float('-inf'))
    key = float_key(x)
    cutoff = tl.load(state + row * 3).to(tl.uint32)
    equal_budget = tl.load(state + row * 3 + 1)
    gt_before = tl.load(prefixes + (row * NS + seg) * 2)
    eq_before = tl.load(prefixes + (row * NS + seg) * 2 + 1)
    equal = valid & (key == cutoff)
    keep = valid & ((key > cutoff) | (equal & (eq_before + tl.cumsum(equal.to(tl.int32), 0) <= equal_budget)))
    index = gt_before + tl.minimum(eq_before, equal_budget) + tl.cumsum(keep.to(tl.int32), 0) - 1
    tl.store(out + row * os + index, col.to(tl.int32), keep & (index < K))


class SelectionWorkspace:
    """Fixed capacity, one independently executable stream/runner's scratch."""
    def __init__(self, rows, width, device):
        self.rows, self.width = rows, width
        self.segments = triton.cdiv(max(width, 1), SEGMENT)
        self.hist = torch.empty((rows, self.segments, RADIX), dtype=torch.int32, device=device)
        self.totals = torch.empty((rows, RADIX), dtype=torch.int32, device=device)
        self.state = torch.empty((rows, 3), dtype=torch.int64, device=device)
        self.counts = torch.empty((rows, self.segments, 2), dtype=torch.int32, device=device)
        self.prefixes = torch.empty_like(self.counts)

    @staticmethod
    def bytes(rows, width):
        segments = triton.cdiv(max(width, 1), SEGMENT)
        return rows * (segments * (RADIX + 4) * 4 + RADIX * 4 + 3 * 8)


def select(scores, n_vis, out, workspace, *, first_histogram=False):
    """Write canonical IDs directly into the persistent sharing buffer.

    ``first_histogram`` means the score producer has populated the high-byte
    histogram in exactly this workspace layout. Visibility is bounded by the
    caller; nonfinite arithmetic is rejected by validation, not repaired here.
    """
    rows, width = scores.shape
    if rows > workspace.rows or width > workspace.width:
        raise ValueError('selection exceeds reserved capacity')
    if rows == 0:
        return
    k = out.shape[1]
    if k == 0:
        return
    ns = triton.cdiv(max(width, 1), SEGMENT)
    _selection_init[(rows, triton.cdiv(k, 256))](out, workspace.state, n_vis, k,
        out.stride(0), ns, 256)
    for shift in (24, 16, 8, 0):
        if shift != 24 or not first_histogram:
            _radix_hist[(rows, ns)](scores, n_vis, workspace.state, workspace.hist,
                width, scores.stride(0), ns, shift, SEGMENT)
        _hist_reduce[(rows, 256)](workspace.hist, workspace.totals, ns,
            triton.next_power_of_2(ns))
        _radix_cutoff[(rows,)](workspace.totals, workspace.state, shift)
    _selection_counts[(rows, ns)](scores, n_vis, workspace.state, workspace.counts,
        width, scores.stride(0), ns, SEGMENT)
    _selection_prefix[(rows,)](workspace.counts, workspace.prefixes, workspace.state,
        ns, triton.next_power_of_2(ns))
    _selection_scatter[(rows, ns)](scores, n_vis, workspace.state, workspace.prefixes,
        out, width, scores.stride(0), out.stride(0), ns, k, SEGMENT)
