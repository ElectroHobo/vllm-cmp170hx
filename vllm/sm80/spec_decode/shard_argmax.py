# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy DSpark Markov sampling with a per-rank argmax, without the
all-gather of the draft logits.

The DSpark drafter samples its block left to right: at step i the draft
logits of position i are its base logits plus a Markov bias computed from the
token drafted at step i - 1, and a greedy draft is their argmax. Upstream
builds the base logits with the vocabulary-parallel head and all-gathers
them over the TP group (``compute_draft_logits``), only to take that argmax.

With ``VLLM_DEEPSEEK_V41_DSPARK_SHARD_ARGMAX`` the drafter's
``compute_logits_local`` keeps each rank's own columns of the base logits,
and every step:

1. adds the Markov bias's columns of that shard, with the same torch
   addition the gathered path runs over the whole row;
2. :func:`chunk_slots` reduces each ``BLOCK``-column chunk to the first index
   of its largest value and writes it into the chunk's 16-byte slot of a BF16
   buffer, every other rank's slots zero;
3. the buffer is summed over the TP group (``tensor_model_parallel_all_reduce``);
4. :func:`pick` takes, per row, the slot with the largest value, the lowest
   slot on ties, and returns its token.

Exactness. The draft is torch.argmax's over the gathered row: ATen's
``GreaterOrNan`` prefers a NaN to any number, the lower index between two
NaNs or two equal values, and compares -0.0 equal to +0.0. The kernels
compare a 32-bit key built from the value's bits alone: -0.0 becomes +0.0,
every NaN the largest key, and otherwise the key keeps the IEEE order. No
floating-point comparison runs, so no flush of subnormals can merge two
values. A chunk takes the lowest index among its largest keys, and the slots
lie in global column order (rank, then chunk), so the lowest slot among equal
keys holds the lowest index. A slot carries the key in four BF16 lanes of one
byte each and the global index in three (the eighth is zero): integers up to
255 are exact in BF16, and adding zeros is exact under any all-reduce
algorithm or accumulation dtype, so the buffer arrives intact. Columns at or
beyond the vocabulary size (padding) are never read, as the gathered path
cuts them off.
"""

import functools
from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

BLOCK = 4096  # columns per chunk; one slot per chunk
LANES = 8  # BF16 lanes per slot: 4 key bytes, 3 index bytes, 1 zero
MAX_VOCAB = 1 << 24  # three index bytes
SWITCH = "VLLM_DEEPSEEK_V41_DSPARK_SHARD_ARGMAX"


@triton.jit
def _lane_shift(lane):
    # Key bytes in lanes 0-3 and index bytes in lanes 4-6, most significant
    # first; lane 7 carries nothing.
    return tl.where(lane < 4, 24 - 8 * lane, tl.where(lane < 7, 48 - 8 * lane, 0))


@triton.jit
def _chunk_slots_kernel(
    logits_ptr,
    logits_stride,
    slots_ptr,
    n_cols,
    col_start,
    rank,
    TP: tl.constexpr,
    TP_P2: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_cols
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=valid, other=0.0)
    bits = x.to(tl.int32, bitcast=True)
    bits = tl.where(bits == -2147483648, 0, bits)  # -0.0 -> +0.0
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    wide = bits.to(tl.int64)
    # Non-negative floats above every negative one; negative floats ordered
    # by decreasing magnitude; NaN above +inf (0xFF800000).
    key = tl.where(wide >= 0, wide + 2147483648, -1 - wide)
    key = tl.where(is_nan, 4294967295, key)
    key = tl.where(valid, key, -1)
    best = tl.max(key, axis=0)
    first = tl.min(tl.where(key == best, offs, 2147483647), axis=0)
    # A chunk without columns sends key 0, below every value's key (-inf's is
    # 0x007FFFFF), so it is never picked.
    empty = best < 0
    enc_key = tl.where(empty, 0, best)
    index = tl.where(empty, 0, col_start + first.to(tl.int64))
    lane = tl.arange(0, 8)
    shift = _lane_shift(lane).to(tl.int64)
    src = tl.where(lane < 4, enc_key, index)
    byte = tl.where(lane < 7, (src >> shift) & 255, 0)
    r = tl.arange(0, TP_P2)
    vals = tl.where(r[:, None] == rank, byte[None, :], 0)
    ptrs = (
        slots_ptr
        + row * (TP * CHUNKS * 8)
        + (r[:, None] * CHUNKS + chunk) * 8
        + lane[None, :]
    )
    tl.store(ptrs, vals.to(tl.float32).to(tl.bfloat16), mask=r[:, None] < TP)


@triton.jit
def _pick_kernel(slots_ptr, out_ptr, SLOTS: tl.constexpr, SLOTS_P2: tl.constexpr):
    row = tl.program_id(0)
    s = tl.arange(0, SLOTS_P2)
    lane = tl.arange(0, 8)
    v = tl.load(
        slots_ptr + row * (SLOTS * 8) + s[:, None] * 8 + lane[None, :],
        mask=s[:, None] < SLOTS,
        other=0.0,
    )
    part = v.to(tl.float32).to(tl.int64) << _lane_shift(lane).to(tl.int64)[None, :]
    key = tl.sum(tl.where(lane[None, :] < 4, part, 0), axis=1)
    index = tl.sum(tl.where((lane[None, :] >= 4) & (lane[None, :] < 7), part, 0), axis=1)
    key = tl.where(s < SLOTS, key, -1)
    best = tl.max(key, axis=0)
    first = tl.min(tl.where(key == best, s, SLOTS_P2), axis=0)
    tl.store(out_ptr + row, tl.sum(tl.where(s == first, index, 0), axis=0))


def chunk_slots(
    logits: torch.Tensor, col_start: int, rank: int, tp_size: int, chunks: int
) -> torch.Tensor:
    """This rank's slots of one sampling step.

    ``logits`` [rows, n] FP32 holds global columns ``col_start ..
    col_start + n - 1`` (n <= chunks * BLOCK). Returns [rows, tp_size * chunks
    * LANES] BF16: this rank's ``chunks`` slots filled, every other rank's
    zero, ready to be summed over the TP group.
    """
    rows, n_cols = logits.shape
    if logits.dtype != torch.float32 or logits.stride(-1) != 1:
        raise ValueError("chunk_slots takes FP32 logits with unit column stride")
    if n_cols > chunks * BLOCK or col_start + n_cols > MAX_VOCAB:
        raise ValueError(f"{n_cols} columns from {col_start} exceed the slots")
    slots = torch.empty(
        rows, tp_size * chunks * LANES, dtype=torch.bfloat16, device=logits.device
    )
    if rows:
        _chunk_slots_kernel[(rows, chunks)](
            logits,
            logits.stride(0),
            slots,
            n_cols,
            col_start,
            rank,
            TP=tp_size,
            TP_P2=triton.next_power_of_2(tp_size),
            CHUNKS=chunks,
            BLOCK=BLOCK,
            num_warps=8,
        )
    return slots


def pick(slots: torch.Tensor) -> torch.Tensor:
    """The token of each row of summed slots ([rows, slots * LANES] BF16):
    the index in the slot with the largest key, the lowest slot on ties."""
    rows = slots.shape[0]
    num_slots = slots.shape[1] // LANES
    out = torch.empty(rows, dtype=torch.int64, device=slots.device)
    if rows:
        _pick_kernel[(rows,)](
            slots,
            out,
            SLOTS=num_slots,
            SLOTS_P2=triton.next_power_of_2(num_slots),
            num_warps=1,
        )
    return out


def refusal(speculator) -> str | None:
    """Why the per-rank argmax cannot replace this speculator's sampling."""
    model = speculator.model
    if not hasattr(model, "compute_logits_local"):
        return (
            f"{type(model).__name__} has no vocabulary-shard logits "
            "(compute_logits_local)"
        )
    if speculator.draft_logits is not None:
        return "probabilistic drafting samples the full draft logits"
    if speculator._draft_topk is not None:
        return "draft top-k (dspark_draft_topk) selects on the full draft logits"
    if speculator.acceptance_estimator is not None:
        return "the acceptance estimator reads the full draft logits"
    if model.draft_id_to_target_id is not None:
        return "a reduced draft vocabulary maps draft ids to target ids"
    if model.logits_processor.org_vocab_size > MAX_VOCAB:
        return f"a vocabulary above {MAX_VOCAB} tokens"
    return None


def _enabled(speculator) -> bool:
    on = getattr(speculator, "_shard_argmax", None)
    if on is None:
        on = bool(getattr(envs, SWITCH))
        if on:
            reason = refusal(speculator)
            if reason is not None:
                raise ValueError(f"{SWITCH}: {reason}")
            logger.info_once(
                "DSpark greedy Markov sampling takes a per-rank argmax "
                "(%s=1): no all-gather of the draft logits.",
                SWITCH,
            )
        speculator._shard_argmax = on
    return on


def sample_sequential(speculator, num_reqs: int, head_hidden: torch.Tensor) -> None:
    """``DSparkSpeculator._sample_sequential`` for greedy drafts, with the
    argmax taken per rank (see the module docstring)."""
    s = speculator
    model = s.model
    n_spec = s.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[s.sample_indices[:num_sample]]
    local = model.compute_logits_local(sample_hidden)
    width = local.shape[-1]
    local = local.view(num_reqs, n_spec, width)
    # The gathered row is the ranks' columns in rank order, cut at the
    # vocabulary size.
    rank, tp_size = model.lm_head.tp_rank, model.lm_head.tp_size
    col_start = rank * width
    n_cols = min(max(model.logits_processor.org_vocab_size - col_start, 0), width)
    chunks = triton.cdiv(width, BLOCK)
    confidence_markov_embeds = []

    prev = s.input_buffers.input_ids[s._anchor_idx[:num_reqs]]
    for i in range(n_spec):
        markov_embed = model.markov_embed(prev)
        if s.use_confidence_head:
            confidence_markov_embeds.append(markov_embed)
        bias = model.markov_bias(markov_embed)
        logits_i = local[:, i, :n_cols] + bias[:, col_start : col_start + n_cols]
        if logits_i.dtype != torch.float32:
            logits_i = logits_i.float()  # exact; keeps order, ties and NaN
        slots = chunk_slots(logits_i, col_start, rank, tp_size, chunks)
        if tp_size > 1:
            slots = tensor_model_parallel_all_reduce(slots)
        draft_sampled_i = pick(slots)
        s.draft_tokens[:num_reqs, i] = draft_sampled_i
        prev = draft_sampled_i

    if s.use_confidence_head:
        confidence = model.compute_confidence(
            sample_hidden,
            torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
        )
        s.draft_token_confidence_probs[:num_reqs] = confidence.view(num_reqs, n_spec)


def shard_argmax_sampling(
    sample: Callable[..., None],
) -> Callable[..., None]:
    """Decorates ``DSparkSpeculator._sample_sequential``: with
    ``VLLM_DEEPSEEK_V41_DSPARK_SHARD_ARGMAX`` the greedy drafts come from
    :func:`sample_sequential`; the switch refuses a speculator it cannot
    serve (:func:`refusal`)."""

    @functools.wraps(sample)
    def wrapper(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if not _enabled(self):
            return sample(self, num_reqs, head_hidden)
        sample_sequential(self, num_reqs, head_hidden)

    return wrapper
