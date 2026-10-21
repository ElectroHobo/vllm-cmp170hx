# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark row mask, draft side: the confidence head's outputs per request slot.

With ``VLLM_DSPARK_ROW_MASK`` set, ``create_dspark_speculator`` mixes
:class:`RowMaskConfidenceMixin` in front of the DSpark speculator class. It

* turns the drafter's confidence head on after the draft model loads (and
  refuses a drafter without one), and creates the shared ``RowMaskState``
  (``dspark_row_mask.py``);
* at the end of every draft step's sequential sampling (the default, top-k
  and per-rank-argmax samplers alike, at any draft width) publishes the
  step's confidences into ``RowMaskState.conf_by_slot``: request r's row goes
  to its request-state slot, its first ``width`` columns are the head's
  outputs p_1 .. p_width and the columns past the drafted width are 0 (a
  draft that does not exist is never accepted);
* on a ``propose`` call that drafts nothing (a zero-draft step stores only the
  context K/V, ``zero_draft.py``), resets the batch's slots to 1.0, so no
  later verify step reads an older step's confidences.

The publish is one small kernel recorded inside the captured draft step. It
reads the slots from ``sample_idx_mapping``, the drafter's persistent
per-sample request-state index (column 0 of each request's ``width`` sample
rows): ``prepare_dflash_inputs`` writes it before every draft step, eager or
replayed, with -1 for every request past the batch up to ``max_num_reqs``,
and ``capture`` fills it with -1. A request with slot -1 (graph padding)
writes nothing. The width and the strides are fixed per captured graph: each
draft width has its own graphs, captured at that width.

Drafted tokens do not change: the head runs on the sampler's existing
inputs, and the publish only reads them.

Composes with ``VLLM_DSPARK_CONFIDENCE_DUMP``: both mixins only read the
head's outputs.
"""

from __future__ import annotations

import inspect
from functools import cache

import torch

from vllm.logger import init_logger
from vllm.sm80.spec_decode.dspark_row_mask import RowMaskState, init_row_mask_state
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _publish_kernel(
    conf_ptr,
    conf_stride,
    slot_ptr,
    slot_stride,
    out_ptr,
    out_stride,
    num_slots,
    width,
    max_draft,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    slot = tl.load(slot_ptr + req * slot_stride)
    if (slot >= 0) & (slot < num_slots):
        cols = tl.arange(0, BLOCK)
        conf = tl.load(conf_ptr + req * conf_stride + cols, mask=cols < width, other=0.0)
        tl.store(out_ptr + slot * out_stride + cols, conf, mask=cols < max_draft)


def publish_confidences(
    confidences: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    num_reqs: int,
    width: int,
    conf_by_slot: torch.Tensor,
) -> None:
    """Write ``confidences[r, :width]`` (zeros past it) into
    ``conf_by_slot[slot_r]`` for each request r < ``num_reqs`` whose slot
    ``slot_r = sample_idx_mapping[r * width]`` is not -1.

    ``confidences``: [>= num_reqs, >= width] float32, unit column stride;
    ``conf_by_slot``: [num_slots, max_draft] float32, contiguous."""
    if num_reqs == 0:
        return
    max_draft = conf_by_slot.shape[1]
    assert confidences.stride(1) == 1 and conf_by_slot.stride(1) == 1
    assert width <= max_draft
    _publish_kernel[(num_reqs,)](
        confidences,
        confidences.stride(0),
        sample_idx_mapping,
        width,
        conf_by_slot,
        conf_by_slot.stride(0),
        conf_by_slot.shape[0],
        width,
        max_draft,
        BLOCK=triton.next_power_of_2(max_draft),
    )


def drafts_nothing(speculator, num_speculative_tokens: int | None) -> bool:
    """True when ``propose`` returns before the draft step
    (``zero_draft.draft_context_only``)."""
    return num_speculative_tokens == 0 and speculator.dp_size <= 1


class RowMaskConfidenceMixin:
    """Mixed in front of a DSpark speculator class by ``row_mask_class``."""

    _row_mask: RowMaskState | None = None

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        if getattr(model.model, "confidence_head", None) is None:
            raise ValueError(
                "VLLM_DSPARK_ROW_MASK is set but the DSpark drafter has no "
                "confidence head (the checkpoint ships none, or it did not load)."
            )
        self.use_confidence_head = True
        # The target forward's token rows: the runner pads a step to at most
        # max_num_batched_tokens, the speculator's max_num_tokens.
        self._row_mask = init_row_mask_state(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            max_draft=self.max_width,
            device=self.device,
        )
        assert self._row_mask is not None
        return model

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        # The full [max_num_reqs, max_width] buffer: the per-width drafter
        # swaps in its [:, :width] view only while it samples.
        confidences = self.draft_token_confidence_probs
        super()._sample_sequential(num_reqs, head_hidden)
        state = self._row_mask
        if state is not None:
            publish_confidences(
                confidences,
                self.sample_idx_mapping,
                num_reqs,
                self.num_speculative_steps,
                state.conf_by_slot,
            )

    def propose(self, input_batch, *args, **kwargs):
        result = super().propose(input_batch, *args, **kwargs)
        state = self._row_mask
        if state is not None:
            if "num_speculative_tokens" in kwargs:
                width = kwargs["num_speculative_tokens"]
            else:
                width = _propose_signature().bind(
                    self, input_batch, *args, **kwargs
                ).arguments.get("num_speculative_tokens")
            if drafts_nothing(self, width):
                num_reqs = input_batch.num_reqs
                state.conf_by_slot.index_fill_(
                    0, input_batch.idx_mapping[:num_reqs], 1.0
                )
        return result


@cache
def _propose_signature() -> inspect.Signature:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    return inspect.signature(DFlashSpeculator.propose)


@cache
def row_mask_class(base: type) -> type:
    """``base`` with the confidence head on, published per request slot."""
    return type(f"RowMask{base.__name__}", (RowMaskConfidenceMixin, base), {})
