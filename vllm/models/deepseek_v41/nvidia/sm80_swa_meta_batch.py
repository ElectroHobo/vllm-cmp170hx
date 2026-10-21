# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every SWA group's sliding-window metadata of a step in one launch
(``VLLM_DEEPSEEK_V41_SWA_META_BATCH``, sm_80 DeepSeek-V4.1).

DeepSeek-V4.1 puts each layer's sliding-window cache in a KV cache group of
its own (one layer per group: the single-layer compressed and index-key
buckets set the group size, ``vllm/v1/core/kv_cache_planning.py``), and every
group draws its own blocks, so ``build_attn_metadata``
(``vllm/v1/worker/gpu/attn_utils.py``) builds the window metadata once per
group, from that group's block table and slot mapping, with three launches
each (decode-only steps) or five (with prefill rows). The runner keeps those
inputs in persistent buffers (``BlockTables.input_block_tables`` and the
stacked ``BlockTables.slot_mappings``, ``vllm/v1/worker/gpu/block_table.py``),
so their addresses do not change between steps.

:func:`build` (called by ``Sm80SWAMetadataBuilder.build`` with the switch on)
learns, from one step's builds, which builders build together in one
``build_attn_metadata`` call and the addresses of their inputs. On a later
step the first of them launches ``ops/sm80_swa_meta.swa_meta_batch`` for all
of them, writing each group's buffers as its own build would; each builder
then returns the metadata its own build returns, over the same buffers. A
builder takes the batched result only if its inputs are the learned ones and
the batch-level inputs are the first builder's; anything else (another step
shape, a changed buffer, DSpark's non-causal draft metadata, image spans,
CUDA graph capture and drafting builds) takes the per-group build. No host
synchronization: the learned tables are uploaded once per change of the
learned inputs, from pinned memory.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.models.deepseek_v41.nvidia.ops.sm80_swa_meta import OUT_ROWS, swa_meta_batch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.sparse_swa import (
    _LAYER_TYPE_C1A,
    _LAYER_TYPE_C2A,
    _LAYER_TYPE_C4A,
    _LAYER_TYPE_C128A,
    _LAYER_TYPE_SWAONLY,
    DeepseekSparseSWAMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

__all__ = ["build", "reset"]


def _end(t: torch.Tensor) -> int:
    """One past the last byte of ``t``'s storage."""
    storage = t.untyped_storage()
    return storage.data_ptr() + storage.nbytes()


@dataclass
class _Member:
    builder: Any
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    def key(self) -> tuple:
        return (
            id(self.builder),
            self.block_table.data_ptr(),
            self.block_table.stride(0),
            self.slot_mapping.data_ptr(),
        )


def _uniform(builders: list[Any]) -> bool:
    """Whether one launch serves every builder: one shape, no image spans."""

    def shape(b) -> tuple:
        return (
            b.window_size,
            b.block_size,
            b.prefill_index_width,
            b.decode_threshold,
            b.max_image_tokens,
            b.device,
            tuple(b.decode_swa_indices.shape),
            tuple(b.prefill_swa_indices.shape),
            b.decode_swa_lens.shape[0],
            b.prefill_swa_lens.shape[0],
            b.is_valid_token.shape[0],
        )

    first = shape(builders[0])
    return (
        first[4] == 0
        and first[2] == first[0]
        and all(shape(b) == first for b in builders[1:])
        and all(
            b.decode_swa_indices.is_contiguous()
            and b.prefill_swa_indices.is_contiguous()
            and b.is_valid_token.dtype == torch.bool
            for b in builders
        )
    )


class _Family:
    """Builders that build together, with their learned inputs and the
    device tables of one launch."""

    def __init__(self, members: list[_Member]) -> None:
        self.members = members
        self.keys = [m.key() for m in members]
        self.position = {id(m.builder): i for i, m in enumerate(members)}
        builders = [m.builder for m in members]
        device = builders[0].device
        # Rows readable from each learned input: the batch must fit them all.
        self.max_slots = min(
            (_end(m.slot_mapping) - m.slot_mapping.data_ptr()) // 8 for m in members
        )
        self.max_rows = min(
            (_end(m.block_table) - m.block_table.data_ptr())
            // (4 * max(m.block_table.stride(0), 1))
            for m in members
        )
        values = [
            [m.slot_mapping.data_ptr() for m in members],
            [m.block_table.data_ptr() for m in members],
            [m.block_table.stride(0) for m in members],
            *([getattr(b, row).data_ptr() for b in builders] for row in OUT_ROWS),
        ]
        host = torch.tensor(values, dtype=torch.int64)
        if device.type == "cuda":
            host = host.pin_memory()
        self._host = host  # alive until the copy has run
        tables = torch.empty_like(host, device=device)
        tables.copy_(host, non_blocking=True)
        self.slot_ptrs = tables[0]
        self.block_table_ptrs = tables[1]
        self.block_table_strides = tables[2]
        self.out_ptrs = tables[3:]
        first = builders[0]
        self.gather_lens = torch.zeros(
            first.no_replay_start.shape[0], dtype=torch.int32, device=device
        )


@dataclass
class _Step:
    """The current ``build_attn_metadata`` call."""

    # Its batch-level inputs, held so that a later call's are other objects.
    common: tuple
    observed: list[_Member] = field(default_factory=list)
    family: _Family | None = None
    # Of the launch: (split, token_to_req_indices, replay_start, num_slots).
    launch: tuple | None = None


@dataclass
class _State:
    step: _Step | None = None
    # Learned families by their first builder.
    families: dict[int, _Family | None] = field(default_factory=dict)


_STATE = _State()


def reset() -> None:
    """Forget every learned family (tests)."""
    _STATE.step = None
    _STATE.families.clear()


def _common(cam: CommonAttentionMetadata) -> tuple:
    return (
        cam.query_start_loc,
        cam.query_start_loc_cpu,
        cam.seq_lens,
        cam.seq_lens_cpu_upper_bound,
        cam.is_prefilling,
    )


def _same_common(a: tuple, b: tuple) -> bool:
    return all(x is y for x, y in zip(a, b))


def _scalars(cam: CommonAttentionMetadata) -> tuple:
    return (
        cam.num_reqs,
        cam.num_actual_tokens,
        cam.max_query_len,
        cam.slot_mapping.shape[0],
    )


def _finish(step: _Step) -> None:
    """Learn the family of a finished call, keeping its tables if unchanged."""
    if not step.observed:
        return
    first = id(step.observed[0].builder)
    old = _STATE.families.get(first)
    keys = [m.key() for m in step.observed]
    if old is not None and old.keys == keys:
        return
    builders = [m.builder for m in step.observed]
    family = None
    if (
        len(builders) > 1
        and len({id(b) for b in builders}) == len(builders)
        and _uniform(builders)
    ):
        family = _Family(step.observed)
    _STATE.families[first] = family


def _launch(
    family: _Family,
    builder: Any,
    cam: CommonAttentionMetadata,
    replay_start: torch.Tensor | None,
) -> tuple | None:
    split = split_decodes_and_prefills(cam, decode_threshold=builder.decode_threshold)
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split
    num_slots = cam.slot_mapping.shape[0]
    if (
        num_decode_tokens + num_prefill_tokens > num_slots
        or num_slots > family.max_slots
        or cam.num_reqs > family.max_rows
        or num_prefills > family.gather_lens.shape[0]
        or (num_prefills > 0 and cam.seq_lens_cpu_upper_bound is None)
        or cam.block_table_tensor.dtype != torch.int32
        or cam.slot_mapping.dtype != torch.int64
    ):
        return None
    token_to_req = cam.token_to_req_indices(builder.token_to_req_indices)
    rs = builder.no_replay_start if replay_start is None else replay_start
    swa_meta_batch(
        family.slot_ptrs,
        family.block_table_ptrs,
        family.block_table_strides,
        family.out_ptrs,
        family.gather_lens,
        cam.query_start_loc,
        cam.seq_lens,
        token_to_req,
        rs,
        num_slots=num_slots,
        num_decode_tokens=num_decode_tokens,
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        lens_capacity=builder.decode_swa_lens.shape[0],
        window_size=builder.window_size,
        decode_width=builder.decode_swa_indices.shape[-1],
        prefill_width=builder.prefill_index_width,
        block_size=builder.block_size,
    )
    return split, _scalars(cam), _replay_key(replay_start)


def _replay_key(replay_start: torch.Tensor | None) -> tuple | None:
    if replay_start is None:
        return None
    return (replay_start.data_ptr(), tuple(replay_start.shape), replay_start.dtype)


def _metadata(
    builder: Any,
    cam: CommonAttentionMetadata,
    split: tuple[int, int, int, int],
    replay_start: torch.Tensor | None,
    gather_lens: torch.Tensor,
) -> DeepseekSparseSWAMetadata:
    """What the per-group ``build`` returns for a causal batch without image
    spans, over the buffers the launch filled."""
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split
    seq_lens = cam.seq_lens
    query_start_loc_cpu = cam.query_start_loc_cpu
    token_to_req_indices = cam.token_to_req_indices(builder.token_to_req_indices)
    if replay_start is None:
        replay_start = builder.no_replay_start
    deepseek_v4_fields: dict[str, Any] = {}
    if num_prefills > 0:
        seq_lens_cpu = cam.seq_lens_cpu_upper_bound
        deepseek_v4_fields = dict(
            prefill_seq_lens=seq_lens[num_decodes:],
            prefill_seq_lens_cpu=seq_lens_cpu[num_decodes:],
            prefill_gather_lens=gather_lens[:num_prefills],
            prefill_query_lens_cpu=(
                query_start_loc_cpu[num_decodes + 1 : num_decodes + num_prefills + 1]
                - query_start_loc_cpu[num_decodes : num_decodes + num_prefills]
            ).to(dtype=torch.int32),
            prefill_window_size=builder.window_size,
            prefill_max_model_len=builder.max_model_len,
            prefill_max_num_batched_tokens=builder.max_num_batched_tokens,
        )
    tile_sched = builder.build_tile_scheduler(num_decode_tokens)
    return DeepseekSparseSWAMetadata(
        seq_lens=seq_lens,
        query_start_loc=cam.query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        block_table=cam.block_table_tensor,
        slot_mapping=cam.slot_mapping,
        is_valid_token=builder.is_valid_token[: cam.slot_mapping.shape[0]],
        token_to_req_indices=token_to_req_indices,
        decode_swa_indices=builder.decode_swa_indices[:num_decode_tokens],
        decode_swa_lens=builder.decode_swa_lens[:num_decode_tokens],
        decode_swa_width=builder.window_size,
        prefill_swa_indices=(
            builder.prefill_swa_indices[:num_prefill_tokens]
            if num_prefill_tokens > 0
            else None
        ),
        prefill_swa_lens=(
            builder.prefill_swa_lens[:num_prefill_tokens]
            if num_prefill_tokens > 0
            else None
        ),
        prefill_left_visible=None,
        prefill_right_visible=None,
        replay_start=replay_start,
        block_size=builder.block_size,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        num_decode_tokens=num_decode_tokens,
        num_prefill_tokens=num_prefill_tokens,
        max_decode_query_len=min(cam.max_query_len, builder.decode_threshold),
        tile_sched_swaonly=tile_sched[_LAYER_TYPE_SWAONLY],
        tile_sched_c4a=tile_sched[_LAYER_TYPE_C4A],
        tile_sched_c128a=tile_sched[_LAYER_TYPE_C128A],
        tile_sched_c1a=tile_sched[_LAYER_TYPE_C1A],
        tile_sched_c2a=tile_sched[_LAYER_TYPE_C2A],
        **deepseek_v4_fields,  # type: ignore[arg-type]
    )


def build(
    builder: Any,
    common_prefix_len: int,
    cam: CommonAttentionMetadata,
    fast_build: bool,
    replay_start: torch.Tensor | None,
) -> DeepseekSparseSWAMetadata:
    """``builder.build`` with the switch on: the batched result when it is the
    per-group build's, else the per-group build."""
    common = _common(cam)
    step = _STATE.step
    if step is None or not _same_common(step.common, common):
        # The first build of a new build_attn_metadata call.
        if step is not None:
            _finish(step)
        step = _Step(common)
        _STATE.step = step
        family = _STATE.families.get(id(builder))
        if (
            family is not None
            and family.keys[0] == _Member(builder, cam.block_table_tensor, cam.slot_mapping).key()
            and cam.causal is True
        ):
            step.launch = _launch(family, builder, cam, replay_start)
            if step.launch is not None:
                step.family = family
    member = _Member(builder, cam.block_table_tensor, cam.slot_mapping)
    position = len(step.observed)
    step.observed.append(member)

    family = step.family
    if (
        family is not None
        and step.launch is not None
        and position < len(family.keys)
        and family.keys[position] == member.key()
        and cam.causal is True
    ):
        split, scalars, replay_key = step.launch
        if scalars == _scalars(cam) and replay_key == _replay_key(replay_start):
            return _metadata(builder, cam, split, replay_start, family.gather_lens)
    return builder.build_per_group(common_prefix_len, cam, fast_build, replay_start)
