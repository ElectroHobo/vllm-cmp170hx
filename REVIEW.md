# Attention pipeline

## Intent and interface

`VLLM_DEEPSEEK_V41_ATTN_PIPELINE` defaults off. It is independent of
`VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE`; the latter remains the control.
Construction records static eligibility separately from the mutable active
arm. The supported domain is scheduler decode/verify and actual drafter
calls, bounded by `ATTN_PIPELINE_MAX_TOKENS` (default 48), never target
prefill merely because its row count is small. Context writes and
`forward_mqa` retain their existing contracts.

## Bodies and lifetime

- `sm80_attn_pipeline.py`: compressor latent read, separate compressed/ring
  store barrier, combined main/index-query and index-key postprocessing,
  index score producer with first-byte statistics and block maxima.
- `sm80_index_topk.py`: exact four-byte radix cutoff and bounded integer
  count/prefix/scatter; ties use the earliest positions and IDs finish in
  ascending order. Signed zeros tie; visible rejected positions remain
  `-FLT_MAX`, distinct from invisible `-inf`. Newest candidate block is pinned.
- `sm80_attn_fused.py`: the existing ordered-value schedule writes inverse
  RoPE/QDQ output directly, retaining BF16 probability and output stores.
- `sm80_index_scores.py`: one shared score tile preserves head/rank arithmetic.
- Runner workspace resources are separated by execution owner, lane and
  stream; maximum configured context bounds all scratch. Source top-k and
  candidates remain persistent until their final sharing consumer.
- Core modes are explicit: seed fused returns pre-RoPE queries; pipeline
  returns post-RoPE queries. Selecting both modes raises before launching.

No new upstream-owned call site is introduced. The switch-off path does not
allocate pipeline scratch unless both arms were explicitly requested at boot.

## Validation written

The selector tests cover exact widths, ties, signed zero, sentinels, candidate
source/mask semantics, newest pin, long capacity and int64 paged offsets.
Pipeline tests cover ratios, source/sharing consumers, first-token predecessor,
ring wrap/tail ownership, null and padded rows, FP64 boundaries, repeated graph
replay, alternating row sizes, storage identity, delayed stream progress and
host-sync observation. Existing seed behavior is retained.

`fp64_cases.py` adds `index_select`, `attn_pipeline` and
`attn_wo_a_consumer`. The coupled exporter rejects absent pre-state, changed
hashes, duplicate sources, missing coverage and a non-unique or non-byte-linked
actual grouped output consumer. Independent FP64 references retain BF16
storage points and official weights. Preselected captures can screen the
output epilogue only; they are never complete pipeline evidence.

## Known limits and required review

GPU tests are written, not executed. No speed or numerical PASS is claimed.
Fresh coupled hybrid captures are required for full preparation/index and
actual hybrid grouped-output qualification. Inspect Triton resource usage,
all reached specialization keys, both-arm graph accounting and the complete
captured case matrix before enabling the switch. The staged pipeline is a
starting implementation, not a qualified speed default. It removes torch
compressor/postprocessing/mask/top-k/full candidate-sort launches only on its
supported path; existing oracles and unsupported paths remain intentional.
