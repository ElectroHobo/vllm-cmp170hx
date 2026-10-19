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
  execution stream; maximum configured context bounds all scratch before
  graph profiling. Temporary profiling/capture streams transport the same
  serialized runner/lane and reuse that reservation; independently executable
  callers retain stream-disjoint resources. Source top-k and candidates remain
  persistent until their final sharing consumer.
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
actual grouped output consumer. Coverage includes every reached supported
padded count in each rank's scheduler census, with target/draft short/long
contexts. Aliased physical cache blocks are checked and compacted; exports
are immutable and use the existing captured_sha256/concurrency protocol. Independent FP64 references retain BF16
storage points and official weights. Preselected captures can screen the
output epilogue only; they are never complete pipeline evidence.

## Known limits and required review

GPU tests are written, not executed. No speed or numerical PASS is claimed.
Fresh coupled hybrid captures are required for full preparation/index and
actual hybrid grouped-output qualification. Inspect Triton resource usage,
all reached specialization keys, both-arm graph accounting and the complete
captured case matrix before enabling the switch. The pipeline remains
unqualified and default-off until those GPU gates pass. It removes torch
compressor/postprocessing/mask/top-k/full candidate-sort launches only on its
supported path; existing oracles and unsupported paths remain intentional.

# Hybrid arithmetic and local consumer captures

## Intent and scope

Default-off diagnostic hooks record actual hybrid projection calls, before-pack loader fingerprints, byte-linked successor arguments/results, compact paged pre/post state and actual TP boundaries. No synthetic serving activations, EXL3 weight denominator, input padding or analytical tolerance substitutes for captured V2. The original hybrid launch implementation and policy are not changed by this branch.

`hybrid_linear_cases.py` independently decodes official E4M3FN/UE8M0 bytes and TP/group slices. Candidate replay loads and repacks the actual ModelOpt Marlin method; comparator uses the same decoded weights in stock BF16 GEMM with the served BF16/FP32 output dtype. TF32 and reduced-precision reference reductions are disabled and reported. Captured-output replay, row invariance and repeated graph replay are hard failures.

`hybrid_consumer_cases.py` replays real local norm/RoPE/QDQ, shared SwiGLU/down/combine, compressor, index/selection/attention, gate/mHC and greedy consumers where byte-linked local sources exist. Required intermediate storage dtypes are retained. Discrete flips and writes outside observed owned slots fail. Actual TP reductions/gathers and their dependent paths are captured but explicitly deferred to distributed replay; the local verdict does not claim a collective or token-level proof. Each new projection may start from its actual captured activation after a TP boundary.

## Files and interfaces

- `vllm/models/deepseek_v41/nvidia/hybrid_capture.py`: capture-only wrappers and prepack fingerprints. Links are cleared at new steps/graph bindings and do not retain dead outputs. Explicit `point_phase` and scoped draft-model ownership distinguish eager drafter calls from a target prefill without guessing from row count; quota keys, filenames and metadata use the same phase, while scheduler `step_phase` is preserved. Large cache pools are never cloned wholesale; addressed blocks and real adjacent guard blocks are preserved and aliases validated during export.
- `vllm/sm80/capture.py`, `vllm/sm80/kernels/exl3/{loader,quant}.py`: existing owned capture/loader extension points. No additional upstream-owned hook.
- `tests/models/deepseek_v41/sm80/{fp64_cases,hybrid_linear_cases,hybrid_consumer_cases}.py`: existing family protocol and FP64 driver, not a second result schema.
- `tools/sm80/export_hybrid_cases.py`: existing `served_cases/1`; validates capture hashes, actual links, rank/role/phase/context coverage, immutable frozen bytes and state aliases. Missing cells refuse export. TP-dependent cells are enumerated separately, never counted as local passes.
- Existing `test_fp64_cases.py` and `test_hybrid_checkpoint.py`: independent FP8 decoding, loader-byte corruption, missing links, weak references, reference settings, TP deferral and observed-state write guards.

## Checks and known limits

CPU tests are added but not executed here; the integration owner should run the focused hybrid/FP64 suites under the CPU/RAM gate and the exact import/native dry preflight. No GPU work or numerical verdict has been performed. The local driver requires real hybrid captures; no served-case files are fabricated before that capture. V2 acceptance remains the existing mean/max judge with a declared complete case set, not driver exit status alone.

The full attention prepare-state capture is integrated separately through the attention pipeline boundary, in the same diagnostic boot. The capture tree must contain that boundary before its row passes preflight. Instrumented boots make no timing claim. Old EXL3 shared-down/head dimensions are not silently sliced or padded into hybrid inputs. Whole-model quality, distributed TP replay and token drift remain separate obligations.

## Removed or unchanged

No production default changes, no compatibility aliases, no native source changes, and no routed-expert arithmetic changes. The retained FP32 router is captured as a downstream consumer, not relabelled as a changed FP8/BF16 projection; its promoted weight bytes are not compared to the source storage dtype as though a hybrid GEMM changed them. Existing capture APIs and the stock judge remain the single conventions.
