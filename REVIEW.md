# Bounded prefill capture selection

`VLLM_SM80_CAPTURE_PREFILL_PER_KEY` is an opt-in integer control, default
zero. Zero retains the existing sparse prefill ordinals and non-prefill
limits. Positive values retain the first N eligible attempts per family,
point, exact prefix, rows, actual phase, writer owner and existing short/long
context. Existing prefix/minimum-row selectors still apply. Actual global
prefill call ordinals and the full reached-row census remain unchanged.

Quota refusals stay explicitly unsaved and consume an attempt in this mode,
so an oversized repeated call cannot keep copying tensors indefinitely.
Saved-byte accounting only advances on a successful write. Semantic-key
filenames prevent different points or sanitized prefixes overwriting each
other. No tensor, hook, driver, numerical threshold or disabled-path work
is added; all changes are in fork-owned capture configuration and tests.

Changed files: `vllm/sm80/capture.py`, `vllm/sm80/envs.py`,
`tests/kernels/quantization/test_sm80_capture.py`, and the explicit config
construction in `tests/models/deepseek_v41/sm80/test_fp64_cases.py`.
The focused CPU checks cover all capture families, unseen rows/points and
contexts, bounded repeats/refusals, exact prefix identity, writer owners,
actual draft phase, selectors, census and unchanged ordinal defaults.
The focused batch passed, including the existing graph ownership and plain
GLM/Qwen materialization controls. CPU import preparation is not a native
build or GPU qualification.
No GPU or integrated-capture coverage is claimed; missing real source
classes and byte-quota refusals must still fail downstream coverage checks.

# Consolidated DeepSeek-V4.1 execution paths

## Intent and ownership

The existing default-off attention, MoE prefill, attention-pipeline, hybrid
checkpoint, per-K draft width, reduction, early-Engram and expert-placement
paths share one implementation tree. The scheduler progress correction is
unconditional; the earlier PP state split is already in the base.

Runtime-safe modes use the allowlisted adapters in `vllm/sm80/dev_ab.py`.
MoE V2, MoE prefill and dense prefill must be constructed enabled to reserve
and warm their resources before an off/on comparison. mHC finish requires
its fused parent. Checkpoint/weight layout, draft-width construction,
reduction ownership and communicator setup remain boot-bound.

The capture guard is now an owned decorator, with the original capture
body retained exactly. Ordinary and early-gather Engram projection share
`nvidia/sm80_engram_projection.py`; no projection implementation remains in
the upstream class and no common-to-adapter import cycle is introduced.
Existing constructor/load/profile/output lifetimes retain content-scoped
exceptions, each with a reason, owner and removal condition in
`tools/sm80/upstream_edit_exceptions.yaml`; there are no blanket file waivers.

## CPU contracts and limits

The CPU commands cover `tests/sm80`, `tests/models/deepseek_v41/sm80`, the
quantization MoE-prefill/placement/dense-prefill/capture tests, DSpark width
and zero-draft tests, scheduler progress/fairness/alignment, Qwen PLE replay
and deferred state copies. Source-identity tests retain the original shared
graph bodies; unset controls do not touch the runner. Both GLM and Qwen
plain EXL3 capture paths retain the original host materialization.

The config registration fixture now starts from the original class even
when earlier tests have installed its idempotent wrapper. FP64 metadata
assertions include the actual reference arithmetic settings, and dense
prefill warmup assertions cover each supported output dtype. No numerical
tolerance or production default is changed to satisfy these tests.

No GPU numerical, identity, quality or speed verdict is claimed. The shared
executor/output hooks, dev_ab, deferred capture, scheduler and EXL3 paths
require integrated code inspection and GLM/Qwen default-configuration GPU
identity before adoption. The feature-specific numerical and speed gates
remain required before any default is enabled.

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
are immutable and use the existing captured_sha256/concurrency protocol.
The unified diagnostic boot retains only pre-write cache slots in each graph;
model-owned deferred materialization gathers unique visible physical blocks at
the synchronized drain and reconstructs pre-state. It does not retain one
full-capacity, duplicate-padded cache shadow per layer and graph shape. Independent FP64 references retain BF16
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

- `vllm/models/deepseek_v41/nvidia/hybrid_capture.py`: capture-only wrappers and prepack fingerprints. Links are cleared at new steps/graph bindings and do not retain dead outputs. Explicit `point_phase` and scoped draft-model ownership distinguish eager drafter calls from a target prefill without guessing from row count; quota keys, filenames and metadata use the same phase, while scheduler `step_phase` is preserved. Read-only index/attention caches are materialized once per unique consumed physical block at the existing synchronized deferred drain; unused block-table capacity is never expanded into graph-resident cache copies. Metadata is captured in stream order on each replay. Mutating stores/compressors retain their bounded pre/post snapshots and real adjacent write guards, with aliases validated during export.
- `vllm/sm80/capture.py`, `vllm/sm80/kernels/exl3/{loader,quant}.py`: existing owned capture/loader extension points. No additional upstream-owned hook.
- `tests/models/deepseek_v41/sm80/{fp64_cases,hybrid_linear_cases,hybrid_consumer_cases}.py`: existing family protocol and FP64 driver, not a second result schema.
- `tools/sm80/export_hybrid_cases.py`: existing `served_cases/1`; validates capture hashes, actual links, rank/role/phase/context coverage, immutable frozen bytes and state aliases. Missing cells refuse export. Raw and frozen tensor payloads are memory-mapped, and source hashes stream from disk, rather than allocating the full captured corpus as anonymous memory. TP-dependent cells are enumerated separately, never counted as local passes.
- Existing `test_fp64_cases.py` and `test_hybrid_checkpoint.py`: independent FP8 decoding, loader-byte corruption, missing links, weak references, reference settings, TP deferral and observed-state write guards.

## Checks and known limits

The focused hybrid/FP64 CPU suites and exact native-bound import must pass under the CPU/RAM gate; their result must identify the tested source and actual native manifest. No GPU work or numerical verdict has been performed. The local driver requires real hybrid captures; no served-case files are fabricated before that capture. V2 acceptance remains the existing mean/max judge with a declared complete case set, not driver exit status alone.

The full attention prepare-state capture is integrated separately through the attention pipeline boundary, in the same diagnostic boot. The capture tree must contain that boundary before its row passes preflight. Instrumented boots make no timing claim. Old EXL3 shared-down/head dimensions are not silently sliced or padded into hybrid inputs. Whole-model quality, distributed TP replay and token drift remain separate obligations.

## Removed or unchanged

No production default changes, no compatibility aliases, no native source changes, and no routed-expert arithmetic changes. The retained FP32 router is captured as a downstream consumer, not relabelled as a changed FP8/BF16 projection; its promoted weight bytes are not compared to the source storage dtype as though a hybrid GEMM changed them. Existing capture APIs and the stock judge remain the single conventions.

## Deferred-reader allocation correction (2026-10-20)

`test_hybrid_capture.py` exercises real index-score and sparse-attention consumers after compacting repeated/null blocks, poisoned unused table tails, request-specific visibility, no/empty compressed selections, and negative window slots. It also checks metadata ownership, unchanged mutating pre/post state and guard blocks, producer/successor links, and absence of capture-time reader-cache gathers or clones. `test_hybrid_capture_gpu.py` adds actual-consumer CUDA graph replay with changing cache contents and metadata; it is written, not run by this repair.

The targeted CPU command is `python -B -m pytest -q -n 4 -p no:cacheprovider tests/models/deepseek_v41/sm80/test_hybrid_capture.py tests/models/deepseek_v41/sm80/test_fp64_cases.py -k hybrid`. Run only on the frozen successor with the genuine native binding, `CUDA_VISIBLE_DEVICES=""`, and the lab RAM/CPU wrappers. The repair handoff records the one authorized batch and its actual results; no earlier result substitutes for it.

The existing `Record.defer` API, family/op/consumer-link schema, sampling policy, capture matrix and production numerical path are unchanged. Borrowed caches must remain stable until the owning call/graph drains; point-local metadata is independently snapshotted. The repaired payload expansion is not an attribution of the entire failed graph-memory estimate: allocator free-in-reserved segments, other graph intermediates and cold retention remain separately measured. Main must review the source and authorize actual replay/whole-boot memory and complete capture/export coverage proof before any successor GPU run. No fallback, memory-reserve reduction or coverage cut is introduced.
