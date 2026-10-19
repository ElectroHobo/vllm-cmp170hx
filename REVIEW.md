# Hybrid arithmetic and local consumer captures

## Intent and scope

Default-off diagnostic hooks record actual hybrid projection calls, before-pack loader fingerprints, byte-linked successor arguments/results, compact paged pre/post state and actual TP boundaries. No synthetic serving activations, EXL3 weight denominator, input padding or analytical tolerance substitutes for captured V2. The original hybrid launch implementation and policy are not changed by this branch.

`hybrid_linear_cases.py` independently decodes official E4M3FN/UE8M0 bytes and TP/group slices. Candidate replay loads and repacks the actual ModelOpt Marlin method; comparator uses the same decoded weights in stock BF16 GEMM with the served BF16/FP32 output dtype. TF32 and reduced-precision reference reductions are disabled and reported. Captured-output replay, row invariance and repeated graph replay are hard failures.

`hybrid_consumer_cases.py` replays real local norm/RoPE/QDQ, shared SwiGLU/down/combine, compressor, index/selection/attention, gate/mHC and greedy consumers where byte-linked local sources exist. Required intermediate storage dtypes are retained. Discrete flips and writes outside observed owned slots fail. Actual TP reductions/gathers and their dependent paths are captured but explicitly deferred to distributed replay; the local verdict does not claim a collective or token-level proof. Each new projection may start from its actual captured activation after a TP boundary.

## Files and interfaces

- `vllm/models/deepseek_v41/nvidia/hybrid_capture.py`: capture-only wrappers and prepack fingerprints. Links are cleared at new steps/graph bindings and do not retain dead outputs. Explicit `point_phase` and scoped draft-model ownership distinguish eager drafter calls from a target prefill without guessing from row count; quota keys, filenames and metadata use the same phase, while scheduler `step_phase` is preserved. Large cache pools are never cloned wholesale; addressed blocks and real adjacent guard blocks are preserved and aliases validated during export.
- `vllm/sm80/capture.py`, `vllm/sm80/kernels/exl3/{loader,quant}.py`: existing owned capture/loader extension points. No additional upstream-owned hook.
- `tests/models/deepseek_v41/sm80/{fp64_cases,hybrid_linear_cases,hybrid_consumer_cases}.py`: existing family protocol and FP64 driver, not a second result schema.
- `tools/sm80/export_hybrid_cases.py`: existing `served_cases/1`; validates capture hashes, actual links, rank/role/phase/context coverage, immutable frozen bytes and state aliases. Missing cells refuse export. Raw and frozen tensor payloads are memory-mapped, and source hashes stream from disk, rather than allocating the full captured corpus as anonymous memory. TP-dependent cells are enumerated separately, never counted as local passes.
- Existing `test_fp64_cases.py` and `test_hybrid_checkpoint.py`: independent FP8 decoding, loader-byte corruption, missing links, weak references, reference settings, TP deferral and observed-state write guards.

## Checks and known limits

CPU tests are added but not executed here; the integration owner should run the focused hybrid/FP64 suites under the CPU/RAM gate and the exact import/native dry preflight. No GPU work or numerical verdict has been performed. The local driver requires real hybrid captures; no served-case files are fabricated before that capture. V2 acceptance remains the existing mean/max judge with a declared complete case set, not driver exit status alone.

The full attention prepare-state capture is integrated separately through the attention pipeline boundary, in the same diagnostic boot. The capture tree must contain that boundary before its row passes preflight. Instrumented boots make no timing claim. Old EXL3 shared-down/head dimensions are not silently sliced or padded into hybrid inputs. Whole-model quality, distributed TP replay and token drift remain separate obligations.

## Removed or unchanged

No production default changes, no compatibility aliases, no native source changes, and no routed-expert arithmetic changes. The retained FP32 router is captured as a downstream consumer, not relabelled as a changed FP8/BF16 projection; its promoted weight bytes are not compared to the source storage dtype as though a hybrid GEMM changed them. Existing capture APIs and the stock judge remain the single conventions.
