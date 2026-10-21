# DSpark per-round verification row mask (default off)

`VLLM_DSPARK_ROW_MASK=<dspark-row-mask-costs/1 JSON>` lets each DSpark verify
step keep, per request, only the drafts its trained confidence head makes
worth verifying. The rows past them take no routed expert in the target's MoE
layers, which shrinks the busiest rank's distinct experts, and their drafts
count as rejected. Empty (the default) changes nothing: no class wrap, no state,
no extra kernel; the runner hooks return at once.

Pieces, all reading `dspark_row_mask.RowMaskState` (fixed-address buffers):

1. `dspark_row_mask_conf.py`: the drafter computes its confidence head in
   every draft step and one Triton launch inside the draft graph publishes
   each request's per-position outputs into `conf_by_slot` at the request's
   state slot (from `sample_idx_mapping`, which `prepare_dflash_inputs`
   rewrites before every replay; padding requests write nothing; columns past
   the drafted width are 0). A zero-draft step (no draft step runs) resets its
   batch's slots to 1.0, so no later verify reads stale confidences.
2. `dspark_row_mask_kernel.py`: before the target forward of a uniform
   speculative decode step (one call in the runner, eager on the forward
   stream before the graph replay), one Triton launch computes per request
   `live_len = argmax_k E(k) / cost_r(k)`, with
   `E(k) = 1 + sum_{i<=k} prod_{j<=i} p_j` and
   `cost_r(k) = T_b(K) - (T_b(K) - T_b(k)) / b` from the cost table, and
   writes `row_live` (rows `0..live_len` of each request live). Any other
   step resets every row to live.
3. `sm80_moe.py`, `ops/sm80_route.py`: target routers (marked by
   `adopt_row_mask` in `DeepseekV4Model.adopt`; the drafter's layers are never
   marked) give masked rows expert id -1 and weight +0.0: inside the fused
   route kernel (`ROW_MASK` constexpr, no extra launch) or with `masked_fill`
   on the torch path. The EXL3 MoE kernels skip id -1 at every stage.
4. `dspark_row_mask_reject.py`: in the runner's `sample`, masked rows are
   ORed into the rejection sampler's existing `invalid_drafts`, so drafts past
   `live_len` are verified as -1 (rejected; the token at row `live_len` is the
   target's). `num_rejected` then excludes them from the drafter's context.

Exactness: a row's routed-expert output in `moe_out`/`moe_out_v2` depends only
on that row (kernel reading; GPU test listed below), causal attention never
reads a later row, and the all-reduces are per row, so rows `0..live_len` are
bit-identical to the unmasked step. With greedy target sampling the emitted
tokens are the unmasked step's truncated at `min(accepted, live_len)` drafts
plus the target token at that row. Masked rows' KV and compressor-ring writes
are positions past the accepted length, handled as for rejected rows. With
sampled (probabilistic) drafts the cap would bias the verified distribution;
the served drafter is greedy.

Upstream-owned files: `vllm/v1/worker/gpu/model_runner.py` gets two imports and
two calls: `begin_target_step(input_batch, dummy_run)` after
`step_timing.forward_start()` (the mask must be written before the target
forward) and `cap_drafts(input_batch.logits_indices, invalid_drafts)` before
rejection sampling (after the forward, so the forward's inputs keep the real
drafts). Both return immediately with the switch off. Seam rows pin
`prepare_dflash_inputs` and `RejectionSampler.__call__`.

Tests (CPU): `test_dspark_row_mask_conf.py`, `test_dspark_row_mask_kernel.py`,
`test_row_mask_moe.py`, `test_dspark_row_mask_reject.py`, the fused-route
test, seams. GPU, not run: `test_row_mask_moe.py`'s CUDA cases (fused vs torch
bytes, graph replay reading `row_live` in place, live rows of `moe_out` and
`moe_out_v2` bit-identical with other rows masked at 6/24/48 rows).

Known limits: the cost table is an estimate until re-fitted from measured
per-depth step times. `adaptive_k`'s acceptance statistics would count masked
drafts as rejections. Spec-decode counters show masked drafts as proposed and
rejected. The pipeline-parallel remote draft tail is not covered (not served).

# Grouped MXFP8 wo_a as one Triton GEMM

Intent: with `VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM=1` the FP8 `wo_a` of a
hybrid checkpoint (block-diagonal, one Marlin W8A16 GEMM per local group
and a concat) runs as one grouped Triton W8A16 GEMM over all groups that
writes the `[T, groups * o_lora_rank]` input of `wo_b` directly. Adapted
from GNmac-lab's `patches/dsv41_wo_a.py` (MIT, notice in the file): their
E4M3-to-BF16 register decode and grouped tile loop, reworked for MXFP8
(E8M0 per 32 inputs of a row, decoded in the same PTX block) and for any
row count (16-row MMA tiles sharing one decoded weight tile up to 64 rows,
larger tiles for prefill), one launch, no split-K reduce.

Files:
- `vllm/models/deepseek_v41/nvidia/ops/sm80_wo_a_grouped.py` (new): the
  kernel, `wo_a_grouped`, `WoAGroupedGemmMethod` (a `GroupedLinearMethod`
  that keeps the loaded codes and per-row scales and skips Marlin's
  repack; with `keep_groups` it also builds the per-group Marlin layers and
  `gemm` picks the arm per call).
- `sm80_attention.py`: the adoption takes the method under the flag or the
  dev A/B switch; refused for a non-MXFP8 or unsupported `wo_a` and for a
  hybrid capture boot (the capture records `GroupedLinearMethod.apply_grouped`).
- `vllm/sm80/envs.py` (flag), `vllm/sm80/dev_ab.py` (switch
  `dsv41_wo_a_grouped_gemm`, not output-identical; boot with
  `VLLM_DEV_AB_SWITCH` naming it, which keeps both forms of the weight);
  `dsv41_wo_a_strided` refuses a server whose `wo_a` runs the grouped GEMM
  (strided rows apply to the per-group Marlin GEMMs only).
GLM and Qwen do not adopt `GroupedLinearMethod`; the shared class is
unchanged.

Memory: the codes and scales are the loaded tensors (8 MiB + 256 KiB per
layer and rank at TP4), the same bytes Marlin's repack would hold; nothing
extra. A server started with the dev switch holds both forms (one more
copy per layer).

Numerics: the decoded weights are Marlin's (exact E4M3 x 2**(e-127), E8M0 0
as 0), the products are exact, the sums are FP32 on the BF16 tensor cores
in K order, rounded once to BF16. Outputs differ from Marlin's only by the
order of the FP32 sums (V2/V6). A row's bytes do not depend on the call's
other rows (each 16-row MMA tile sums K in the same order).

Tests (CPU, `tests/models/deepseek_v41/sm80/test_sm80_wo_a_grouped.py`,
31 cases, all failing on the previous head): the decode PTX block run by an
emulator of its own instructions over every finite code in every lane,
exact against the FP64 E4M3/UE8M0 definitions (mutations of a selector,
shift or bias fail it); the kernel compiled for sm_80 at every row tile
(BF16 MMA, FP32 accumulation, none twice, one decode per weight for 16-row
tiles, one rounding at the store, no ftz, no spills); the launch (one,
strides of a padded or strided attention output read in place, the output
the only allocation); the method (codes kept without a copy, no Marlin
repack, both arms under the dev switch, refusals); the switch default and
the dev A/B switches (the strided one refused beside the grouped GEMM).
`test_hybrid_checkpoint.py`: the flag and the dev
switch reach every target and drafter `wo_a` at TP4. GPU,
`test_sm80_wo_a_grouped_gpu.py` (unrun): against Marlin and FP64 for rows
1-48, 64 and prefill rows (65-8193), 2 and 4 groups (mean error <= 1.10x,
max <= 1.25x Marlin's, every output within half a BF16 ulp plus the FP32
summation bound), row independence, padded rows, graph replay, the Marlin
path of a method that keeps both forms.

Known limits:
- Speed is unmeasured: the per-call cost (GNmac measured 26.0 -> 14.8 us per
  layer for their split-K version) and the prefill tiles (the register
  decode keeps Triton from pipelining the weight loads with cp.async; the
  prefill tiles decode each weight twice) need the screen.
- Decode configs were chosen at compile level (registers, no spills,
  no duplicated decode), not tuned.
- With the flag on, a hybrid capture boot is refused.

Removes: nothing (Marlin stays the default).

# Copy-engine ring all-reduce for DeepSeek-V4.1 prefill sums

# Engram `wkv` replicated for DeepSeek-V4.1 prefill

Intent: with `VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL=1`, a prefill
step projects the Engram `wkv` (6144 inputs, 25600 outputs, MXFP8 on Marlin
W8A16, layers 1 and 14) on every TP rank from all four shards instead of
projecting its own shard and all-gathering the BF16 output. At TP4 a
2048-row chunk drops a 78.6 MB gather per Engram layer per rank for three
more shard GEMMs (about 0.8 ms each at 2048 rows).

How:
- After the layer's own method has processed this rank's shard (Marlin
  repack, permuted E8M0 scales), the processed bytes of every parameter are
  all-gathered once over the TP group into one `[tp, ...]` tensor. The
  layer's parameters become views of its own slot: no second copy of the
  shard, and decode reads the same bytes as before.
- A step of at least `min_rows` rows applies the method once per slot (the
  GEMM that slot's rank runs: same input, same shard bytes, same N=6400) and
  concatenates in rank order, as the all-gather does. The output is the
  gathered output byte for byte. A single N=25600 Marlin launch would split
  the work differently and is not used.
- `min_rows = max(max_num_seqs * (1 + num_speculative_tokens),
  max_cudagraph_capture_size) + 1` (49 for 8 sequences, 5 drafts, captures
  up to 40): decode steps and every captured graph keep the sharded GEMM and
  the all-gather; only eager prefill steps take the replica.

Files:
- `vllm/models/deepseek_v41/nvidia/sm80_engram_wkv.py` (new):
  `prefill_min_rows`, `ReplicatedWkvMethod` (wraps the layer's method; gathers
  the replica after processing), `install_replica`, `ReplicatedPrefillWkv`
  (row-count dispatch), `adopt_replicated_wkv` (TP only: refuses sequence
  parallelism, the batched Engram projection and DP-sharded tables).
- `vllm/models/deepseek_v41/nvidia/sm80_model.py`: adopts it beside the early
  Engram gather when the flag is on.
- `vllm/sm80/envs.py`: the flag, default off.
- `vllm/sm80/dev_ab.py`: switch `dsv41_engram_wkv_replicated_prefill`
  (needs the flag set at start; switching it off clears `min_rows` and keeps
  the replica in memory; not graph-bound, output-identical).

Tests:
- `tests/models/deepseek_v41/sm80/test_engram_wkv_replica.py` (CPU): four
  ranks of a real `ColumnParallelLinear` load one checkpoint through their
  weight loaders, process in threads joined by an in-memory all-gather; the
  replica slots are each rank's processed bytes, the layer's parameters are
  views of its slot, prefill rows run no collective and give the gathered
  bytes, decode and capture rows keep the shard and the gather; refusals,
  backbone adoption, the switch.
- `tests/models/deepseek_v41/sm80/test_engram_wkv_replica_gpu.py` (one
  card): the real MXFP8 Marlin layer at the served shape, 1 to 2048 rows,
  replicated output equal in bytes to the gathered shard outputs.
- `tests/sm80/test_dev_ab.py`, `tests/sm80/test_dev_ab_dynamic.py`: allowlist
  and default-off entries.

Known limits: every card holds the other three shards of both Engram
layers: 3 x 40.55 MB x 2 = 0.227 GiB per card less KV cache. Pre-processed
(weight-cache) loads still call the wrapper's post-load processing: the
inner method returns at once and the wrapper gathers the already-processed
shards, so they get the replica too. Refused: a step budget
(`max_num_batched_tokens`) below `min_rows`, where no step could use the
replica, and a hybrid capture, whose shard copies would each carry this
rank's capture facts. Prefill only: decode is unchanged by construction.

Removes: nothing; the sharded path stays for decode and for the flag off.

Removes: nothing.

# Delayed mHC finish v2

`VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2` (default off) speeds up DeepSeek-V4.1's
delayed mHC call (kernel A + finish, 86 calls per decode step per rank). With
the finish kernel selected (`VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH=1`), a call in
the validated configuration (hc 4, hidden 5120, 20 Sinkhorn iterations,
1..2048 rows, with the norm, sm_80) runs kernel A with `HS_REUSE` from 17
rows (two shared-memory staging pairs reused: 24 KB per CTA instead of 32, the
same products in the same order) and the Gluon `_delayed_finish_v2_kernel`
instead of the Triton `_delayed_finish_kernel`: the same contract, outputs
and rounding points; FP64 `1/x` and `rsqrt` by MUFU approximation and Newton
steps, the mean as a product with an FP64 constant, one CTA barrier for the
partial reduction, the Sinkhorn in scale-vector form, one pass over the
streams. `mhc_ops` decides the gate once (banner when open, a line with the
reason when closed) and registers the pre-capture JIT warmup.

Files: `vllm/models/deepseek_v41/nvidia/ops/sm80_mhc_finish_v2.py` (new),
`ops/sm80_mhc_fused.py` (dispatch, `finish_v2_unavailable`), `sm80_model.py`
(`mhc_ops`), `vllm/sm80/kernels/mhc/decode.py` (`HS_REUSE`, `hs_reuse`; shared
with GLM-5.3-Flash, which never passes it), `vllm/sm80/envs.py`,
`vllm/sm80/dev_ab.py` (switch `dsv41_fused_mhc_finish_v2`, not
output-identical), the consumer family `mhc_finish_consumer` of
`fp64_cases.py` (`mhc_consumer_cases.py`, `export_mhc_consumer_cases.py`).
Tests: `test_sm80_mhc_finish_v2.py` (CPU: switch default, unchanged off-path
launches, the gate in every direction, reasons, banner through the vLLM
logger, warmup keys, kernel A without `HS_REUSE` pinned to its previous sm_80
SASS for both models' options, sm_80 compiles), `test_sm80_mhc_finish_v2_gpu.py`
(kernel A bits with and without the reuse, the FP64 rule against the served
finish, kernel names, determinism, graph replay equal to eager with no
allocation growth, no compile after the warmup; GPU, unrun),
`test_mhc_consumer_cases.py`, the dev A/B tests.

Numerics: order-changing. The coefficients are held to the FP64 rule against
the served finish (errors 1.000x of it on the captured cells), but the
arithmetic differs: expect about a 2^-16 per-element rate of one-ulp FP32
flips against the served path, and the BF16 outputs downstream of them.

Known limits: the GPU tests, the FP64 rule on captured inputs, the consumer
check (`mhc_finish_consumer`, bounds declared before any GPU run), a boot
and a matched served profile are not run yet; no forced-tile switch (the
finish's tiles are fixed inside the gate and `HS_REUSE` changes no bits).

Removes: nothing (the served finish stays the default and the fallback).
