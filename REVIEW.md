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
layers: 3 x 40.55 MB x 2 = 0.227 GiB per card less KV cache. A weight-cache
load that skips processing leaves no replica; the first prefill step then
raises. Prefill only: decode is unchanged by construction.

Removes: nothing; the sharded path stays for decode and for the flag off.
