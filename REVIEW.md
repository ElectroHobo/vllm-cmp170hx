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
  `VLLM_DEV_AB_SWITCH` naming it, which keeps both forms of the weight).
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
30 cases, all failing on the previous head): the decode PTX block run by an
emulator of its own instructions over every finite code in every lane,
exact against the FP64 E4M3/UE8M0 definitions (mutations of a selector,
shift or bias fail it); the kernel compiled for sm_80 at every row tile
(BF16 MMA, FP32 accumulation, none twice, one decode per weight for 16-row
tiles, one rounding at the store, no ftz, no spills); the launch (one,
strides of a padded or strided attention output read in place, the output
the only allocation); the method (codes kept without a copy, no Marlin
repack, both arms under the dev switch, refusals); the switch default and
the dev A/B switch. `test_hybrid_checkpoint.py`: the flag and the dev
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

Intent: with `VLLM_DEEPSEEK_V41_PREFILL_CE_ALLREDUCE=1`, the prefill-sized TP
sums that the host-staged and custom all-reduce do not take move by
copy-engine copies instead of SM kernels (NCCL's ring, or the custom
all-reduce in pieces). Both keep SMs busy for the whole transfer, and the
compute beside them (the prefill split reduction's overlap) slows 2-3x.
Two algorithms share one scratch and flag block:
- `ring_ops` (after b12x's PCIe DMA all-reduce, Apache-2.0): reduce-scatter +
  all-gather ring, one add per step in the message dtype. It carries the FP32
  `wo_b` sum and the MoE sum (runner path).
- `round_once_ops` (owner sum): every rank copies shard j to rank j, rank j
  sums the world partials in FP32 in rank order and rounds once, then copies
  the result to every rank. Same bytes as the ring. It carries the round-once
  contract (`VLLM_DEEPSEEK_V41_WO_B_BF16_PREFILL_ALLREDUCE`, and
  `VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE` when on) in place of the
  custom all-reduce in pieces, which split-reduce's 1024-row slices took
  (two pieces of < 8 MiB each), leaving two of the three overlap windows per
  layer beside SM all-reduce kernels.
Default off; decode, graph capture and messages the custom all-reduce takes
whole are unchanged.

Files:
- `vllm/sm80/distributed/ce_ring_all_reduce.py` (new): `ring_steps`,
  `emulate` / `emulate_owner` (host bits of each algorithm), `ring_ops`,
  `round_once_ops`, the device backend (Triton system-scope release flag
  store, system-scope acquire flag wait bounded by `WAIT_TIMEOUT_NS`,
  elementwise add, FP32 owner sum of up to eight partials;
  `cudaMemcpyAsync` through the loaded CUDA runtime), `CeRingAllReduce`
  (`all_reduce`, `round_once`; IPC scratch: 2(world-1) slots for a
  `max_num_batched_tokens` x hidden FP32 message, plus a flag block;
  collective construction), `init_ce_ring` / `ce_ring`.
- `vllm/sm80/distributed/step_errors.py`: the sticky error word is
  bit-coded (1 round-once span, 2 copy-engine ring wait timeout); the raised
  message names every set bit.
- `sm80_attention.py`: `_ce_ring_for` (the ring when it is built, takes the
  message, and neither the host-staged nor the custom all-reduce does);
  `prefill_all_reduce` uses its ring path, `large_round_once_all_reduce` its
  owner sum (before the grouped and pieces transports); the FP32 `wo_b` sum
  goes through `prefill_all_reduce`.
- `sm80_moe.py`: the layer-reduced MoE's runner-path sum goes through
  `prefill_all_reduce`; the switch adopts the layer-reduced MoE.
- `sm80_model.py`: the backbone adoption builds the ring under the switch.
- `vllm/sm80/envs.py`: the switch. `FORK.md` regenerated.

Protocol. Ring, per call (epoch = call count, the same on every rank): store
the epoch into the previous rank's ready flag, wait for this rank's; per ring
step copy the send shard into the next rank's slot, store its step flag, wait
for this rank's step flag, add or copy. Owner sum: store the epoch into every
sender's ready flag for this receiver and wait for every receiver's; copy each
shard to its owner's slot (indexed by sender offset) and store its flag; wait
for every sender's flag and sum; copy the result to every peer's gather slot
and store its flag; wait for every owner's flag and copy the shard out.
Flags are monotonic epochs shared by both algorithms, so calls of either kind
interleave; a rank writes into a peer's slot only after that peer's ready
store for the call, which follows its reads of the previous call on its
stream. Flag stores are `st.release.sys`; waits load with `ld.acquire.sys`;
a wait that exceeds 60 s ORs bit 2 into the producing stream's sticky error
word and returns (the step fails at its output copy; no hang, no trap).

Bits: the ring sums each element in its owner's ring order in the message
dtype (FP32 for `wo_b`, BF16 per add for the MoE, as NCCL's BF16 ring). The
owner sum adds the partials in FP32 in rank order and rounds once, the
round-once switches' contract; its bits do not depend on how the message is
split. Both give every rank the same bits; the order differs from NCCL's and
from the custom all-reduce's, so outputs change against those builds (V6).

Tests (CPU): `tests/sm80/test_ce_ring_all_reduce.py`: ring schedule
properties; emulations against the FP64 sum; `ring_ops` for every rank on a
host memory model over three back-to-back calls; owner-sum and ring calls
interleaved on one scratch, flag block and epoch sequence (world 2/3/4/8,
FP32 and BF16): each call's emulated bits on every rank, every copy's bytes
checked where they land; owner-sum split invariance; ready-before-write for
both algorithms; the kernels compiled for sm_80 on the host (system-scope
release/acquire flags, a bounded wait that ORs into the error word with no
trap, the owner sum's FP32 accumulation and one BF16 rounding); the error
word's messages. Dispatch:
`tests/models/deepseek_v41/sm80/test_sm80_ce_allreduce_dispatch.py`
(including the BF16 `wo_b` large path to the owner sum). The owner-sum tests
fail on the previous head (14 failed).

Four-card evidence (ring only, check r3, 10-21): 88 calls bit-exact on all
ranks; FP32 41.9 MB 9.93 ms vs NCCL 10.93 ms; a persistent Marlin W4A16 GEMM
0.86x beside the ring and its flag waits vs 2.29x beside NCCL, a BF16 GEMM
0.87x vs 1.98x. The owner sum has no four-card run yet.

Known limits: one stream, one piece per step (no copy/add pipelining); the
owner sum's copies run one after another on that stream; 63 MiB of device
memory per rank at TP4 with 2048-row steps; decode sums are unaffected by
construction (custom all-reduce sizes, graphs).

Removes: nothing.

# Grouped wo_a reads its groups' rows in place

`VLLM_DEEPSEEK_V41_WO_A_STRIDED` drops the two copy kernels per attention
layer in front of the FP8 `wo_a` of a hybrid checkpoint (one Marlin W8A16
GEMM per local group): `GroupedLinearMethod.apply_grouped` hands each group's
rows of the `[T, groups, input]` attention output to the method as a view
(row stride `groups * input`) instead of `x[:, g].contiguous()`. Marlin takes
`A.stride(0)` as its row stride (`lda`, `csrc/quantization/marlin/marlin.cu`:
checks at 528-532, passed to `marlin_mm` at 723; used only to address rows,
`marlin_template.h` 451-504), and the tile choice depends on m, n and k
only, so the GEMMs and their bytes are unchanged. The outputs keep their
`torch.cat`: Marlin's `c` must be contiguous (`marlin.cu` 572), is not a
mutable argument of the op schema (`csrc/torch_bindings.cpp` 83), and the
registered `vllm::marlin_gemm` passes none, so a group cannot write into its
column slice of a `[T, groups * output]` result for T > 1.

The option is `GroupedLinearMethod(strided_input=...)`, default off; only
DeepSeek-V4.1 attention adopts `GroupedLinearMethod` (`sm80_attention.py`),
and only it passes the switch. GLM and Qwen do not use the class. Files:
`vllm/sm80/layers/grouped_linear.py`, `sm80_attention.py` (adoption),
`vllm/sm80/envs.py`, `vllm/sm80/dev_ab.py` (switch `dsv41_wo_a_strided`,
output-identical). Tests: `test_wo_a_strided.py` (CPU: no copy op, the
inputs are views of the attention output with its strides, the copy path's
bytes, for 1/6/48 rows and 2/4 groups; default adoption keeps the copies;
the switch default and the dev A/B switch over target and drafter; CUDA:
the real Marlin GEMM of an MXFP8 layer gives the copy path's bytes for 1-48
rows at 2 and 4 groups, GPU, unrun) and `test_hybrid_checkpoint.py` (the
switch reaches every target and drafter `wo_a` of the hybrid model at TP4).

Known limits: a single row is already contiguous, so steps of one row gain
nothing; the `torch.cat` (1.9 us per layer) stays. An inner method that
cannot read a row-strided input must not be given the switch (Marlin
refuses a misaligned one: its row stride and group offset must be multiples
of 8 elements; DeepSeek-V4.1's group input is 4096).

Removes: nothing (the copy path stays the default).

# Replicated hot routed experts

Intent: cut the all-reduce wait for the busiest EP rank in decode MoE calls.
That wait is routed-expert count imbalance: about 30 µs per extra distinct
expert per target layer. `VLLM_EXL3_EXPERT_REPLICAS` (a SHA-pinned plan, empty
by default) copies each layer's listed hot experts onto every rank. Per MoE
call of 1..64 rows, one owner rank per used listed expert computes its
assignments. Rule and limits are in the `replicas.py` docstring and in
`docs/features/quantization/exl3.md` ("Replicated hot experts").

Files:
- `vllm/sm80/kernels/exl3/replicas.py` (new): plan, layout, owner rule, its
  reference, and the Triton per-call map kernel.
- `vllm/sm80/kernels/exl3/quant.py`: replica slots after the home slots in
  `create_weights`; the operator gates check the home slots; `Exl3Experts`
  takes the per-call map for calls of 1..64 rows.
- `vllm/sm80/envs.py`: the switch and its digest.
- `docs/features/quantization/exl3.md`.
- `tests/kernels/quantization/test_exl3_replicas.py` (new).

No upstream-owned file changes. No native change: the kernels already size
local experts from the pointer table and global experts from the map length,
and the map length does not change.

Numerics: order-changing. Expert arithmetic, router ids, weights and top-k
order are unchanged, but an expert that leaves its home rank lands in another
rank's partial sum, so the reduction order differs (V6 or V2's rounding
check). A call whose used experts are already balanced keeps ordinary
placement exactly.

Tests (CPU):
- plan validation and refusals;
- the owner rule: exactly one rank per assignment, the same owners on every
  rank, slots holding the right expert, never busier than ordinary placement,
  balanced calls unchanged;
- exact rank sums equal to ordinary placement for integer-valued experts;
- the real loader holding each copy's checkpoint bytes, and the replica byte
  ledger;
- the experts' dispatch by row count;
- the kernel against the reference in the Triton interpreter.

Known limits:
- The kernel's GPU run, graph replay and its per-layer launch cost (about
  3–5 µs per MoE layer [EST]) are unmeasured.
- Owner choice counts distinct experts, not row tiles.
- Replica bytes come out of the KV pool.

Removes: nothing.
