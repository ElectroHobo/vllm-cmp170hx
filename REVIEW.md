# Copy-engine ring all-reduce for DeepSeek-V4.1 prefill sums

Intent: with `VLLM_DEEPSEEK_V41_PREFILL_CE_ALLREDUCE=1`, the TP sums of the
FP32 `wo_b` partial and of the MoE output that the host-staged and custom
all-reduce do not take (eager prefill-sized messages) run on a copy-engine
ring instead of NCCL. NCCL's ring moves the bytes with SM kernels that stay
resident for the whole transfer, and the compute beside them (the prefill
split reduction's overlap) slows 2-3x; copy-engine copies do not occupy SMs.
The design follows b12x's PCIe DMA all-reduce (Apache-2.0): reduce-scatter +
all-gather ring, `cudaMemcpyAsync` into the next rank's IPC scratch,
one-block flag kernels, one add per reduce-scatter step. Default off; decode,
graph capture and messages the custom all-reduce takes are unchanged.

Files:
- `vllm/sm80/distributed/ce_ring_all_reduce.py` (new): `ring_steps`,
  `emulate` (the host's bits of the ring), `ring_ops` (one call's copies,
  flag stores, flag waits and adds on a backend), the device backend (Triton
  flag store, volatile-load flag wait, elementwise add; `cudaMemcpyAsync`
  through the loaded CUDA runtime), `CeRingAllReduce` (IPC scratch: one slot
  per ring step for a `max_num_batched_tokens` x hidden FP32 message, plus a
  flag block; collective construction), `init_ce_ring` / `ce_ring`.
- `sm80_attention.py`: `prefill_all_reduce` (the ring when it takes the
  message and neither the host-staged nor the custom all-reduce does, else
  `tensor_model_parallel_all_reduce`); the FP32 `wo_b` sum goes through it.
- `sm80_moe.py`: the layer-reduced MoE's runner-path sum goes through
  `prefill_all_reduce`; the switch adopts the layer-reduced MoE.
- `sm80_model.py`: the backbone adoption builds the ring under the switch.
- `vllm/sm80/envs.py`: the switch. `FORK.md` regenerated.

Protocol per call (epoch = call count, the same on every rank): store the
epoch into the previous rank's ready flag, wait for this rank's; then per ring
step copy the send shard into the next rank's slot, store its step flag, wait
for this rank's step flag, add (reduce-scatter: `out = input + slot`) or copy
(all-gather). Flags are monotonic, slots are per step, the ready flag keeps a
rank from writing a slot its neighbour may still read.

Bits: each element is summed in its owner's ring order in the message dtype
(FP32 for `wo_b`, BF16 per add for the MoE, as NCCL's BF16 ring); every rank
gets the same bits, `emulate` reproduces them. The order differs from NCCL's,
so outputs change against the NCCL build (V6).

Tests (CPU): `tests/sm80/test_ce_ring_all_reduce.py`: schedule properties
(world 2/3/4/8), emulation against the FP64 sum, `ring_ops` for every rank on
a host memory model over three back-to-back calls (world 2 and 4, FP32 and
BF16): emulated bits on every rank and every copy's bytes checked where they
land; ready-before-write ordering; host tensors refused. Dispatch:
`tests/models/deepseek_v41/sm80/test_sm80_ce_allreduce_dispatch.py`. With
`tests/models/deepseek_v41/sm80` and `tests/sm80/test_seams.py`: 695 passed,
403 skipped (GPU-only). The ring test file fails to collect on the base (no
module).

Known limits: one stream, one piece per step (no copy/add pipelining); the
flag wait spins one block while a peer copy is in flight (the four-card
check measures Marlin beside it); 63 MiB of device memory per rank at TP4
with 2048-row steps; decode sums are unaffected by construction (custom
all-reduce sizes, graphs). Four-card bits and timing are outside pytest.

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
