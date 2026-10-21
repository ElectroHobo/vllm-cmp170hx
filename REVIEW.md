# B1: DeepSeek-V4.1 step fusion, attention pipeline fixes and prefill split-reduce

Intent: one tree for the B1 bundle on base 7b31d6b252. Every item is behind
its own default-off switch; B1 turns them on together:
`VLLM_DEEPSEEK_V41_FUSED_ROUTE`, `VLLM_DEEPSEEK_V41_ATTN_PIPELINE`,
`VLLM_DEEPSEEK_V41_PREFILL_SPLIT_REDUCE`,
`VLLM_DEEPSEEK_V41_WO_B_BF16_PREFILL_ALLREDUCE` (an existing 7b31 switch)
and the switches the current candidate already runs. Sections below, one per
slice, in commit order.

# MoE routing after the gate GEMM as one kernel

`VLLM_DEEPSEEK_V41_FUSED_ROUTE` replaces the fifteen torch kernels of
`sm80_moe.route` that follow the FP32 gate GEMM (image-token bias select,
sqrt-softplus, bias add, top-k and its sort, gather, row sum, `+ 1e-20`,
divide, scale, int32 ids) with one Triton kernel
(`ops/sm80_route.py`). The FP32 operations and their order are torch's:
libdevice `exp`/`log1p` compiled without flush-to-zero, `sqrt_rn`, `div_rn`,
and the row sum in the order of torch's CUDA reduction kernel for a short
contiguous row; the outputs are the torch path's bytes for finite logits
(the order of exactly equal biased scores aside, which torch leaves open).
`gate_temp != 1` is refused. Files: `ops/sm80_route.py` (new),
`sm80_moe.py` (router attribute and dispatch), `vllm/sm80/envs.py`,
`vllm/sm80/dev_ab.py` (switch `dsv41_fused_route`). Tests:
`test_sm80_route_fused.py` (interpreter vs torch, bitwise vs a NumPy model of
the CUDA order, sm_80 compile and PTX checks, switch default) and
`test_sm80_route_fused_gpu.py` (bytes vs the torch path on CUDA, the order
model vs `torch.sum`, graph replay; GPU, unrun).

# Attention pipeline: index work bounded by visibility

In a FULL graph the pipeline's index width is the capacity
(`max_model_len // ratio`); the score producer and the radix selector
processed every column of every row each step. Segments past a row's visible
count now take a fast path in the producer (the same -inf scores, block
maxima and empty histogram, no scoring) and are skipped by every selector
pass; grids keep the capacity width. Scores, maxima and selections are
unchanged bit for bit. Files: `ops/sm80_attn_pipeline.py`,
`ops/sm80_index_topk.py`. Tests: `test_sm80_index_visible.py` (interpreter,
with the GPU `tl.histogram` semantics installed: the stock interpreter counts
the out-of-range sentinel in the last bin), and a capacity-width case in
`test_sm80_index_topk_gpu.py`.

# Pipeline compressor circular predecessor

The default-off attention pipeline now wraps a position-zero predecessor
within its own ring, matching the existing Torch compressor. Triton lowers
signed integer remainder to LLVM `srem`; `(position - 1) % capacity` is -1
at position zero. The first row of a dummy request reads that predecessor
even when its compressed-cache slot is padding. Clamping the ring slot to
zero therefore previously read one row before the null ring block.
Adding the capacity before subtracting one preserves the circular index for
all supported nonnegative positions without changing pooling arithmetic,
workspace ownership, graph mode, supported limits or flag defaults.

The only production change is in `sm80_attn_pipeline.py`. Direct sibling
ring stores require a nonnegative slot; the Torch reference already uses
nonnegative remainder. No shared or upstream-owned production file changes.

`test_sm80_pipeline_lifecycle.py` drives the actual launch wrapper and
Triton interpreter through both predecessor loads, checking their addresses
before dereferencing. It covers position zero, wrap boundaries, maximum
context, null/owned blocks and padded strides. It also exercises actual
manager-owned workspace allocations, ownership separation and locking.
The address probe deliberately stops before unsupported libdevice math;
it is not a GPU lifetime or numerical qualification. A focused CUDA test in
`test_sm80_attn_pipeline_gpu.py` covers a fresh ring and graph pool before
and after retirement with persistent production scratch; it remains unrun.
The observed asynchronous CUDA fault still needs targeted device attribution.


# Attention pipeline: graph-capture warm-up inputs test

`test_sm80_attn_pipeline_gpu.py::test_capture_warmup_inputs_eager_then_graph`
runs the pipeline with the capture warm-up's dummy inputs (zero positions,
all-zero block tables at width 262144, padding or null-block slots, 4/6/48
rows, all roles, production scratch shape) eagerly, then captured and
replayed. Only the padding form is checked bitwise between eager and replay:
with null-block slots every row writes the same null-block window row. Run
on one card: 34 passed with the ring and capacity-width tests.

# Prefill split-reduce: DeepSeek-V4.1 prefill all-reduces beside compute, in two row slices

Intent: in a TP4 prefill step each of the 20 full-row decoder layers runs two
large NCCL all-reduces (the FP32 `wo_b` sum and the BF16 MoE sum) between the
compute that makes and the compute that needs them, so communication and
compute add up. With `VLLM_DEEPSEEK_V41_PREFILL_SPLIT_REDUCE=1` an eager step
of at least 1024 rows runs each target layer's `wo_a`/`wo_b`, FFN mHC, MoE and
both all-reduces in two row slices cut at a multiple of 512 rows, issuing the
all-reduces on one side stream in program order: `wo_b(B)` runs beside the
sum of `wo_b(A)`, `FFN(A)` beside the sum of `wo_b(B)`, `FFN(B)` beside the
sum of `FFN(A)`. The attention runs on every row as before, and the layer's
outputs are whole again at the join, so the next layer, the Engram layers and
the decoder replay are unchanged. Default off; switch read at load.

Files:
- `vllm/models/deepseek_v41/nvidia/sm80_split_reduce.py` (new): the cut
  (`cut_rows`, `split_point`), the side stream and its hand-offs
  (`SideStream`, `_Pending`), the split post-attention schedule
  (`post_attention`), the adoption (`adopt_split_reduce`).
- `sm80_model.py`: the decoder layer takes the split path when its
  `_split_reduce` is set and `split_point` cuts; the backbone adoption marks
  its layers under the switch.
- `sm80_attention.py`: `forward` is `_o_proj(attend(...))`; `_o_proj` is
  `reduce_wo_b(wo_b_partial(...))` when `reduces_wo_b()`, as before.
- `sm80_moe.py`: the layer-reduced MoE (`Sm80DeepseekV41MoE`) is now
  `reduce(forward_partial(...))`; the switch adopts it with the runner's own
  all-reduce (`_adopt_layer_all_reduce(round_once=False)`), the round-once
  switch as before (`round_once=True`).
- `vllm/sm80/envs.py`: the switch. `FORK.md` regenerated.

Tests (CPU, `CUDA_VISIBLE_DEVICES=""`):
- `tests/models/deepseek_v41/sm80/test_sm80_split_reduce.py`: the cut for
  2044/2048/4096/1536/1100/1024 rows and refusals; `split_point` refuses graph
  capture and replay, the kernel capture, MegaMoE metadata, short steps and
  slices the host-staged or custom all-reduce would take; two split decoder
  layers (model entry and a fused boundary, torch mHC) return the unsplit
  layers' outputs bit for bit with row-wise sums, and issue attention,
  `wo_b(A)`, its sum, `wo_b(B)`, its sum, `FFN(A)`, its sum, `FFN(B)`, its sum;
  the side-stream hand-off order (wait for the current stream, run, record,
  keep the input alive on the side stream; the consumer waits for the event
  and keeps the output alive); adoption marks only layers that issue their own
  sums; default off.
- `test_sm80_moe_allreduce.py`: adoption in both modes, `forward` equals
  `reduce(forward_partial)`.
- `test_sm80_attention_wiring.py`: `reduces_wo_b`, and `_o_proj` equals the
  concatenated row-slice `reduce_wo_b(wo_b_partial)`.
- `tests/models/deepseek_v41/sm80` with `tests/sm80/test_seams.py` and the
  guard tests: 650 passed, 357 skipped (GPU-only).

Bits: equal to the unsplit step when the per-row kernels do not depend on the
call's row count and the all-reduce is split-invariant. The cut keeps the MoE
prefill kernel's 512-row chunks and Marlin's 64-row blocks; the round-once
all-reduces are split-invariant, NCCL's ring is not (an element's sum order
follows its place in the message), so with the default FP32/BF16 ring sums the
bits may differ.

Known limits: the speed gain depends on how much the compute beside an NCCL
all-reduce slows (unmeasured for these kernels); the two slices' FFN mHC
outputs (streams, mixes, next pre-mix) are concatenated, one extra copy of
the streams per layer; a step whose slices the host-staged or custom
all-reduce would take runs whole. Four-card correctness and speed are outside
pytest.

Removes: nothing.
