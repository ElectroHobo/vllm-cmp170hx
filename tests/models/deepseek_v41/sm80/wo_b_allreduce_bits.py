# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Do the TP4 all-reduces of BF16 wo_b partials give the same bits as the
FP32 path the sm_80 DeepSeek-V4.1 attention uses today?

Today ``DeepseekV41Sm80Attention._o_proj`` all-reduces ``partial.float()``
and rounds the sum to BF16 once. The served kernels (flags two-shot, custom
1/2-stage, host-staged) accumulate BF16 inputs in FP32 in a fixed order and
round once, so all-reducing the BF16 partial directly may give the same bits
at half the bytes. For every row count, on rank-distinct random partials of
the wo_b output shape ``[rows, 5120]``, this compares

* ``bf16``: ``all_reduce(partial)`` (BF16 message);
* ``fp32``: ``all_reduce(partial.float()).bfloat16()`` (today's path);
* ``exact``: the FP64 sum of the four gathered partials rounded to BF16;

* ``switched``: ``wo_b_all_reduce(partial, True)``, the layer's own path
  with VLLM_DSV41_WO_B_BF16_ALLREDUCE on (BF16 only where the custom
  all-reduce takes the message);

reports which backend each message takes, bitwise equality, mismatch counts
and the largest BF16-ulp distance, and the median time of each all-reduce.
``switched`` must equal ``fp32`` everywhere (verdict line ``SWITCHED``).
Rank 0 prints a table and writes JSON to ``--out``. The exit code is 0 when
every case ran; the verdict line says whether ``bf16`` equals ``fp32``.

Not a pytest module: it needs four peer-connected sm_80 GPUs, one process
each, under the served all-reduce switches::

    VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1 VLLM_CUSTOM_ALLREDUCE_FLAGS=1 \\
    VLLM_CUSTOM_ALLREDUCE_ALGO=2stage VLLM_HOST_ALLREDUCE=1 NCCL_P2P_LEVEL=SYS \\
    torchrun --standalone --nproc-per-node 4 \\
        tests/models/deepseek_v41/sm80/wo_b_allreduce_bits.py --out result.json
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

ROWS = [1, 2, 4, 8, 16, 24, 25, 26, 32, 51, 52, 64, 128, 256, 512, 1024, 2048]
HIDDEN = 5120


def _ordered(t: torch.Tensor) -> torch.Tensor:
    bits = t.view(torch.int16).int() & 0xFFFF
    return torch.where(bits >= 0x8000, 0x8000 - bits, bits)


def _ulps(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((_ordered(a) - _ordered(b)).abs().max()) if a.numel() else 0


def _backend(comm, t: torch.Tensor) -> str:
    """Which backend the fork communicator dispatches ``t`` to (its own
    predicates, evaluated in its dispatch order)."""
    host = getattr(comm, "hostshm_comm", None)
    if host is not None and host.should_host_ar(t):
        return "hostshm"
    ca = getattr(comm, "ca_comm", None)
    if ca is not None and not ca.disabled and ca.should_custom_ar(t):
        flags = getattr(ca, "_flags", None)
        if flags is not None and flags.eligible(t):
            return "flags"
        return "custom"
    return "other(nccl/symm)"


def _time(fn, iters: int) -> float:
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
        tensor_model_parallel_all_reduce,
    )
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.models.deepseek_v41.nvidia.sm80_attention import (
        custom_all_reduce_takes,
        wo_b_all_reduce,
    )

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world == 4, "run with --nproc-per-node 4"
    torch.cuda.set_device(local_rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method="env://",
            backend="nccl",
        )
        ensure_model_parallel_initialized(world, 1)
    comm = get_tp_group().device_communicator
    dev = torch.device("cuda", local_rank)

    results = []
    for rows in ROWS:
        for seed in range(args.seeds):
            g = torch.Generator(device=dev).manual_seed(1000 * seed + rows + 7 * rank)
            # wo_b partials: sums of ~2048 products, magnitudes around 1 with
            # a heavy tail; a scale per row so exponents vary across rows.
            scale = torch.exp2(
                torch.randint(-6, 4, (rows, 1), generator=g, device=dev).float()
            )
            partial = (
                torch.randn(rows, HIDDEN, generator=g, device=dev) * scale
            ).to(torch.bfloat16)

            bf16 = tensor_model_parallel_all_reduce(partial.clone())
            fp32 = tensor_model_parallel_all_reduce(partial.float()).to(torch.bfloat16)
            switched = wo_b_all_reduce(partial.clone(), True)
            gathered = [torch.empty_like(partial) for _ in range(world)]
            dist.all_gather(gathered, partial)
            exact = torch.stack(gathered).double().sum(0).to(torch.bfloat16)

            case = {
                "rows": rows,
                "seed": seed,
                "backend_bf16": _backend(comm, partial),
                "backend_fp32": _backend(comm, partial.float()),
                "switched_bf16": custom_all_reduce_takes(partial),
                "switched_eq_fp32": bool(torch.equal(switched, fp32)),
                "bf16_eq_fp32": bool(torch.equal(bf16, fp32)),
                "bf16_ne_fp32": int((bf16.view(torch.int16) != fp32.view(torch.int16)).sum()),
                "ulps_bf16_fp32": _ulps(bf16, fp32),
                "bf16_ne_exact": int((bf16.view(torch.int16) != exact.view(torch.int16)).sum()),
                "fp32_ne_exact": int((fp32.view(torch.int16) != exact.view(torch.int16)).sum()),
            }
            if seed == 0:
                case["us_bf16"] = _time(
                    lambda: tensor_model_parallel_all_reduce(partial), args.iters
                )
                pf = partial.float()
                case["us_fp32"] = _time(
                    lambda: tensor_model_parallel_all_reduce(pf).to(torch.bfloat16),
                    args.iters,
                )
                case["us_switched"] = _time(
                    lambda: wo_b_all_reduce(partial, True), args.iters
                )
            results.append(case)

    # Every rank must hold the same bits; check rank agreement on the flags.
    flags = torch.tensor(
        [c["bf16_ne_fp32"] for c in results], dtype=torch.int64, device=dev
    )
    agree = flags.clone()
    dist.all_reduce(agree, op=dist.ReduceOp.MAX)
    if rank == 0:
        equal = all(c["bf16_eq_fp32"] for c in results)
        switched_ok = all(c["switched_eq_fp32"] for c in results)
        for c in results:
            t = (
                f" bf16 {c['us_bf16']:.1f} us fp32 {c['us_fp32']:.1f} us"
                f" switched {c['us_switched']:.1f} us"
                if "us_bf16" in c
                else ""
            )
            print(
                f"rows {c['rows']:5d} seed {c['seed']} "
                f"{c['backend_bf16']:>16s}/{c['backend_fp32']:<16s} "
                f"switched {'bf16' if c['switched_bf16'] else 'fp32'} "
                f"eq {c['switched_eq_fp32']!s:5s} "
                f"bf16==fp32 {c['bf16_eq_fp32']!s:5s} ne {c['bf16_ne_fp32']:7d} "
                f"ulps {c['ulps_bf16_fp32']:3d} | vs exact bf16 "
                f"{c['bf16_ne_exact']:7d} fp32 {c['fp32_ne_exact']:7d}{t}"
            )
        verdict = (
            "EQUAL: the BF16 all-reduce gives the FP32 path's bits"
            if equal
            else "DIFFERS: the BF16 all-reduce changes bits"
        )
        print(f"VERDICT {verdict}; max ne over ranks {int(agree.max())}")
        n_bf16 = sum(c["switched_bf16"] for c in results)
        state = "EQUAL to" if switched_ok else "DIFFERS from"
        print(
            f"SWITCHED {state} the FP32 path "
            f"(BF16 messages: {n_bf16}/{len(results)})"
        )
        with open(args.out, "w") as f:
            json.dump(
                {
                    "verdict": verdict,
                    "equal": equal,
                    "switched_equal": switched_ok,
                    "cases": results,
                },
                f,
                indent=1,
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
