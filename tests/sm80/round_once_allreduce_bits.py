# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bits and time of the grouped round-once all-reduce on real GPUs
(vllm/sm80/distributed/round_once_all_reduce.py), against NCCL's BF16 ring.

For every row count and hidden size, on rank-distinct random BF16 partials
(per-rank, per-row scales 2^[-6, 3]), this compares
* ``grouped``: ``nccl_round_once_all_reduce(partial)`` on the TP group's
  PyNccl communicator;
* ``ring``: the TP group's PyNccl all-reduce of the BF16 partial (NCCL's
  ring, one rounding per hop);
* ``exact``: the exact sum of the gathered partials, correctly rounded
  (tests/sm80/test_round_once_all_reduce.py's reference);
records mismatch counts against ``exact``, whether every rank holds the same
bits, and the median time of the grouped all-reduce, the BF16 ring and the
FP32 ring at seed 0. ``grouped`` must equal ``exact`` everywhere (verdict
line ``GROUPED``). Rank 0 writes JSON to ``--out``; the exit code is 0 when
every case ran.

Not a pytest module: it needs peer-connected GPUs, one process each
(test_round_once_all_reduce_gpu.py runs it at TP4 under the served settings
and judges its JSON)::

    torchrun --standalone --nproc-per-node 4 \\
        tests/sm80/round_once_allreduce_bits.py --out result.json
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

ROWS = [1, 7, 819, 820, 1024, 2047, 2048]
HIDDEN = [5120, 4096]


def _time(fn, iters: int) -> float:
    times = []
    for _ in range(iters):
        dist.barrier()
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
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    from tests.sm80.test_round_once_all_reduce import exact_sum, reference_round
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.plugins import load_general_plugins
    from vllm.sm80.distributed.round_once_all_reduce import (
        grouped_round_once_takes,
        nccl_round_once_all_reduce,
    )

    load_general_plugins()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
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
    pynccl = comm.pynccl_comm
    dev = torch.device("cuda", local_rank)
    results = []
    for hidden in HIDDEN:
        for rows in ROWS:
            for seed in range(args.seeds):
                g = torch.Generator(device=dev).manual_seed(
                    1000 * seed + rows + 7 * rank + hidden
                )
                scale = torch.exp2(
                    torch.randint(-6, 4, (rows, 1), generator=g, device=dev).float()
                )
                partial = (
                    torch.randn(rows, hidden, generator=g, device=dev) * scale
                ).to(torch.bfloat16)
                assert grouped_round_once_takes(partial, comm)
                grouped = nccl_round_once_all_reduce(partial, pynccl)
                ring = pynccl.all_reduce(partial)
                gathered = [torch.empty_like(partial) for _ in range(world)]
                dist.all_gather(gathered, partial)
                exact = reference_round(exact_sum(torch.stack(gathered)), torch.bfloat16)
                bits = grouped.view(torch.int16)
                same = [torch.empty_like(bits) for _ in range(world)]
                dist.all_gather(same, bits.contiguous())
                case = {
                    "rows": rows,
                    "hidden": hidden,
                    "seed": seed,
                    "grouped_ne_exact": int((bits != exact.view(torch.int16)).sum()),
                    "ring_ne_exact": int(
                        (ring.view(torch.int16) != exact.view(torch.int16)).sum()
                    ),
                    "ranks_agree": all(torch.equal(s, same[0]) for s in same),
                    "elements": rows * hidden,
                }
                if seed == 0:
                    pf = partial.float()
                    case["us_grouped"] = _time(
                        lambda: nccl_round_once_all_reduce(partial, pynccl), args.iters
                    )
                    case["us_ring_bf16"] = _time(
                        lambda: pynccl.all_reduce(partial), args.iters
                    )
                    case["us_ring_fp32"] = _time(lambda: pynccl.all_reduce(pf), args.iters)
                results.append(case)
    if rank == 0:
        equal = all(c["grouped_ne_exact"] == 0 and c["ranks_agree"] for c in results)
        for c in results:
            t = (
                f" grouped {c['us_grouped']:.1f} us ring bf16 {c['us_ring_bf16']:.1f} us"
                f" ring fp32 {c['us_ring_fp32']:.1f} us"
                if "us_grouped" in c
                else ""
            )
            print(
                f"hidden {c['hidden']} rows {c['rows']:5d} seed {c['seed']} "
                f"grouped ne {c['grouped_ne_exact']} ring ne {c['ring_ne_exact']:8d}/"
                f"{c['elements']} agree {c['ranks_agree']}{t}"
            )
        state = "EQUAL to" if equal else "DIFFERS from"
        print(f"GROUPED {state} the correctly rounded exact sum")
        with open(args.out, "w") as f:
            json.dump(
                {"world": world, "equal": equal, "cases": results}, f, indent=1
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
