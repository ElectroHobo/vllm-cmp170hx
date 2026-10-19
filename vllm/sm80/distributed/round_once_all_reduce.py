# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Round-once all-reduce of large messages over grouped NCCL send/recv.

Why this exists
---------------
NCCL's ring all-reduce of a BF16 (or FP16) message stores every hop's partial
sum in the message dtype, so with N ranks an element is rounded N - 1 times
and the result depends on the ring order. At TP4 about a third of the elements
of a prefill-sized message differ from the correctly rounded sum (CPU emulation
of the ring, calibrated against a measured TP4 run). Two ranks are the
exception: one hop, one rounding.

What it does
------------
The same bytes on the wire as the ring, one rounding:

  1. reduce-scatter as one grouped set of NCCL send/recv: the flat message is
     cut into N equal chunks (the last one zero-padded) and rank r receives
     every rank's chunk r, in the message dtype;
  2. rank r sums its N chunks in FP64 in rank order and rounds the sum once to
     the message dtype (round to odd into FP32, then round to nearest even:
     a correct rounding of the FP64 sum);
  3. all-gather of the rounded chunks (NCCL), in the message dtype.

Exactness: the FP64 sum of N BF16 (FP16) values is exact while their nonzero
magnitudes lie within a factor 2^(45 - ceil(log2 N)) (2^(42 - ceil(log2 N)))
of each other. Outside that span a small summand can be lost (for example,
BF16 [1, 2**-60, -1, 0] sums to zero in this order). The result is always the
rank-ordered FP64 sum rounded once, deterministic and independent of message
splitting, not independent of rank order or TP size. Messages of other dtypes
are not taken.

Used only outside CUDA graph capture (the call sites ask for it on eager
prefill steps), with the TP group's PyNccl communicator. Model call sites opt
in through their own switches; VLLM_ROUND_ONCE_ALLREDUCE_GROUPED selects this
transport for them (see vllm/sm80/envs.py).
"""

from collections.abc import Callable

import torch

ROUND_ONCE_DTYPES = (torch.bfloat16, torch.float16)


def round_from_fp64(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``x`` (FP64) correctly rounded to ``dtype`` (BF16 or FP16; round to
    nearest, ties to even): first rounded to odd into FP32 (truncate towards
    zero, then set the last bit when inexact), which keeps the conversion to
    the narrower format correctly rounded, FP32 having at least 13 more
    significand bits."""
    f = x.float()
    back = f.double()
    inexact = back != x
    away = inexact & (back.abs() > x.abs())
    bits = f.view(torch.int32)
    bits = torch.where(away, bits - 1, bits)
    bits = torch.where(inexact, bits | 1, bits)
    return bits.view(torch.float32).to(dtype)


# exchange(send_chunks, recv_chunks): rank r sends send_chunks[j] to rank j and
# receives rank j's chunk r into recv_chunks[j], for every j != r (one group).
Exchange = Callable[[list[torch.Tensor], list[torch.Tensor]], None]
# all_gather(out, chunk): out = the ranks' chunks in rank order.
AllGather = Callable[[torch.Tensor, torch.Tensor], None]


def round_once_all_reduce_with(
    t: torch.Tensor,
    rank: int,
    world: int,
    exchange: Exchange,
    all_gather: AllGather,
) -> torch.Tensor:
    """The round-once all-reduce of ``t`` over ``world`` ranks with the given
    transports (NCCL in serving, gloo in the CPU tests). Returns a new tensor
    like ``t``; ``t`` is not modified."""
    if t.dtype not in ROUND_ONCE_DTYPES:
        raise TypeError(f"round-once all-reduce of {t.dtype}")
    flat = t.reshape(-1)
    numel = flat.numel()
    chunk = -(-numel // world)
    if chunk * world != numel:
        padded = torch.zeros(chunk * world, dtype=t.dtype, device=t.device)
        padded[:numel] = flat
        flat = padded
    sends = list(flat.view(world, chunk).unbind(0))
    recv = torch.empty((world, chunk), dtype=t.dtype, device=t.device)
    recvs = list(recv.unbind(0))
    recvs[rank].copy_(sends[rank])
    if world > 1:
        exchange(sends, recvs)
    acc = recv[0].double()
    for j in range(1, world):
        acc += recv[j].double()
    mine = round_from_fp64(acc, t.dtype)
    gathered = torch.empty(chunk * world, dtype=t.dtype, device=t.device)
    if world > 1:
        all_gather(gathered, mine)
    else:
        gathered.copy_(mine)
    return gathered[:numel].view(t.shape)


def nccl_round_once_all_reduce(t: torch.Tensor, pynccl) -> torch.Tensor:
    """:func:`round_once_all_reduce_with` over a vLLM ``PyNcclCommunicator``
    on the current stream: one NCCL group of sends and receives, then NCCL's
    all-gather."""
    rank, world = pynccl.rank, pynccl.world_size
    stream = torch.cuda.current_stream()

    def exchange(sends, recvs):
        pynccl.group_start()
        for j in range(world):
            if j != rank:
                pynccl.send(sends[j], j, stream)
                pynccl.recv(recvs[j], j, stream)
        pynccl.group_end()

    def all_gather(out, chunk):
        pynccl.all_gather(out, chunk, stream)

    return round_once_all_reduce_with(t, rank, world, exchange, all_gather)


def grouped_round_once_takes(t: torch.Tensor, comm) -> bool:
    """Whether :func:`nccl_round_once_all_reduce` can reduce ``t`` on the
    device communicator ``comm``: a contiguous message of a round-once dtype,
    an enabled PyNccl communicator of more than one rank on ``t``'s device,
    and no CUDA graph being captured."""
    pynccl = getattr(comm, "pynccl_comm", None)
    return (
        t.is_cuda
        and t.dtype in ROUND_ONCE_DTYPES
        and t.is_contiguous()
        and t.numel() > 0
        and pynccl is not None
        and not pynccl.disabled
        and pynccl.world_size > 1
        and t.device == pynccl.device
        and not torch.cuda.is_current_stream_capturing()
    )
