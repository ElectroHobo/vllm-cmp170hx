# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 MoE routing on sm_80, as the reference ``Gate`` computes it.

Upstream routes DSv4.1 with an FP32-weight-less gate (on sm_80 GateLinear
stores the FP16 checkpoint weight in BF16 and multiplies in BF16) and fused
kernels (``dsv4_topk``, ``topk_hash_softplus_sqrt``) whose softplus,
normalisation and scale differ from the reference in rounding (fast
``__logf``/``__expf``, ``log(1 + exp)``, ``w * (scale / sum)``, no ``+1e-20``).
:func:`adopt_moe` keeps the upstream modules, parameter names and EXL3
experts, and changes two things before the weights load:

- the gate weight is kept in FP32 (the FP16 checkpoint values exactly) and
  the gate computes ``x.float() @ weight.T`` in FP32;
- the runner's router is :class:`Sm80DeepseekV41Router`, whose
  :func:`route` is the reference ``Gate.forward``: ``sqrt(softplus(logits /
  gate_temp))``, the correction bias (``bias_vl`` for image tokens) used only
  to select the top-k, the weights gathered from the unbiased scores,
  divided by their sum ``+ 1e-20`` and multiplied by the routed scaling
  factor, all in FP32 torch ops (graph-safe, no host synchronisation).

The experts and the SwiGLU clamp run in the EXL3 MoE kernels
(vllm/sm80/kernels/exl3/quant.py, ``swiglu_limit``). With
``VLLM_DEEPSEEK_V41_MOE_PREFILL_TC`` the experts may take the tensor-core
prefill operator ``moe_out_prefill_v2`` for prefill calls (its gate decides
when the weights have loaded).
"""

import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.sm80.kernels.exl3.quant import set_moe_prefill_v2
from vllm.models.deepseek_v41.nvidia import model as upstream_model
from vllm.models.deepseek_v41.nvidia.sm80_attention import (
    large_round_once_all_reduce,
    round_once_transport,
)

logger = init_logger(__name__)

# Image-span tokens carry five consecutive sentinel ids from this one on
# (common/mm_preprocess.py), the ids the upstream routers test for bias_vl.
IMAGE_SENTINEL_IDS = 5
# GateLinear's specialised GEMM tiers; none may run an FP32 gate weight with
# BF16 input, so the adopted gate takes the FP32 fallback.
_GATE_TIERS = (
    "allow_ll_bf16_gemm",
    "allow_fp32_router_gemm",
    "allow_ll_fp32w_gemm",
    "allow_bf16x3_router_gemm",
    "allow_cublas_router_gemm",
    "allow_rocm_bf16x3_router_gemm",
)


def image_token_mask(input_ids: torch.Tensor | None, sentinel_lo: int):
    """Tokens inside an image span, or None for a text-only model."""
    if input_ids is None or sentinel_lo <= 0:
        return None
    return (input_ids >= sentinel_lo) & (input_ids < sentinel_lo + IMAGE_SENTINEL_IDS)


def route(
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    *,
    bias_vl: torch.Tensor | None = None,
    image_mask: torch.Tensor | None = None,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    gate_temp: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference ``Gate.forward`` after its GEMM, for FP32 ``logits``
    [tokens, experts]: FP32 weights and int64 expert ids [tokens, top_k]."""
    scores = logits if gate_temp == 1.0 else logits / gate_temp
    scores = F.softplus(scores).sqrt()
    if image_mask is not None and bias_vl is not None:
        bias = torch.where(image_mask.unsqueeze(-1), bias_vl, bias)
    indices = (scores + bias).topk(top_k, dim=-1)[1]
    weights = scores.gather(1, indices)
    if renormalize and top_k > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return weights, indices


class Sm80DeepseekV41Router(FusedTopKBiasRouter):
    """The upstream DSv4.1 router with the reference routing arithmetic."""

    gate_temp = 1.0

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if router_logits.dtype != torch.float32:
            raise ValueError("DeepSeek-V4.1 routing needs FP32 router logits")
        weights, ids = route(
            router_logits,
            self.e_score_correction_bias,
            self.top_k,
            bias_vl=self.bias_vl,
            image_mask=image_token_mask(input_ids, self.image_sentinel_lo),
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
            gate_temp=self.gate_temp,
        )
        return weights, ids.to(torch.int32 if indices_type is None else indices_type)


def _fp32_gate_weight(gate: nn.Module) -> None:
    """Hold the gate weight in FP32 so the checkpoint's FP16 values load
    exactly, and route the gate's forward to its FP32 fallback."""
    weight = gate.weight
    if weight.dtype != torch.float32:
        fp32 = nn.Parameter(
            torch.empty_like(weight, dtype=torch.float32), requires_grad=False
        )
        for key, value in vars(weight).items():
            setattr(fp32, key, value)
        gate.weight = fp32
    for tier in _GATE_TIERS:
        if getattr(gate, tier, False):
            setattr(gate, tier, False)
    if gate.out_dtype != torch.float32:
        raise ValueError("DeepSeek-V4.1 gates produce FP32 logits")


def adopt_moe(ffn: nn.Module, config) -> None:
    """Give an upstream-built DeepseekV4MoE the reference gate and router
    (before its weights load)."""
    if ffn.use_native_mega_moe:
        raise ValueError("DeepSeek-V4.1 on sm_80 runs the modular EXL3 MoE")
    if ffn.gate.tid2eid is not None or ffn.gate.e_score_correction_bias is None:
        raise ValueError("DeepSeek-V4.1 routes every layer with noaux_tc")
    if ffn.scoring_func != "sqrtsoftplus":
        raise ValueError(f"Unsupported DeepSeek-V4.1 scoring: {ffn.scoring_func}")
    _fp32_gate_weight(ffn.gate)
    router = ffn.experts.router
    if type(router) is not FusedTopKBiasRouter or router.num_fused_shared_experts:
        raise TypeError(f"Unexpected DeepSeek-V4.1 router: {type(router).__name__}")
    router.__class__ = Sm80DeepseekV41Router
    router.gate_temp = float(getattr(config, "gate_temp", 1.0))
    if envs.VLLM_DEEPSEEK_V41_MOE_PREFILL_TC:
        reason = set_moe_prefill_v2(ffn.experts.routed_experts)
        if reason is not None:
            logger.info_once(
                "VLLM_DEEPSEEK_V41_MOE_PREFILL_TC is set but the gate is closed: "
                "%s; these MoE layers keep their kernels in prefill steps.",
                reason,
            )
    if envs.VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE:
        _adopt_round_once_all_reduce(ffn)


def moe_all_reduce(out: torch.Tensor) -> torch.Tensor:
    """The MoE output's TP all-reduce: rounded once per element when the fast
    paths do not take ``out`` whole (``large_round_once_all_reduce`` of
    sm80_attention); otherwise the runner's own ``tensor_model_parallel_all_reduce``."""
    reduced = large_round_once_all_reduce(out)
    if reduced is not None:
        return reduced
    return tensor_model_parallel_all_reduce(out)


class Sm80DeepseekV41MoE(upstream_model.DeepseekV4MoE):
    """The upstream DSv4.1 MoE with its runner's final all-reduce moved here
    (``skip_final_all_reduce``) as :func:`moe_all_reduce`."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        mega_gate_metadata=None,
    ) -> torch.Tensor:
        out = super().forward(hidden_states, input_ids, mega_gate_metadata)
        return moe_all_reduce(out)


def _adopt_round_once_all_reduce(ffn: nn.Module) -> None:
    """VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE: the layer, not the runner,
    all-reduces the MoE output, where the runner would."""
    if type(ffn) is not upstream_model.DeepseekV4MoE:
        raise TypeError(f"Unexpected DeepSeek-V4.1 MoE: {type(ffn).__name__}")
    moe_config = ffn.experts.moe_config
    if moe_config.is_sequence_parallel or moe_config.skip_final_all_reduce:
        raise ValueError("the MoE output is not all-reduced by the runner")
    if moe_config.tp_size == 1 and moe_config.ep_size == 1:
        return
    moe_config.skip_final_all_reduce = True
    ffn.__class__ = Sm80DeepseekV41MoE
    logger.info_once(
        "DeepSeek-V4.1 sm_80: MoE outputs too large for the custom all-reduce "
        "whole are all-reduced with one rounding per element on eager steps, "
        "over %s (VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE=1; set 0 for the "
        "runner's all-reduce)",
        round_once_transport(),
        scope="global",
    )
