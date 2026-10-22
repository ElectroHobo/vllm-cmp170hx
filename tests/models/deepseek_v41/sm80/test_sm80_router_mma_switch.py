# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DEEPSEEK_V41_ROUTER_MMA (vllm/sm80/envs.py, read in
vllm/models/deepseek_v41/nvidia/sm80_moe.py) on CPU: declared, off by
default, read through vllm.envs, and adopt_moe hands the layer to
_adopt_router_mma only when it is set.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_router_mma_switch.py
"""

from types import SimpleNamespace

import torch

SWITCH = "VLLM_DEEPSEEK_V41_ROUTER_MMA"


def _ffn(monkeypatch):
    from tests.models.deepseek_v41.sm80 import construct
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        FusedTopKBiasRouter,
    )
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    construct.set_groups(monkeypatch, 1, 0)
    gate = GateLinear(64, 8, bias=False, out_dtype=torch.float32,
                      prefix="layers.0.ffn.gate")
    gate.e_score_correction_bias = torch.zeros(8)
    gate.tid2eid = None
    router = object.__new__(FusedTopKBiasRouter)
    router.num_fused_shared_experts = 0
    return SimpleNamespace(use_native_mega_moe=False, gate=gate,
                           scoring_func="sqrtsoftplus",
                           experts=SimpleNamespace(router=router))


def test_switch_declared_and_off_by_default(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv(SWITCH, raising=False)
    assert envs.VLLM_DEEPSEEK_V41_ROUTER_MMA is False
    monkeypatch.setenv(SWITCH, "1")
    assert envs.VLLM_DEEPSEEK_V41_ROUTER_MMA is True
    monkeypatch.setenv(SWITCH, "0")
    assert envs.VLLM_DEEPSEEK_V41_ROUTER_MMA is False


def test_adoption_takes_the_router_only_with_the_switch(monkeypatch):
    from vllm.models.deepseek_v41.nvidia import sm80_moe

    for switch in (None, "0", "1"):
        if switch is None:
            monkeypatch.delenv(SWITCH, raising=False)
        else:
            monkeypatch.setenv(SWITCH, switch)
        adopted = []
        monkeypatch.setattr(
            sm80_moe, "_adopt_router_mma",
            lambda ffn, router: adopted.append((ffn, router)),
        )  # fmt: skip
        ffn = _ffn(monkeypatch)
        sm80_moe.adopt_moe(ffn, SimpleNamespace())
        if switch == "1":
            assert adopted == [(ffn, ffn.experts.router)]
        else:
            assert adopted == []
        assert type(ffn.experts.router) is sm80_moe.Sm80DeepseekV41Router
        assert ffn.experts.router.mma_gate is None
        assert ffn.gate.mma is False
