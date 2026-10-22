# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The EXL3 MoE switches VLLM_EXL3_MOE_V2_RING and
VLLM_EXL3_MOE_PREFILL_UNCHUNKED (vllm/sm80/envs.py,
vllm/sm80/kernels/exl3/ops.py) on CPU: declared, off by default, read
through vllm.envs, and open only for the configuration they cover.

    pytest -q tests/kernels/quantization/test_exl3_moe_v2_switches.py
"""

import pytest

import vllm.envs as envs
from vllm.sm80.kernels.exl3 import ops as exl3_ops

SWITCHES = ("VLLM_EXL3_MOE_V2_RING", "VLLM_EXL3_MOE_PREFILL_UNCHUNKED")


@pytest.mark.parametrize("name", SWITCHES)
def test_switch_declared_and_off_by_default(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is False
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name) is True
    monkeypatch.setenv(name, "0")
    assert getattr(envs, name) is False


def test_ring_opens_only_with_its_switch_for_its_configuration(monkeypatch):
    config = exl3_ops.MOE_V2_TRY_CONFIG
    monkeypatch.delenv("VLLM_EXL3_MOE_V2_RING", raising=False)
    assert not any(exl3_ops.moe_v2_try_open(*config, b) for b in (6, 4))
    monkeypatch.setenv("VLLM_EXL3_MOE_V2_RING", "1")
    assert all(exl3_ops.moe_v2_try_open(*config, b) for b in (6, 4))
    assert not exl3_ops.moe_v2_try_open(*config, 8)
    for other in (
        (2560, 640, 10, 256, 512),  # the hidden-2560 layers
        (5120, 640, 6, 384, 384),  # the tensor-parallel chunks
        (5120, 2304, 3, 32, 128),  # the DSpark drafter's experts
    ):
        assert not exl3_ops.moe_v2_try_open(*other, 6), other


def test_unchunked_prefill_is_read_through_envs_once(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_MOE_PREFILL_UNCHUNKED", raising=False)
    assert exl3_ops.TRY_MOE_PREFILL is False
    # Read once at import: setting it later changes nothing in this process.
    monkeypatch.setenv("VLLM_EXL3_MOE_PREFILL_UNCHUNKED", "1")
    assert exl3_ops.TRY_MOE_PREFILL is False
    assert not exl3_ops._try_moe_prefill(5120, 2304, 6, 96)


def test_unchunked_prefill_covers_the_whole_expert_shape_only(monkeypatch):
    monkeypatch.setattr(exl3_ops, "TRY_MOE_PREFILL", True)
    assert exl3_ops._try_moe_prefill(5120, 2304, 6, 96)
    for other in ((5120, 640, 6, 384), (5120, 2304, 3, 32), (2560, 640, 10, 256)):
        assert not exl3_ops._try_moe_prefill(*other), other
    assert exl3_ops.TRY_MOE_PREFILL_ROWS == (65, 2048)
