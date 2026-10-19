# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel call capture (vllm/sm80/capture.py, VLLM_SM80_CAPTURE_DIR), CPU
only: off by default; eager prefill calls are kept by prefix and ordinal;
decode calls by prefix up to the per-key count; records made while a CUDA
graph is captured are bound to that graph and written after its replays,
with the replaying manager as owner; dummy steps record nothing."""

import json
from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from vllm.sm80 import capture as cap
from vllm.sm80.kernels.exl3 import capture_points

PREFIX = "layers.3.attn.wq_a"


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_SM80_CAPTURE_DIR", raising=False)
    import importlib

    assert importlib.reload(cap).active is False


@pytest.fixture
def on(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_SM80_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SM80_CAPTURE_MATCH", r"layers\.3\.")
    monkeypatch.setenv("VLLM_SM80_CAPTURE_ROWS", "6")
    monkeypatch.setenv("VLLM_SM80_CAPTURE_PER_KEY", "1")
    monkeypatch.setenv("VLLM_SM80_CAPTURE_PREFILL_CALLS", "2")
    monkeypatch.setenv("VLLM_SM80_CAPTURE_PREFILL_MIN_ROWS", "65")
    monkeypatch.setattr(cap, "active", True)
    monkeypatch.setattr(cap, "_cfg", None)
    monkeypatch.setattr(cap, "_install", lambda: None)
    monkeypatch.setattr(cap, "_capturing", lambda: False)
    for name, value in (("_pending", []), ("_counts", {}), ("_prefill_calls", {}),
                        ("_bytes", [0]), ("_weights", {})):
        monkeypatch.setattr(cap, name, value)
    monkeypatch.setattr(
        cap, "_step", {"real": False, "step": 0, "phase": None, "query_lens": []}
    )
    return tmp_path / "r0"


def _index(root):
    path = root / "index.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _step(lens, dummy=False, context=()):
    output = SimpleNamespace(
        num_scheduled_tokens=dict(enumerate(lens)),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(num_computed_tokens=list(context)),
    )
    cap.set_step(output, 6, dummy)


def test_prefill_ordinals_and_dummy_steps(on):
    x = torch.randn(80, 4, dtype=torch.bfloat16)
    _step([80], dummy=True)
    assert cap.begin("dense", "exl3_linear", PREFIX, 80) is None
    _step([2048])
    assert cap._step["phase"] == "prefill"
    assert cap.begin("dense", "exl3_linear", PREFIX, 80) is None  # call 1
    rec = cap.begin("dense", "exl3_linear", PREFIX, 80, bits_x2=6)  # call 2
    rec.put(x=x, out=x[:, :2]).done()
    assert cap.begin("dense", "exl3_linear", PREFIX, 64) is None  # too few rows
    assert cap.begin("dense", "exl3_linear", "layers.4.attn.wq_a", 80) is None
    (line,) = _index(on)
    assert line["saved"] and line["phase"] == "prefill" and line["owner"] == "eager"
    obj = torch.load(line["path"], weights_only=True)
    assert obj["schema"] == cap.SCHEMA and obj["family"] == "exl3_linear"
    assert obj["meta"]["prefill_call"] == 2 and obj["meta"]["bits_x2"] == 6
    assert obj["meta"]["query_lens"] == [2048]
    assert torch.equal(obj["tensors"]["x"], x)


def test_decode_per_key(on):
    _step([1, 1])
    assert cap._step["phase"] == "decode"
    for _ in range(3):
        rec = cap.begin("moe", "exl3_moe_out_v2", "model.layers.3.ffn.experts", 2)
        if rec is not None:
            rec.put(x=torch.ones(2, 3)).done()
    assert len(_index(on)) == 1
    _step([1, 1], context=[cap.LONG_CONTEXT, 10])  # a long-context key of its own
    rec = cap.begin("moe", "exl3_moe_out_v2", "model.layers.3.ffn.experts", 2)
    rec.put(x=torch.ones(2, 3)).done()
    assert [x["context"] for x in _index(on)] == ["short", "long"]
    assert cap.begin("moe", "exl3_moe_out_v2", "model.layers.3.ffn.experts", 2) is None
    _step([6])
    assert cap._step["phase"] == "verify"
    assert cap.begin("moe", "exl3_moe_out_v2", "model.layers.3.ffn.experts", 6)


class Graph:
    pass


def _graph_record(manager, desc, monkeypatch, prefix="model.layers.3.attn"):
    """A record captured with ``manager``'s graph ``desc``, stored with it."""
    cap._bind(manager)
    monkeypatch.setattr(cap, "_capturing", lambda: True)
    rec = cap.begin("attn", "dsv41_attn_layer", prefix, desc.num_tokens)
    rec.put(q=torch.zeros(desc.num_tokens, 2))
    rec.done()
    manager.graphs[desc] = Graph()
    monkeypatch.setattr(cap, "_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda: SimpleNamespace(synchronize=lambda: None))
    return rec


Desc = namedtuple("Desc", "num_tokens num_reqs uniform_token_count max_query_len cg_mode")


def test_graph_records_follow_their_graph(on, monkeypatch):
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    class TargetManager(ModelCudaGraphManager):
        def __init__(self):
            self.graphs = {}

    manager = TargetManager()
    desc = Desc(6, 1, 6, 6, "FULL")
    monkeypatch.setattr(cap, "_capturing", lambda: True)
    assert cap.begin("attn", "dsv41_attn_layer", "model.layers.3.attn", 8) is None
    rec = _graph_record(manager, desc, monkeypatch)
    assert cap._pending == []
    cap._drain(manager, desc)  # no real step yet: nothing written
    assert _index(on) == []
    _step([6])
    rec.tensors["q"].fill_(7)  # what a replay leaves in the graph's copy
    cap._drain(manager, desc)
    cap._drain(manager, desc)  # per key 1
    (line,) = _index(on)
    assert line["owner"] == "TargetManager" and line["phase"] == "verify"
    assert line["step_phase"] == "verify"
    obj = torch.load(line["path"], weights_only=True)
    assert obj["meta"]["graph"]["num_tokens"] == 6
    assert torch.equal(obj["tensors"]["q"], torch.full((6, 2), 7.0))


def test_drafter_graph_in_prefill_step_is_draft_and_bounded(on, monkeypatch):
    """A speculator's graph replayed in a prefill step: phase draft (the
    step's phase in step_phase), per-key bound applied."""

    class DrafterManager:
        def __init__(self):
            self.graphs = {}

    manager = DrafterManager()
    desc = Desc(6, 2, 3, None, "FULL")  # rows in VLLM_SM80_CAPTURE_ROWS
    _graph_record(manager, desc, monkeypatch, prefix="model.layers.3.attn")
    _step([2048])
    for _ in range(3):
        cap._drain(manager, desc)
    (line,) = _index(on)
    assert (line["phase"], line["step_phase"]) == ("draft", "prefill")
    obj = torch.load(line["path"], weights_only=True)
    assert (obj["meta"]["phase"], obj["meta"]["step_phase"]) == ("draft", "prefill")


def test_moe_point_names_operator(on):
    import functools

    def moe_out_v2(*a, **k):
        pass

    pointers = torch.zeros(2, 3, 3, dtype=torch.int64)
    capture_points.register_moe(pointers, "model.layers.3.ffn.experts")
    _step([1])
    out = torch.zeros(1, 4)
    capture_points.moe(
        functools.partial(moe_out_v2, bits_x2=6), torch.ones(1, 4),
        torch.zeros(1, 6, dtype=torch.int32), torch.ones(1, 6), pointers,
        torch.zeros(2, 3, 4, dtype=torch.int64), torch.arange(4, dtype=torch.int32),
        2304, 10.0, out,
    )
    (line,) = _index(on)
    obj = torch.load(line["path"], weights_only=True)
    assert obj["family"] == "exl3_moe_out_v2"
    assert obj["tensors"]["bits_x2"] == 6 and obj["tensors"]["swiglu_limit"] == 10.0
    assert obj["meta"]["module_prefix"] == "model.layers.3.ffn.experts"
