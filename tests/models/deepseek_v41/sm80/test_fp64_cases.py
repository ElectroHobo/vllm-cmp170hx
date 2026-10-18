# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp64_cases.py on synthetic sources in the captured format.

CPU: every family except the EXL3 MoE runs end to end through
``fp64_cases.run`` (on CPU the candidate is the torch path): one well-formed
line per case, the exact families exact against their independent
definitions, wrong or row-dependent candidates reported, a case that cannot
run left out and named. The EXL3 MoE's checkpoint reading and FP64
reference (with the SwiGLU clamp) run on the tiny checkpoint against a
direct sum. GPU (skips without CUDA): the same cases through the CUDA paths.

    pytest -q tests/models/deepseek_v41/sm80/test_fp64_cases.py
"""

import json
import math

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core

from . import exl3_fixture as fixture
from . import fp64_cases as F
from .test_sm80_kernels_gpu import _attn_case, _index_case

BF16 = torch.bfloat16
CPU = torch.device("cpu")
KEYS = {"cand_mean", "cand_max", "ref_mean", "ref_max"}
EXACT = ("qdq_fp8", "qdq_fp4_e8m0", "qdq_fp4_e4m3", "engram")


def _source(directory, name, family, tensors):
    path = directory / f"{name}.pt"
    meta = {"step": 0, "num_reqs": 1, "layer": 0, "rank": 0, "phase": "decode"}
    torch.save({"family": family, "meta": meta, "tensors": tensors}, path)
    return str(path)


def _attention(g, compressed=True):
    q, swa, swa_idx, comp, table, local, req, sink = _attn_case(g)
    t = dict(q=q, swa_cache=swa, swa_indices=swa_idx, comp_cache=comp,
             comp_block_table=table, comp_local=local, req=req, sink=sink,
             scale=512**-0.5)
    if not compressed:
        t.update(comp_cache=None, comp_block_table=None, comp_local=None)
    return t


def _index(g):
    q, w, n_vis, req, idxk, table = _index_case(g)
    return dict(q=q, w=w, n_vis=n_vis, req=req, idxk=idxk, block_table=table,
                n_cols=int(n_vis.max()), heads_per_rank=8)


def _activations(g, d=512):
    x = torch.randn(13, 4, d, generator=g) * torch.logspace(-6, 3, 13)[:, None, None]
    x[0, 0] = 0.0
    x[1, 1, :4] = torch.tensor([448.0, -448.0, 6.0, 1e-30])
    return x.to(BF16)


def _compress(g, ratio):
    """Two requests: one whose groups close inside its chunk, one whose
    first token closes a group begun in the previous chunk (ring)."""
    d, capacity = 64, 4
    pos = torch.tensor([4, 5, 6, 7, 8, 9, 7, 8, 9, 10, 11])
    req = torch.tensor([0] * 6 + [1] * 5)
    t = dict(
        kv_score=torch.randn(pos.shape[0], ratio * d, generator=g),
        norm_weight=(1 + 0.1 * torch.randn(d, generator=g)).to(BF16),
        positions=pos,
        token_to_req=req,
        query_start_loc=torch.tensor([0, 6, 11]),
        comp_slots=torch.arange(pos.shape[0]),
        compress_ratio=ratio,
        head_dim=d,
        eps=1e-6,
    )
    if ratio == 1:
        t["comp_slots"] = torch.where(pos % 3 == 0, t["comp_slots"], -1)
    else:
        blocks = torch.tensor([2, 0])
        t["ring"] = torch.randn(3, capacity, 2 * d, generator=g)
        t["ring_slots"] = blocks[req] * capacity + pos % capacity
    return t


def _index_keys(g, n=9, d=128):
    return dict(
        wk_out=torch.randn(n, d, generator=g).to(BF16),
        k_norm=(1 + 0.1 * torch.randn(d, generator=g)).to(BF16),
        freqs=core.rope_freqs(64, 0, 10000.0, 1.0, 32, 1),
        group_pos=torch.randint(0, 5000, (n,), generator=g),
        eps=1e-6,
    )


def _engram(g, n=21, dim=256):
    values = torch.randint(0, 256, (n, dim), generator=g).to(torch.uint8)
    values[0, :2] = torch.tensor([0x7F, 0xFF], dtype=torch.uint8)  # NaN patterns
    scales = torch.randint(110, 140, (n, dim // 32), generator=g).to(torch.uint8)
    scales[0, 1] = 0  # 2**-127: products in the FP32 denormals
    return dict(values=values, scales=scales)


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    d = tmp_path_factory.mktemp("sources")
    g = torch.Generator().manual_seed(7)
    sources = {
        "attention": [
            _source(d, "attn0", "sparse_attention", _attention(g)),
            _source(d, "attn1", "sparse_attention", _attention(g, compressed=False)),
        ],
        "index_scores": [_source(d, "index", "index_scores", _index(g))],
        "qdq_fp8": [_source(d, "fp8", "qdq_fp8", dict(x=_activations(g), block=32))],
        "qdq_fp4_e8m0": [
            _source(d, "fp4a", "qdq_fp4", dict(x=_activations(g), block=32, scale="e8m0"))
        ],
        "qdq_fp4_e4m3": [
            _source(d, "fp4b", "qdq_fp4", dict(x=_activations(g), block=16, scale="e4m3"))
        ],
        "compress": [
            _source(d, "comp1", "compress", _compress(g, 1)),
            _source(d, "comp2", "compress", _compress(g, 2)),
        ],
        "index_keys": [_source(d, "keys", "index_keys", _index_keys(g))],
        "engram": [_source(d, "engram", "engram_dequant", _engram(g))],
    }
    return [
        {"id": k, "concurrency": 1, "phase": "decode", "captured_sources": v}
        for k, v in sources.items()
    ]


def _run(tmp_path, cases, device=CPU, schema="served_cases/1"):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"schema": schema, "model": "m", "family": "f",
                                "cases": cases}))
    out = tmp_path / "out.jsonl"
    rc = F.run(path, out, device)
    if rc:
        return rc, None, None
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert set(lines[0]) == {"header"}
    return rc, lines[0]["header"], {r["case"]: r for r in lines[1:]}


def _well_formed(row):
    assert isinstance(row["bitwise"], bool) and isinstance(row["row_invariance"], bool)
    assert isinstance(row["numel"], int) and row["numel"] > 0
    assert set(row["fp64"]) == KEYS
    for v in row["fp64"].values():
        assert isinstance(v, float) and math.isfinite(v) and v >= 0


def test_every_family_runs_on_cpu(tmp_path, cases):
    rc, head, rows = _run(tmp_path, cases)
    assert rc == 0
    assert set(head) == {"torch", "triton", "cuda", "gpu", "sm_count"}
    assert set(rows) == {c["id"] for c in cases}
    for cid, row in rows.items():
        _well_formed(row)
        assert row["row_invariance"], cid
        e = row["fp64"]
        if cid == "attention":
            # The torch definition against the reference kernel's arithmetic:
            # both within BF16 rounding of the FP64 value.
            assert 0 < e["cand_max"] < 0.05 and 0 < e["ref_max"] < 0.05
            continue
        assert row["bitwise"], cid  # on CPU both paths are the torch definition
        assert (e["cand_mean"], e["cand_max"]) == (e["ref_mean"], e["ref_max"]), cid
        if cid.startswith("qdq"):
            # Exact: the definition equals the independent one everywhere.
            assert e["cand_max"] == 0.0, cid
        if cid in ("index_scores", "compress"):
            assert e["cand_max"] > 0, cid  # their BF16 rounding is measured
    assert rows["compress"]["family"] == "compress"


def test_a_wrong_candidate_shows_in_its_bits_and_error(tmp_path, cases, monkeypatch):
    real = core.qdq_fp8

    def off(x, block=32):
        y = real(x, block).clone()
        y.view(-1)[5] = y.view(-1)[5] * 2 + 1
        return y

    monkeypatch.setattr(core, "qdq_fp8", off)
    _, _, rows = _run(tmp_path, [c for c in cases if c["id"] == "qdq_fp8"])
    row = rows["qdq_fp8"]
    assert not row["bitwise"]
    assert row["fp64"]["cand_max"] > 0 == row["fp64"]["ref_max"]


def test_a_row_dependent_candidate_is_reported(tmp_path, cases, monkeypatch):
    real = core.sparse_attention

    def batch_dependent(q, *args):
        return real(q, *args) + (q.shape[0] > 1)

    monkeypatch.setattr(core, "sparse_attention", batch_dependent)
    _, _, rows = _run(tmp_path, [c for c in cases if c["id"] == "attention"])
    assert not rows["attention"]["row_invariance"]


def test_a_case_that_cannot_run_is_left_out_and_named(tmp_path, cases, capsys):
    bad = tmp_path / "bad.pt"
    torch.save({"family": "unknown", "meta": {}, "tensors": {}}, bad)
    good = [c for c in cases if c["id"] == "engram"]
    extra = [
        {"id": "no_family", "concurrency": 1, "phase": "decode",
         "captured_sources": [str(bad)]},
        {"id": "no_sources", "concurrency": 2, "phase": "prefill"},
    ]
    rc, _, rows = _run(tmp_path, good + extra)
    assert rc == 0 and set(rows) == {"engram"}
    err = capsys.readouterr().err
    assert "case no_family: ValueError" in err and "'unknown'" in err
    assert "case no_sources: ValueError: no captured_sources" in err


def test_a_file_of_another_schema_is_refused(tmp_path, cases):
    assert _run(tmp_path, cases, schema="capml/1")[0] == 2


def test_abs_errors():
    exact = torch.tensor([1.5, float("nan"), float("inf")], dtype=torch.float64)
    assert F.abs_errors(torch.tensor([1.0, float("nan"), float("inf")]), exact) == (
        0.5, 0.5, 1,
    )
    assert F.abs_errors(torch.tensor([1.5, 0.0, float("inf")]), exact)[1] == math.inf
    assert F.abs_errors(torch.tensor([float("inf")]), torch.tensor([1.0]))[1] == math.inf
    with pytest.raises(ValueError, match="shape"):
        F.abs_errors(torch.zeros(2), torch.zeros(3))


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU refusal")
def test_main_refuses_without_cuda(tmp_path, capsys):
    assert F.main(["--cases", str(tmp_path / "c"), "--out", str(tmp_path / "o")]) == 2
    assert "no CUDA device" in capsys.readouterr().err


# ---------------------------------------------------------------- EXL3 MoE


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny")
    fixture.write_checkpoint(directory, fixture.tiny_config())
    return directory


def test_moe_experts_from_the_checkpoint_and_their_fp64_output(checkpoint):
    from tests.kernels.quantization import exl3_real_cases as real
    from tests.kernels.quantization.test_exl3_kernels import effective_weight

    ckpt = real.Checkpoint(checkpoint)
    local = (3, 7)
    names = [[f"layers.0.ffn.experts.{e}.{w}" for w in ("w1", "w3", "w2")] for e in local]
    meta = torch.tensor(
        [[[m.padded_in, m.padded_out, m.bits_x2, m.codebook]
          for m in map(ckpt.matrix, row)] for row in names]
    )
    weights, pointers, got = F.expert_set(str(checkpoint), names, meta, CPU)
    assert torch.equal(got, meta) and pointers.shape == (2, 3, 3)
    with pytest.raises(ValueError, match="layout"):
        F.expert_set(str(checkpoint), names, meta + 1, CPU)

    g = torch.Generator().manual_seed(3)
    hidden, inter = fixture.TEXT["hidden_size"], fixture.TEXT["moe_intermediate_size"]
    expert_map = torch.full((fixture.TEXT["n_routed_experts"],), -1, dtype=torch.int32)
    expert_map[3], expert_map[7] = 0, 1
    x = torch.randn(4, hidden, generator=g).to(BF16)
    ids = torch.tensor([[3, 7, -1, 5, 0, 9], [7, 1, 2, 4, 6, 8],
                        [10, 11, 12, 13, 14, 15], [3, 3, 7, 16, -1, 2]], dtype=torch.int32)
    tw = torch.rand(4, 6, generator=g)
    dense = [
        [effective_weight(*(p.numpy() for p in weights[3 * e + j]), int(m[2]), int(m[3]))
         for j, m in enumerate(meta[e].tolist())]
        for e in range(2)
    ]
    plain = F.moe_exact(x, ids, tw, expert_map, weights, meta, inter, 0.0)
    gate_values = np.concatenate(
        [x.double().numpy() @ dense[e][0][:, :inter] for e in range(2)]
    )
    limit = float(np.median(np.abs(gate_values)))
    clamped = F.moe_exact(x, ids, tw, expert_map, weights, meta, inter, limit)
    assert not torch.equal(plain, clamped)
    x64 = x.double().numpy()
    for t in range(x.shape[0]):
        want = np.zeros(hidden)
        for gid, w in zip(ids[t].tolist(), tw[t].double().tolist()):
            if not 0 <= gid < len(expert_map) or expert_map[gid] < 0:
                continue
            gate, up, down = dense[int(expert_map[gid])]
            a = np.minimum(x64[t] @ gate[:, :inter], limit)
            b = np.clip(x64[t] @ up[:, :inter], -limit, limit)
            want += w * ((a / (1.0 + np.exp(-a)) * b) @ down[:inter, :hidden])
        np.testing.assert_allclose(clamped[t].numpy(), want, rtol=1e-11, atol=1e-13)


# ---------------------------------------------------------------- GPU


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_paths_on_the_synthetic_cases(tmp_path, cases):
    rc, head, rows = _run(tmp_path, cases, torch.device("cuda"))
    assert rc == 0 and head["sm_count"] > 0
    assert set(rows) == {c["id"] for c in cases}
    for cid, row in rows.items():
        _well_formed(row)
        assert row["row_invariance"], cid
        e = row["fp64"]
        if cid in EXACT or cid == "index_scores":
            assert row["bitwise"], cid
            assert e["cand_mean"] == e["ref_mean"] and e["cand_max"] == e["ref_max"], cid
        if cid in EXACT[:3]:
            assert e["cand_max"] == 0.0, cid
