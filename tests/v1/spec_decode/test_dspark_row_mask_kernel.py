# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark verification row mask kernel (``dspark_row_mask_kernel``).

- The torch reference against hand-computed ``live_len`` at b = 1 and b = 8:
  all-ones confidences keep every draft, a near-zero first confidence keeps
  none, exact ties take the larger k.
- ``row_live``: request r's rows r*(K+1) .. r*(K+1)+K are live up to its
  ``live_len``; rows past the step that an earlier, wider step masked are
  live again; rows no step can mask are not written.
- ``begin_target_step``: a uniform speculative decode step computes the mask,
  any other step (prefill rows, mixed or zero drafts, dummy runs, GPU-side
  draft reallocation, a token count that is not num_reqs*(K+1)) leaves every
  row live; nothing happens without the state.
- The Triton kernel in the interpreter (a subprocess with
  ``TRITON_INTERPRET=1``) writes bitwise the reference's ``live_len`` and
  ``row_live`` on random confidences and cost tables.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sm80.spec_decode import dspark_row_mask
from vllm.sm80.spec_decode import dspark_row_mask_kernel as rmk
from vllm.sm80.spec_decode.dspark_row_mask import RowMaskState

pytestmark = pytest.mark.cpu_test

ROOT = Path(__file__).resolve().parents[3]
MAX_REQS = 8
MAX_DRAFT = 5
MAX_TOKENS = 64


def make_state(max_draft: int = MAX_DRAFT, max_tokens: int = MAX_TOKENS) -> RowMaskState:
    return RowMaskState(MAX_REQS, max_tokens, max_draft, torch.device("cpu"))


def set_conf(state: RowMaskState, slots: list[int], rows: list[list[float]]) -> torch.Tensor:
    for slot, row in zip(slots, rows):
        state.conf_by_slot[slot, : len(row)] = torch.tensor(row, dtype=torch.float32)
    return torch.tensor(slots, dtype=torch.int64)


# b = 1: cost_r(k) = T_1(k).
T1 = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
# b = 8, K = 5: cost_r(k) = 80 - (80 - T_8(k)) / 8 = 75 + k.
T8 = [40.0, 48.0, 56.0, 64.0, 72.0, 80.0]


def test_reference_b1_hand_computed():
    state = make_state()
    state.cost[1] = torch.tensor(T1)
    cases = [
        # E(k) = 1 + k, score (1 + k) / (10 + k) rises with k: keep all 5.
        ([1.0] * 5, 5),
        # E(k) ~ 1: 1/10 beats (1 + 1e-6)/11.
        ([1e-6, 1.0, 1.0, 1.0, 1.0], 0),
        # E = 1, 1.9, 2.35, 2.395, ...: scores 0.1, 0.1727, 0.1958, 0.1842: 2.
        ([0.9, 0.5, 0.1, 0.1, 0.1], 2),
    ]
    for conf, want in cases:
        idx = set_conf(state, [3], [conf])
        rmk.reference_row_mask(state, idx, 1, 5)
        assert int(state.live_len[0]) == want, (conf, want, state.live_len)
        assert state.active


def test_reference_b1_exact_ties_take_larger_k():
    state = make_state()
    # k=0: 1/1 = 1; k=1 with p_1 = 1: 2/2 = 1 (tie); k=2 with p_2 = 0: 2/3.
    state.cost[1] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    idx = set_conf(state, [0], [[1.0, 0.0, 0.0, 0.0, 0.0]])
    rmk.reference_row_mask(state, idx, 1, 5)
    assert int(state.live_len[0]) == 1
    # k=0: 1/2; k=1 with p_1 = 0.5: 1.5/3 = 1/2 (tie); k>=2: 1.5/4 and less.
    state.cost[1] = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    set_conf(state, [0], [[0.5, 0.0, 0.0, 0.0, 0.0]])
    rmk.reference_row_mask(state, idx, 1, 5)
    assert int(state.live_len[0]) == 1


def test_reference_b8_hand_computed():
    state = make_state()
    state.cost[8] = torch.tensor(T8)
    slots = [7, 2, 5, 0, 1, 6, 3, 4]
    rows = [
        [1.0] * 5,  # (1 + k)/(75 + k) rises: 5
        [1e-6, 1.0, 1.0, 1.0, 1.0],  # 1/75 beats 1.000001/76: 0
        [0.5, 1e-6, 1.0, 1.0, 1.0],  # 1/75=.01333, 1.5/76=.01974, 1.5000005/77: 1
        [1.0, 1.0, 1e-6, 1.0, 1.0],  # 3/77 = .03896 beats 3.000001/78: 2
        [1.0, 1.0, 1.0, 1e-6, 1.0],  # 4/78 beats 4.000001/79: 3
        [1.0, 1.0, 1.0, 1.0, 1e-6],  # 5/79 beats 5.000001/80: 4
        [0.02, 1.0, 1.0, 1.0, 1.0],  # see below: 5
        [0.01, 0.01, 0.01, 0.01, 0.01],  # 1/75 = .013333 > 1.01/76 = .013289: 0
    ]
    idx = set_conf(state, slots, rows)
    rmk.reference_row_mask(state, idx, 8, 5)
    # Row 6: E(k) = 1 + 0.02k, scores 1/75 = .013333, 1.02/76 = .013421,
    # 1.04/77 = .013506, 1.06/78 = .013590, 1.08/79 = .013671, 1.1/80 = .01375.
    assert state.live_len.tolist() == [5, 0, 1, 2, 3, 4, 5, 0]


def test_reference_b8_k3_all_ties_take_k():
    state = make_state()
    # K = 3: cost_r(k) = 9 - (9 - T_8(k)) / 8 = [8, 9, 9, 9].
    state.cost[8] = torch.tensor([1.0, 9.0, 9.0, 9.0, 9.0, 9.0])
    # E = [1, 1.125, 1.125, 1.125]: every score is exactly 0.125.
    idx = set_conf(state, list(range(8)), [[0.125, 0.0, 0.0, 1.0, 1.0]] * 8)
    rmk.reference_row_mask(state, idx, 8, 3)
    assert state.live_len.tolist() == [3] * 8


def test_row_live_layout_and_tail():
    state = make_state()
    state.cost[1] = torch.tensor(T1)
    state.cost[2] = torch.tensor(T1)
    span = MAX_REQS * (MAX_DRAFT + 1)
    # A sentinel past the maskable span: no step may write it.
    state.row_live[span:] = 7
    # A wide step masks rows across 0 .. 11.
    idx = set_conf(state, [4, 1], [[1.0] * 5, [1e-6] + [1.0] * 4])
    rmk.reference_row_mask(state, idx, 2, 5)
    assert state.live_len[:2].tolist() == [5, 0]
    assert state.row_live[:12].tolist() == [1] * 6 + [1, 0, 0, 0, 0, 0]
    # A narrower step (1 request, K = 3) rewrites rows 0 .. 3; rows 4 .. 11
    # of the earlier step are live again.
    idx = set_conf(state, [2], [[0.9, 0.5, 0.1]])
    rmk.reference_row_mask(state, idx, 1, 3)
    # cost_r = T_1 = 10, 11, 12, 13: E = 1, 1.9, 2.35, 2.395: keep 2.
    assert int(state.live_len[0]) == 2
    assert state.row_live[:4].tolist() == [1, 1, 1, 0]
    assert state.row_live[4:span].tolist() == [1] * (span - 4)
    assert state.row_live[span:].tolist() == [7] * (MAX_TOKENS - span)


def step(num_drafts, *, has_prefill=False, num_tokens=None, max_query_len=None, slots=None):
    n = len(num_drafts)
    slots = list(range(n)) if slots is None else slots
    drafts = None if num_drafts is None else np.asarray(num_drafts, dtype=np.int32)
    if num_tokens is None:
        num_tokens = int(sum(k + 1 for k in num_drafts))
    return SimpleNamespace(
        num_reqs=n,
        idx_mapping=torch.tensor(slots, dtype=torch.int64),
        num_tokens=num_tokens,
        num_draft_tokens_per_req=drafts,
        has_prefill=has_prefill,
        max_query_len=max_query_len,
    )


def masked(state: RowMaskState) -> None:
    state.cost[2] = torch.tensor(T1)
    set_conf(state, [0, 1], [[1e-6] + [1.0] * 4, [1e-6] + [1.0] * 4])
    rmk.begin_step(state, step([5, 5]), dummy_run=False)
    assert state.active
    assert state.live_len[:2].tolist() == [0, 0]
    assert state.row_live[:12].tolist() == [1, 0, 0, 0, 0, 0] * 2


@pytest.mark.parametrize(
    "batch, dummy",
    [
        (dict(num_drafts=[5, 5], has_prefill=True), False),
        (dict(num_drafts=[5, 3]), False),
        (dict(num_drafts=[0, 0]), False),
        (dict(num_drafts=[5, 5]), True),
        (dict(num_drafts=[5, 5], max_query_len=6), False),
        (dict(num_drafts=[5, 5], num_tokens=11), False),
        (dict(num_drafts=[6, 6]), False),
    ],
    ids=["prefill", "mixed", "zero", "dummy", "reallocated", "tokens", "too-wide"],
)
def test_non_uniform_steps_reset(batch, dummy):
    state = make_state()
    masked(state)
    rmk.begin_step(state, step(**batch), dummy_run=dummy)
    assert not state.active
    assert bool((state.row_live == 1).all())


def test_no_drafts_scheduled_resets():
    state = make_state()
    masked(state)
    batch = step([0, 0])
    batch.num_draft_tokens_per_req = None
    rmk.begin_step(state, batch, dummy_run=False)
    assert not state.active
    assert bool((state.row_live == 1).all())


def test_uniform_step_masks_in_batch_order():
    state = make_state()
    state.cost[3] = torch.tensor(T1)
    # Slots 6, 0, 3 in batch order; only slot 0's request drops its drafts.
    set_conf(state, [6, 0, 3], [[1.0] * 3, [1e-6, 1.0, 1.0], [1.0] * 3])
    rmk.begin_step(state, step([3, 3, 3], slots=[6, 0, 3]), dummy_run=False)
    assert state.active
    assert state.live_len[:3].tolist() == [3, 0, 3]
    assert state.row_live[:12].tolist() == [1] * 4 + [1, 0, 0, 0] + [1] * 4


def test_runner_entry_is_a_no_op_without_state(monkeypatch):
    monkeypatch.setattr(dspark_row_mask, "_STATE", None)
    # A batch whose fields would fail if read: nothing is read.
    rmk.begin_target_step(object(), False)


def test_runner_entry_uses_the_shared_state(monkeypatch):
    state = make_state()
    monkeypatch.setattr(dspark_row_mask, "_STATE", state)
    state.cost[2] = torch.tensor(T1)
    set_conf(state, [0, 1], [[1e-6] + [1.0] * 4, [1.0] * 5])
    rmk.begin_target_step(step([5, 5]), False)
    assert state.live_len[:2].tolist() == [0, 5]
    rmk.begin_target_step(step([5, 5], has_prefill=True), False)
    assert not state.active and bool((state.row_live == 1).all())


def interpreted_results() -> dict:
    """Run in the Triton interpreter: per random case, the kernel's buffers
    next to the reference's."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    from vllm.sm80.spec_decode import dspark_row_mask_kernel as kernel

    gen = torch.Generator().manual_seed(0)
    out = []
    for max_draft in (5, 8):
        for b in range(1, MAX_REQS + 1):
            for k in sorted({1, 3, max_draft}):
                for trial in range(3):
                    states = [RowMaskState(MAX_REQS, 96, max_draft, torch.device("cpu"))
                              for _ in range(2)]
                    conf = torch.rand(MAX_REQS, max_draft, generator=gen)
                    if trial == 1:
                        conf = conf.sqrt().sqrt()  # near-one confidences
                    if trial == 2:
                        conf[: b // 2] = 1.0
                    steps = torch.rand(MAX_REQS + 1, max_draft + 1, generator=gen) + 0.05
                    cost = 20.0 + steps.cumsum(1)
                    cost[0] = 0.0
                    if trial == 2:
                        cost[:, 1:] = cost[:, :1] + torch.arange(1, max_draft + 1)
                    idx = torch.randperm(MAX_REQS, generator=gen)[:b]
                    if trial == 0:
                        conf[idx[0], 0] = 1e-4  # drops every draft
                    for s in states:
                        s.conf_by_slot.copy_(conf)
                        s.cost.copy_(cost)
                        s.row_live[: 6 * 9].copy_(torch.randint(0, 2, (54,), generator=gen))
                        s.live_len.fill_(-3)
                    states[1].row_live.copy_(states[0].row_live)
                    kernel.compute_row_mask(states[0], idx, b, k)
                    kernel.reference_row_mask(states[1], idx, b, k)
                    out.append({
                        "case": [max_draft, b, k, trial],
                        "kernel": [states[0].live_len.tolist(), states[0].row_live.tolist()],
                        "reference": [states[1].live_len.tolist(), states[1].row_live.tolist()],
                        "active": states[0].active,
                    })
    return {"cases": out}


@pytest.fixture(scope="module")
def interpreted():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = ("import json; from tests.v1.spec_decode.test_dspark_row_mask_kernel import "
            "interpreted_results as f; print('RESULT ' + json.dumps(f()))")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_interpreted_kernel_matches_reference(interpreted):
    cases = interpreted["cases"]
    assert len(cases) == 2 * MAX_REQS * 3 * 3
    for case in cases:
        assert case["kernel"] == case["reference"], case["case"]
        assert case["active"], case["case"]
    # The cases cover masked and unmasked requests.
    lens = [n for c in cases for n in c["kernel"][0][: c["case"][1]]]
    assert min(lens) == 0 and max(lens) == 8
