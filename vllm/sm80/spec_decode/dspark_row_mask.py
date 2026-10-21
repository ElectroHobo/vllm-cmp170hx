# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-round DSpark verification row mask: shared state and cost table.

With ``VLLM_DSPARK_ROW_MASK=<cost table JSON>`` each verify step keeps, per
request, only the first ``live_len`` drafts its DSpark confidence head makes
worth verifying, and the rows past them (rows ``live_len + 1 .. K`` of the
request's ``K + 1`` verify rows) take no routed expert in the target's MoE
layers. The rejection sampler treats those drafts as rejected, so a request
emits at most ``live_len`` drafts plus the target's token from row
``live_len``: rows ``0 .. live_len`` are computed exactly as without the mask,
and greedy output is the unmasked output truncated at ``live_len``.

Pieces (each in its own module, all reading the one ``RowMaskState``):

* the draft step writes the confidence head's per-position outputs, keyed by
  request-state slot, into ``conf_by_slot``;
* before the target forward of a uniform speculative decode step, a kernel
  turns ``conf_by_slot`` and ``cost`` into ``live_len`` (batch order) and
  ``row_live`` (token order); any other step leaves every row live;
* the target MoE routing gives rows with ``row_live == 0`` the expert id -1,
  which the EXL3 MoE kernels skip on every rank;
* the rejection sampler rejects the drafts past ``live_len``.

The cost table (schema ``dspark-row-mask-costs/1``) gives, per batch size b,
``T_b(k)``: the step time in ms when every request of a b-request uniform
decode batch verifies k drafts, k = 0 .. max_draft::

    {"schema": "dspark-row-mask-costs/1", "max_draft": 5,
     "by_batch": {"1": [T(0), ..., T(5)], "8": [...]}, "source": "..."}

Batch sizes not listed are interpolated linearly between the nearest listed
ones (the nearest one outside them). Each row must be positive and
nondecreasing in k.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

SCHEMA = "dspark-row-mask-costs/1"


def load_cost_table(path: str | Path, max_num_reqs: int, max_draft: int) -> torch.Tensor:
    """``T_b(k)`` as a CPU float32 tensor [max_num_reqs + 1, max_draft + 1];
    row 0 is unused (zeros). ValueError on a malformed table."""
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema must be {SCHEMA!r}, got {raw.get('schema')!r}")
    table_draft = raw.get("max_draft")
    if not isinstance(table_draft, int) or table_draft < max_draft:
        raise ValueError(
            f"{path}: max_draft {table_draft!r} does not cover the drafter's "
            f"{max_draft} drafts"
        )
    rows: dict[int, list[float]] = {}
    for key, values in (raw.get("by_batch") or {}).items():
        b = int(key)
        if b < 1:
            raise ValueError(f"{path}: batch size {key!r} is not positive")
        if not isinstance(values, list) or len(values) != table_draft + 1:
            raise ValueError(
                f"{path}: batch {b} needs {table_draft + 1} costs (k = 0 .. "
                f"{table_draft}), got {values!r}"
            )
        row = [float(v) for v in values]
        if any(not v > 0 for v in row) or any(a > b_ for a, b_ in zip(row, row[1:])):
            raise ValueError(f"{path}: batch {b} costs must be positive and nondecreasing: {row}")
        rows[b] = row[: max_draft + 1]
    if not rows:
        raise ValueError(f"{path}: by_batch is empty")
    known = sorted(rows)
    table = torch.zeros(max_num_reqs + 1, max_draft + 1, dtype=torch.float32)
    for b in range(1, max_num_reqs + 1):
        if b <= known[0]:
            row = rows[known[0]]
        elif b >= known[-1]:
            row = rows[known[-1]]
        else:
            lo = max(k for k in known if k <= b)
            hi = min(k for k in known if k >= b)
            w = 0.0 if hi == lo else (b - lo) / (hi - lo)
            row = [x + (y - x) * w for x, y in zip(rows[lo], rows[hi])]
        table[b] = torch.tensor(row, dtype=torch.float32)
    return table


@dataclass
class RowMaskState:
    """Persistent device buffers shared by the row-mask pieces (fixed
    addresses: CUDA graphs read and write them in place)."""

    max_num_reqs: int
    max_num_tokens: int
    max_draft: int
    device: torch.device
    # Per request-state slot, the confidence head's output for each drafted
    # position of the slot's latest draft step (1.0: no confidence yet).
    conf_by_slot: torch.Tensor = field(init=False)
    # T_b(k), [max_num_reqs + 1, max_draft + 1].
    cost: torch.Tensor = field(init=False)
    # Per request of the current verify step (batch order): drafts kept.
    live_len: torch.Tensor = field(init=False)
    # Per token row of the current target forward: 1 live, 0 masked.
    row_live: torch.Tensor = field(init=False)
    # Host flag: row_live may hold a 0 (set by the mask kernel's caller,
    # cleared when the rows are reset to live).
    active: bool = False

    def __post_init__(self) -> None:
        d = self.device
        self.conf_by_slot = torch.ones(
            self.max_num_reqs, self.max_draft, dtype=torch.float32, device=d
        )
        self.cost = torch.zeros(
            self.max_num_reqs + 1, self.max_draft + 1, dtype=torch.float32, device=d
        )
        self.live_len = torch.full(
            (self.max_num_reqs,), self.max_draft, dtype=torch.int32, device=d
        )
        self.row_live = torch.ones(self.max_num_tokens, dtype=torch.int8, device=d)

    def reset_live(self) -> None:
        """Every row live again (no-op unless a mask was applied)."""
        if self.active:
            self.row_live.fill_(1)
            self.live_len.fill_(self.max_draft)
            self.active = False


_STATE: RowMaskState | None = None


def enabled() -> bool:
    return bool(envs.VLLM_DSPARK_ROW_MASK)


def init_row_mask_state(
    *, max_num_reqs: int, max_num_tokens: int, max_draft: int, device: torch.device
) -> RowMaskState | None:
    """Create the state once (later calls must agree and return it); None
    when the switch is off."""
    global _STATE
    if not enabled():
        return None
    if _STATE is not None:
        same = (
            _STATE.max_num_reqs == max_num_reqs
            and _STATE.max_num_tokens == max_num_tokens
            and _STATE.max_draft == max_draft
            and _STATE.device == torch.device(device)
        )
        if not same:
            raise ValueError("DSpark row mask state was created with other sizes")
        return _STATE
    state = RowMaskState(max_num_reqs, max_num_tokens, max_draft, torch.device(device))
    state.cost.copy_(load_cost_table(envs.VLLM_DSPARK_ROW_MASK, max_num_reqs, max_draft))
    _STATE = state
    logger.info(
        "DSpark row mask on (VLLM_DSPARK_ROW_MASK=%s): %d drafts, %d request slots",
        envs.VLLM_DSPARK_ROW_MASK,
        max_draft,
        max_num_reqs,
    )
    return state


def get_row_mask_state() -> RowMaskState | None:
    """The state, or None when the switch is off or nothing created it yet."""
    return _STATE


def _reset_for_tests() -> None:
    global _STATE
    _STATE = None
