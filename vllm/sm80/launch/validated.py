# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The validated statement: is this launch the one a release gate passed?

Inputs come only from the settings directory (``validated/<layout>.json``
and ``validated/receipt.json`` of one generation), the target tree, the
checkpoints and the environment; nothing is read from any release state.

The validated layout file (schema validated-launch/1)::

    {"schema", "model", "layout", "argv_functional", "env_functional",
     "functional_sha256", "engine_commit", "requires", "checkpoints",
     "gate_record", "engine_equivalence"?}

The receipt (schema validation-receipt/1)::

    {"schema", "model",
     "layouts": {L: {"validated_sha256", "functional_sha256", "gate_record",
                     "verdict_revision", "gate_revision", "requires"}},
     "pins": {ENGINE_COMMIT, TARGET_REPO, TARGET_REVISION, ...},
     "engines": [{"commit", "role", "native_manifest_sha256",
                  "build_owned_sha256", "engine_equivalence"?}],
     "checkpoints": {name: {"repo", "revision", "manifest_sha256"}},
     "promoted_at", "user_yes", "tools_commit"}

Roles: ``final`` settings must carry a receipt (a missing one is refused,
never read as a candidate); ``candidate`` settings must not; ``any`` reports
what it finds.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from canon import canonical_sha256, sha256_bytes
from errors import Refusal
from settings_dir import SettingsDir, check_schema

ROLES = ("any", "candidate", "final")
RECEIPT = "receipt.json"


@dataclass
class Generation:
    directory: str | None
    receipt: bytes | None
    layout_file: bytes | None


def read_generation(sd: SettingsDir, layout: str,
                    between: Callable[[], None] | None = None) -> Generation:
    """Read the receipt and the layout file from one generation.

    In the lab ``validated`` is a symlink to ``validated.d/<gen>/``; the
    reader holds a shared flock on ``validated.d/.lock`` from resolving the
    pointer to its last read, and closes it before returning, so the server
    never inherits it. ``between`` runs between the two reads (tests)."""
    pointer = sd.path("validated")
    gens = sd.path("validated.d")
    fd = None
    if os.path.isdir(gens):
        lock = os.path.join(gens, ".lock")
        try:
            fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        except OSError:
            fd = os.open(lock, os.O_RDONLY | os.O_CLOEXEC)
        fcntl.flock(fd, fcntl.LOCK_SH)
    try:
        if not os.path.isdir(pointer):
            return Generation(None, None, None)
        directory = os.path.realpath(pointer)
        receipt = _read(os.path.join(directory, RECEIPT))
        if between is not None:
            between()
        layout_file = _read(os.path.join(directory, f"{layout}.json"))
        return Generation(directory, receipt, layout_file)
    finally:
        if fd is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _read(path: str) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


def _json(raw: bytes, schema: str, where: str) -> dict:
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise Refusal(f"{where}: {exc}") from None
    if not isinstance(data, dict):
        raise Refusal(f"{where}: not an object")
    check_schema(data, schema, where)
    return data


@dataclass
class Evidence:
    layout: str
    functional_sha256: str
    pins: dict[str, str]
    tree_commit: str | None
    tree_dirty: bool
    build_owned_sha256: str | None
    native_manifest_sha256: str | None
    cross_commit_artifacts: bool
    adopted: bool
    requires: dict[str, str]
    checkpoints: dict[str, dict]            # name -> {repo, revision, manifest_sha256, state, identity}
    not_validated: list[str] = field(default_factory=list)


def evaluate(gen: Generation, ev: Evidence, *, role: str = "any") -> dict:
    """Criteria 1-5; every failing reason is listed."""
    if role not in ROLES:
        raise Refusal(f"unknown role {role!r}")
    reasons: list[str] = []
    out: dict = {"state": "NOT_VALIDATED", "reasons": reasons, "functional_match": False,
                 "role": role, "generation": gen.directory and os.path.basename(gen.directory)}
    pin = ev.pins.get("ENGINE_COMMIT")
    # checked in every mode, with or without a receipt
    if not pin:
        reasons.append("no engine pin")
    elif ev.tree_commit != pin:
        reasons.append("tree ≠ pin")
    if gen.receipt is None:
        if role == "final":
            raise Refusal("final settings carry no receipt (validated/receipt.json); "
                          "a final launch is never treated as a candidate")
        reasons.insert(0, "no receipt")
        out["receipt"] = None
        _common(ev, reasons)
        return out
    if role == "candidate":
        raise Refusal("candidate settings must not carry a receipt (validated/ must be empty)")
    receipt = _json(gen.receipt, "validation-receipt", "validated/receipt.json")
    out["receipt_sha256"] = sha256_bytes(gen.receipt)
    entry = receipt.get("layouts", {}).get(ev.layout)
    # 1. validated file ↔ receipt ↔ this launch
    if gen.layout_file is None or entry is None:
        reasons.append(f"layout {ev.layout} not validated")
    else:
        vfile = _json(gen.layout_file, "validated-launch", f"validated/{ev.layout}.json")
        own = canonical_sha256({"argv": vfile.get("argv_functional"), "env": vfile.get("env_functional")})
        c1 = True
        if sha256_bytes(gen.layout_file) != entry.get("validated_sha256"):
            reasons.append("validated file ≠ receipt")
            c1 = False
        if vfile.get("functional_sha256") != entry.get("functional_sha256") \
                or own != vfile.get("functional_sha256"):
            reasons.append("receipt ≠ validated file")
            c1 = False
        if vfile.get("functional_sha256") != ev.functional_sha256:
            reasons.append("functional ≠ validated")
            c1 = False
        out["functional_match"] = c1
        out["gate_record"] = vfile.get("gate_record") or entry.get("gate_record")
        out["engine_commit"] = vfile.get("engine_commit")
        if vfile.get("requires", {}) != ev.requires:
            reasons.append("requires ≠ validated")
    # 2. engine and pins bound to the receipt
    rpins = receipt.get("pins", {})
    if any(ev.pins.get(k) != v for k, v in rpins.items()):
        reasons.append("pin ≠ receipt")
    engines = {e.get("commit"): e for e in receipt.get("engines", []) if isinstance(e, dict)}
    if pin not in engines or ev.tree_commit not in engines:
        reasons.append("engine not in receipt")
    # 3. tree clean and its build identity equal to its engines entry
    if ev.tree_dirty:
        reasons.append("dirty tree")
    if ev.adopted:
        reasons.append("adopted tree")
    eng = engines.get(ev.tree_commit)
    if eng is not None:
        if eng.get("build_owned_sha256") != ev.build_owned_sha256:
            reasons.append("cross-commit artefacts" if ev.cross_commit_artifacts else "build ≠ receipt")
        if eng.get("native_manifest_sha256") != ev.native_manifest_sha256:
            reasons.append("natives ≠ receipt")
    elif ev.cross_commit_artifacts:
        reasons.append("cross-commit artefacts")
    # 4a. checkpoints bound to the receipt
    rcps = receipt.get("checkpoints", {})
    for name, cp in ev.checkpoints.items():
        r = rcps.get(name)
        if r is None or r.get("manifest_sha256") != cp.get("manifest_sha256") \
                or r.get("repo") != cp.get("repo") or r.get("revision") != cp.get("revision"):
            reasons.append("checkpoint ≠ receipt")
            break
    if set(rcps) - set(ev.checkpoints):
        reasons.append("checkpoint ≠ receipt")
    _common(ev, reasons)
    out["reasons"] = list(dict.fromkeys(reasons))
    if not out["reasons"]:
        out["state"] = "VALIDATED"
    return out


def _common(ev: Evidence, reasons: list[str]) -> None:
    """4b and 5, which do not need the receipt."""
    for name, cp in ev.checkpoints.items():
        if cp.get("state") in ("not present", "incomplete"):
            reasons.append("checkpoint not present")
        elif cp.get("identity") != "verified":
            reasons.append("checkpoint not verified")
    reasons.extend(ev.not_validated)


def compare_expected(expected_raw: bytes, argv_functional: list[str],
                     env_functional: dict[str, str], where: str) -> dict:
    """Compare this launch with a frozen expected launch (a resolved launch
    or a validated layout file): candidate consistency, apart from VALIDATED."""
    try:
        exp = json.loads(expected_raw)
    except ValueError as exc:
        raise Refusal(f"{where}: {exc}") from None
    schema = exp.get("schema", "") if isinstance(exp, dict) else ""
    if schema not in ("launch-resolved/2", "validated-launch/1"):
        raise Refusal(f"{where}: expected a launch-resolved/2 or validated-launch/1 file")
    e_argv = exp.get("argv_functional")
    e_env = exp.get("env_functional")
    if not isinstance(e_argv, list) or not isinstance(e_env, dict):
        raise Refusal(f"{where}: no argv_functional/env_functional")
    sha = canonical_sha256({"argv": e_argv, "env": e_env})
    mine = canonical_sha256({"argv": argv_functional, "env": env_functional})
    env_diff = sorted(k for k in set(e_env) | set(env_functional)
                      if e_env.get(k) != env_functional.get(k))
    return {
        "source": where,
        "functional_sha256": sha,
        "functional_match": sha == mine,
        "argv_equal": e_argv == argv_functional,
        "env_differs": env_diff,
    }
