# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""model.toml: what applies to every layout of one model.

Fields (schema launch-model/1)::

    model = "<model id>"
    hardware = "<hardware file name>"
    env_prefixes = ["MYMODEL_"]        # model-owned functional env prefixes
    arg_order = ["served-model-name", ...]   # argv order; other args follow
    baseline = { release = "...", gate_record = "...", bundle = "...", engine_commit = "..." }
    fixed = ["args.max-model-len", "tp", "pp", "cards"]
    controlled_args = ["tensor-parallel-size", ...]
    [layouts] allowed = [...], default = "..."
    [checkpoint.<name>] repo_key, revision_key, dir, index?, weights?, required, manifest
    [args] / [env] / [location]        # values: plain or { value = ..., why = "..." }
    [[refuse]] when = { "env.X" = "0", ... }, why = "..."
    [requires] files = [ { path = "vllm/_x*.so", fork_only = true } ]
    [user.<KNOB>] target = "args.port" | "env.X" | "card_ids" | "extra_args",
                  variant = "<name>", choices = [...], divide_by = N, type = "int",
                  secret = true
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from errors import Refusal
from settings_dir import SettingsDir

_KNOB = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass
class ModelSettings:
    data: dict
    model: str
    hardware: str
    env_prefixes: list[str]
    arg_order: list[str]
    allowed_layouts: list[str]
    default_layout: str | None
    checkpoints: dict[str, dict]
    fixed: set[str]
    controlled_args: set[str]
    refuse: list[dict]
    requires: list[dict]
    user: dict[str, dict]
    baseline: dict = field(default_factory=dict)


def load_model_settings(sd: SettingsDir) -> ModelSettings:
    d = sd.toml("model.toml", "launch-model")
    for key in ("model", "hardware"):
        if not isinstance(d.get(key), str) or not d[key]:
            raise Refusal(f"settings:model.toml: {key} is required")
    layouts = d.get("layouts", {})
    allowed = list(layouts.get("allowed", []))
    if not allowed:
        raise Refusal("settings:model.toml: [layouts] allowed is empty")
    default = layouts.get("default")
    if default is not None and default not in allowed:
        raise Refusal(f"settings:model.toml: default layout {default!r} is not allowed")
    cps = d.get("checkpoint", {})
    for name, spec in cps.items():
        for key in ("dir", "manifest", "repo_key", "revision_key"):
            if not isinstance(spec.get(key), str):
                raise Refusal(f"settings:model.toml: checkpoint.{name}.{key} is required")
    user = d.get("user", {})
    for knob, spec in user.items():
        if not _KNOB.match(knob):
            raise Refusal(f"settings:model.toml: user knob {knob!r} is not an env-style name")
        if not isinstance(spec, dict) or not (spec.get("target") or spec.get("variant")):
            raise Refusal(f"settings:model.toml: user.{knob} needs target or variant")
    requires = []
    for ent in d.get("requires", {}).get("files", []):
        if isinstance(ent, str):
            ent = {"path": ent}
        if not isinstance(ent, dict) or not isinstance(ent.get("path"), str):
            raise Refusal("settings:model.toml: requires.files entries need a path")
        requires.append(ent)
    return ModelSettings(
        data=d,
        model=d["model"],
        hardware=d["hardware"],
        env_prefixes=list(d.get("env_prefixes", [])),
        arg_order=[a.replace("_", "-") for a in d.get("arg_order", [])],
        allowed_layouts=allowed,
        default_layout=default,
        checkpoints=cps,
        fixed=set(d.get("fixed", [])),
        controlled_args={a.lstrip("-").replace("_", "-") for a in d.get("controlled_args", [])},
        refuse=list(d.get("refuse", [])),
        requires=requires,
        user=user,
        baseline=dict(d.get("baseline", {})),
    )
