# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve the settings layers into one final object.

Precedence, lowest first::

    defaults.toml[runtime] < model.toml < hardware < layout
      < variants (in the order given) < overlays (incl. the community
      settings directory) < user .env ([user] keys only)
      < ambient (merge policy only) < CLI --set

Every key of the final object records its value, layer, source file and
``why``. argv, the child environment, the printed configuration and
``functional_sha256`` are all derived from that object and nothing else.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
import tomllib
from dataclasses import dataclass, field
from typing import Any

from canon import canonical_sha256
from envclass import ClassTable, always_refused
from errors import Refusal
from model import ModelSettings
from registry import Registry
from settings_dir import check_arg_spellings, normalize_arg_name

LOCKED_LAYERS = ("overlay", "user", "ambient", "cli")   # [fixed] needs --experiment
RUNTIMES = ("native", "container")
_SUBST = re.compile(r"\$\{([^}]*)\}")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOP_KEYS = ("tp", "pp", "cards", "card_ids")
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("", "0", "false", "no", "off")


@dataclass
class Entry:
    value: Any
    layer: str
    source: str
    why: str | None = None


@dataclass
class Final:
    entries: dict[str, Entry] = field(default_factory=dict)
    unset: dict[str, str] = field(default_factory=dict)
    whys: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        e = self.entries.get(key)
        return default if e is None else e.value

    def put(self, key: str, value: Any, layer: str, source: str, why: str | None = None) -> None:
        old = self.entries.pop(key, None)
        if why is None and old is not None and old.value == value:
            why = old.why
        self.entries[key] = Entry(value, layer, source, why)
        self.unset.pop(key, None)

    def drop(self, key: str, layer: str) -> None:
        self.entries.pop(key, None)
        self.unset[key] = layer


def split_value(v: Any, where: str, whys: dict | None = None, path: str = "") -> tuple[Any, str | None]:
    """A plain value or { value = ..., why = "..." }; nested tables keep
    their own value/why forms, whose whys are collected into ``whys``."""
    if isinstance(v, dict) and "value" in v and set(v) <= {"value", "why"}:
        why = v.get("why")
        if why is not None and not isinstance(why, str):
            raise Refusal(f"{where}: why must be a string")
        inner, _ = split_value(v["value"], where, whys, path)
        return inner, why
    if isinstance(v, dict):
        out = {}
        for k, sub in v.items():
            sub_path = f"{path}.{k}" if path else k
            val, why = split_value(sub, where, whys, sub_path)
            if why and whys is not None:
                whys[sub_path] = why
            out[k] = val
        return out, None
    if isinstance(v, list):
        return [split_value(x, where, whys, path)[0] for x in v], None
    return v, None


def _env_value(v: Any, where: str) -> str:
    if isinstance(v, bool) or not isinstance(v, (str, int)):
        raise Refusal(f"{where}: environment values are strings or integers")
    return str(v)


def apply_table(final: Final, table: dict, layer: str, source: str, *,
                layout: str | None = None, runtime: str | None = None) -> None:
    """Apply one settings table (layout, variant, overlay, hardware, model).

    A [location] value may be ``{ native = "...", container = "..." }``:
    only the current runtime's entry applies (none if it is missing)."""
    for key in _TOP_KEYS:
        if key in table:
            v = table[key]
            if key == "card_ids":
                v = str(v)
            elif isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise Refusal(f"{source}: {key} must be a positive integer")
            final.put(key, v, layer, source)
    if "tp" in table:
        final.put("args.tensor-parallel-size", table["tp"], layer, source)
    if "pp" in table:
        final.put("args.pipeline-parallel-size", table["pp"], layer, source)
    args = table.get("args", {})
    check_arg_spellings(args, source)
    for k, v in args.items():
        key = "args." + normalize_arg_name(k)
        sub_whys: dict[str, str] = {}
        val, why = split_value(v, source, sub_whys)
        final.put(key, val, layer, source, why)
        for p, w in sub_whys.items():
            final.whys[f"{key}.{p}"] = w
    for section, prefix in (("env", "env."), ("location", "loc.")):
        for k, v in table.get(section, {}).items():
            if not _ENV_NAME.match(k):
                raise Refusal(f"{source}: {k!r} is not an environment name")
            val, why = split_value(v, source)
            if section == "location" and isinstance(val, dict):
                if not set(val) <= set(RUNTIMES):
                    raise Refusal(f"{source}: location {k} keys must be native/container")
                if runtime not in val:
                    continue
                val = val[runtime]
            final.put(prefix + k, _env_value(val, f"{source}:{k}"), layer, source, why)
    for key in table.get("unset", []):
        if not isinstance(key, str):
            raise Refusal(f"{source}: unset entries are key strings")
        if key.startswith("args."):
            key = "args." + normalize_arg_name(key[5:])
        final.drop(key, layer)
    if layout is not None:
        sub = table.get("layout", {}).get(layout)
        if sub:
            apply_table(final, sub, layer, f"{source}[layout.{layout}]", runtime=runtime)


def substitute(value: Any, subst: dict[str, str], where: str) -> Any:
    if isinstance(value, str):
        def rep(m: re.Match) -> str:
            name = m.group(1)
            if name not in subst:
                raise Refusal(f"{where}: unknown substitution ${{{name}}}")
            return subst[name]
        return _SUBST.sub(rep, value)
    if isinstance(value, dict):
        return {k: substitute(v, subst, where) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, subst, where) for v in value]
    return value


def render_arg(name: str, value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if value is True:
        return [f"--{name}"]
    if isinstance(value, dict):
        return [f"--{name}", json.dumps(value, separators=(",", ":"), ensure_ascii=False)]
    if isinstance(value, list):
        return [f"--{name}", *[str(v) for v in value]]
    if isinstance(value, float):
        return [f"--{name}", repr(value)]
    return [f"--{name}", str(value)]


def parse_set(item: str) -> tuple[str, Any]:
    key, sep, raw = item.partition("=")
    if not sep or not key:
        raise Refusal(f"--set {item!r}: expected KEY=VALUE")
    if not key.startswith("args.") and key not in ("tp", "pp"):
        return key, raw
    try:
        return key, tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return key, raw


def set_nested(final: Final, target: str, value: Any, layer: str, source: str) -> None:
    """args.a.b = value: set key b inside the table argument a."""
    parts = target.split(".")
    key = "args." + normalize_arg_name(parts[1])
    if len(parts) == 2:
        final.put(key, value, layer, source)
        return
    current = copy.deepcopy(final.get(key) or {})
    if not isinstance(current, dict):
        raise Refusal(f"{source}: {key} is not a table argument")
    node = current
    for p in parts[2:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    final.put(key, current, layer, source)


@dataclass
class Inputs:
    runtime: str
    runtime_cfg: dict
    layout: str
    variants: list[str]
    overlays: list[tuple[str, dict]]     # (source, table)
    community: list[tuple[str, dict]]    # files from the community settings directory
    env_file: dict[str, str]
    ambient: dict[str, str]
    sets: list[str]
    extra_args: list[str]
    policy: str
    experiment: str | None
    capability: str
    tree: str
    venv: str
    home: str
    checkpoint_paths: dict[str, str]


@dataclass
class Resolved:
    final: Final
    variants: list[str]
    argv: list[str]
    argv_functional: list[str]
    location_keys: dict[str, Any]
    harness: list[str]
    child_env: dict[str, dict]
    env_functional: dict[str, str]
    ambient_equal: list[str]
    ambient_layer: list[str]
    dropped: list[str]
    ignored: list[str]
    fork_absent: list[str]
    not_validated: list[str]
    functional_sha256: str
    exe: str
    secret_names: set[str]


def user_values(ms: ModelSettings, inp: Inputs) -> dict[str, str]:
    """[user] knob values: the .env file, plus (in a container, where the
    .env arrives as the process environment) the ambient knobs."""
    values = dict(inp.env_file)
    if inp.runtime == "container":
        for knob in ms.user:
            if knob in inp.ambient and knob not in values:
                values[knob] = inp.ambient[knob]
    return values


def _knob_variant(knob: str, spec: dict, raw: str) -> str | None:
    """The variant a knob value selects (None: none)."""
    sel = spec["variant"]
    if isinstance(sel, dict):
        if raw not in sel:
            raise Refusal(f"{knob}={raw!r}: expected one of {', '.join(sel)}")
        return sel[raw] or None
    v = raw.strip().lower()
    if v in _TRUE:
        return sel
    if v in _FALSE:
        return None
    raise Refusal(f"{knob}={raw!r}: expected 1 or 0")


def selected_variants(ms: ModelSettings, inp: Inputs, values: dict[str, str] | None = None) -> list[str]:
    variants = list(inp.variants)
    values = user_values(ms, inp) if values is None else values
    for knob, spec in ms.user.items():
        if spec.get("variant") and knob in values:
            name = _knob_variant(knob, spec, values[knob])
            if name and name not in variants:
                variants.append(name)
    if inp.capability == "plain-upstream" and "plain-upstream" not in variants:
        variants.append("plain-upstream")
    return variants


def resolve(ms: ModelSettings, files: dict[str, dict], registry: Registry,
            classes: ClassTable, harness_args: list[str], inp: Inputs) -> Resolved:
    """files: 'hardware', 'layout', 'variant:<name>' → parsed tables."""
    final = Final()
    not_validated: list[str] = []
    rt = inp.runtime_cfg
    run = inp.runtime
    harness_args = list(harness_args)
    # runtime layer
    final.put("args.host", rt["host"], "runtime", f"defaults.toml[runtime.{run}]")
    for k, v in rt.get("location", {}).items():
        final.put("loc." + k, str(v), "runtime", f"defaults.toml[runtime.{run}.location]")
    # model, hardware, layout
    apply_table(final, {k: ms.data[k] for k in ("args", "env", "location") if k in ms.data},
                "model", "model.toml", runtime=run)
    hw = files["hardware"]
    apply_table(final, {k: hw[k] for k in ("env", "location") if k in hw}, "hardware",
                f"hardware/{ms.hardware}.toml", runtime=run)
    apply_table(final, files["layout"], "layout", f"layouts/{inp.layout}.toml", runtime=run)
    # variants (user knobs that select a variant are read first)
    values = user_values(ms, inp)
    variants = selected_variants(ms, inp, values)
    for v in variants:
        table = files.get(f"variant:{v}")
        if table is None:
            raise Refusal(f"variant {v!r} has no variants/{v}.toml")
        apply_table(final, table, "variant", f"variants/{v}.toml", layout=inp.layout, runtime=run)
    # what the settings themselves fixed; later layers may not change it
    settled = {k: e.value for k, e in final.entries.items() if _fixed(k, ms)}
    # overlays, then the community settings directory (same level)
    for source, table in inp.overlays:
        if table.get("class") == "harness":
            harness_args += [normalize_arg_name(k) for k in table.get("args", {})]
        apply_table(final, table, "overlay", source, layout=inp.layout, runtime=run)
    for source, table in inp.community:
        apply_table(final, table, "overlay", source, layout=inp.layout, runtime=run)
    # user layer
    extra_args = list(inp.extra_args)
    secret_names: set[str] = set()
    for knob, raw in values.items():
        spec = ms.user.get(knob)
        if spec is None:
            raise Refusal(f".env: {knob} is not a documented setting of this model "
                          f"(known: {', '.join(sorted(ms.user)) or 'none'})")
        if spec.get("target"):
            _apply_user(final, knob, spec, raw, extra_args, secret_names, variants)
    sets = [parse_set(s) for s in inp.sets]
    _check_fixed(final, ms, inp, sets, settled)
    # ambient
    amb = _ambient(final, ms, registry, classes, inp)
    for key, value in amb["layer"].items():
        final.put(key, value, "ambient", "environment")
    # CLI layer
    for key, value in sets:
        if key in ms.user:
            if not ms.user[key].get("target"):
                raise Refusal(f"--set {key}: selects a variant; use --variant")
            _apply_user(final, key, ms.user[key], str(value), extra_args, secret_names, variants,
                        layer="cli", source="--set")
        elif key == "cards":
            final.put("card_ids", str(value), "cli", "--set")
        elif key.startswith("args."):
            set_nested(final, key, value, "cli", "--set")
        elif key.startswith("env."):
            if not _ENV_NAME.match(key[4:]):
                raise Refusal(f"--set {key}: not an environment name")
            final.put(key, str(value), "cli", "--set")
        elif key in ("tp", "pp"):
            apply_table(final, {key: value}, "cli", "--set")
        else:
            raise Refusal(f"--set {key}: expected env.NAME, args.NAME, cards, tp, pp or a [user] knob")
    if extra_args:
        _check_controlled(ms, extra_args, inp.experiment)
        final.put("extra_args", list(extra_args), "user", "EXTRA_ARGS")

    _check_geometry(final, hw, inp.layout)
    for rule in ms.refuse:
        when = rule.get("when", {})
        if when and all(str(final.get(k, "")) == str(v) for k, v in when.items()):
            raise Refusal(f"refused combination {when}: {rule.get('why', '')}".strip())
    fork_absent = _check_env_names(final, registry, classes, inp.capability)
    return _render(final, ms, inp, rt, harness_args, amb, variants, fork_absent,
                   secret_names, not_validated, classes)


def _check_geometry(final: Final, hw: dict, layout: str) -> None:
    hw_count = hw.get("count")
    tp = final.get("tp", 1)
    pp = final.get("pp", 1)
    cards = final.get("cards", tp * pp)
    if cards != tp * pp:
        raise Refusal(f"layout {layout}: cards {cards} != tp {tp} x pp {pp}")
    if isinstance(hw_count, int) and cards > hw_count:
        raise Refusal(f"layout {layout}: needs {cards} cards, hardware has {hw_count}")
    card_ids = final.get("card_ids")
    if card_ids is not None:
        ids = [c.strip() for c in str(card_ids).split(",") if c.strip()]
        if len(ids) != cards or len(set(ids)) != len(ids):
            raise Refusal(f"cards {card_ids!r}: layout {layout} needs exactly {cards} distinct cards")
        if isinstance(hw_count, int) and any(not i.isdigit() or int(i) >= hw_count for i in ids):
            raise Refusal(f"cards {card_ids!r}: not cards of this hardware (0..{hw_count - 1})")


def _check_env_names(final: Final, registry: Registry, classes: ClassTable,
                     capability: str) -> list[str]:
    """Settings env names: functional or secret class; VLLM_* registered."""
    fork_absent: list[str] = []
    for key in list(final.entries):
        if not key.startswith("env."):
            continue
        name = key[4:]
        source = final.entries[key].source
        if name.upper().startswith("VLLM_"):
            problem = registry.check(name)
            if problem:
                if capability == "plain-upstream" and name == name.upper() \
                        and name not in registry.retired:
                    fork_absent.append(name)
                    final.drop(key, "capability")
                    continue
                raise Refusal(f"{source}: {problem}")
        cls = classes.classify(name)
        if cls not in ("functional", "secret"):
            raise Refusal(f"{source}: {name} is class {cls}; settings env holds functional switches only")
    return fork_absent


def _render(final: Final, ms: ModelSettings, inp: Inputs, rt: dict, harness_args: list[str],
            amb: dict, variants: list[str], fork_absent: list[str], secret_names: set[str],
            not_validated: list[str], classes: ClassTable) -> Resolved:
    subst = {"home": inp.home, "venv": inp.venv, "tree": inp.tree,
             "models_dir": substitute(rt["models_dir"], {"home": inp.home}, "runtime")}
    for name, path in inp.checkpoint_paths.items():
        subst[f"checkpoint.{name}.path"] = path
    model_value = final.get("args.model")
    if model_value is None:
        raise Refusal("model.toml: args.model (the served checkpoint) is not set")
    exe = os.path.join(inp.venv, "bin", "vllm")
    argv = [exe, "serve", substitute(model_value, subst, "args.model")]
    argv_functional: list[str] = []
    harness: list[str] = []
    location_keys: dict[str, Any] = {"args.model": argv[2]}
    harness_set = {normalize_arg_name(h) for h in harness_args}
    ordered = [f"args.{k}" for k in ms.arg_order if k != "model" and f"args.{k}" in final.entries]
    ordered += [k for k in final.entries
                if k.startswith("args.") and k != "args.model" and k not in ordered]
    for key in ordered:
        name = key[5:]
        value = final.get(key)
        tokens = render_arg(name, substitute(value, subst, key))
        argv += tokens
        if name in ("host", "port"):
            location_keys[key] = value
        elif name in harness_set:
            harness += tokens
        else:
            # placeholders stay unexpanded: paths are location, not function
            argv_functional += render_arg(name, value)
    for tok in final.get("extra_args", []) or []:
        argv.append(tok)
        argv_functional.append(tok)

    child: dict[str, dict] = {}
    env_functional: dict[str, str] = {}
    for key, e in final.entries.items():
        if key.startswith("env."):
            name = key[4:]
            cls = classes.classify(name)
            child[name] = {"value": substitute(e.value, subst, key), "class": cls}
            if cls == "functional":
                env_functional[name] = e.value
        elif key.startswith("loc."):
            name = key[4:]
            child[name] = {"value": substitute(e.value, subst, key), "class": "location"}
            location_keys[key] = child[name]["value"]
    card_ids = final.get("card_ids")
    if card_ids is not None:
        child["CUDA_VISIBLE_DEVICES"] = {"value": str(card_ids), "class": "location"}
        location_keys["CUDA_VISIBLE_DEVICES"] = str(card_ids)
    for name, value in amb["code"].items():
        child[name] = {"value": value, "class": "code-loading"}
    for name, (value, cls) in amb["passed"].items():
        child.setdefault(name, {"value": value, "class": cls})
    for name in secret_names:
        if name in child:
            child[name]["class"] = "secret"
    functional_sha256 = canonical_sha256({"argv": argv_functional, "env": env_functional})
    return Resolved(
        final=final, variants=variants, argv=argv, argv_functional=argv_functional,
        location_keys=location_keys, harness=harness, child_env=child,
        env_functional=env_functional, ambient_equal=amb["equal"],
        ambient_layer=sorted(amb["layer"]), dropped=amb["dropped"],
        ignored=amb["ignored"], fork_absent=fork_absent,
        not_validated=not_validated + amb["not_validated"],
        functional_sha256=functional_sha256, exe=exe, secret_names=secret_names,
    )


def _apply_user(final: Final, knob: str, spec: dict, raw: str, extra_args: list[str],
                secret_names: set[str], variants: list[str], *, layer: str = "user",
                source: str | None = None) -> None:
    source = source or f".env:{knob}"
    only = spec.get("variants")
    if only and not set(only) & set(variants):
        raise Refusal(f"{knob} applies only with variant {' or '.join(only)} "
                      f"(active: {', '.join(variants) or 'none'})")
    choices = spec.get("choices")
    if choices is not None and raw not in [str(c) for c in choices]:
        raise Refusal(f"{knob}={raw!r}: expected one of {', '.join(map(str, choices))}")
    value: Any = raw
    kind = spec.get("type", "str")
    try:
        if spec.get("divide_by"):
            value = int(raw) // int(spec["divide_by"])
        elif kind == "int":
            value = int(raw)
    except ValueError:
        raise Refusal(f"{knob}={raw!r}: expected an integer") from None
    if kind == "bool":
        low = raw.strip().lower()
        if low not in _TRUE + _FALSE[1:]:
            raise Refusal(f"{knob}={raw!r}: expected 1 or 0")
        value = low in _TRUE
    targets = spec["target"] if isinstance(spec["target"], list) else [spec["target"]]
    for target in targets:
        if target == "extra_args":
            extra_args.extend(shlex.split(raw))
        elif target == "card_ids":
            final.put("card_ids", raw, layer, source)
        elif target.startswith("args."):
            set_nested(final, target, value, layer, source)
        elif target.startswith("env."):
            final.put(target, str(raw if kind == "bool" else value), layer, source)
            if spec.get("secret"):
                secret_names.add(target[4:])
        else:
            raise Refusal(f"model.toml: user.{knob} target {target!r} is not args.*, env.*, card_ids or extra_args")


def _check_controlled(ms: ModelSettings, extra: list[str], experiment: str | None) -> None:
    for tok in extra:
        if tok.startswith("--"):
            name = normalize_arg_name(tok[2:].split("=", 1)[0])
            if name in ms.controlled_args and not experiment:
                raise Refusal(f"extra argument --{name} is set by the settings; "
                              "change it with --set under --experiment")


def _fixed(key: str, ms: ModelSettings) -> bool:
    if key.startswith("args."):
        key = "args." + normalize_arg_name(key[5:])
    return key in ms.fixed


def _check_fixed(final: Final, ms: ModelSettings, inp: Inputs, sets: list,
                 settled: dict[str, Any]) -> None:
    """A [fixed] key the settings set may be changed by an overlay, the
    community directory, the user .env or the CLI only under --experiment.
    Repeating the same value is no change."""
    if inp.experiment:
        return
    for key, e in final.entries.items():
        if e.layer in LOCKED_LAYERS and key in settled and settled[key] != e.value:
            raise Refusal(f"{key} is fixed for this model ({e.source}); it changes only under --experiment")
    for key, value in sets:
        targets = [key]
        if key == "cards":
            targets = ["card_ids"]
        elif key in ms.user and ms.user[key].get("target"):
            t = ms.user[key]["target"]
            targets = t if isinstance(t, list) else [t]
        for k in targets:
            if _fixed(k, ms) and str(final.get(k)) != str(value):
                raise Refusal(f"{k} is fixed for this model (--set); it changes only under --experiment")


def _ambient(final: Final, ms: ModelSettings, registry: Registry, classes: ClassTable,
             inp: Inputs) -> dict:
    """Classify the ambient environment and apply the policy."""
    out: dict = {"equal": [], "layer": {}, "dropped": [], "ignored": [], "passed": {},
                 "code": {}, "not_validated": []}
    strict = inp.policy == "strict"
    container = inp.runtime == "container"
    built_code = {
        "PATH": ":".join([os.path.join(inp.venv, "bin"),
                          os.path.join(inp.runtime_cfg["cuda_home"], "bin"),
                          "/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin",
                          "/sbin", "/bin"]),
        "PYTHONPATH": inp.tree,
        "PYTHONNOUSERSITE": "1",
        "CUDA_HOME": inp.runtime_cfg["cuda_home"],
    }
    out["code"] = dict(built_code)
    refusals: list[str] = []
    for name in sorted(inp.ambient):
        value = inp.ambient[name]
        if container and name in ms.user:
            continue  # read as a [user] knob
        if name.upper().startswith("VLLM_") and name != name.upper():
            refusals.append(registry.check(name) or f"{name}: not a registered name")
            continue
        cls = classes.classify(name)
        if cls == "secret":
            problem = registry.check(name) if name.startswith("VLLM_") else None
            if problem:
                refusals.append(problem)
            else:
                out["passed"][name] = (value, cls)
        elif cls == "ownership":
            if container:
                out["dropped"].append(name)
            else:
                out["passed"][name] = (value, cls)
        elif cls in ("container-runtime", "inert"):
            out["passed"][name] = (value, cls)
        elif cls == "code-loading":
            if always_refused(name):
                refusals.append(f"{name} in the environment is refused: it changes what the server loads")
            elif name == "PATH":
                out["ignored"].append(name)
            elif built_code.get(name) == value:
                out["equal"].append(name)
            elif strict:
                refusals.append(f"{name}={value!r} in the environment differs from the launcher's "
                                f"{built_code.get(name)!r} (strict policy)")
            else:
                out["code"][name] = value
                out["not_validated"].append(f"ambient {name}")
        elif cls == "location":
            if name == "CUDA_VISIBLE_DEVICES":
                want = final.get("card_ids")
                if want is None:
                    final.put("card_ids", value, "ambient", "environment")
                elif str(want) == value:
                    out["equal"].append(name)
                elif strict:
                    refusals.append(f"CUDA_VISIBLE_DEVICES={value!r} differs from the resolved cards {want!r}")
                else:
                    final.put("card_ids", value, "ambient", "environment")
            elif final.get("loc." + name) == value:
                out["equal"].append(name)
            else:
                out["ignored"].append(name)
        elif cls == "functional":
            if name.startswith("VLLM_"):
                problem = registry.check(name)
                if problem:
                    refusals.append(problem)
                    continue
            key = "env." + name
            declared = key in final.entries
            if declared and final.get(key) == value:
                out["equal"].append(name)
            elif _fixed(key, ms) and not inp.experiment:
                refusals.append(f"{name} is fixed for this model; it changes only under --experiment")
            elif strict:
                if declared:
                    refusals.append(f"{name}={value!r} in the environment differs from the settings "
                                    f"value {final.get(key)!r} (strict policy: use --set or --overlay)")
                else:
                    refusals.append(f"{name} in the environment is not declared by the settings "
                                    "(strict policy: use --set or --overlay)")
            else:
                out["layer"][key] = value
                out["not_validated"].append(f"ambient {name}")
        else:
            out["dropped"].append(name)
    if refusals:
        raise Refusal(*refusals)
    return out
