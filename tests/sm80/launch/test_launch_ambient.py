# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ambient environment: classes, policies and the exec'd child."""

from __future__ import annotations

import json
import os

import pytest

from helpers import World, promote, read_record, write

# names /bin/sh (the stub server) adds to its own environment
SHELL_ADDED = {"PWD", "SHLVL", "_", "OLDPWD"}


@pytest.fixture
def w(tmp_path):
    w = World(tmp_path)
    assert w.verify() == 0
    promote(w, ["tp2"])
    return w


def reasons(d):
    return d["validated"]["reasons"]


@pytest.mark.parametrize("policy", ["strict", "merge"])
def test_retired_name_refused(w, policy):
    msg = w.refused("--policy", policy, environ=w.env(VLLM_OLD_THIN_GEMM="1"))
    assert "renamed to VLLM_THIN_GEMM" in msg


@pytest.mark.parametrize("policy", ["strict", "merge"])
def test_unknown_name_refused(w, policy):
    assert "not registered" in w.refused("--policy", policy, environ=w.env(VLLM_FOO="1"))


@pytest.mark.parametrize("name,value", [("VLLM_KERNEL_X", "0"), ("NCCL_ALGO", "Ring"),
                                        ("PYTORCH_NO_CUDA_MEMORY_CACHING", "1"),
                                        ("CUBLAS_WORKSPACE_CONFIG", ":0:0"),
                                        ("PYTHONOPTIMIZE", "1"), ("PYTHONHASHSEED", "1"),
                                        ("MALLOC_ARENA_MAX", "2"), ("OMP_PROC_BIND", "true")])
def test_undeclared_functional(w, name, value):
    assert "not declared" in w.refused(environ=w.env(**{name: value}))
    d = w.resolve("--policy", "merge", environ=w.env(**{name: value}))
    assert d["validated"]["state"] == "NOT_VALIDATED"
    assert f"ambient {name}" in reasons(d)
    assert d["ambient_layer"] == [f"env.{name}"]
    assert d["child_env"][name]["value"] == value


@pytest.mark.parametrize("policy", ["strict", "merge"])
def test_equal_value_stays_validated(w, policy):
    environ = w.env(VLLM_THIN_GEMM="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                    CUDA_VISIBLE_DEVICES="0,1")
    d = w.resolve("--policy", policy, environ=environ)
    assert d["validated"]["state"] == "VALIDATED", reasons(d)
    assert set(d["ambient_equal"]) >= {"VLLM_THIN_GEMM", "CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES"}
    assert d["resolved"]["env.VLLM_THIN_GEMM"]["layer"] == "model"


def test_declared_different_value(w):
    assert "differs from the settings" in w.refused(environ=w.env(VLLM_THIN_GEMM="0"))
    d = w.resolve("--policy", "merge", environ=w.env(VLLM_THIN_GEMM="0"))
    assert d["resolved"]["env.VLLM_THIN_GEMM"]["layer"] == "ambient"
    assert "ambient VLLM_THIN_GEMM" in reasons(d)


def test_inert_passes_not_compared(w):
    d = w.resolve(environ=w.env(TERM="dumb", LANG="C"))
    assert d["child_env"]["TERM"] == {"value": "dumb", "class": "inert"}
    assert d["validated"]["state"] == "VALIDATED"


def test_fixed_key_from_ambient(w):
    model = open(os.path.join(w.settings, "model.toml")).read().replace(
        'fixed = ["args.', 'fixed = ["env.VLLM_FAST_PATH", "args.')
    write(os.path.join(w.settings, "model.toml"), model)
    msg = w.refused("--policy", "merge", environ=w.env(VLLM_FAST_PATH="0"))
    assert "fixed" in msg


def test_ownership_names(w):
    own = {"GPU_LOCK_TOKENS": "a b", "RAM_LEASE_TOKEN": "t", "RAM_JOB_MARKER": "m", "SESSION_JOB": "j"}
    d = w.resolve(environ=w.env(**own))
    for k, v in own.items():
        assert d["child_env"][k] == {"value": v, "class": "ownership"}
    assert d["validated"]["state"] == "VALIDATED"
    base = w.resolve()
    assert d["functional_sha256"] == base["functional_sha256"]
    p = w.run(environ=w.env(**own))
    assert p.returncode == 0, p.stderr
    _, env = read_record(w.venv)
    assert all(env[k] == v for k, v in own.items())


@pytest.mark.parametrize("policy", ["strict", "merge"])
@pytest.mark.parametrize("name", ["LD_PRELOAD", "LD_AUDIT", "PYTHONHOME", "PYTHONSTARTUP"])
def test_code_loading_always_refused(w, policy, name):
    assert "refused" in w.refused("--policy", policy, environ=w.env(**{name: "/x"}))


def test_pythonpath_other_tree(w):
    environ = w.env(PYTHONPATH="/elsewhere")
    assert "strict" in w.refused(environ=environ)
    d = w.resolve("--policy", "merge", environ=environ)
    assert "ambient PYTHONPATH" in reasons(d)


def test_path_ignored_and_rebuilt(w):
    d = w.resolve(environ=w.env(PATH="/evil/bin:/usr/bin"))
    assert "PATH" in d["ignored"]
    assert d["child_env"]["PATH"]["value"].startswith(os.path.join(w.venv, "bin") + ":")
    assert d["validated"]["state"] == "VALIDATED"


def test_cards_from_lock(w):
    assert "differs from the resolved cards" in w.refused(environ=w.env(CUDA_VISIBLE_DEVICES="1,0"))


def test_container_runtime_names(w):
    d = w.resolve(environ=w.env(NVIDIA_VISIBLE_DEVICES="all"))
    assert d["child_env"]["NVIDIA_VISIBLE_DEVICES"]["class"] == "container-runtime"
    assert d["validated"]["state"] == "VALIDATED"


def test_unclassified_dropped_from_child(w):
    environ = w.env(FOO_DEBUG="1", __NV_PRIME_RENDER_OFFLOAD="1")
    d = w.resolve(environ=environ)
    assert set(d["dropped"]) == {"FOO_DEBUG", "__NV_PRIME_RENDER_OFFLOAD"}
    assert d["validated"]["state"] == "VALIDATED"
    p = w.run(environ=environ)
    assert p.returncode == 0, p.stderr
    _, env = read_record(w.venv)
    assert "FOO_DEBUG" not in env and "__NV_PRIME_RENDER_OFFLOAD" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert set(env) - SHELL_ADDED == set(d["child_env"])


def test_isolated_interpreter_output_identical(w, tmp_path):
    shadow = tmp_path / "shadow"
    write(str(shadow / "tomllib.py"), "raise SystemExit('shadowed')\n")
    base = w.run("--print-config", "--json")
    hostile = w.run("--print-config", "--json",
                    environ=w.env(PYTHONOPTIMIZE="2", PYTHONPATH=str(shadow)))
    # the ambient names are refused (strict); the resolution itself is not altered
    assert base.returncode == 0
    assert hostile.returncode == 3 and "shadowed" not in hostile.stderr
    merged = w.run("--print-config", "--json", "--policy", "merge",
                   environ=w.env(PYTHONOPTIMIZE="2", PYTHONPATH=str(shadow)))
    d0, d1 = json.loads(base.stdout), json.loads(merged.stdout)
    assert d0["argv"] == d1["argv"] and d0["functional_sha256"] != d1["functional_sha256"]


def test_container_drops_ownership(tmp_path):
    w = World(tmp_path)
    d = w.resolve("--runtime", "container", "--tree", w.tree, environ={"RAM_LEASE_TOKEN": "t"})
    assert "RAM_LEASE_TOKEN" in d["dropped"]
