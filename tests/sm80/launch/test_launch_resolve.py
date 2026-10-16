# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolution, precedence, refusals and the synthetic three-layout model."""

from __future__ import annotations

import glob
import json
import os
import re

import pytest

from helpers import LAUNCH, World, promote, read_record, write


@pytest.fixture
def w(tmp_path):
    return World(tmp_path)


# -- synthetic model --------------------------------------------------------------

@pytest.mark.parametrize("layout,tp,pp", [("tp1", 1, 1), ("tp2", 2, 1), ("pp2", 1, 2)])
def test_every_layout_resolves(w, layout, tp, pp):
    d = w.resolve(layout=layout)
    assert d["schema"] == "launch-resolved/2"
    a = d["argv"]
    assert a[1] == "serve" and a[2] == os.path.join(w.models, "synth-dense-1b")
    assert a[a.index("--tensor-parallel-size") + 1] == str(tp)
    assert a[a.index("--pipeline-parallel-size") + 1] == str(pp)
    assert d["engine_target"]["commit"] and d["registry"]["source"] == "arm-tree"
    assert d["validated"]["state"] == "NOT_VALIDATED" and "no receipt" in d["validated"]["reasons"]


def test_arg_order_and_functional_excludes_location(w):
    d = w.resolve()
    a = d["argv"]
    assert a[3:7] == ["--served-model-name", "synth-dense", "--tensor-parallel-size", "2"]
    f = d["argv_functional"]
    assert "--port" not in f and "--host" not in f and a[2] not in f
    assert d["location_keys"]["args.port"] == 8123


def test_why_and_layers_recorded(w):
    d = w.resolve()
    r = d["resolved"]
    assert r["args.max-num-batched-tokens"]["layer"] == "layout"
    assert r["args.max-model-len"]["why"].startswith("Long enough")
    assert r["args.host"]["layer"] == "runtime"


def test_text_and_explain(w, capsys):
    import main
    assert main.main(w.base_args("--print-config", "--explain"), w.env()) == 0
    out = capsys.readouterr().out
    assert out.startswith("NOT VALIDATED (no receipt")
    assert "# Faster small-M GEMMs" in out


def test_launcher_names_no_real_model():
    text = "".join(open(p, encoding="utf-8").read().lower()
                   for p in glob.glob(os.path.join(LAUNCH, "**", "*.*"), recursive=True)
                   if p.endswith((".py", ".toml")))
    for word in ("glm", "qwen", "deepseek", "llama", "dflash"):
        assert word not in text, word


def test_variant_with_layout_subtable_and_unset(w):
    d = w.resolve("--variant", "lean", layout="pp2")
    assert "--trust-remote-code" not in d["argv"]
    assert d["argv"][d["argv"].index("--block-size") + 1] == "32"
    assert d["unset"]["args.trust-remote-code"] == "variant"


def test_user_knobs(w, tmp_path):
    env = write(str(tmp_path / "user.env"),
                "PORT=9001\nPLACEMENT=ram\nMM_CAP=1\nMM_MAX_PIXELS=1568\nAPI_KEY='s3cret'\n"
                "EXTRA_ARGS=--enable-log-requests\n")
    d = w.resolve("--env-file", env)
    a = d["argv"]
    assert a[a.index("--port") + 1] == "9001"
    assert json.loads(a[a.index("--mm-processor-kwargs") + 1]) == {"max_image_tokens": 2}
    assert d["child_env"]["VLLM_PLACEMENT"]["value"] == "ram"
    assert d["child_env"]["VLLM_API_KEY"] == {"value": "***", "class": "secret"}
    assert "s3cret" not in json.dumps(d)
    assert a[-1] == "--enable-log-requests" and d["argv_functional"][-1] == "--enable-log-requests"
    assert "mm-cap" in d["variants"]


# -- precedence ------------------------------------------------------------------

def test_precedence_each_layer(w, tmp_path):
    key = "VLLM_FAST_PATH"
    ov = write(str(tmp_path / "ov.toml"), f'schema = "launch-overlay/1"\n[env]\n{key} = "3"\n')
    layers = [((), "1", "layout"),
              (("--variant", "x"), "2", "variant"),
              (("--variant", "x", "--overlay", ov), "3", "overlay"),
              (("--variant", "x", "--overlay", ov, "--set", f"env.{key}=5"), "5", "cli")]
    write(os.path.join(w.settings, "variants", "x.toml"), f'schema = "launch-variant/1"\n[env]\n{key} = "2"\n')
    for extra, value, layer in layers:
        d = w.resolve(*extra)
        assert d["child_env"][key]["value"] == value
        assert d["resolved"][f"env.{key}"]["layer"] == layer
        assert d["env_functional"][key] == value


def test_ambient_vs_cli_conflict_reaches_child(w):
    """Ambient (merge) and --set on one key: the CLI value is printed, hashed and exec'd."""
    environ = w.env(VLLM_THIN_GEMM="7")
    d = w.resolve("--policy", "merge", "--set", "env.VLLM_THIN_GEMM=9", environ=environ)
    assert d["child_env"]["VLLM_THIN_GEMM"]["value"] == "9"
    assert d["env_functional"]["VLLM_THIN_GEMM"] == "9"
    ref = w.resolve("--set", "env.VLLM_THIN_GEMM=9")
    assert d["functional_sha256"] == ref["functional_sha256"]
    assert w.verify() == 0
    p = w.run("--policy", "merge", "--set", "env.VLLM_THIN_GEMM=9", environ=environ)
    assert p.returncode == 0, p.stderr
    argv, env = read_record(w.venv)
    assert env["VLLM_THIN_GEMM"] == "9"
    assert argv[1:] == d["argv"][1:]


# -- refusals --------------------------------------------------------------------

@pytest.mark.parametrize("extra,needle", [
    (("--set", "env.VLLM_NOPE=1"), "not registered"),
    (("--set", "env.VLLM_OLD_THIN_GEMM=1"), "renamed to VLLM_THIN_GEMM"),
    (("--set", "env.vllm_thin_gemm=1"), "case-sensitive"),
    (("--set", "args.max-model-len=1024"), "fixed"),
    (("--set", "cards=0"), "exactly 2"),
    (("--set", "cards=0,7"), "not cards of this hardware"),
    (("--set", "env.VLLM_MOE_SKIP_PADDING=0", "--set", "env.VLLM_MOE_MASK_PADDING=1"), "Masking"),
    (("--", "--tensor-parallel-size", "1"), "--tensor-parallel-size"),
])
def test_refusals(w, extra, needle):
    assert needle in w.refused(*extra)


def test_fixed_key_from_env_file(w, tmp_path):
    model = open(os.path.join(w.settings, "model.toml")).read()
    write(os.path.join(w.settings, "model.toml"),
          model + '\n[user.MAX_LEN]\ntarget = "args.max-model-len"\ntype = "int"\n')
    env = write(str(tmp_path / "u.env"), "MAX_LEN=1024\n")
    assert "fixed" in w.refused("--env-file", env)
    d = w.resolve("--env-file", env, "--experiment", "exp-1")
    assert any("experiment" in r for r in d["validated"]["reasons"])


def test_unknown_env_file_key(w, tmp_path):
    env = write(str(tmp_path / "u.env"), "VLLM_THIN_GEMM=0\n")
    assert "not a documented setting" in w.refused("--env-file", env)


def test_layout_cards_exceed_hardware(w):
    write(os.path.join(w.settings, "layouts", "tp4.toml"),
          'schema = "launch-layout/1"\ntp = 4\npp = 1\ncards = 4\n')
    model = open(os.path.join(w.settings, "model.toml")).read().replace(
        'allowed = ["tp1", "tp2", "pp2"]', 'allowed = ["tp1", "tp2", "pp2", "tp4"]')
    write(os.path.join(w.settings, "model.toml"), model)
    assert "hardware has 2" in w.refused(layout="tp4")


def test_card_count_not_tp_times_pp(w):
    write(os.path.join(w.settings, "layouts", "tp2.toml"),
          'schema = "launch-layout/1"\ntp = 2\npp = 1\ncards = 1\n')
    assert "cards 1 != tp 2" in w.refused()


def test_missing_requires_file(w):
    model = open(os.path.join(w.settings, "model.toml")).read()
    write(os.path.join(w.settings, "model.toml"),
          model.replace("[args]\n", '[requires]\nfiles = [{ path = "vllm/_x_C*.so", fork_only = true }]\n\n[args]\n', 1))
    assert "_x_C*.so is missing" in w.refused()
    write(os.path.join(w.tree, "vllm", "_x_C.abi3.so"), "lib")
    d = w.resolve()
    assert list(d["requires"]) == ["vllm/_x_C.abi3.so"]
    assert "unowned" in d["engine_target"]["dirty_paths"]


def test_sm80_capability_needs_fork_envs(tmp_path):
    w = World(tmp_path, sm80=False)
    assert "no vllm/sm80/envs.py" in w.refused()


def test_plain_upstream_on_fork_tree_refused(w):
    assert "carries the sm80 plugin" in w.refused("--capability", "plain-upstream")


def test_plain_upstream_drops_fork_names(tmp_path):
    w = World(tmp_path, sm80=False)
    d = w.resolve("--capability", "plain-upstream", layout="pp2")
    assert sorted(d["fork_switches_absent"]) == ["VLLM_PP_SPLIT", "VLLM_THIN_GEMM"]
    assert "--block-size" not in d["argv"]
    assert d["validated"]["state"] is None and d["validated"]["reasons"] == ["reference"]
    assert d["registry"]["sm80_envs_sha256"] is None


def test_no_option_names_another_registry():
    import main
    with pytest.raises(SystemExit):
        main.parse_args(["--settings", "x", "--launcher-tree", "y"])
    src = open(os.path.join(LAUNCH, "registry.py")).read()
    assert re.search(r"def read_registry\(tree: str, capability: str\)", src)


# -- validated --------------------------------------------------------------------

def test_validated_after_promote(w):
    assert w.verify() == 0
    promote(w, ["tp2", "pp2"])
    d = w.resolve()
    assert d["validated"]["state"] == "VALIDATED", d["validated"]
    assert d["validated"]["functional_match"] is True
    import main
    assert main.banner(d).startswith("VALIDATED synth-dense tp2 (engine ")
    assert w.resolve(layout="tp1")["validated"]["reasons"] == ["layout tp1 not validated"]


def test_cli_change_breaks_functional_match(w):
    assert w.verify() == 0
    promote(w, ["tp2"])
    d = w.resolve("--set", "env.VLLM_FAST_PATH=0")
    assert d["validated"]["functional_match"] is False
    assert "functional ≠ validated" in d["validated"]["reasons"]


def test_harness_overlay_excluded(w, tmp_path):
    assert w.verify() == 0
    promote(w, ["tp2"])
    ov = write(str(tmp_path / "h.toml"),
               'schema = "launch-overlay/1"\nclass = "harness"\n[args]\nenable-prompt-tokens-details = true\n')
    d = w.resolve("--overlay", ov)
    assert d["harness"] == ["--enable-prompt-tokens-details"]
    assert "--enable-prompt-tokens-details" in d["argv"]
    assert d["validated"]["state"] == "VALIDATED"
