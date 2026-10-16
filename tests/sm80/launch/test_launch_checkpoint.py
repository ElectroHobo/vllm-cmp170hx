# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint completeness, identity, stamps and the verify command."""

from __future__ import annotations

import builtins
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout

import pytest

import checkpoint as ck
from helpers import CHECKPOINT_FILES, LAUNCH, World, make_checkpoint, promote, write
from model import load_model_settings
from settings_dir import SettingsDir


@pytest.fixture
def w(tmp_path):
    return World(tmp_path)


class OpenCounter:
    """Counts opens of files under one directory (builtins.open and os.open)."""

    def __init__(self, root: str) -> None:
        self.root = os.path.realpath(root)
        self.hits: list[str] = []

    def __enter__(self):
        self._open, self._osopen = builtins.open, os.open
        counter = self

        def bopen(file, *a, **k):
            counter._see(file)
            return counter._open(file, *a, **k)

        def oopen(path, *a, **k):
            counter._see(path)
            return counter._osopen(path, *a, **k)
        builtins.open, os.open = bopen, oopen
        return self

    def _see(self, path) -> None:
        if isinstance(path, (str, bytes, os.PathLike)):
            p = os.path.realpath(os.fsdecode(path))
            if p.startswith(self.root + os.sep):
                self.hits.append(p)

    def __exit__(self, *exc):
        builtins.open, os.open = self._open, self._osopen


def verify_cli(w, *extra) -> tuple[int, dict]:
    out = io.StringIO()
    with redirect_stdout(out):
        code = ck.main(["verify", "--settings", w.settings, "--models-dir", w.models,
                        "--stamp-dir", w.stamps, *extra])
    return code, json.loads(out.getvalue() or "{}")


def same_size_edit(path: str) -> None:
    data = open(path).read()
    write(path, data[:-1] + ("X" if data[-1] != "X" else "Y"))


def test_print_config_unstamped_reads_no_checkpoint_file(w):
    with OpenCounter(w.cp) as c:
        d = w.resolve()
    assert d["checkpoint"]["target"]["identity"] == "unverified"
    assert "checkpoint not verified" in d["validated"]["reasons"]
    assert c.hits == []


def test_verify_then_stamp_boot_hashes_nothing(w):
    code, rep = verify_cli(w)
    assert code == 0 and rep["target"]["state"] == "verified" and rep["target"]["hashed"] == 8
    import main
    doc = main.build(main.parse_args(w.base_args()), w.env())
    with OpenCounter(w.cp) as c:
        main.preflight_checkpoints(doc, "strict")
    assert c.hits == []
    assert w.resolve()["checkpoint"]["target"]["identity"] == "verified"


def test_verify_without_gpu(w):
    env = {"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HOME": w.home}
    p = subprocess.run([sys.executable, "-I", os.path.join(LAUNCH, "checkpoint.py"), "verify",
                        "--settings", w.settings, "--models-dir", w.models, "--stamp-dir", w.stamps],
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["target"]["state"] == "verified"


def test_mismatch_and_incomplete_exits(w):
    path = os.path.join(w.cp, "tokenizer.json")
    original = open(path).read()
    write(path, original + " ")
    code, rep = verify_cli(w)
    assert code == ck.EXIT_CHECKPOINT_INCOMPLETE and "tokenizer.json: size" in rep["target"]["files"][0]
    write(path, original)
    same_size_edit(path)
    code, rep = verify_cli(w)
    assert code == ck.EXIT_CHECKPOINT_MISMATCH and rep["target"]["files"] == ["tokenizer.json"]


@pytest.mark.parametrize("rel", ["chat_template.jinja", "tokenizer.json"])
def test_altered_file_is_mismatch(w, rel):
    assert verify_cli(w)[0] == 0
    same_size_edit(os.path.join(w.cp, rel))
    d = w.resolve()
    assert d["checkpoint"]["target"]["identity"] == "unverified"       # stat changed
    code, rep = verify_cli(w)
    assert code == ck.EXIT_CHECKPOINT_MISMATCH and rep["target"]["files"] == [rel]
    d = w.resolve()
    assert d["checkpoint"]["target"]["identity"] == "mismatch"
    p = w.run()
    assert p.returncode == 3 and "differs from its manifest" in p.stderr
    p = w.run("--policy", "merge")
    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize("case", ["empty", "truncated", "missing_tokenizer", "incomplete_blob"])
def test_incomplete_refused_every_mode(w, case):
    if case == "empty":
        write(os.path.join(w.cp, "model-00001-of-00002.safetensors"), "")
    elif case == "truncated":
        write(os.path.join(w.cp, "model-00001-of-00002.safetensors"), "A" * 10)
    elif case == "missing_tokenizer":
        os.unlink(os.path.join(w.cp, "tokenizer.json"))
    else:
        write(os.path.join(w.cp, "blob.incomplete"), "x")
    assert verify_cli(w)[0] == ck.EXIT_CHECKPOINT_INCOMPLETE
    for policy, mode in (("strict", "stamp"), ("merge", "boot")):
        p = w.run("--policy", policy, "--checkpoint-verify", mode)
        assert p.returncode == 3 and "not complete" in p.stderr
    d = w.resolve()
    assert d["checkpoint"]["target"]["state"] == "incomplete"
    assert "checkpoint not present" in d["validated"]["reasons"]


def test_other_revision_is_mismatch(w, tmp_path):
    files = dict(CHECKPOINT_FILES)
    files["config.json"] = files["config.json"].replace("synth", "SYNTH")
    other = make_checkpoint(str(tmp_path / "other"), files)
    code, rep = verify_cli(w, "--checkpoint", "target", "--path", other)
    assert code == ck.EXIT_CHECKPOINT_MISMATCH and rep["target"]["files"] == ["config.json"]
    d = w.resolve("--checkpoint-path", f"target={other}")
    assert d["checkpoint"]["target"]["identity"] == "mismatch"
    assert "checkpoint not verified" in d["validated"]["reasons"]
    assert w.run("--checkpoint-path", f"target={other}").returncode == 3


def test_not_present(w, tmp_path):
    code, _ = verify_cli(w, "--checkpoint", "target", "--path", str(tmp_path / "nope"))
    assert code == ck.EXIT_CHECKPOINT_ABSENT
    d = w.resolve("--checkpoint-path", f"target={tmp_path / 'nope'}")
    assert d["checkpoint"]["target"]["state"] == "not present"
    assert "checkpoint not present" in d["validated"]["reasons"]


def test_indexless_single_file(tmp_path):
    d = make_checkpoint(str(tmp_path / "drafter"), {"config.json": "{}", "model.safetensors": "W" * 32})
    os.unlink(os.path.join(d, "model.safetensors.index.json"))
    m = ck.make_manifest("r", "v", {"required": ["config.json", "model.safetensors"]}, d)
    assert sorted(m["files"]) == ["config.json", "model.safetensors"]
    sha = ck.sha256_bytes(json.dumps(m).encode())
    state, bad, _ = ck.verify(m, sha, d, str(tmp_path / "s.json"))
    assert state == ck.VERIFIED and bad == []


def test_stale_stamp_policies(w):
    assert verify_cli(w)[0] == 0
    shard = os.path.join(w.cp, "model-00002-of-00002.safetensors")
    os.utime(shard, ns=(1, 1))
    p = w.run()
    assert p.returncode == 3 and "checkpoint.py verify" in p.stderr
    import main
    doc = main.build(main.parse_args(w.base_args("--checkpoint-verify", "boot")), w.env())
    with OpenCounter(w.cp) as c:
        main.preflight_checkpoints(doc, "strict")
    assert c.hits == [os.path.realpath(shard)]


def test_interrupt_writes_no_stamp(w):
    m, sha = ck.load_manifest(os.path.join(w.settings, "checkpoints", "synth-dense-1b.manifest.json"))
    sfile = ck.stamp_path(w.stamps, w.cp)
    with pytest.raises(ck.Interrupted):
        ck.verify(m, sha, w.cp, sfile, stop=[True])
    assert not os.path.exists(sfile)


def test_new_pin_with_verified_weights_fails_receipt(w, tmp_path):
    assert verify_cli(w)[0] == 0
    promote(w, ["tp2"])
    assert w.resolve()["validated"]["state"] == "VALIDATED"
    files = dict(CHECKPOINT_FILES)
    files["config.json"] = files["config.json"].replace("synth", "rev02")
    other = make_checkpoint(str(tmp_path / "rev2"), files)
    pins = open(os.path.join(w.settings, "pins.env")).read().replace(
        "TARGET_REVISION=0123", "TARGET_REVISION=9123")
    write(os.path.join(w.settings, "pins.env"), pins)
    spec = load_model_settings(SettingsDir(w.settings)).checkpoints["target"]
    revision = pins.split("TARGET_REVISION=")[1].strip()
    m = ck.make_manifest("example/synth-dense-1b", revision, spec, other)
    write(os.path.join(w.settings, "checkpoints", "synth-dense-1b.manifest.json"),
          json.dumps(m, indent=1, sort_keys=True) + "\n")
    assert verify_cli(w, "--checkpoint", "target", "--path", other)[0] == 0
    d = w.resolve("--checkpoint-path", f"target={other}")
    assert d["checkpoint"]["target"]["identity"] == "verified"
    # TARGET_REVISION is a receipt pin, so criterion 2 fires beside 4a
    assert set(d["validated"]["reasons"]) == {"pin ≠ receipt", "checkpoint ≠ receipt"}
