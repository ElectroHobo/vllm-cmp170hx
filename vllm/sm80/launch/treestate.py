# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State of the engine tree being launched.

Scope is the whole tree, any depth, any file type. Every untracked or
ignored file is one of:

* build-owned: listed in ``.sm80-build-owned.json`` (schema build-owned/1)
  and compared with its manifest hash (resolved content);
* provenance metadata: exactly ``.sm80-build-owned.json``,
  ``.sm80-build.json`` and, in an image, ``provenance.json``; each is
  trusted only if its content checks out;
* exempt: a path in ``data/tree_exempt.toml``;
* unowned: anything else. An unowned file makes the tree dirty.

Two verification contracts: a tree with Git checks the stamp against
``HEAD``; a Git-free image tree checks it against ``provenance.json``.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field

from canon import HashCache, canonical_sha256, sha256_file
from errors import Refusal
from settings_dir import load_toml

MANIFEST = ".sm80-build-owned.json"
STAMP = ".sm80-build.json"
PROVENANCE = "provenance.json"
VERSION_FILE = "vllm/_version.py"
SOURCES = ("built_in_place", "linked", "adopted")

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def glob_regex(pattern: str) -> re.Pattern:
    """Tree-path glob: no '/' → basename at any depth; '**' spans
    components; '*' and '?' stay inside one component."""
    if "/" not in pattern.rstrip("/"):
        pattern = "**/" + pattern
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("/**", i) and i + 3 == len(pattern):
            out.append("/.*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def load_patterns(name: str, schema: str) -> list[re.Pattern]:
    data = load_toml(os.path.join(_DATA, name), schema)
    return [glob_regex(p) for p in data.get("patterns", [])]


def matches(regexes: list[re.Pattern], rel: str) -> bool:
    return any(r.match(rel) for r in regexes)


def build_owned_sha256(files: dict[str, str]) -> str:
    return canonical_sha256({k: v for k, v in files.items() if k != VERSION_FILE})


def _git(tree: str, *args: str) -> str:
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1",
           "LC_ALL": "C", "HOME": os.environ.get("HOME", "/")}
    proc = subprocess.run(["git", "-C", tree, *args], capture_output=True, env=env,
                          text=True, check=False)
    if proc.returncode != 0:
        raise Refusal(f"git {' '.join(args)} failed in {tree}: {proc.stderr.strip()}")
    return proc.stdout


def _read_json(path: str) -> dict | None:
    import json
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


@dataclass
class TreeState:
    tree: str
    commit: str | None = None
    has_git: bool = False
    dirty: bool = False
    dirty_paths: dict[str, list] = field(default_factory=dict)
    exempt: list[str] = field(default_factory=list)
    source: str | None = None
    build_owned_sha256: str | None = None
    native_manifest_sha256: str | None = None
    manifest_sha256: str | None = None
    linked_from: dict | None = None
    cross_commit_artifacts: bool = False
    adopted: bool = False
    provenance: dict = field(default_factory=dict)

    def add(self, kind: str, item) -> None:
        self.dirty_paths.setdefault(kind, []).append(item)
        if kind != "cross_commit_artifacts":
            self.dirty = True

    def describe(self) -> dict:
        return {
            "tree": self.tree,
            "commit": self.commit,
            "dirty": self.dirty,
            "dirty_paths": self.dirty_paths,
            "exempt": len(self.exempt),
            "cross_commit_artifacts": self.cross_commit_artifacts,
            "build_owned": {
                "source": self.source,
                "build_owned_sha256": self.build_owned_sha256,
                "native_manifest_sha256": self.native_manifest_sha256,
                "manifest_sha256": self.manifest_sha256,
                "linked_from": self.linked_from,
            },
            "adopted": self.adopted,
            "provenance": self.provenance,
        }


def _untracked_files(tree: str) -> list[str]:
    """Every untracked or ignored path (files and symlinks), via Git."""
    out = _git(tree, "ls-files", "--others", "-z")
    return [p for p in out.split("\0") if p]


def _walk_all(tree: str) -> list[str]:
    """Every file under the tree (Git-free image). Symlinks listed as entries."""
    paths = []
    for dirpath, dirnames, filenames in os.walk(tree):
        rel_dir = os.path.relpath(dirpath, tree)
        for name in filenames:
            paths.append(os.path.normpath(os.path.join(rel_dir, name)))
        for d in list(dirnames):
            if os.path.islink(os.path.join(dirpath, d)):
                paths.append(os.path.normpath(os.path.join(rel_dir, d)))
                dirnames.remove(d)
    return paths


def _expand(tree: str, rel: str) -> list[str]:
    """A path, or every file below it when it is a (linked) directory."""
    full = os.path.join(tree, rel)
    if os.path.isdir(full):
        out = []
        for dirpath, _dirs, files in os.walk(full, followlinks=True):
            for name in files:
                out.append(os.path.relpath(os.path.join(dirpath, name), tree))
        return out
    return [rel]


def tree_state(tree: str, *, image: bool = False, cache: HashCache | None = None) -> TreeState:
    """Classify every untracked file of ``tree`` and check its metadata."""
    tree = os.path.abspath(tree)
    cache = cache or HashCache(None)
    st = TreeState(tree=tree)
    exempt_rx = load_patterns("tree_exempt.toml", "tree-exempt")
    st.has_git = os.path.exists(os.path.join(tree, ".git"))
    if image and st.has_git:
        st.add("metadata_invalid", "image tree carries .git")
    if st.has_git:
        st.commit = _git(tree, "rev-parse", "HEAD").strip()
        tracked = _git(tree, "status", "--porcelain", "--untracked-files=no", "-z")
        for entry in tracked.split("\0"):
            if entry.strip():
                st.add("tracked", entry[3:])
        candidates = _untracked_files(tree)
    else:
        # Image contract: the tracked bytes are bound by the image digest;
        # only the owned files and the provenance metadata are checked.
        # A native tree without Git has no checkable state at all.
        candidates = []
        if not image:
            st.add("tracked", "no Git checkout: tree state unknown")
    manifest = _read_json(os.path.join(tree, MANIFEST))
    stamp = _read_json(os.path.join(tree, STAMP))
    prov = _read_json(os.path.join(tree, PROVENANCE)) if image else None
    files: dict[str, str] = {}
    if os.path.exists(os.path.join(tree, MANIFEST)):
        if manifest is None or manifest.get("schema") != "build-owned/1" or \
                manifest.get("source") not in SOURCES or not isinstance(manifest.get("files"), dict):
            st.add("metadata_invalid", f"{MANIFEST}: unreadable or wrong schema")
            manifest = None
        else:
            files = manifest["files"]
            st.source = manifest["source"]
            st.adopted = st.source == "adopted"
            st.native_manifest_sha256 = manifest.get("native_manifest_sha256")
            st.build_owned_sha256 = build_owned_sha256(files)
            st.manifest_sha256 = sha256_file(os.path.join(tree, MANIFEST))
            st.linked_from = manifest.get("linked_from")
            if manifest.get("build_owned_sha256") != st.build_owned_sha256:
                st.add("metadata_invalid", f"{MANIFEST}: build_owned_sha256 does not match its map")
            if st.source == "linked":
                _check_linked(st, manifest, files, exempt_rx)

    provenance_names = {MANIFEST, STAMP} | ({PROVENANCE} if image else set())
    owned_seen: set[str] = set()
    for rel in sorted(candidates):
        if rel in provenance_names:
            continue
        if matches(exempt_rx, rel):
            st.exempt.append(rel)
            continue
        for sub in _expand(tree, rel):
            if matches(exempt_rx, sub):
                st.exempt.append(sub)
                continue
            expected = files.get(sub)
            if expected is None:
                st.add("unowned", {"path": sub, "sha256": _safe_hash(cache, tree, sub)})
                continue
            owned_seen.add(sub)
            actual = _safe_hash(cache, tree, sub)
            if actual != expected:
                st.add("modified_owned", {"path": sub, "sha256": actual, "expected": expected})
    for rel in sorted(set(files) - owned_seen):
        if not os.path.exists(os.path.join(tree, rel)):
            st.add("modified_owned", {"path": rel, "sha256": None, "expected": files[rel]})
        elif image or rel not in candidates:
            # an owned path Git tracks (or an image file) still must match
            actual = _safe_hash(cache, tree, rel)
            if actual != files[rel]:
                st.add("modified_owned", {"path": rel, "sha256": actual, "expected": files[rel]})

    if image and manifest is None:
        st.add("metadata_invalid", f"{MANIFEST}: missing in an image tree")
    # build stamp
    if image or stamp is not None or os.path.exists(os.path.join(tree, STAMP)):
        _check_stamp(st, stamp, manifest, image, prov)
    if image:
        if prov is None:
            st.add("metadata_invalid", f"{PROVENANCE}: missing or unreadable")
        else:
            st.provenance["engine_commit"] = prov.get("engine_commit")
            if st.commit is None:
                st.commit = prov.get("engine_commit")
    for name in sorted(provenance_names):
        p = os.path.join(tree, name)
        if os.path.isfile(p):
            st.provenance[name] = sha256_file(p)
    return st


def _safe_hash(cache: HashCache, tree: str, rel: str) -> str | None:
    try:
        return cache.sha256(os.path.join(tree, rel))
    except OSError:
        return None


def _check_linked(st: TreeState, manifest: dict, files: dict, exempt_rx) -> None:
    lf = manifest.get("linked_from")
    if not isinstance(lf, dict) or not lf.get("tree") or not lf.get("commit"):
        st.add("metadata_invalid", f"{MANIFEST}: linked without linked_from")
        return
    src_manifest = _read_json(os.path.join(lf["tree"], MANIFEST))
    if src_manifest is None or src_manifest.get("schema") != "build-owned/1":
        st.add("metadata_invalid", f"linked_from {lf['tree']}: no build-owned manifest")
    else:
        src_files = src_manifest.get("files", {})
        if build_owned_sha256(src_files) != lf.get("build_owned_sha256"):
            st.add("metadata_invalid", "linked_from: source manifest changed since the link (re-link)")
        for rel, sha in files.items():
            if rel in src_files and src_files[rel] != sha:
                st.add("metadata_invalid", f"{rel}: manifest hash differs from the source tree's")
    if st.commit and lf["commit"] != st.commit:
        changed = _git(st.tree, "diff", "--name-only", f"{lf['commit']}..{st.commit}").split("\n")
        inputs = load_patterns("build_inputs.toml", "build-inputs")
        hits = [p for p in changed if p and matches(inputs, p)]
        if hits:
            st.cross_commit_artifacts = True
            st.dirty_paths.setdefault("cross_commit_artifacts", []).extend(hits[:50])


def _check_stamp(st: TreeState, stamp: dict | None, manifest: dict | None, image: bool,
                 prov: dict | None) -> None:
    if stamp is None or stamp.get("schema") != "build-stamp/1":
        st.add("metadata_invalid", f"{STAMP}: unreadable or wrong schema")
        return
    if image:
        if prov is None or stamp.get("commit") != prov.get("engine_commit"):
            st.add("metadata_invalid", f"{STAMP}: commit differs from {PROVENANCE}")
    elif stamp.get("commit") != st.commit:
        st.add("metadata_invalid", f"{STAMP}: commit differs from HEAD")
    if manifest is None:
        st.add("metadata_invalid", f"{STAMP}: present without {MANIFEST}")
        return
    if stamp.get("build_owned_sha256") != st.build_owned_sha256:
        st.add("metadata_invalid", f"{STAMP}: build_owned_sha256 differs from the manifest (stamp ≠ manifest)")
    if stamp.get("native_manifest_sha256") != st.native_manifest_sha256:
        st.add("metadata_invalid", f"{STAMP}: native_manifest_sha256 differs from the manifest")
    if image and st.commit is None:
        st.commit = stamp.get("commit")
