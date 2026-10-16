# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State of the engine tree being launched.

Scope is the whole tree, any depth, any file type. Every file that is not
tracked (natively: by Git; in a Git-free image: by the manifest's
``tracked`` inventory) is one of:

* build-owned: listed in ``.sm80-build-owned.json`` (schema build-owned/1)
  and compared with its manifest hash (resolved content);
* provenance metadata: exactly ``.sm80-build-owned.json``,
  ``.sm80-build.json`` and, in an image, ``provenance.json``; each is
  trusted only if its content checks out;
* exempt: a path in ``data/tree_exempt.toml``;
* unowned: anything else. An unowned file makes the tree dirty.

The owned set is checked against where it came from: a ``linked`` tree may
own only its source tree's files (same hashes) and the link step's own
outputs; a ``built_in_place`` tree must name a retained build record whose
file list equals its map; a non-null native manifest digest must name the
native manifest file, which is re-hashed. Two verification contracts: a
tree with Git checks the stamp against ``HEAD``; a Git-free image tree
checks it against ``provenance.json``.

build-owned/1 fields: schema, source, commit, files, native_manifest_sha256,
native_manifest_path?, build_owned_sha256, linked_from? (linked),
build_record? = {path, sha256} (built_in_place), tracked? (images).
"""

from __future__ import annotations

import json
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
# what the link step itself writes into a linked tree, beside the links
LINK_WRITTEN = ("plugins/sm80/build/**", "plugins/sm80/*.egg-info/**", "/*.dist-info/**",
                "vllm/_version.py", "vllm/_ampere_marlin_C*.so", "vllm/_exl3_C*.so")

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def glob_regex(pattern: str) -> re.Pattern:
    """Tree-path glob: no '/' → basename at any depth; a leading '/' anchors
    at the tree root; '**' spans components; '*' and '?' stay inside one."""
    if pattern.startswith("/"):
        pattern = pattern[1:]
    elif "/" not in pattern.rstrip("/"):
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

    def state_sha256(self) -> str:
        """Identity of the whole tree state (changes when any dirty content does)."""
        return canonical_sha256({"commit": self.commit, "dirty_paths": self.dirty_paths,
                                 "build_owned_sha256": self.build_owned_sha256,
                                 "native_manifest_sha256": self.native_manifest_sha256,
                                 "provenance": self.provenance})

    def describe(self) -> dict:
        return {
            "tree": self.tree,
            "commit": self.commit,
            "dirty": self.dirty,
            "dirty_paths": self.dirty_paths,
            "state_sha256": self.state_sha256(),
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


def _safe_hash(cache: HashCache, path: str) -> str | None:
    try:
        return cache.sha256(path)
    except OSError:
        return None


def _tracked_changes(st: TreeState, tree: str, cache: HashCache) -> None:
    """Tracked changes with their kind and content hash (renames parsed)."""
    raw = _git(tree, "status", "--porcelain=v1", "-z", "--untracked-files=no")
    parts = raw.split("\0")
    i = 0
    while i < len(parts):
        entry = parts[i]
        i += 1
        if len(entry) < 4:
            continue
        xy, path = entry[:2], entry[3:]
        item = {"path": path, "kind": xy.strip() or xy,
                "sha256": _safe_hash(cache, os.path.join(tree, path))}
        if "R" in xy or "C" in xy:
            item["from"] = parts[i] if i < len(parts) else None
            i += 1
        st.add("tracked", item)


def _untracked_files(tree: str) -> list[str]:
    """Every untracked or ignored path (files and symlinks), via Git."""
    out = _git(tree, "ls-files", "--others", "-z")
    return [p for p in out.split("\0") if p]


def _walk_all(tree: str) -> list[str]:
    """Every file under a Git-free tree. Symlinks are listed as entries."""
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
    """Classify every file of ``tree`` and check its metadata."""
    tree = os.path.abspath(tree)
    cache = cache or HashCache(None)
    st = TreeState(tree=tree)
    exempt_rx = load_patterns("tree_exempt.toml", "tree-exempt")
    st.has_git = os.path.exists(os.path.join(tree, ".git"))
    manifest = _read_json(os.path.join(tree, MANIFEST))
    stamp = _read_json(os.path.join(tree, STAMP))
    prov = _read_json(os.path.join(tree, PROVENANCE)) if image else None
    tracked_inventory: dict[str, str] = {}
    if image and st.has_git:
        st.add("metadata_invalid", "image tree carries .git")
    if st.has_git and not image:
        st.commit = _git(tree, "rev-parse", "HEAD").strip()
        _tracked_changes(st, tree, cache)
        candidates = _untracked_files(tree)
    elif image:
        inv = manifest.get("tracked") if manifest else None
        if not isinstance(inv, dict) or not inv:
            st.add("metadata_invalid", f"{MANIFEST}: no tracked inventory in an image tree")
        else:
            tracked_inventory = inv
        candidates = _walk_all(tree)
    else:
        candidates = []
        st.add("tracked", {"path": "", "kind": "no Git checkout: tree state unknown", "sha256": None})

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
            _check_source(st, manifest, files)
            _check_native_manifest(st, manifest, cache)
    elif image:
        st.add("metadata_invalid", f"{MANIFEST}: missing in an image tree")

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
            actual = _safe_hash(cache, os.path.join(tree, sub))
            if sub in files:
                owned_seen.add(sub)
                if actual != files[sub]:
                    st.add("modified_owned", {"path": sub, "sha256": actual, "expected": files[sub]})
            elif sub in tracked_inventory:
                if actual != tracked_inventory[sub]:
                    st.add("tracked", {"path": sub, "kind": "M", "sha256": actual})
            else:
                st.add("unowned", {"path": sub, "sha256": actual})
    for rel in sorted(set(tracked_inventory) - set(_expand_all(tree, candidates))):
        st.add("tracked", {"path": rel, "kind": "D", "sha256": None})
    for rel in sorted(set(files) - owned_seen):
        actual = _safe_hash(cache, os.path.join(tree, rel))
        if actual != files[rel]:
            st.add("modified_owned", {"path": rel, "sha256": actual, "expected": files[rel]})

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


def _expand_all(tree: str, rels: list[str]) -> list[str]:
    out = []
    for rel in rels:
        out += _expand(tree, rel)
    return out


def _resolve(tree: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(tree, path)


def _check_native_manifest(st: TreeState, manifest: dict, cache: HashCache) -> None:
    digest = manifest.get("native_manifest_sha256")
    if digest is None:
        return
    path = manifest.get("native_manifest_path")
    if not isinstance(path, str) or not path:
        st.add("metadata_invalid", f"{MANIFEST}: native_manifest_sha256 without native_manifest_path")
        return
    actual = _safe_hash(cache, _resolve(st.tree, path))
    if actual != digest:
        st.add("metadata_invalid", f"native manifest {path}: sha256 {actual} != {digest}")


def _check_source(st: TreeState, manifest: dict, files: dict) -> None:
    source = manifest["source"]
    if source == "linked":
        _check_linked(st, manifest, files)
    elif source == "built_in_place":
        rec = manifest.get("build_record")
        if not isinstance(rec, dict) or not isinstance(rec.get("path"), str):
            st.add("metadata_invalid", f"{MANIFEST}: built_in_place without a retained build_record")
            return
        path = _resolve(st.tree, rec["path"])
        data = _read_json(path)
        if data is None or data.get("schema") != "build-record/1":
            st.add("metadata_invalid", f"build record {rec['path']}: missing or wrong schema")
        elif sha256_file(path) != rec.get("sha256"):
            st.add("metadata_invalid", f"build record {rec['path']}: sha256 differs from the manifest")
        elif data.get("files") != files:
            st.add("metadata_invalid", "owned map differs from the retained build record")


def _check_linked(st: TreeState, manifest: dict, files: dict) -> None:
    lf = manifest.get("linked_from")
    if not isinstance(lf, dict) or not lf.get("tree") or not lf.get("commit"):
        st.add("metadata_invalid", f"{MANIFEST}: linked without linked_from")
        return
    src_manifest = _read_json(os.path.join(lf["tree"], MANIFEST))
    if src_manifest is None or src_manifest.get("schema") != "build-owned/1":
        st.add("metadata_invalid", f"linked_from {lf['tree']}: no build-owned manifest")
        src_files: dict[str, str] = {}
    else:
        src_files = src_manifest.get("files", {})
        if build_owned_sha256(src_files) != lf.get("build_owned_sha256"):
            st.add("metadata_invalid", "linked_from: source manifest changed since the link (re-link)")
    written = [glob_regex(p) for p in LINK_WRITTEN]
    for rel, sha in files.items():
        if rel in src_files:
            if src_files[rel] != sha:
                st.add("metadata_invalid", f"{rel}: manifest hash differs from the source tree's")
        elif not matches(written, rel):
            st.add("metadata_invalid", f"{rel}: claimed as owned but neither linked nor written by the link step")
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
        st.add("metadata_invalid", f"{STAMP}: missing, unreadable or wrong schema")
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
