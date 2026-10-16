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

The manifest rules are those of the shared provenance verifier
(``build_owned.py verify``), applied to the tree's manifest and, for a
``linked`` tree, to its source manifest against the source tree:

* structure: schema, source, a 40-hex artefact commit, a path → sha256
  map whose hash is ``build_owned_sha256``, no exempt or provenance path in
  the map, ``linked_from`` exactly on linked manifests, the ``adopted``
  block on adopted ones, the ``tracked`` inventory only in an image;
* a non-null native manifest digest names the native manifest file
  (``native_manifest_path``), which is re-hashed; a path without a digest
  is invalid;
* ``built_in_place`` names a retained build record whose file list equals
  the map;
* ``vllm/_version.py``, which ``build_owned_sha256`` leaves out because its
  text names the checkout, holds version metadata only (literal
  assignments to the version names); anything else in it is invalid;
* ``linked``: the source manifest is itself valid and not linked, the link
  recorded its commit and map hash, the tree's artefact commit is the
  link's, every source file under ``vllm/`` the tree does not track is
  linked, the tree owns nothing else but the link step's outputs
  (``data/link_written.toml``), and its native identity is the source's
  unless it carries link-written natives, which must be exactly the
  artefacts of the native manifest it names;
* the build stamp is required and names the tree's commit (natively
  ``HEAD``; in a Git-free image ``provenance.json``) and the manifest's
  hashes;
* artefacts of another commit are compatible only when no build input
  changed between the two commits (``data/build_inputs.toml``).

build-owned/1 fields: schema, source, commit, files, native_manifest_sha256,
native_manifest_path?, build_owned_sha256, linked_from? (linked),
adopted? (adopted), build_record? = {path, sha256} (built_in_place),
tracked? (images).
"""

from __future__ import annotations

import ast
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
LINKED_PREFIX = "vllm/"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
VERSION_NAMES = frozenset({"__all__", "__version__", "version", "__version_tuple__",
                           "version_tuple", "__commit_id__", "commit_id"})
_ANNOTATION_NAMES = frozenset({"str", "int", "tuple"})

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


def _data(name: str, schema: str) -> dict:
    return load_toml(os.path.join(_DATA, name), schema)


def load_patterns(name: str, schema: str) -> list[re.Pattern]:
    return [glob_regex(p) for p in _data(name, schema).get("patterns", [])]


def matches(regexes: list[re.Pattern], rel: str) -> bool:
    return any(r.match(rel) for r in regexes)


def build_owned_sha256(files: dict[str, str]) -> str:
    return canonical_sha256({k: v for k, v in files.items() if k != VERSION_FILE})


def _git_env() -> dict[str, str]:
    return {"PATH": "/usr/local/bin:/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1",
            "LC_ALL": "C", "HOME": os.environ.get("HOME", "/")}


def _git(tree: str, *args: str) -> str:
    proc = subprocess.run(["git", "-C", tree, *args], capture_output=True, env=_git_env(),
                          text=True, check=False)
    if proc.returncode != 0:
        raise Refusal(f"git {' '.join(args)} failed in {tree}: {proc.stderr.strip()}")
    return proc.stdout


def _git_blob(tree: str, commit: str, path: str) -> str:
    """The text of ``path`` at ``commit``; empty when the path is absent there."""
    proc = subprocess.run(["git", "-C", tree, "show", f"{commit}:{path}"], capture_output=True,
                          env=_git_env(), check=False)
    return proc.stdout.decode("utf-8", errors="replace") if proc.returncode == 0 else ""


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
    native = st.has_git and not image
    manifest = _read_json(os.path.join(tree, MANIFEST))
    stamp = _read_json(os.path.join(tree, STAMP))
    prov = _read_json(os.path.join(tree, PROVENANCE)) if image else None
    tracked_inventory: dict[str, str] = {}
    if image and st.has_git:
        st.add("metadata_invalid", "image tree carries .git")
    if native:
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
        problems = manifest_problems(manifest, tree, exempt_rx, image=image, cache=cache)
        for problem in problems:
            st.add("metadata_invalid", f"{MANIFEST}: {problem}")
        if manifest is not None and _is_map(manifest.get("files")):
            files = manifest["files"]
            st.source = manifest.get("source")
            st.adopted = st.source == "adopted"
            st.native_manifest_sha256 = manifest.get("native_manifest_sha256")
            st.build_owned_sha256 = build_owned_sha256(files)
            st.manifest_sha256 = sha256_file(os.path.join(tree, MANIFEST))
            st.linked_from = manifest.get("linked_from")
            if not problems and st.source == "linked":
                _check_linked(st, manifest, exempt_rx, cache)
    elif native or image:
        st.add("metadata_invalid", f"{MANIFEST}: missing")
        manifest = None

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

    if native or image:
        _check_stamp(st, stamp, manifest, image, prov)
    if image:
        if prov is None:
            st.add("metadata_invalid", f"{PROVENANCE}: missing or unreadable")
        else:
            st.provenance["engine_commit"] = prov.get("engine_commit")
            if st.commit is None:
                st.commit = prov.get("engine_commit")
    if native and manifest is not None:
        _check_artefact_commit(st, manifest.get("commit"))
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


def _is_map(files) -> bool:
    return isinstance(files, dict) and all(isinstance(v, str) for v in files.values())


# -- the shared manifest rules -------------------------------------------------------------

def manifest_problems(manifest: dict | None, tree: str, exempt_rx: list[re.Pattern], *,
                      image: bool = False, cache: HashCache | None = None) -> list[str]:
    """Consistency of a build-owned/1 manifest of ``tree`` on its own and against
    the files it names (native manifest, build record, version file); [] when
    it checks out."""
    if manifest is None:
        return ["unreadable or not a JSON object"]
    cache = cache or HashCache(None)
    problems = []
    if manifest.get("schema") != "build-owned/1":
        problems.append(f"schema {manifest.get('schema')!r}")
    if manifest.get("source") not in SOURCES:
        problems.append(f"source {manifest.get('source')!r}")
    if not isinstance(manifest.get("commit"), str) or not HEX40.match(manifest["commit"]):
        problems.append(f"commit {manifest.get('commit')!r} is not a 40-hex commit")
    files = manifest.get("files")
    if not _is_map(files):
        return problems + ["files is not a path -> sha256 map"]
    if manifest.get("build_owned_sha256") != build_owned_sha256(files):
        problems.append("build_owned_sha256 does not match its map")
    provenance = {MANIFEST, STAMP} | ({PROVENANCE} if image else set())
    for rel in files:
        if rel in provenance or matches(exempt_rx, rel):
            problems.append(f"{rel}: exempt or provenance path in the owned map")
    if "tracked" in manifest and not image:
        problems.append("tracked inventory present: image manifests are checked under the image contract")
    problems += _native_problems(manifest, tree, cache)
    if manifest.get("source") == "linked":
        lf = manifest.get("linked_from")
        if not isinstance(lf, dict) or not all(isinstance(lf.get(k), str)
                                               for k in ("tree", "commit", "build_owned_sha256")):
            problems.append("linked without linked_from {tree, commit, build_owned_sha256}")
    elif "linked_from" in manifest:
        problems.append("linked_from on a non-linked manifest")
    if manifest.get("source") == "adopted" and not isinstance(manifest.get("adopted"), dict):
        problems.append("adopted without its adopted block")
    if manifest.get("source") == "built_in_place":
        problems += _build_record_problems(manifest, tree)
    if VERSION_FILE in files:
        problem = version_file_problem(os.path.join(tree, VERSION_FILE))
        if problem:
            problems.append(problem)
    return problems


def _native_problems(manifest: dict, tree: str, cache: HashCache) -> list[str]:
    nsha, npath = manifest.get("native_manifest_sha256"), manifest.get("native_manifest_path")
    if nsha is None:
        return [] if npath is None else ["native_manifest_path without native_manifest_sha256"]
    if not isinstance(nsha, str) or not HEX64.match(nsha):
        return [f"native_manifest_sha256 {nsha!r} is not a sha256"]
    if not isinstance(npath, str) or not npath:
        return ["native_manifest_sha256 without native_manifest_path"]
    actual = _safe_hash(cache, _resolve(tree, npath))
    return [] if actual == nsha else [f"native manifest {npath}: sha256 {actual} is not {nsha}"]


def _build_record_problems(manifest: dict, tree: str) -> list[str]:
    rec = manifest.get("build_record")
    if not isinstance(rec, dict) or not isinstance(rec.get("path"), str) \
            or not isinstance(rec.get("sha256"), str):
        return ["built_in_place without build_record {path, sha256}"]
    path = _resolve(tree, rec["path"])
    try:
        actual = sha256_file(path)
    except OSError:
        actual = None
    if actual != rec["sha256"]:
        return [f"build record {rec['path']}: sha256 {actual} is not {rec['sha256']}"]
    data = _read_json(path)
    if data is None or data.get("schema") != "build-record/1" or not isinstance(data.get("files"), dict):
        return [f"build record {rec['path']}: not a build-record/1 file list"]
    if data["files"] != manifest["files"]:
        return [f"build record {rec['path']}: files differ from the owned map"]
    return []


def _literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is None or type(node.value) in (str, int)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_literal(e) for e in node.elts)
    return False


def _annotation(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _ANNOTATION_NAMES
    if isinstance(node, ast.Constant):
        return node.value is None or node.value is Ellipsis
    if isinstance(node, ast.Subscript):
        return _annotation(node.value) and _annotation(node.slice)
    if isinstance(node, ast.Tuple):
        return all(_annotation(e) for e in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation(node.left) and _annotation(node.right)
    return False


def _version_statement(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Expr):
        return isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)
    if isinstance(stmt, ast.ImportFrom):
        return stmt.module == "__future__" and stmt.level == 0 and \
            [(a.name, a.asname) for a in stmt.names] == [("annotations", None)]
    if isinstance(stmt, ast.Assign):
        return all(isinstance(t, ast.Name) and t.id in VERSION_NAMES for t in stmt.targets) \
            and _literal(stmt.value)
    if isinstance(stmt, ast.AnnAssign):
        return stmt.simple == 1 and isinstance(stmt.target, ast.Name) \
            and stmt.target.id in VERSION_NAMES and _annotation(stmt.annotation) \
            and (stmt.value is None or _literal(stmt.value))
    return False


def version_file_problem(path: str) -> str | None:
    """The version file is left out of build_owned_sha256 because its text
    names the checkout, so it may hold version metadata only: a docstring,
    ``from __future__ import annotations``, and literal assignments (with
    plain annotations) to the version names. Anything else could run code
    the build identity does not cover. An unreadable file is reported as a
    modified owned file."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    try:
        module = ast.parse(raw.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError, ValueError) as exc:
        return f"{VERSION_FILE}: not version metadata ({type(exc).__name__})"
    for stmt in module.body:
        if not _version_statement(stmt):
            return (f"{VERSION_FILE}: not version metadata (line {stmt.lineno}; only literal "
                    f"assignments to {', '.join(sorted(VERSION_NAMES))})")
    return None


def load_source_manifest(src: str, exempt_rx: list[re.Pattern], cache: HashCache) -> dict:
    """The manifest of a linked tree's source tree, valid on its own and not linked."""
    if not os.path.exists(os.path.join(src, MANIFEST)):
        raise Refusal(f"source tree {src}: {MANIFEST} missing")
    manifest = _read_json(os.path.join(src, MANIFEST))
    problems = manifest_problems(manifest, src, exempt_rx, cache=cache)
    if problems:
        raise Refusal(f"source tree {src}: {MANIFEST} " + "; ".join(problems))
    if manifest["source"] == "linked":
        raise Refusal(f"source tree {src} is itself a linked tree")
    return manifest


def _is_native(rel: str, written: list[re.Pattern]) -> bool:
    """A link-written compiled extension (the fork natives the native build installs)."""
    return rel.endswith(".so") and matches(written, rel)


def _check_linked(st: TreeState, manifest: dict, exempt_rx: list[re.Pattern],
                  cache: HashCache) -> None:
    lf = manifest["linked_from"]
    files = manifest["files"]
    try:
        srcm = load_source_manifest(lf["tree"], exempt_rx, cache)
    except Refusal as exc:
        st.add("metadata_invalid", f"linked_from: {exc}")
        return
    src_files = srcm["files"]
    if lf["build_owned_sha256"] != srcm["build_owned_sha256"] or lf["commit"] != srcm["commit"]:
        st.add("metadata_invalid", "linked_from: the source manifest changed since the link (re-link)")
    if manifest["commit"] != lf["commit"]:
        st.add("metadata_invalid", f"{MANIFEST}: commit {manifest['commit']} is not linked_from.commit")
    written = load_patterns("link_written.toml", "link-written")
    natives = {rel: sha for rel, sha in files.items() if rel not in src_files and _is_native(rel, written)}
    if natives:
        _check_link_natives(st, manifest, natives)
    elif manifest.get("native_manifest_sha256") != srcm.get("native_manifest_sha256"):
        st.add("metadata_invalid", f"{MANIFEST}: native_manifest_sha256 differs from the source tree's")
    for rel, sha in files.items():
        if rel in src_files:
            if src_files[rel] != sha:
                st.add("metadata_invalid", f"{rel}: manifest hash differs from the source tree's")
        elif not matches(written, rel):
            st.add("metadata_invalid", f"{rel}: claimed as owned but neither linked nor written by the link step")
    if st.has_git:
        tracked = set(_git(st.tree, "ls-files", "-z").split("\0"))
        missing = sorted(rel for rel in src_files
                         if rel.startswith(LINKED_PREFIX) and rel not in tracked and rel not in files)
        for rel in missing[:50]:
            st.add("metadata_invalid", f"{rel}: in the source tree but not linked")


def _check_link_natives(st: TreeState, manifest: dict, natives: dict[str, str]) -> None:
    """Natives the link step installed are exactly the artefacts of the native
    manifest the tree names (whose digest is checked with the manifest)."""
    npath = manifest.get("native_manifest_path")
    if manifest.get("native_manifest_sha256") is None or not isinstance(npath, str):
        st.add("metadata_invalid", f"{MANIFEST}: link-written natives without a native manifest")
        return
    data = _read_json(_resolve(st.tree, npath))
    arts = data.get("artefacts") if data else None
    if not isinstance(arts, dict) or not all(isinstance(e, dict) for e in arts.values()):
        st.add("metadata_invalid", f"native manifest {npath}: no artefacts table")
        return
    recorded = {rel: e.get("sha256") for rel, e in arts.items()}
    if recorded != natives:
        diff = sorted(set(recorded.items()) ^ set(natives.items()))
        st.add("metadata_invalid", f"native manifest {npath}: artefacts differ from the linked natives at "
               + ", ".join(sorted({str(p) for p, _ in diff})[:20]))


def _env_entries(text: str, names: set[str]) -> list | None:
    """(name, value source) of every dict entry keyed by one of ``names``;
    None when the text does not parse."""
    try:
        module = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    out = []
    for node in ast.walk(module):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value in names:
                    out.append((k.value, ast.get_source_segment(text, v)))
    return sorted(out)


def _check_artefact_commit(st: TreeState, art) -> None:
    """Artefacts built at another commit: compatible only when no build input
    changed in between (data/build_inputs.toml: path patterns, and for the
    environment modules the entries of the build variables)."""
    if not isinstance(art, str) or not HEX40.match(art) or st.commit is None or art == st.commit:
        return
    try:
        changed = _git(st.tree, "diff", "--name-only", f"{art}..{st.commit}").split("\n")
    except Refusal:
        st.add("metadata_invalid", f"artefact commit {art} is not in this repository")
        return
    data = _data("build_inputs.toml", "build-inputs")
    inputs = [glob_regex(p) for p in data.get("patterns", [])]
    env = data.get("env_entries", {})
    env_paths, names = set(env.get("paths", [])), set(env.get("names", []))
    hits = []
    for p in changed:
        if not p:
            continue
        if matches(inputs, p):
            hits.append(p)
        elif p in env_paths:
            old = _env_entries(_git_blob(st.tree, art, p), names)
            new = _env_entries(_git_blob(st.tree, st.commit, p), names)
            if old is None or new is None or old != new:
                hits.append(p)
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
    if manifest is None or not _is_map(manifest.get("files")):
        return
    if stamp.get("build_owned_sha256") != st.build_owned_sha256:
        st.add("metadata_invalid", f"{STAMP}: build_owned_sha256 differs from the manifest (stamp ≠ manifest)")
    if stamp.get("native_manifest_sha256") != st.native_manifest_sha256:
        st.add("metadata_invalid", f"{STAMP}: native_manifest_sha256 differs from the manifest")
    if image and st.commit is None:
        st.commit = stamp.get("commit")
