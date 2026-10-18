#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build an override sidecar: source-checkpoint tensors an export stored lossily.

An export (EXL3 or any safetensors re-encode) may store unquantised tensors at
a dtype that cannot hold every source value (an FP32 router bias as FP16, a
BF16 gate as FP16). This tool copies those tensors' bytes from the pinned
source checkpoint into one safetensors sidecar and writes a manifest beside
it; the loader (VLLM_EXL3_OVERRIDE, pinned by VLLM_EXL3_OVERRIDE_SHA256)
verifies the manifest and every tensor's sha256 and loads the source bytes in
place of the export's.

    python tools/sm80/build_override_sidecar.py \\
        --repo ORG/NAME --revision COMMIT --export EXPORT_DIR --out OUT_DIR \\
        (--rule lossy-dtype | --names NAMES_FILE) [--source-dir LOCAL_SOURCE]

--rule lossy-dtype selects every unquantised tensor present under the same
name and shape in both checkpoints whose export dtype differs from the source
dtype and does not represent every source value. --names takes one tensor name
per line instead. The source is read from the repository's resolve URL at the
pinned revision with HTTP Range requests (only headers and selected tensors),
or from --source-dir, a local copy of that revision. Only the standard
library is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

FORMAT = "override-sidecar/1"
SIDECAR = "override.safetensors"
MANIFEST = "override.json"
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "F32": 4,
    "I64": 8,
    "F64": 8,
}
# (source, export) dtype pairs whose export holds every source value exactly.
EXACT = {
    ("F16", "F32"),
    ("BF16", "F32"),
    ("F16", "F64"),
    ("BF16", "F64"),
    ("F32", "F64"),
}
# Quantised-tensor parts: never plain tensors.
PACKED_SUFFIXES = (".trellis", ".suh", ".svh", ".mcg", ".mul1")


class Source:
    """Byte ranges of the source checkpoint's files at one revision."""

    def __init__(self, repo: str, revision: str, local: Path | None):
        self.base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        self.local = local
        self.headers: dict[str, tuple[dict, int]] = {}

    def read(self, filename: str, start: int, end: int) -> bytes:
        """Bytes [start, end) of one file."""
        if end <= start:
            return b""
        if self.local is not None:
            with (self.local / filename).open("rb") as file:
                file.seek(start)
                data = file.read(end - start)
        else:
            request = urllib.request.Request(
                self.base + filename, headers={"Range": f"bytes={start}-{end - 1}"}
            )
            with urllib.request.urlopen(request) as response:
                if response.status != 206:
                    raise ValueError(f"{filename}: range request not honoured")
                data = response.read()
        if len(data) != end - start:
            raise ValueError(f"{filename}: short read at {start}")
        return data

    def index(self) -> dict[str, str]:
        if self.local is not None:
            raw = (self.local / "model.safetensors.index.json").read_bytes()
        else:
            with urllib.request.urlopen(
                self.base + "model.safetensors.index.json"
            ) as response:
                raw = response.read()
        return json.loads(raw)["weight_map"]

    def header(self, filename: str) -> tuple[dict, int]:
        """A shard's parsed header and the offset of its payload."""
        if filename not in self.headers:
            length = struct.unpack("<Q", self.read(filename, 0, 8))[0]
            header = json.loads(self.read(filename, 8, 8 + length))
            self.headers[filename] = (header, 8 + length)
        return self.headers[filename]

    def tensor(self, filename: str, name: str) -> tuple[dict, bytes]:
        header, start = self.header(filename)
        entry = header[name]
        begin, end = entry["data_offsets"]
        data = self.read(filename, start + begin, start + end)
        size = DTYPE_BYTES[entry["dtype"]]
        for dim in entry["shape"]:
            size *= dim
        if len(data) != size:
            raise ValueError(f"{name}: extent does not match dtype and shape")
        return entry, data


def export_headers(directory: Path) -> dict[str, tuple[str, dict]]:
    """Every tensor of a local safetensors checkpoint: name -> (file, entry)."""
    tensors = {}
    for path in sorted(directory.glob("*.safetensors")):
        with path.open("rb") as file:
            length = struct.unpack("<Q", file.read(8))[0]
            header = json.loads(file.read(length))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"Duplicate export tensor: {name}")
            tensors[name] = (path.name, entry)
    if not tensors:
        raise ValueError(f"No safetensors files in {directory}")
    return tensors


def select(
    rule: str | None,
    names: list[str] | None,
    source_map: dict[str, str],
    export: dict[str, tuple[str, dict]],
    source: Source,
) -> list[str]:
    if names is not None:
        for name in names:
            if name not in source_map or name not in export:
                raise ValueError(f"{name}: not in both checkpoints")
            if name.endswith(PACKED_SUFFIXES):
                raise ValueError(f"{name}: a quantised part, not a plain tensor")
        return sorted(set(names))
    if rule != "lossy-dtype":
        raise ValueError(f"Unknown rule: {rule}")
    chosen = []
    for name in sorted(source_map.keys() & export.keys()):
        if name.endswith(PACKED_SUFFIXES):
            continue
        entry = source.header(source_map[name])[0][name]
        exported = export[name][1]
        if entry["dtype"] == exported["dtype"]:
            continue
        if (entry["dtype"], exported["dtype"]) in EXACT:
            continue
        if entry["shape"] != exported["shape"]:
            raise ValueError(
                f"{name}: lossy dtype {entry['dtype']} -> {exported['dtype']} "
                f"with a different shape {entry['shape']} -> {exported['shape']}"
            )
        chosen.append(name)
    return chosen


def write_safetensors(path: Path, tensors: dict[str, tuple[dict, bytes]], meta) -> None:
    header: dict = {"__metadata__": meta}
    offset = 0
    for name, (entry, data) in tensors.items():
        header[name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    partial = path.with_suffix(".partial")
    with partial.open("wb") as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        for _, data in tensors.values():
            file.write(data)
    partial.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict:
    source = Source(args.repo, args.revision, args.source_dir)
    source_map = source.index()
    export = export_headers(args.export)
    names = None
    if args.names is not None:
        names = [
            line.strip()
            for line in args.names.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    chosen = select(args.rule, names, source_map, export, source)
    if not chosen:
        raise ValueError("No tensor selected")
    tensors = {}
    records = []
    for name in chosen:
        entry, data = source.tensor(source_map[name], name)
        exported = export[name][1]
        if entry["shape"] != exported["shape"]:
            raise ValueError(f"{name}: source and export shapes differ")
        tensors[name] = (entry, data)
        records.append(
            {
                "name": name,
                "shape": entry["shape"],
                "source_shard": source_map[name],
                "source_dtype": entry["dtype"],
                "export_shard": export[name][0],
                "export_dtype": exported["dtype"],
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    args.out.mkdir(parents=True, exist_ok=True)
    sidecar = args.out / SIDECAR
    write_safetensors(
        sidecar,
        tensors,
        {"source_repo": args.repo, "source_revision": args.revision},
    )
    manifest = {
        "format": FORMAT,
        "source": {"repo": args.repo, "revision": args.revision},
        "export": args.export.name,
        "selection": args.rule or f"names:{args.names.name}",
        "sidecar": SIDECAR,
        "sidecar_sha256": sha256_file(sidecar),
        "tensors": records,
    }
    (args.out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="source repository id")
    parser.add_argument("--revision", required=True, help="source commit")
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    how = parser.add_mutually_exclusive_group(required=True)
    how.add_argument("--rule", choices=["lossy-dtype"])
    how.add_argument("--names", type=Path, help="file with one tensor name per line")
    parser.add_argument(
        "--source-dir", type=Path, help="local copy of the source at --revision"
    )
    args = parser.parse_args(argv)
    manifest = build(args)
    for record in manifest["tensors"]:
        print(
            f"{record['name']}\t{record['source_dtype']} (export "
            f"{record['export_dtype']})\t{record['sha256']}"
        )
    print(f"{len(manifest['tensors'])} tensors")
    print(f"{SIDECAR} sha256 {manifest['sidecar_sha256']}")
    print(f"{MANIFEST} sha256 {sha256_file(args.out / MANIFEST)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
