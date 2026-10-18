# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-checkpoint bytes for the plain tensors an export stored lossily.

An override sidecar (built by tools/sm80/build_override_sidecar.py) is a
safetensors file plus a JSON manifest naming the source repository and
revision and, per tensor, its name, shape, source shard and dtype, the
export's dtype and the sha256 of the source bytes. VLLM_EXL3_OVERRIDE names
the manifest and VLLM_EXL3_OVERRIDE_SHA256 pins it. Every listed tensor is
verified and then read from the sidecar, in the source dtype, wherever the
loader would have read the export's copy.
"""

import hashlib
import json
from pathlib import Path

from vllm.logger import init_logger
from vllm.sm80.kernels.exl3.utils import (
    Exl3Manifest,
    _unique_object,
    read_headers,
)

logger = init_logger(__name__)
FORMAT = "override-sidecar/1"
_PACKED_SUFFIXES = (".trellis", ".suh", ".svh", ".mcg", ".mul1")


def apply_override(manifest: Exl3Manifest, path: str, sha256: str) -> list[str]:
    """Point ``manifest`` at the sidecar's tensors; return the replaced names.

    Refuses, before replacing anything, a manifest that differs from the pin,
    a sidecar whose tensors differ from the manifest's list, a tensor the
    export lacks or holds at another dtype or shape than recorded, a packed
    tensor, and any tensor whose bytes do not hash to the recorded sha256.
    """
    if not sha256:
        raise ValueError("VLLM_EXL3_OVERRIDE needs VLLM_EXL3_OVERRIDE_SHA256")
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256.lower():
        raise ValueError(
            f"Override manifest sha256 {digest} differs from the pinned {sha256}"
        )
    spec = json.loads(raw, object_pairs_hook=_unique_object)
    if spec.get("format") != FORMAT:
        raise ValueError(f"Unsupported override manifest format: {spec.get('format')}")
    entries = spec["tensors"]
    names = [entry["name"] for entry in entries]
    sidecar = read_headers(manifest_path.parent / spec["sidecar"])
    if len(set(names)) != len(names) or set(names) != sidecar.keys():
        raise ValueError("Override sidecar and manifest list different tensors")
    for entry in entries:
        name = entry["name"]
        shape = tuple(entry["shape"])
        export = manifest.tensors.get(name)
        if export is None:
            raise ValueError(f"Override tensor is not in the checkpoint: {name}")
        if name.endswith(_PACKED_SUFFIXES):
            raise ValueError(f"Override tensor is a packed EXL3 part: {name}")
        if export.dtype != entry["export_dtype"] or export.shape != shape:
            raise ValueError(
                f"Checkpoint tensor differs from the override manifest: {name} "
                f"is {export.dtype}{list(export.shape)}, recorded "
                f"{entry['export_dtype']}{list(shape)}"
            )
        source = sidecar[name]
        if source.dtype != entry["source_dtype"] or source.shape != shape:
            raise ValueError(f"Override sidecar tensor differs from manifest: {name}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Override tensor sha256 mismatch: {name}")
    for entry in entries:
        manifest.tensors[entry["name"]] = sidecar[entry["name"]]
    logger.info(
        "EXL3 override from %s@%s: %d tensors loaded at source dtype: %s",
        spec["source"]["repo"],
        spec["source"]["revision"],
        len(entries),
        ", ".join(
            f"{e['name']} {e['export_dtype']}->{e['source_dtype']}" for e in entries
        ),
    )
    return names
