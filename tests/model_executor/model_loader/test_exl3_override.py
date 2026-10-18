# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Override sidecars: the builder selects the lossy plain tensors, the loader
reads the source bytes in the source dtype and refuses any hash mismatch."""

import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from vllm.sm80.kernels.exl3.override import apply_override
from vllm.sm80.kernels.exl3.utils import Exl3Manifest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "build_override_sidecar", REPO / "tools" / "sm80" / "build_override_sidecar.py"
)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)
REVISION = "0123456789abcdef0123456789abcdef01234567"


def _raw(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


_DTYPE = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16"}


def _write(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header, payload = {}, bytearray()
    for name, tensor in tensors.items():
        start = len(payload)
        payload.extend(_raw(tensor))
        header[name] = {
            "dtype": _DTYPE[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


@pytest.fixture
def checkpoints(tmp_path):
    """A two-shard source and an export that stored two tensors lossily."""
    generator = torch.Generator().manual_seed(0)

    def rand(*shape, dtype):
        return (torch.randn(*shape, generator=generator) * 7).to(dtype)

    source = {
        "layers.0.ffn.gate.bias": rand(16, dtype=torch.float32),
        "layers.0.ffn.gate.weight": rand(16, 8, dtype=torch.bfloat16),
        "layers.0.attn_norm.weight": rand(8, dtype=torch.bfloat16),
        "embed.weight": rand(32, 8, dtype=torch.bfloat16),
        "layers.0.attn.wq.weight": rand(8, 8, dtype=torch.bfloat16),
    }
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    shards = {
        "model-1.safetensors": ["layers.0.ffn.gate.bias", "embed.weight"],
        "model-2.safetensors": [
            "layers.0.ffn.gate.weight",
            "layers.0.attn_norm.weight",
            "layers.0.attn.wq.weight",
        ],
    }
    for shard, names in shards.items():
        _write(source_dir / shard, {name: source[name] for name in names})
    (source_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {n: shard for shard, ns in shards.items() for n in ns}}
        )
    )
    # layers.0.attn.wq.weight is quantised in an export: no plain tensor of
    # that name.
    export = {
        "layers.0.ffn.gate.bias": source["layers.0.ffn.gate.bias"].half(),
        "layers.0.ffn.gate.weight": source["layers.0.ffn.gate.weight"].half(),
        "layers.0.attn_norm.weight": source["layers.0.attn_norm.weight"],
        # Widened: holds every source value, so no override.
        "embed.weight": source["embed.weight"].float(),
    }
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write(export_dir / "model.safetensors", export)
    out = tmp_path / "override"
    assert (
        builder.main(
            [
                "--repo",
                "org/model",
                "--revision",
                REVISION,
                "--export",
                str(export_dir),
                "--out",
                str(out),
                "--rule",
                "lossy-dtype",
                "--source-dir",
                str(source_dir),
            ]
        )
        == 0
    )
    path = out / builder.MANIFEST
    return source, export_dir, path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_builder_selects_every_lossy_plain_tensor(checkpoints):
    source, _, path, _ = checkpoints
    spec = json.loads(path.read_text())
    assert spec["source"] == {"repo": "org/model", "revision": REVISION}
    assert [
        (t["name"], t["source_dtype"], t["export_dtype"]) for t in spec["tensors"]
    ] == [
        ("layers.0.ffn.gate.bias", "F32", "F16"),
        ("layers.0.ffn.gate.weight", "BF16", "F16"),
    ]
    for record in spec["tensors"]:
        expected = hashlib.sha256(_raw(source[record["name"]])).hexdigest()
        assert record["sha256"] == expected


def test_loaded_tensors_equal_source_bytes(checkpoints):
    source, export_dir, path, digest = checkpoints
    manifest = Exl3Manifest(export_dir)
    replaced = apply_override(manifest, str(path), digest)
    assert replaced == ["layers.0.ffn.gate.bias", "layers.0.ffn.gate.weight"]
    for name in replaced:
        header = manifest.tensors[name]
        # The loader reads plain tensors through each header's file.
        with safe_open(str(header.path), framework="pt", device="cpu") as file:
            loaded = file.get_tensor(name)
        assert loaded.dtype == source[name].dtype
        assert _raw(loaded) == _raw(source[name])
    untouched = manifest.tensors["layers.0.attn_norm.weight"]
    assert untouched.path == export_dir / "model.safetensors"
    # The FP32 gate parameters keep every source bit.
    param = torch.empty(16, dtype=torch.float32)
    with safe_open(
        str(manifest.tensors["layers.0.ffn.gate.bias"].path), "pt", device="cpu"
    ) as file:
        param.copy_(file.get_tensor("layers.0.ffn.gate.bias"))
    assert _raw(param) == _raw(source["layers.0.ffn.gate.bias"])


def test_refuses_tensor_sha256_mismatch(checkpoints):
    _, export_dir, path, digest = checkpoints
    sidecar = path.parent / builder.SIDECAR
    data = bytearray(sidecar.read_bytes())
    data[-1] ^= 1
    sidecar.write_bytes(bytes(data))
    manifest = Exl3Manifest(export_dir)
    before = dict(manifest.tensors)
    with pytest.raises(ValueError, match="sha256 mismatch: layers.0.ffn.gate"):
        apply_override(manifest, str(path), digest)
    assert manifest.tensors == before


def test_refuses_unpinned_or_changed_manifest(checkpoints):
    _, export_dir, path, digest = checkpoints
    manifest = Exl3Manifest(export_dir)
    with pytest.raises(ValueError, match="needs VLLM_EXL3_OVERRIDE_SHA256"):
        apply_override(manifest, str(path), "")
    with pytest.raises(ValueError, match="differs from the pinned"):
        apply_override(manifest, str(path), "0" * 64)


def test_refuses_export_other_than_recorded(checkpoints, tmp_path):
    source, _, path, digest = checkpoints
    other = tmp_path / "other"
    other.mkdir()
    _write(
        other / "model.safetensors",
        {
            "layers.0.ffn.gate.bias": source["layers.0.ffn.gate.bias"].bfloat16(),
            "layers.0.ffn.gate.weight": source["layers.0.ffn.gate.weight"].half(),
        },
    )
    with pytest.raises(ValueError, match="differs from the override manifest"):
        apply_override(Exl3Manifest(other), str(path), digest)
