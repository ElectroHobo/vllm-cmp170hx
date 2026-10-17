# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm_80 Engram table storage: SSD row reads at int64 offsets, the host
row cache, TP head ownership, telemetry, dequantization against the
reference formula, the real shards (DSV41_ENGRAM_DIR) and CUDA graph replay
with changing ids (GPU only)."""

import json
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open

from vllm.models.deepseek_v41.nvidia import sm80_engram
from vllm.models.deepseek_v41.nvidia.ops.sm80_engram_dequant import (
    dequant_engram_rows,
    dequant_engram_rows_reference,
)
from vllm.models.deepseek_v41.nvidia.sm80_engram import (
    EngramRowCache,
    EngramRowSource,
    EngramRowStore,
    Sm80EngramEmbedding,
    engram_owner_range,
    engram_table_dir,
    locate_engram_table,
)

DIM = 256
SCALE_DIM = DIM // 32
TP = 4

# A table past 2**32 bytes of values: rows from 2**23 sit past byte 2**31,
# rows from 2**24 past byte 2**32.
BIG_HEADS = (2_097_161,) * 8
BIG_ROWS = sum(BIG_HEADS) + 12
BIG_LAYER = 1
SMALL_HEADS = (600,) * 8
SMALL_ROWS = sum(SMALL_HEADS) + 200
SMALL_LAYER = 14


def _row_bytes(rows: np.ndarray, width: int, salt: int) -> np.ndarray:
    """Deterministic bytes of the given rows (every byte value occurs)."""
    rows = rows.astype(np.uint64)[:, None]
    cols = np.arange(width, dtype=np.uint64)[None, :]
    mixed = (
        rows * np.uint64(0x9E3779B97F4A7C15)
        + cols * np.uint64(0xBF58476D1CE4E5B9)
        + np.uint64(salt)
    )
    mixed ^= mixed >> np.uint64(31)
    mixed *= np.uint64(0x94D049BB133111EB)
    return (mixed >> np.uint64(56)).astype(np.uint8)


def _write_table(directory, layer, rows, written, file_name, extra_index=None):
    """A base-checkpoint shard in the real layout (values, then scales, then
    a small tensor), sparse: only `written` rows hold data, the rest are
    holes that read as zeros."""
    values_name = f"layers.{layer}.engram.embed.weight"
    scales_name = f"layers.{layer}.engram.embed.scale"
    q_name = f"layers.{layer}.engram.q_weight"
    values_end = rows * DIM
    scales_end = values_end + rows * SCALE_DIM
    header = {
        values_name: {
            "dtype": "F8_E4M3",
            "shape": [rows, DIM],
            "data_offsets": [0, values_end],
        },
        scales_name: {
            "dtype": "F8_E8M0",
            "shape": [rows, SCALE_DIM],
            "data_offsets": [values_end, scales_end],
        },
        q_name: {
            "dtype": "BF16",
            "shape": [4, 16],
            "data_offsets": [scales_end, scales_end + 128],
        },
    }
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    start = 8 + len(raw)
    path = directory / file_name
    written = np.unique(np.asarray(written, dtype=np.int64))
    values = _row_bytes(written, DIM, 1)
    scales = _row_bytes(written, SCALE_DIM, 2)
    with open(path, "wb") as file:
        file.write(struct.pack("<Q", len(raw)))
        file.write(raw)
        file.truncate(start + scales_end + 128)
        fd = file.fileno()
        for i, row in enumerate(written.tolist()):
            os.pwrite(fd, values[i].tobytes(), start + row * DIM)
            os.pwrite(fd, scales[i].tobytes(), start + values_end + row * SCALE_DIM)
    index_path = directory / "model.safetensors.index.json"
    index = (
        json.loads(index_path.read_text())
        if index_path.exists()
        else {"metadata": {}, "weight_map": {}}
    )
    for name in (values_name, scales_name, q_name):
        index["weight_map"][name] = file_name
    index["weight_map"].update(extra_index or {})
    index_path.write_text(json.dumps(index))
    return path


def _owner_ranges(heads):
    return [engram_owner_range(heads, rank, TP) for rank in range(TP)]


def _big_written():
    rng = np.random.default_rng(0)
    rows = [rng.integers(0, BIG_ROWS, 4000)]
    boundary = [
        0,
        1,
        2**23 - 1,
        2**23,
        2**23 + 1,
        2**24 - 1,
        2**24,
        2**24 + 1,
        BIG_ROWS - 1,
    ]
    for _, _, lo, hi in _owner_ranges(BIG_HEADS):
        boundary += [lo, lo + 1, hi - 1]
    return np.concatenate([np.concatenate(rows), np.asarray(boundary)])


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    directory = tmp_path_factory.mktemp("engram_tables")
    big_written = _big_written()
    _write_table(
        directory, BIG_LAYER, BIG_ROWS, big_written, "model-00047-of-00048.safetensors"
    )
    _write_table(
        directory,
        SMALL_LAYER,
        SMALL_ROWS,
        np.arange(SMALL_ROWS),
        "model-00048-of-00048.safetensors",
    )
    return SimpleNamespace(
        dir=directory,
        big=locate_engram_table(directory, BIG_LAYER, BIG_ROWS, DIM),
        small=locate_engram_table(directory, SMALL_LAYER, SMALL_ROWS, DIM),
        big_written=np.unique(big_written),
    )


def _independent_rows(path, layer, rows):
    """Values and scales bytes of `rows` through the safetensors library."""
    values, scales = [], []
    with safe_open(str(path), framework="pt") as file:
        weight = file.get_slice(f"layers.{layer}.engram.embed.weight")
        scale = file.get_slice(f"layers.{layer}.engram.embed.scale")
        for row in np.asarray(rows).tolist():
            values.append(weight[row : row + 1].view(torch.uint8))
            scales.append(scale[row : row + 1].view(torch.uint8))
    return torch.cat(values).numpy(), torch.cat(scales).numpy()


def _fetch(store, ids):
    ids = np.asarray(ids, dtype=np.int64)
    values = np.full((len(ids), DIM), 0xAA, dtype=np.uint8)
    scales = np.full((len(ids), SCALE_DIM), 0xAA, dtype=np.uint8)
    store.fetch(ids, values, scales)
    return values, scales


def _expected(files, layer, ids, lo, hi):
    ids = np.asarray(ids, dtype=np.int64)
    values = np.zeros((len(ids), DIM), dtype=np.uint8)
    scales = np.full((len(ids), SCALE_DIM), 127, dtype=np.uint8)
    owned = (ids >= lo) & (ids < hi)
    if owned.any():
        v, s = _independent_rows(files.values_path, layer, ids[owned])
        values[owned], scales[owned] = v, s
    return values, scales


# ---------------------------------------------------------------------------
# Table files
# ---------------------------------------------------------------------------


def test_locate_engram_table_offsets(tables):
    files = tables.big
    with open(files.values_path, "rb") as file:
        (length,) = struct.unpack("<Q", file.read(8))
    assert files.values_offset == 8 + length
    assert files.scales_offset == 8 + length + BIG_ROWS * DIM
    assert files.values_path == files.scales_path
    assert files.scales_offset > 2**32


def test_locate_engram_table_refuses_mismatch(tables, tmp_path):
    with pytest.raises(ValueError, match="expected"):
        locate_engram_table(tables.dir, BIG_LAYER, BIG_ROWS + 1, DIM)
    with pytest.raises(ValueError, match="lists no"):
        locate_engram_table(tables.dir, 2, BIG_ROWS, DIM)
    path = _write_table(tmp_path, 3, 64, np.arange(64), "short.safetensors")
    os.truncate(path, os.path.getsize(path) - 200)
    with pytest.raises(ValueError, match="truncated"):
        locate_engram_table(tmp_path, 3, 64, DIM)


def test_engram_table_dir_from_model_config():
    config = SimpleNamespace(
        hf_config=SimpleNamespace(engram_table_dir=None),
        hf_text_config=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="engram_table_dir"):
        engram_table_dir(config)
    config.hf_config.engram_table_dir = "/tables"
    assert engram_table_dir(config) == "/tables"


def test_owner_ranges_match_upstream_head_sharding():
    heads = tuple(range(100, 124))
    ranges = _owner_ranges(heads)
    assert [r[:2] for r in ranges] == [(6, 0), (6, 6), (6, 12), (6, 18)]
    assert ranges[0][2] == 0 and ranges[-1][3] == sum(heads)
    for (_, _, _, hi), (_, _, lo, _) in zip(ranges, ranges[1:]):
        assert hi == lo
    with pytest.raises(ValueError, match="without hash heads"):
        engram_owner_range((1, 2, 3), 0, 4)


# ---------------------------------------------------------------------------
# Reader parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank", range(TP))
def test_reader_row_parity_per_rank(tables, rank):
    _, _, lo, hi = engram_owner_range(BIG_HEADS, rank, TP)
    store = EngramRowStore(tables.big, lo, hi, placement="ssd", cache_bytes=1 << 20)
    try:
        written = tables.big_written
        mine = written[(written >= lo) & (written < hi)]
        holes = np.array([lo + 7, hi - 7])
        others = written[(written < lo) | (written >= hi)][:50]
        ids = np.concatenate([mine, holes, others, mine[:20], [-1, BIG_ROWS + 5]])
        for _ in range(2):  # cold, then warm through the cache
            values, scales = _fetch(store, ids)
            want_values, want_scales = _expected(tables.big, BIG_LAYER, ids, lo, hi)
            np.testing.assert_array_equal(values, want_values)
            np.testing.assert_array_equal(scales, want_scales)
        counts = store.telemetry.snapshot()
        assert counts["cache_hits"] > 0
        assert counts["rows_unowned"] == 2 * (len(others) + 2)
    finally:
        store.close()


def test_reader_reads_past_int32_offsets(tables):
    rows = np.array(
        [2**23 - 1, 2**23, 2**23 + 1, 2**24 - 1, 2**24, 2**24 + 1, BIG_ROWS - 1]
    )
    source = EngramRowSource(
        tables.big.values_path, tables.big.values_offset, DIM, 0, BIG_ROWS
    )
    try:
        dest = np.empty((len(rows), DIM), dtype=np.uint8)
        source.read(rows, dest)
        want, _ = _independent_rows(tables.big.values_path, BIG_LAYER, rows)
        np.testing.assert_array_equal(dest, want)
        assert dest.any()
        with pytest.raises(ValueError, match="outside"):
            source.read(np.array([BIG_ROWS]), np.empty((1, DIM), dtype=np.uint8))
    finally:
        source.close()


def test_native_reader_parity(tables):
    try:
        from vllm.sm80.kernels.exl3.ops import ensure_extension

        ensure_extension()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"native EXL3 extension unavailable: {exc}")
    _, _, lo, hi = engram_owner_range(BIG_HEADS, 2, TP)
    store = EngramRowStore(
        tables.big, lo, hi, placement="ssd", cache_bytes=0, native_read=True
    )
    try:
        written = tables.big_written
        ids = written[(written >= lo) & (written < hi)]
        values, scales = _fetch(store, ids)
        want_values, want_scales = _expected(tables.big, BIG_LAYER, ids, lo, hi)
        np.testing.assert_array_equal(values, want_values)
        np.testing.assert_array_equal(scales, want_scales)
        assert store.telemetry.snapshot()["unclassified_rows"] == len(ids)
    finally:
        store.close()


def test_ram_placement_matches_ssd(tables):
    _, _, lo, hi = engram_owner_range(SMALL_HEADS, 1, TP)
    ssd = EngramRowStore(tables.small, lo, hi, placement="ssd", cache_bytes=1 << 16)
    ram = EngramRowStore(tables.small, lo, hi, placement="ram", cache_bytes=0)
    try:
        ids = np.random.default_rng(3).integers(-5, SMALL_ROWS + 5, 3000)
        for a, b in zip(_fetch(ssd, ids), _fetch(ram, ids)):
            np.testing.assert_array_equal(a, b)
        assert ram.cache is None and ram.table.shape == (hi - lo, DIM)
    finally:
        ssd.close()
        ram.close()


# ---------------------------------------------------------------------------
# Row cache
# ---------------------------------------------------------------------------


def _small_cache(num_sets, ways):
    cache = EngramRowCache(num_sets * ways * (DIM + 16), DIM, ways=ways)
    assert cache.num_sets == num_sets
    cache.set_of = lambda rows: rows % num_sets
    return cache


def _data(rows):
    return _row_bytes(np.asarray(rows), DIM, 9)


def test_cache_lru_eviction_and_collisions():
    cache = _small_cache(num_sets=2, ways=2)
    out = np.empty((8, DIM), dtype=np.uint8)
    assert cache.insert(np.array([0, 2]), _data([0, 2])) == (2, 0)
    assert cache.lookup(np.array([0]), out[:1]).tolist() == [True]
    # Set 0 is full: 4 evicts the least recently used row, 2.
    assert cache.insert(np.array([4]), _data([4])) == (1, 1)
    hit = cache.lookup(np.array([0, 2, 4, 1]), out[:4])
    assert hit.tolist() == [True, False, True, False]
    np.testing.assert_array_equal(out[[0, 2]], _data([0, 4]))
    # Three colliding rows in one batch: only two ways, the third is skipped.
    assert cache.insert(np.array([6, 8, 10]), _data([6, 8, 10])) == (2, 2)
    hit = cache.lookup(np.array([6, 8, 10]), out[:3])
    assert hit.tolist() == [True, True, False]
    np.testing.assert_array_equal(out[:2], _data([6, 8]))
    # A row already present is not inserted twice.
    assert cache.insert(np.array([6, 1]), _data([6, 1])) == (1, 0)


def test_cache_disabled_and_bound():
    cache = EngramRowCache(0, DIM)
    assert cache.capacity_rows == 0
    assert not cache.lookup(np.array([1]), np.empty((1, DIM), np.uint8)).any()
    assert cache.insert(np.array([1]), _data([1])) == (0, 0)
    bounded = EngramRowCache(1 << 20, DIM)
    assert bounded.nbytes <= 1 << 20
    assert bounded.capacity_rows == (1 << 20) // (8 * (DIM + 16)) * 8


def test_store_cache_warm_hits_and_eviction(tables):
    _, _, lo, hi = engram_owner_range(SMALL_HEADS, 0, TP)
    # 64 sets x 8 ways: fewer entries than the owned rows.
    store = EngramRowStore(
        tables.small, lo, hi, placement="ssd", cache_bytes=64 * 8 * (DIM + 16)
    )
    try:
        ids = np.arange(lo, lo + 16)
        _fetch(store, ids)
        first = store.telemetry.snapshot()
        assert first["cache_misses"] == 16 and first["cache_hits"] == 0
        values, _ = _fetch(store, np.concatenate([ids, ids]))
        second = store.telemetry.snapshot()
        assert second["cache_hits"] - first["cache_hits"] == 16
        assert second["rows_requested"] == 48 and second["rows_unique"] == 32
        want, _ = _expected(
            tables.small, SMALL_LAYER, np.concatenate([ids, ids]), lo, hi
        )
        np.testing.assert_array_equal(values, want)
        # Overflow every set: rows still arrive, entries are evicted.
        many = np.arange(lo, hi)
        values, scales = _fetch(store, many)
        want_values, want_scales = _expected(tables.small, SMALL_LAYER, many, lo, hi)
        np.testing.assert_array_equal(values, want_values)
        np.testing.assert_array_equal(scales, want_scales)
        third = store.telemetry.snapshot()
        assert third["cache_evictions"] > 0
        assert third["cache_inserts"] <= store.cache.capacity_rows + 16
        read = third["cache_misses"]
        assert read == (
            third["page_cache_rows"]
            + third["storage_rows"]
            + third["unclassified_rows"]
        )
        assert (
            third["page_cache_bytes"]
            + third["storage_bytes"]
            + third["unclassified_bytes"]
            == read * DIM
        )
    finally:
        store.close()


def test_page_cache_classification(tables):
    _, _, lo, hi = engram_owner_range(SMALL_HEADS, 3, TP)
    store = EngramRowStore(tables.small, lo, hi, placement="ssd", cache_bytes=0)
    try:
        resident, total = store.source.page_cache_residency()
        assert 0 <= resident <= total
        # Freshly written rows sit in the page cache.
        _fetch(store, np.arange(lo, lo + 64))
        counts = store.telemetry.snapshot()
        if sm80_engram._preadv2() is not None and store.source._nowait:
            assert counts["unclassified_rows"] == 0
            assert counts["page_cache_rows"] + counts["storage_rows"] == 64
        else:
            assert counts["unclassified_rows"] == 64
    finally:
        store.close()


def test_concurrent_callers(tables):
    _, _, lo, hi = engram_owner_range(SMALL_HEADS, 2, TP)
    store = EngramRowStore(
        tables.small, lo, hi, placement="ssd", cache_bytes=16 * 8 * (DIM + 16)
    )
    everything = np.arange(SMALL_ROWS)
    all_values, all_scales = _expected(tables.small, SMALL_LAYER, everything, lo, hi)
    barrier = threading.Barrier(8)

    def worker(seed):
        rng = np.random.default_rng(seed)
        barrier.wait()
        for _ in range(40):
            ids = rng.integers(lo - 50, hi + 50, 300)
            ids = np.concatenate([ids, ids[:30]])
            values, scales = _fetch(store, ids)
            np.testing.assert_array_equal(values, all_values[ids])
            np.testing.assert_array_equal(scales, all_scales[ids])
        return True

    try:
        with ThreadPoolExecutor(8) as pool:
            assert all(pool.map(worker, range(8)))
        counts = store.telemetry.snapshot()
        assert counts["lookups"] == 320
        assert counts["cache_hits"] > 0 and counts["cache_evictions"] > 0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# The module and dequantization
# ---------------------------------------------------------------------------


def _reference_lookup(files, layer, heads, rank, indices):
    """ParallelEngramEmbedding.forward of the reference for this rank's heads:
    out-of-range ids give zero rows."""
    part, head_start, lo, hi = engram_owner_range(heads, rank, TP)
    cols = indices[:, head_start : head_start + part].reshape(-1).numpy()
    values, scales = _expected(files, layer, cols, lo, hi)
    values = torch.from_numpy(values).view(torch.float8_e4m3fn)
    scales = torch.from_numpy(scales).view(torch.float8_e8m0fnu)
    rows = values.float().unflatten(-1, (-1, 32)) * scales.float().unsqueeze(-1)
    owned = torch.from_numpy((cols >= lo) & (cols < hi))
    rows = rows.flatten(-2).to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)
    return rows.view(indices.shape[0], part, DIM)


def _hash_like_indices(heads, written, tokens, seed):
    """Per-head ids inside each head's bucket range, from the written rows,
    with dead (-1) entries."""
    rng = np.random.default_rng(seed)
    offsets = np.cumsum((0, *heads[:-1]))
    columns = []
    for head, (offset, size) in enumerate(zip(offsets, heads)):
        inside = written[(written >= offset) & (written < offset + size)]
        pick = rng.choice(inside, tokens) if len(inside) else np.full(tokens, offset)
        columns.append(pick)
    indices = np.stack(columns, axis=1)
    indices[rng.random(indices.shape) < 0.05] = -1
    return torch.from_numpy(indices.astype(np.int32))


@pytest.mark.parametrize("rank", range(TP))
def test_module_lookup_matches_reference(tables, rank):
    table = Sm80EngramEmbedding(
        BIG_ROWS,
        DIM,
        BIG_HEADS,
        tables.big,
        tp_rank=rank,
        tp_size=TP,
        max_tokens=64,
        placement="ssd",
        cache_bytes=1 << 20,
    )
    try:
        assert list(table.parameters()) == [] and table.state_dict() == {}
        assert (table.cpu_offload, table.dp_size, table.part_n_hash_cols) == (
            True,
            1,
            2,
        )
        for seed in range(3):
            indices = _hash_like_indices(BIG_HEADS, tables.big_written, 64, seed)
            out = torch.full((64, 2, DIM), float("nan"), dtype=torch.bfloat16)
            table.lookup(indices, out)
            want = _reference_lookup(tables.big, BIG_LAYER, BIG_HEADS, rank, indices)
            assert torch.equal(out.view(torch.int16), want.view(torch.int16))
        with pytest.raises(ValueError, match="capacity"):
            table.lookup(
                torch.zeros(65, 8, dtype=torch.int32),
                torch.empty(65, 2, DIM, dtype=torch.bfloat16),
            )
        assert table.telemetry()["lookups"] == 3
    finally:
        table.close()


def _kernel_decode_emulation(value_bytes, scale_bytes):
    """The Triton kernel's bit formula in numpy FP32 (no torch float8)."""
    x = value_bytes.astype(np.uint32)
    bits = ((x & 0x7F) << 20) | ((x & 0x80) << 24)
    value = bits.view(np.float32) * np.float32(2.0**120)
    value = np.where((x & 0x7F) == 0x7F, np.float32(np.nan), value)
    s = scale_bytes.astype(np.uint32)
    scale = (s << 23).view(np.float32)
    scale = np.where(s == 0, np.array([0x400000], np.uint32).view(np.float32)[0], scale)
    scale = np.where(s == 0xFF, np.float32(np.nan), scale)
    with np.errstate(over="ignore", invalid="ignore"):
        product = (value * scale).astype(np.float32)
    return torch.from_numpy(product).to(torch.bfloat16)


def _assert_bitwise(got, want):
    assert got.dtype == want.dtype and got.shape == want.shape
    bits = torch.int32 if want.dtype == torch.float32 else torch.int16
    nan = want.isnan()
    assert torch.equal(got.isnan(), nan)
    assert torch.equal(got.view(bits)[~nan], want.view(bits)[~nan])


def _all_byte_pairs():
    """Every (value byte, scale byte) pair, laid out as [256 scales, 256]
    rows: row s holds value bytes 0..255 with every block scale s."""
    values = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    scales = np.repeat(np.arange(256, dtype=np.uint8)[:, None], SCALE_DIM, axis=1)
    return values, scales


def test_kernel_decode_formula_matches_reference_exhaustively():
    values, scales = _all_byte_pairs()
    want = dequant_engram_rows_reference(
        torch.from_numpy(values), torch.from_numpy(scales)
    )
    got = _kernel_decode_emulation(values, np.repeat(scales, 32, axis=1))
    _assert_bitwise(got, want)
    # Denormal products, the 2**-127 scale and saturation are all exercised.
    assert (want[0] != 0).any() and want.isinf().any()


def test_cpu_dequant_is_the_reference_formula():
    values, scales = _all_byte_pairs()
    out = torch.empty(256, DIM, dtype=torch.bfloat16)
    dequant_engram_rows(torch.from_numpy(values), torch.from_numpy(scales), out)
    ref = (
        torch.from_numpy(values).view(torch.float8_e4m3fn).float().view(256, 8, 32)
        * torch.from_numpy(scales).view(torch.float8_e8m0fnu).float()[..., None]
    )
    _assert_bitwise(out, ref.view(256, DIM).to(torch.bfloat16))


def test_create_embedding_hook_dispatch(monkeypatch):
    calls = []

    def upstream(engram, layout, index):
        calls.append(("upstream", index))
        return "upstream"

    wrapped = sm80_engram.sm80_engram_embedding(upstream)
    monkeypatch.setattr("vllm.sm80.kernels._platform._SM80_CACHE", False)
    assert wrapped(None, None, 0) == "upstream"
    monkeypatch.setattr("vllm.sm80.kernels._platform._SM80_CACHE", True)
    monkeypatch.setattr(
        sm80_engram,
        "build_sm80_engram_embedding",
        lambda layout, index: ("sm80", index),
    )
    assert wrapped(None, None, 1) == ("sm80", 1)
    assert calls == [("upstream", 0)]


def test_upstream_engram_carries_the_hook():
    from vllm.models.deepseek_v41.common.engram import Engram

    assert Engram._create_embedding.__wrapped__.__qualname__ == (
        "Engram._create_embedding"
    )


# ---------------------------------------------------------------------------
# Real shards
# ---------------------------------------------------------------------------

_REAL_LAYERS = {1: 384_006_168, 14: 384_016_682}
_REAL_FILES = ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")


def _real_dir():
    directory = os.environ.get("DSV41_ENGRAM_DIR")
    if not directory or not all(
        os.path.isfile(os.path.join(directory, name)) for name in _REAL_FILES
    ):
        pytest.skip("DSV41_ENGRAM_DIR with both Engram shards is not set")
    return directory


def _real_head_sizes():
    from vllm.models.deepseek_v41.common.engram import EngramLayout

    config = SimpleNamespace(
        engram_layer_ids=list(_REAL_LAYERS),
        engram_num_embeddings=list(_REAL_LAYERS.values()),
        engram_max_ngram_size=4,
        engram_n_heads=8,
        engram_head_dim=DIM,
        engram_vocab_size=16_000_000,
        engram_compressed_vocab_size=99092,
        engram_pad_token_id=2,
    )
    layout = EngramLayout(config)
    return [tuple(p for order in layer for p in order) for layer in layout.primes]


def test_real_shards_parity():
    directory = _real_dir()
    rng = np.random.default_rng(1017)
    for (layer, rows), heads in zip(_REAL_LAYERS.items(), _real_head_sizes()):
        files = locate_engram_table(directory, layer, rows, DIM)
        assert sum(heads) <= rows
        for rank in range(TP):
            _, _, lo, hi = engram_owner_range(heads, rank, TP)
            sample = np.concatenate([rng.integers(lo, hi, 1024), [lo, hi - 1]])
            if rank == 0:
                sample = np.concatenate([sample, [0]])
            store = EngramRowStore(files, lo, hi, placement="ssd", cache_bytes=0)
            try:
                values, scales = _fetch(store, sample)
            finally:
                store.close()
            want_values, want_scales = _independent_rows(
                files.values_path, layer, sample
            )
            np.testing.assert_array_equal(values, want_values)
            np.testing.assert_array_equal(scales, want_scales)
            got = dequant_engram_rows_reference(
                torch.from_numpy(values), torch.from_numpy(scales)
            )
            assert not got.isnan().any()
        # The table's last row lies past the last head: read it unowned-free.
        source = EngramRowSource(
            files.values_path, files.values_offset, DIM, rows - 1, rows
        )
        try:
            dest = np.empty((1, DIM), dtype=np.uint8)
            source.read(np.array([rows - 1]), dest)
        finally:
            source.close()
        want, _ = _independent_rows(files.values_path, layer, [rows - 1])
        np.testing.assert_array_equal(dest, want)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@needs_cuda
def test_gpu_dequant_kernel_bitwise_exhaustive():
    values, scales = _all_byte_pairs()
    values_t = torch.from_numpy(values).cuda()
    scales_t = torch.from_numpy(scales).cuda()
    out = torch.empty(256, DIM, dtype=torch.bfloat16, device="cuda")
    dequant_engram_rows(values_t, scales_t, out)
    want = dequant_engram_rows_reference(
        torch.from_numpy(values), torch.from_numpy(scales)
    )
    _assert_bitwise(out.cpu(), want)
    # Odd row counts take the masked tail.
    tail = torch.empty(37, DIM, dtype=torch.bfloat16, device="cuda")
    dequant_engram_rows(values_t[:37].contiguous(), scales_t[:37].contiguous(), tail)
    _assert_bitwise(tail.cpu(), want[:37])


@needs_cuda
def test_gpu_lookup_refuses_capture(tables):
    table = Sm80EngramEmbedding(
        SMALL_ROWS,
        DIM,
        SMALL_HEADS,
        tables.small,
        tp_rank=0,
        tp_size=TP,
        max_tokens=8,
        placement="ssd",
        cache_bytes=1 << 20,
        device="cuda",
    )
    try:
        ids = torch.zeros(8, 8, dtype=torch.int32, device="cuda")
        out = torch.empty(8, 2, DIM, dtype=torch.bfloat16, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with (
            pytest.raises(RuntimeError, match="cannot be captured"),
            torch.cuda.graph(graph),
        ):
            table.lookup(ids, out)
    finally:
        table.close()


@needs_cuda
@pytest.mark.parametrize("rank", [0, 3])
def test_gpu_breakable_graph_replay_with_changing_ids(tables, monkeypatch, rank):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        eager_break_during_capture,
    )

    tokens = 16
    table = Sm80EngramEmbedding(
        BIG_ROWS,
        DIM,
        BIG_HEADS,
        tables.big,
        tp_rank=rank,
        tp_size=TP,
        max_tokens=tokens,
        placement="ssd",
        cache_bytes=1 << 20,
        device="cuda",
    )
    prefetch = torch.cuda.Stream()
    done = torch.cuda.Event()
    static_ids = torch.zeros(tokens, 8, dtype=torch.int32, device="cuda")
    rows = torch.empty(tokens, 2, DIM, dtype=torch.bfloat16, device="cuda")

    # The upstream Engram's prefetch and finish, as eager breaks.
    @eager_break_during_capture
    def start(ids, out):
        prefetch.wait_stream(torch.cuda.current_stream())
        ids.record_stream(prefetch)
        with torch.cuda.stream(prefetch):
            table.lookup(ids, out, background=True)
        done.record(prefetch)

    @eager_break_during_capture
    def finish():
        torch.cuda.current_stream().wait_event(done)

    try:
        torch.cuda.synchronize()
        main = torch.cuda.Stream()
        main.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(main):
            capture = BreakableCUDAGraphCapture()
            with capture:
                ids = static_ids + 0  # stands in for the hash kernel
                start(ids, rows)
                finish()
                result = rows.float() * 2
            assert capture.num_eager_breaks == 2
            steps = [
                _hash_like_indices(BIG_HEADS, tables.big_written, tokens, seed)
                for seed in range(10, 16)
            ]
            outputs = []
            for indices in steps:
                static_ids.copy_(indices.to("cuda", non_blocking=False))
                capture.replay()
                outputs.append(result.clone())  # queued; no host sync between
        torch.cuda.synchronize()
        for indices, got in zip(steps, outputs):
            want = _reference_lookup(tables.big, BIG_LAYER, BIG_HEADS, rank, indices)
            _assert_bitwise(got.cpu(), want.float() * 2)
        assert table.telemetry()["lookups"] == len(steps) + 1
    finally:
        table.close()
