# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 attention on sm_80 (compute capability 8.x).

None of the vendor kernels the upstream layers use (FlashMLA, FlashInfer
sparse MLA, DeepGEMM, the FP8/FP4 conversion instructions) exist on Ampere.
This layer keeps upstream's cache topology -- a sliding-window cache per
layer, compressed KV and index keys owned by the kv-source layers, the shared
top-k and candidate-block buffers -- with every cache holding plain BF16 rows,
and computes the reference model's attention through
``ops/sm80_attn_core.py``: the FP8/FP4 rounding of the reference is emulated
in software, so the cached rows carry the reference's quantize-dequantize
values exactly.

Selection is by the platform: :func:`sm80_select_attn_cls` wraps upstream's
``_select_dsv4_attn_cls`` and returns :class:`DeepseekV41Sm80Attention` on
compute capability 8.x; the DSpark draft layers are built by the same
selector. :func:`sm80_dspark_context_kv` wraps the drafter's context-KV
insertion so its window rows go through the same rounding.

The attention region (KV writes, compressor, indexer, top-k, sparse attention,
output RoPE and quantize-dequantize) runs eagerly under a CUDA-graph break;
the input projections and ``wo_a`` / ``wo_b`` stay in the graph.
"""

import functools
from collections.abc import Callable
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.models.deepseek_v41.attention import (
    DeepseekV4Attention,
    DeepseekV4Indexer,
    DeepseekV4IndexerCache,
)
from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4SparseMLABackend,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Backends: upstream's metadata builders, plain BF16 rows, compute capability 8
# ---------------------------------------------------------------------------


class Sm80SparseDsv41Backend(DeepseekV4SparseMLABackend):
    """The kv-source compressed cache: upstream's DSv4.1 sparse metadata
    (block table + compressed slot mapping), BF16 rows, any even block size
    (so the kernel blocks are the manager blocks)."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "SM80_SPARSE_DSV41"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(2)]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class Sm80IndexKBackend(Sm80SparseDsv41Backend):
    """The index-key cache: same metadata, 128-dim BF16 rows."""

    @staticmethod
    def get_name() -> str:
        return "SM80_INDEX_K_DSV41"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]


class Sm80SWAMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    """Upstream's sliding-window metadata without FlashMLA tile plans."""

    def build_tile_scheduler(self, num_decode_tokens: int) -> dict[str, Any]:
        return super().build_tile_scheduler(0)


class Sm80SWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_name() -> str:
        return "SM80_DEEPSEEK_SPARSE_SWA"

    @staticmethod
    def get_builder_cls() -> type[Sm80SWAMetadataBuilder]:
        return Sm80SWAMetadataBuilder


class Sm80IndexKCache(DeepseekV4IndexerCache):
    """Index keys as BF16 rows of ``index_head_dim`` (256 B per key)."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            tokens_per_state=self.compress_ratio,
            alignment=512,
        )

    def get_attn_backend(self) -> type[Sm80IndexKBackend]:
        return Sm80IndexKBackend


# ---------------------------------------------------------------------------
# Indexer: the reference's modules, no upstream scoring op
# ---------------------------------------------------------------------------


class DeepseekV41Sm80Indexer(DeepseekV4Indexer):
    """Holds the indexer's weights and index-key cache; the scoring and the
    top-k run in ``sm80_attn_core`` from :class:`DeepseekV41Sm80Attention`.

    ``wq_b`` and ``weights_proj`` stay replicated (all heads on every rank,
    as upstream); the reference's TP split of the heads is reproduced
    numerically by the head grouping of the scores. ``wq_b`` and ``wk`` take
    the model's quantization config (both are quantized in the EXL3
    checkpoint); ``weights_proj`` stays an unquantized BF16 linear."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config,
        cache_config,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int,
        prefix: str,
        *,
        owns_k: bool,
        k_cache: DeepseekV4IndexerCache,
        main_head_dim: int,
        candidate_block_buffer: torch.Tensor | None = None,
        candidate_block_size: int = 0,
        candidate_write: bool = False,
    ):
        nn.Module.__init__(self)
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk_tokens = config.index_topk
        self.compress_ratio = compress_ratio
        self.owns_k = owns_k
        self.prefix = prefix
        self.topk_indices_buffer = topk_indices_buffer
        self.candidate_block_buffer = candidate_block_buffer
        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        context = vllm_config.compilation_config.static_forward_context
        if owns_k:
            self.wk = ReplicatedLinear(
                main_head_dim,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.wk",
            )
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            # Replace the packed-record cache the base layer registered.
            context.pop(k_cache.prefix)
            k_cache = Sm80IndexKCache(
                head_dim=self.head_dim,
                dtype=torch.bfloat16,
                prefix=k_cache.prefix,
                cache_config=cache_config,
                compress_ratio=compress_ratio,
            )
        elif not isinstance(k_cache, Sm80IndexKCache):
            raise NotImplementedError(
                f"index-key cache {k_cache.prefix} is not an sm_80 cache"
            )
        self.k_cache = k_cache


# ---------------------------------------------------------------------------
# The attention layer
# ---------------------------------------------------------------------------


def _rope_params(config, compress_ratio: int, rope_dim: int) -> tuple:
    """``(dim, original_seq_len, base, factor, beta_fast, beta_slow)`` of the
    reference's ``precompute_freqs_cis`` for this layer type: compressed
    layers use ``compress_rope_theta`` with YaRN, sliding-window layers plain
    RoPE at ``rope_theta``."""
    rp = getattr(config, "rope_parameters", None) or getattr(
        config, "rope_scaling", None
    ) or {}
    if isinstance(rp.get("main"), dict) and isinstance(rp.get("compress"), dict):
        rp = rp["compress" if compress_ratio else "main"]
    factor = float(rp.get("factor", 1.0))
    beta_fast = rp.get("beta_fast", 32)
    beta_slow = rp.get("beta_slow", 1)
    if not compress_ratio:
        return (rope_dim, 0, float(config.rope_theta), factor, beta_fast, beta_slow)
    yarn = rp.get("rope_type", "default") != "default"
    original = int(rp.get("original_max_position_embeddings", 0)) if yarn else 0
    return (
        rope_dim,
        original,
        float(config.compress_rope_theta),
        factor,
        beta_fast,
        beta_slow,
    )


class DeepseekV41Sm80Attention(DeepseekV4Attention):
    """DeepSeek-V4.1 attention layer for compute capability 8.x."""

    backend_cls = Sm80SparseDsv41Backend
    swa_backend_cls = Sm80SWABackend
    use_fp8_ds_mla_layout: ClassVar[bool] = False

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _indexer_cls(self, k_cache) -> type[DeepseekV4Indexer]:
        return DeepseekV41Sm80Indexer

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            vllm_config,
            prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
            candidate_block_buffer=candidate_block_buffer,
        )
        if self.kv_cache_torch_dtype != torch.bfloat16:
            raise ValueError(
                "DeepSeek V4.1 on sm_80 stores BF16 KV rows; got kv-cache dtype "
                f"{self.kv_cache_dtype}."
            )
        config = vllm_config.model_config.hf_config
        if self.compressor is not None:
            # The compressor projection is quantized in the checkpoint, so it
            # takes the model's quantization config (upstream builds it
            # unquantized for its fp32 GEMM).
            c = self.compressor
            c.fused_wkv_wgate = MergedColumnParallelLinear(
                c.hidden_size,
                [c.head_dim] * (2 if c.has_gate else 1),
                bias=False,
                return_bias=False,
                quant_config=vllm_config.quant_config,
                disable_tp=True,
                prefix=f"{c.prefix}.fused_wkv_wgate",
            )
        tp_size = self.n_heads // self.n_local_heads
        index_n_heads = getattr(config, "index_n_heads", 0)
        cand_source = self.candidate_source_layer
        self.sm80_cfg = core.Sm80AttnConfig(
            n_local_heads=self.n_local_heads,
            head_dim=self.head_dim,
            rope_dim=self.rope_head_dim,
            n_local_groups=self.n_local_groups,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            is_kv_source=self.is_kv_source,
            is_index_source=self.is_index_source,
            is_candidate_source=self.is_index_source and self.layer_id == cand_source,
            uses_candidates=self.is_index_source and 0 <= cand_source < self.layer_id,
            index_n_heads=index_n_heads,
            index_head_dim=getattr(config, "index_head_dim", 0),
            index_topk=getattr(config, "index_topk", 0),
            candidate_topk_blocks=self.candidate_topk_blocks,
            candidate_block_size=self.candidate_block_size,
            tp_size=tp_size,
            eps=self.eps,
        )
        if self.is_index_source and index_n_heads % tp_size:
            raise ValueError(
                f"{index_n_heads} index heads do not split over TP {tp_size}"
            )
        self._sm80_rope_params = _rope_params(
            config, self.compress_ratio, self.rope_head_dim
        )
        self._sm80_freqs: dict[torch.device, torch.Tensor] = {}
        self._sm80_chunk_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        self._wo_b_bf16_allreduce = bool(envs.VLLM_DSV41_WO_B_BF16_ALLREDUCE)
        if self._wo_b_bf16_allreduce:
            logger.info_once(
                "DeepSeek-V4.1 sm_80: wo_b partials are all-reduced in BF16 "
                "when the custom all-reduce takes the message, else in FP32 "
                "rounded once (VLLM_DSV41_WO_B_BF16_ALLREDUCE=1; set 0 to "
                "all-reduce every wo_b partial in FP32)",
                scope="global",
            )

    # -- pieces -------------------------------------------------------------

    def _sm80_weights(self, device: torch.device) -> core.Sm80AttnWeights:
        freqs = self._sm80_freqs.get(device)
        if freqs is None:
            freqs = core.rope_freqs(*self._sm80_rope_params, device=device)
            self._sm80_freqs[device] = freqs
        w = core.Sm80AttnWeights(
            q_norm=self.q_norm.weight,
            kv_norm=self.kv_norm.weight,
            attn_sink=self.attn_sink,
            freqs=freqs,
            wq_b=self.wq_b,
        )
        if self.compressor is not None:
            w.comp_norm = self.compressor.norm.weight
        indexer = self.indexer
        if indexer is not None:
            w.index_wq_b = _first(indexer.wq_b)
            if indexer.owns_k:
                w.wk = _first(indexer.wk)
                w.k_norm = indexer.k_norm.weight
        return w

    def _sm80_caches(self) -> core.Sm80AttnCaches:
        assert self.topk_indices_buffer is not None
        return core.Sm80AttnCaches(
            swa=self.swa_cache_layer.kv_cache,
            topk=self.topk_indices_buffer,
            comp=self._compressed_kv_cache() if self.compress_ratio else None,
            idxk=self.indexer.k_cache.kv_cache if self.indexer is not None else None,
            ring=(
                self.compressor.state_cache.kv_cache
                if self.compressor is not None
                and self.compressor.state_cache is not None
                else None
            ),
            candidates=self.candidate_block_buffer,
        )

    def _sm80_batch(self, attn_metadata: dict[str, Any], positions: torch.Tensor):
        """Gather this step's token metadata from the upstream builders."""
        swa = cast(Any, attn_metadata[self.swa_cache_layer.prefix])
        nd, npf = swa.num_decode_tokens, swa.num_prefill_tokens
        T = nd + npf
        parts = []
        if nd:
            parts.append(swa.decode_swa_indices.reshape(nd, -1))
        if npf:
            parts.append(swa.prefill_swa_indices.reshape(npf, -1))
        width = max([p.shape[1] for p in parts] + [1])
        swa_indices = torch.cat(
            [
                torch.nn.functional.pad(p, (0, width - p.shape[1]), value=-1)
                for p in parts
            ]
        ) if parts else torch.empty(0, 1, dtype=torch.int32, device=positions.device)
        batch = core.Sm80AttnBatch(
            positions=positions[:T],
            token_to_req=swa.token_to_req_indices[:T],
            query_start_loc=swa.query_start_loc,
            swa_slots=swa.slot_mapping[:T],
            swa_indices=swa_indices.contiguous(),
        )
        if self.compress_ratio:
            comp = cast(Any, attn_metadata[self.compressed_cache_prefix])
            batch.comp_block_table = comp.block_table
            if self.is_kv_source:
                batch.comp_slots = comp.slot_mapping[:T]
        if self.indexer is not None:
            idxk = cast(Any, attn_metadata[self.indexer.k_cache.prefix])
            batch.idxk_block_table = idxk.block_table
            batch.idxk_slots = idxk.slot_mapping[:T]
        if self.compressor is not None and self.compressor.state_cache is not None:
            ring = cast(Any, attn_metadata[self.compressor.state_cache.prefix])
            batch.ring_slots = ring.slot_mapping[:T]
        return batch

    def _sm80_projections(self, hidden_states: torch.Tensor):
        qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
        kv_score = None
        if self.compressor is not None:
            kv_score = self.compressor.fused_wkv_wgate(hidden_states).float()
        index_weights = None
        if self.indexer is not None:
            index_weights, _ = self.indexer.weights_proj(hidden_states)
        return qr_kv, kv_score, index_weights

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = torch.empty(
            (hidden_states.shape[0], self.n_local_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        qr_kv, kv_score, index_weights = self._sm80_projections(hidden_states)
        self._sm80_attention(qr_kv, kv_score, index_weights, positions, attn_out)
        return self._o_proj(attn_out, positions)

    @eager_break_during_capture
    def _sm80_attention(
        self,
        qr_kv: torch.Tensor,
        kv_score: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            # Profile run: no caches yet. Reserve what the indexer's score
            # chunk (scores, candidate mask, sort) can take at its bound.
            out.zero_()
            if self.is_index_source and out.is_cuda:
                torch.empty(
                    3 * self._sm80_chunk_bytes, dtype=torch.uint8, device=out.device
                )
            return
        batch = self._sm80_batch(attn_metadata, positions)
        q = core.prepare_layer(
            self.sm80_cfg,
            self._sm80_weights(out.device),
            self._sm80_caches(),
            batch,
            qr_kv,
            kv_score,
            index_weights,
            self._sm80_chunk_bytes,
        )
        self._sm80_attend(batch, q, out)

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Sparse attention of the post-RoPE ``q`` over this step's window
        rows and selected compressed rows, already in the caches (``kv`` is
        not read). ``output`` receives the ``wo_a`` input: inverse RoPE and
        FP8 qdq applied, as :meth:`_o_proj` expects."""
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            output.zero_()
            return
        self._sm80_attend(self._sm80_batch(attn_metadata, positions), q, output)

    def _sm80_attend(self, batch, q: torch.Tensor, out: torch.Tensor) -> None:
        core.attend_layer(
            self.sm80_cfg, self._sm80_weights(out.device), self._sm80_caches(),
            batch, q, out,
        )

    def forward_kv(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> None:
        qr_kv, kv_score, _ = self._sm80_projections(hidden_states)
        self._sm80_store_kv(qr_kv, kv_score, positions)

    @eager_break_during_capture
    def _sm80_store_kv(
        self,
        qr_kv: torch.Tensor,
        kv_score: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return
        batch = self._sm80_batch(attn_metadata, positions)
        T = batch.num_tokens
        kv = core.rms_norm(qr_kv[:T, self.q_lora_rank :], self.kv_norm.weight, self.eps)
        core.store_kv(
            self.sm80_cfg,
            self._sm80_weights(qr_kv.device),
            self._sm80_caches(),
            batch,
            kv,
            None if kv_score is None else kv_score[:T],
        )

    @eager_break_during_capture
    def sm80_store_context_kv(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        """DSpark context KV: ``kv_norm(wkv(main_x))`` RoPE'd and FP8 qdq'd
        into this layer's window cache at ``slot_mapping`` (the reference
        ``DSparkAttention`` main KV)."""
        qr_kv = self._fused_wqa_wkv_gemm(main_x)
        if slot_mapping is None:
            return
        n = min(slot_mapping.shape[0], qr_kv.shape[0])
        kv = core.rms_norm(qr_kv[:n, self.q_lora_rank :], self.kv_norm.weight, self.eps)
        freqs = self._sm80_weights(kv.device).freqs
        rows = core.window_kv(kv, positions[:n], freqs)
        core.store_rows(self.swa_cache_layer.kv_cache, slot_mapping[:n], rows)

    def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``wo_a`` per group (the reference's einsum over its block-diagonal
        weight) on the already inverse-RoPE'd, FP8-qdq'd output, then
        ``wo_b``; with TP the ``wo_b`` partials are summed in fp32 and rounded
        once, as the reference's RowParallelLinear does (in BF16 by a kernel
        that does exactly that, with VLLM_DSV41_WO_B_BF16_ALLREDUCE; see
        :func:`wo_b_all_reduce`)."""
        T = attn_out.shape[0]
        o = attn_out.view(T, self.n_local_groups, -1)
        wo_a = self.wo_a
        apply_grouped = getattr(wo_a.quant_method, "apply_grouped", None)
        if apply_grouped is not None:
            z = apply_grouped(wo_a, o)
        else:
            w = wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            z = torch.einsum("tgd,grd->tgr", o, w).flatten(1)
        wo_b = self.wo_b
        if self.gemm_rs is not None or not wo_b.reduce_results or wo_b.tp_size == 1:
            return self._wo_b_proj(z)
        partial = wo_b.quant_method.apply(wo_b, z, None)
        return wo_b_all_reduce(partial, self._wo_b_bf16_allreduce)


# Backends the communicator tries before the custom all-reduce. When one is
# enabled the custom all-reduce's own predicate no longer says which kernel
# reduces a message, so the BF16 switch stays inert.
_PREEMPTING_BACKENDS = (
    "fi_ar_comm",
    "qr_comm",
    "fi_pcie_ipc_ar_comm",
    "aiter_ar_comm",
)


def _preempting_backends(comm: Any) -> list[str]:
    from vllm.distributed.device_communicators.pynccl_allocator import (
        is_symmetric_memory_enabled,
    )

    names = [
        name
        for name in _PREEMPTING_BACKENDS
        if getattr(comm, name, None) is not None
        and not getattr(getattr(comm, name), "disabled", False)
    ]
    pynccl = getattr(comm, "pynccl_comm", None)
    if pynccl is not None and is_symmetric_memory_enabled():
        names.append("nccl_symm_mem")
    return names


def custom_all_reduce_takes(t: torch.Tensor) -> bool:
    """Whether the TP group's custom all-reduce reduces ``t``: the host-staged
    path (dispatched first by the fork communicator) does not take it, no
    backend dispatched before the custom all-reduce is enabled, and the
    custom all-reduce's own ``should_custom_ar`` accepts it."""
    comm = get_tp_group().device_communicator
    host = getattr(comm, "hostshm_comm", None)
    if host is not None and host.should_host_ar(t):
        return False
    ca = getattr(comm, "ca_comm", None)
    if ca is None or ca.disabled or _preempting_backends(comm):
        return False
    return bool(ca.should_custom_ar(t))


def wo_b_all_reduce(partial: torch.Tensor, bf16_when_custom: bool) -> torch.Tensor:
    """Sum the ``wo_b`` partials of the TP ranks, rounding once to BF16: in
    FP32, or with ``bf16_when_custom`` as BF16 when the custom all-reduce
    (FP32 accumulation in a fixed order, one rounding) takes the message."""
    if bf16_when_custom and custom_all_reduce_takes(partial):
        return tensor_model_parallel_all_reduce(partial)
    return tensor_model_parallel_all_reduce(partial.float()).to(partial.dtype)

def _first(linear: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Call a ``(output, bias)``-returning linear for its output."""

    def call(x: torch.Tensor) -> torch.Tensor:
        return linear(x)[0]

    return call


# ---------------------------------------------------------------------------
# Hooks into upstream (one decorator each)
# ---------------------------------------------------------------------------


def _is_sm80() -> bool:
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major == 8


def sm80_select_attn_cls(
    select: Callable[[VllmConfig], type[DeepseekV4Attention]],
) -> Callable[[VllmConfig], type[DeepseekV4Attention]]:
    """Decorates ``_select_dsv4_attn_cls``: compute capability 8.x always
    takes :class:`DeepseekV41Sm80Attention` (no other backend runs there)."""

    @functools.wraps(select)
    def wrapper(vllm_config: VllmConfig) -> type[DeepseekV4Attention]:
        if _is_sm80():
            return DeepseekV41Sm80Attention
        return select(vllm_config)

    return wrapper


def sm80_dspark_context_kv(store: Callable[..., None]) -> Callable[..., None]:
    """Decorates the DSpark drafter's ``precompute_and_store_context_kv``:
    sm_80 draft layers write their context window rows themselves."""

    @functools.wraps(store)
    def wrapper(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        layers = list(self.layers)
        if not layers or not isinstance(layers[0].attn, DeepseekV41Sm80Attention):
            return store(self, main_x, context_positions, context_slot_mappings)
        for i, layer in enumerate(layers):
            slot_mapping = (
                None if context_slot_mappings is None else context_slot_mappings[i]
            )
            layer.attn.sm80_store_context_kv(main_x, context_positions, slot_mapping)
        return None

    return wrapper
