# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 model classes of the sm_80 fork.

The fork plugin (``vllm.sm80.plugin``) registers these in vLLM's model
registry in place of the upstream ``DeepseekV41ForCausalLM`` (the vision
wrapper around the text model) and the DSpark drafter
(``DSparkV41DraftModel``). Each subclasses the upstream class, keeps its
name, and is built by the upstream constructor; the modules it builds are
then given the fork classes below, so every module and parameter keeps its
upstream name (and the EXL3 checkpoint maps onto them through
``sm80_exl3.DeepseekV41Exl3Layout``).

What changes on sm_80:

- mHC runs in the reference's FP32 formulas (``ops/sm80_mhc.py``) instead of
  the TileLang/DeepGEMM kernels: the decoder layers' forward, the backbone's
  final post and the DSpark drafter's forward call them. With
  ``VLLM_DSV41_FUSED_MHC`` the decoder layers' delayed pre and fused post +
  pre run as two kernels per call instead (``ops/sm80_mhc_fused.py``, same
  contracts and rounding points; :func:`mhc_ops`);
- each MoE keeps the upstream modules and EXL3 experts, with an FP32 gate and
  the reference routing (``sm80_moe.adopt_moe``);
- the drafter holds no embedding or LM head of its own (the DSpark loader
  aliases the target's), and keeps its confidence head across the EXL3
  loader's one-tensor-at-a-time ``load_weights`` calls;
- the logits, the DSpark Markov bias and the drafts' logits are FP32, as the
  reference computes them: the plugin defaults ``head_dtype`` to float32
  (``vllm.sm80.plugin.HF_CONFIG_DEFAULTS``), the EXL3 head stores its FP32
  result in that dtype and both models' logits processors serve it
  (``Exl3HeadLogitsProcessor``).

The attention layers are the platform's (``_select_dsv4_attn_cls``); the
layer topology this supports is TP with whole-expert EP and no sequence
parallelism, mHC overlap or fused all-reduce (none of which sm_80 enables).
"""

from itertools import islice

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.models.deepseek_v41.nvidia import dspark as upstream_dspark
from vllm.models.deepseek_v41.nvidia import model as upstream_model
from vllm.models.deepseek_v41.nvidia import vl_model as upstream_vl
from vllm.models.deepseek_v41.nvidia.ops import sm80_mhc
from vllm.models.deepseek_v41.nvidia.ops.sm80_mhc import mhc_post
from vllm.models.deepseek_v41.nvidia.sm80_moe import adopt_moe
from vllm.sm80.kernels.exl3.quant import Exl3HeadLogitsProcessor

logger = init_logger(__name__)


def mhc_ops(config):
    """The module whose delayed mHC ops (``mhc_pre_delayed``,
    ``mhc_fused_post_pre_delayed``, ``mhc_post``) the decoder layers call:
    ``ops/sm80_mhc_fused.py`` with ``VLLM_DSV41_FUSED_MHC``, else the
    reference's torch formulas ``ops/sm80_mhc.py``. The layers look each op
    up on the module at every call."""
    if not envs.VLLM_DSV41_FUSED_MHC:
        return sm80_mhc
    from vllm.models.deepseek_v41.nvidia.ops import sm80_mhc_fused

    reason = sm80_mhc_fused.supports(config.hc_mult, config.hidden_size)
    if reason is not None:
        raise ValueError(f"VLLM_DSV41_FUSED_MHC: {reason}")
    logger.info_once(
        "DeepSeek-V4.1 sm_80: the decoder layers' delayed mHC runs as two "
        "kernels per call (VLLM_DSV41_FUSED_MHC=1)",
        scope="global",
    )
    return sm80_mhc_fused


class DeepseekV4DecoderLayer(upstream_model.DeepseekV4DecoderLayer):
    """Decoder layer whose mHC steps run the module :func:`mhc_ops` picks
    (``self._mhc_ops``).

    ``forward`` is the upstream one for the sm_80 topology: the first layer
    takes the repeated embedding (its coefficients from the whole stream),
    an Engram layer injects into the previous layer's post, every other
    boundary is the previous sublayer's post fused with the next delayed pre.
    """

    @classmethod
    def adopt(cls, layer: upstream_model.DeepseekV4DecoderLayer, config) -> None:
        assert type(layer) is upstream_model.DeepseekV4DecoderLayer, type(layer)
        if layer.use_sequence_parallel or layer.fuse_mhc_all_reduce:
            raise ValueError(
                "DeepSeek-V4.1 on sm_80 runs without sequence parallelism or "
                "the fused mHC all-reduce"
            )
        if layer.mhc_stream is not None:
            raise ValueError("DeepSeek-V4.1 on sm_80 has no mHC overlap stream")
        layer.__class__ = cls
        adopt_moe(layer.ffn, config)
        layer._mhc_ops = mhc_ops(config)

    def _mixes(self, attn: bool) -> tuple:
        if attn:
            fn, scale, base = self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        else:
            fn, scale, base = self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        return (
            fn,
            scale,
            base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )

    def forward(
        self,
        x,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        *,
        capture_previous_aux: bool = False,
        mega_gate_metadata=None,
    ):
        previous_aux = None
        ops = self._mhc_ops
        attn_norm = {
            "norm_weight": self.attn_norm.weight,
            "norm_eps": self.attn_norm.variance_epsilon,
        }
        if residual is None:
            # Model entry: the embedding repeated hc_mult times; a None
            # pre-mix is the identity one (stream zero).
            residual = (
                x.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
                if x.dim() == 2
                else x
            )
            post_mix, res_mix, x, attn_pre = ops.mhc_pre_delayed(
                residual, *self._mixes(True), pre_mix=pre_mix, **attn_norm
            )
        elif self.engram is not None and engram_hashes is not None:
            # Engram injects into the previous sublayer's post, before this
            # block's coefficients are computed from the stream.
            previous_post = ops.mhc_post(x, residual, post_mix, res_mix)
            if capture_previous_aux:
                previous_aux = previous_post.mean(dim=1)
            residual = self.engram(
                previous_post,
                engram_hashes[:, self.engram.layer_hash_index],
                engram_mask,
            )
            post_mix, res_mix, x, attn_pre = ops.mhc_pre_delayed(
                residual, *self._mixes(True), pre_mix=pre_mix, **attn_norm
            )
        else:
            residual, post_mix, res_mix, x, attn_pre, aux = (
                ops.mhc_fused_post_pre_delayed(
                    x,
                    residual,
                    post_mix,
                    res_mix,
                    *self._mixes(True),
                    pre_mix=pre_mix,
                    capture_aux=capture_previous_aux,
                    **attn_norm,
                )
            )
            if capture_previous_aux:
                previous_aux = aux

        x = self.attn(positions, x, None)

        # The FFN collapses with the pre-mix this layer's attention produced.
        residual, post_mix, res_mix, x, ffn_pre, _ = ops.mhc_fused_post_pre_delayed(
            x,
            residual,
            post_mix,
            res_mix,
            *self._mixes(False),
            pre_mix=attn_pre,
            norm_weight=self.ffn_norm.weight,
            norm_eps=self.ffn_norm.variance_epsilon,
        )
        x = self.ffn(x, input_ids, mega_gate_metadata)
        return x, residual, post_mix, res_mix, ffn_pre, previous_aux

    def write_kv(
        self,
        x,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        """Upstream ``write_kv`` with the sm_80 mHC."""
        assert self.engram is None
        *_, x, _, _ = self._mhc_ops.mhc_fused_post_pre_delayed(
            x,
            residual,
            post_mix,
            res_mix,
            *self._mixes(True),
            pre_mix=pre_mix,
            norm_weight=self.attn_norm.weight,
            norm_eps=self.attn_norm.variance_epsilon,
        )
        self.attn.forward_kv(positions, x)


def _adopt_layers(layers, config) -> None:
    for layer in layers:
        if not isinstance(layer, PPMissingLayer):
            DeepseekV4DecoderLayer.adopt(layer, config)


class DeepseekV4Model(upstream_model.DeepseekV4Model):
    """Backbone whose final post runs ``ops/sm80_mhc.py``; the last collapse
    stays ``hc_collapse_triton`` (the reference ``hc_pre`` order: FP32
    products summed per stream, one BF16 rounding)."""

    @classmethod
    def adopt(cls, model: upstream_model.DeepseekV4Model) -> None:
        assert type(model) is upstream_model.DeepseekV4Model, type(model)
        if model.use_sequence_parallel or model.fuse_mhc_all_reduce:
            raise ValueError(
                "DeepSeek-V4.1 on sm_80 runs without sequence parallelism or "
                "the fused mHC all-reduce"
            )
        model.__class__ = cls
        _adopt_layers(islice(model.layers, model.start_layer, model.end_layer), model.config)

    def _collapse(
        self,
        hidden_states,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        aux_hidden_by_layer: dict[int, torch.Tensor],
        full_num_tokens: int,
    ) -> torch.Tensor:
        assert isinstance(hidden_states, torch.Tensor)
        hidden_states = mhc_post(hidden_states, residual, post_mix, res_mix)
        if self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_by_layer[self.end_layer] = hidden_states.mean(dim=1)
        return hidden_states


class DeepseekV41ForCausalLM(upstream_vl.DeepseekV41ForCausalLM):
    """The DeepSeek-V4.1 vision wrapper with the sm_80 text model.

    The vision tower and aligner stay upstream's (BF16, replicated: EXL3 runs
    them data-parallel, ``--mm-encoder-tp-mode data``); their encoder
    attention is the platform's ViT backend (FLASH_ATTN on sm_80)."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        DeepseekV4Model.adopt(self.language_model.model)
        # Upstream builds the head unquantized (the base checkpoint's is
        # BF16); the EXL3 checkpoint's head is packed, so it is built with
        # the model's quantization config.
        config = self.language_model.config
        self.language_model.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{maybe_prefix(prefix, 'language_model')}.lm_head",
        )
        Exl3HeadLogitsProcessor.adopt(self.language_model.logits_processor)


class DSparkDeepseekV4Model(upstream_dspark.DSparkDeepseekV4Model):
    """DSpark stages whose mHC runs ``ops/sm80_mhc.py``."""

    @classmethod
    def adopt(cls, model: upstream_dspark.DSparkDeepseekV4Model) -> None:
        assert type(model) is upstream_dspark.DSparkDeepseekV4Model, type(model)
        if model.use_sequence_parallel or model.use_native_mega_moe:
            raise ValueError(
                "DeepSeek-V4.1 DSpark on sm_80 runs without sequence "
                "parallelism or MegaMoE"
            )
        model.__class__ = cls
        _adopt_layers(model.layers, model.config)
        # Aliased from the target by the DSpark loader (load_dspark_model).
        model.embed_tokens = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        residual = post_mix = res_mix = pre_mix = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix, pre_mix, _ = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
            )
        hidden_states = mhc_post(hidden_states, residual, post_mix, res_mix)
        # The pre-norm head hidden: the last FFN pre-mix collapse
        # (compute_logits applies the norm).
        assert pre_mix is not None
        return hc_collapse_triton(hidden_states, pre_mix)


class DSparkDeepseekV4ForCausalLM(upstream_dspark.DSparkDeepseekV4ForCausalLM):
    """The DeepSeek-V4.1 DSpark drafter with the sm_80 stages."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        DSparkDeepseekV4Model.adopt(self.model)
        # Aliased from the target by the DSpark loader (load_dspark_model).
        self.lm_head = PPMissingLayer()
        # Base logits through the target's EXL3 head; the Markov bias through
        # the upstream unquantized-head path, in the head dtype too.
        Exl3HeadLogitsProcessor.adopt(self.logits_processor)

    def load_weights(self, weights) -> set[str]:
        """Upstream ``load_weights``, without dropping the confidence head
        when one call does not carry it: the EXL3 loader passes one tensor
        per call and refuses a checkpoint whose head parameters stay unloaded.
        """
        confidence_head = self.model.confidence_head
        loaded = super().load_weights(weights)
        self.model.confidence_head = confidence_head
        return loaded
