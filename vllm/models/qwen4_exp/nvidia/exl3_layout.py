# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp EXL3 checkpoint layout (vllm/sm80/kernels/exl3/layout.py).

The checkpoint keeps the HF module names under ``model.language_model.``, the
MTP draft's layers under ``mtp.layers.<i>`` (numbered after the target's in
the model), the compressed PLE table in ``ngram_embedding.safetensors`` and
the MTP mixer in its own patch file.
"""

import re

from vllm.sm80.kernels.exl3.layout import Exl3Layout
from vllm.sm80.kernels.exl3.utils import PLE_AUXILIARIES

# Qwen4Exp QSA projects Q/gate, K, V and the replicated indexer Q/K in one
# layer. It first builds a three-output QKV layer, then replaces it with the
# four-output fused one, and the target, MTP and LoRA packed mappings disagree
# on the indexer, so each layer's output count selects its members.
_QSA_QKV_MEMBERS = ("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj")


class Qwen4ExpExl3Layout(Exl3Layout):
    requires_qwen_aux = True

    def source_prefix(self, config, prefix):
        if prefix.startswith("model.language_model.model."):
            prefix = prefix.replace(
                "model.language_model.model.", "model.language_model.", 1
            )
        elif prefix.startswith("language_model.model."):
            prefix = prefix.replace("language_model.model.", "model.language_model.", 1)
        elif prefix.startswith(("model.layers.", "model.embed_tokens.")):
            prefix = "model.language_model." + prefix.removeprefix("model.")
        for head_prefix in ("model.language_model.lm_head", "language_model.lm_head"):
            if prefix == head_prefix:
                prefix = "lm_head"
        if prefix.startswith("visual."):
            prefix = "model." + prefix
        if prefix.startswith("model.mtp."):
            prefix = prefix.removeprefix("model.")
        if prefix.startswith("mtp.layers."):
            match = re.match(r"mtp.layers.(\d+)\.(.*)", prefix)
            assert match is not None
            index = int(match[1])
            if index >= config.mtp_start_layer_idx:
                prefix = f"mtp.layers.{index - config.mtp_start_layer_idx}.{match[2]}"
        return prefix

    def members(self, config, leaf, source, outputs):
        if leaf == "qkv" and ".visual." in source:
            return ["q_proj", "k_proj", "v_proj"]
        if leaf == "qkv_proj" and outputs:
            return list(_QSA_QKV_MEMBERS[:outputs])
        return None

    def is_draft(self, model_config):
        return model_config.hf_config.model_type == "qwen4_exp_mtp"

    def assign_owners(self, model, config, ledger, model_config, is_draft):
        manifest = config.manifest
        shared: set[str] = set()
        if is_draft and config.mtp_shares_target_embeddings(
            model_config.hf_text_config
        ):
            # The draft reuses the target's embedding and head: no draft owner.
            shared = {
                name
                for name in manifest.tensors
                if name.startswith(("lm_head.", "model.language_model.embed_tokens."))
            }
        for name in manifest.tensors:
            if name in ledger.entries:
                continue
            if name in shared:
                ledger.mark(name, "unused", "shared target embedding/head owner")
            elif name.startswith("mtp.") and not is_draft:
                ledger.mark(name, "unused", "native MTP model owner")
            elif is_draft and not name.startswith("mtp."):
                if name.startswith(("lm_head.", "model.language_model.embed_tokens.")):
                    continue
                ledger.mark(name, "unused", "target model owner")

        from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
            Qwen4ExpNGramEmbedding,
        )

        file_backed = set()
        for name, module in model.named_modules():
            if not isinstance(module, Qwen4ExpNGramEmbedding):
                continue
            source = config.source_prefix(f"{name}.ngram_embedding")
            table = manifest.ple_tables.get(source)
            if table is None:
                raise ValueError(f"Missing EXL3 PLE source for module: {name}")
            expected = {table.name} | {
                f"{source}.{suffix}" for suffix in PLE_AUXILIARIES
            }
            consumed = _setup_ple_file(module, str(table.path), table.name)
            if consumed != expected:
                raise ValueError(
                    f"Incomplete or unexpected EXL3 PLE source join: {source}"
                )
            for key in sorted(consumed):
                ledger.mark(key, "file-backed" if key == table.name else "consumed", name)
            file_backed.add(f"{name}.ngram_embedding.weight")
        return file_backed


def _setup_ple_file(module, path: str, key: str) -> set[str]:
    """Join a file-backed manifest entry to a Qwen4ExpNGramEmbedding and
    consume its small auxiliaries. Reads the module's ngram_embedding,
    ngram_heads_offsets, ngram_heads_vocab_sizes and layer_multipliers
    (pinned in tests/sm80/seams.yaml)."""
    from vllm.models.qwen4_exp.nvidia.exl3_ple import Exl3PLEEmbedding

    if not isinstance(module.ngram_embedding, Exl3PLEEmbedding):
        raise TypeError("File-backed PLE setup requires EXL3 quantization")
    return module.ngram_embedding.setup_file(
        path,
        key,
        module.ngram_heads_offsets,
        module.ngram_heads_vocab_sizes,
        module.layer_multipliers,
    )
