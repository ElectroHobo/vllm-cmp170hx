# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Early Engram gathers and projection on the registered auxiliary stream.

The lookup and WKV projection depend only on this step's hashes. Issue them
before decoder compute and join each layer's own completion at injection.
The registered stream owns EXL3 scratch[1]; main-stream projections own
scratch[0]. Shared-expert projections use the same auxiliary stream, so they
cannot overlap a WKV projection using its scratch. No third scratch is needed.
"""

from itertools import islice

import torch

from vllm.models.deepseek_v41.common.engram import Engram
from vllm.utils.torch_utils import aux_stream, current_stream


class EarlyEngram(Engram):
    """An Engram with one outstanding projection, consumed by its forward."""

    @classmethod
    def adopt(cls, engram: Engram) -> None:
        if engram.use_sequence_parallel or engram._batch is not None:
            raise ValueError("early Engram requires unbatched TP projection")
        if engram.embed_tokens.dp_size != 1:
            raise ValueError("early Engram requires TP-only table sharding")
        stream = aux_stream()
        if stream is None:
            raise ValueError("early Engram requires the registered auxiliary stream")
        engram.__class__ = cls
        engram._prefetch_stream = stream
        engram._prefetch_done = torch.cuda.Event()
        engram._projection_done = torch.cuda.Event()
        engram._early_kv = None

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        if self._early_kv is not None:
            raise RuntimeError("previous Engram projection was not consumed")
        stream = self._prefetch_stream
        assert stream is not None and stream == aux_stream()
        stream.wait_stream(current_stream())
        hash_ids.record_stream(stream)
        with torch.cuda.stream(stream):
            # Keep lookup, both gathers and WKV on the registered stream,
            # including large lookups that the upstream helper runs inline.
            super().prepare_embeddings(hash_ids)
            self._early_kv = super()._project_kv(hash_ids)
            self._projection_done.record(stream)

    def _project_kv(self, hash_ids: torch.Tensor) -> torch.Tensor:
        kv = self._early_kv
        if kv is None or kv.shape[0] != hash_ids.shape[0]:
            raise RuntimeError("Engram projection missing for this step's rows")
        stream = current_stream()
        stream.wait_event(self._projection_done)
        # The tensor was allocated on aux. Its last reader runs on main;
        # retain its storage through that reader before a later aux allocation.
        kv.record_stream(stream)
        self._early_kv = None
        return kv


def adopt_early_engram(model) -> None:
    """Enable early gathers only for a topology with one forward in flight."""
    if model.vllm_config.parallel_config.use_ubatching:
        raise ValueError("early Engram does not support microbatching")
    mode = model.vllm_config.compilation_config.cudagraph_mode
    if mode is not None and mode.has_piecewise_cudagraphs():
        raise ValueError(
            "early Engram requires eager or full CUDA graphs; "
            "piecewise capture cannot retain an auxiliary-stream fork across breaks"
        )
    for layer in islice(model.layers, model.start_layer, model.end_layer):
        engram = getattr(layer, "engram", None)
        if engram is not None:
            EarlyEngram.adopt(engram)
