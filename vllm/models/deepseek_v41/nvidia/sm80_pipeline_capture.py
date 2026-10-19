# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coupled attention snapshots for the existing diagnostic capture service.

Install together with hybrid_capture, never during timing. Fixed-size sorted
block maps preserve cache aliases during CUDA graph replay. Duplicate snapshot
blocks are shadows; only their first representative is a physical post-state.
"""
import dataclasses
import functools
from contextvars import ContextVar

import torch

_current = ContextVar('deepseek_attention_capture', default=None)
_installed = False


def snapshot_cache(cache, *, table=None, slots=None, indices=None):
    if cache is None:
        return None, None, table, slots, indices
    entries = cache.shape[1]
    blocks = [torch.zeros(1, dtype=torch.int64, device=cache.device)]
    if table is not None:
        blocks.append(table.flatten().long().clamp_min(0))
    for value in (slots, indices):
        if value is not None:
            blocks.append(value.flatten().long().clamp_min(0) // entries)
    if slots is not None:
        blocks.append((slots.flatten().long().clamp_min(0) // entries + 1) % cache.shape[0])
    ids = torch.sort(torch.cat(blocks)).values
    def remap(value, is_slot):
        if value is None:
            return None
        block = value.long().clamp_min(0) // entries if is_slot else value.long().clamp_min(0)
        mapped = torch.searchsorted(ids, block.contiguous())
        if is_slot:
            mapped = mapped * entries + value.long().clamp_min(0) % entries
        return torch.where(value >= 0, mapped, -1).to(value.dtype)
    return cache[ids], ids, remap(table, False), remap(slots, True), remap(indices, True)


def install():
    global _installed
    if _installed:
        return
    from vllm.sm80 import capture
    if not capture.active:
        return
    from vllm.forward_context import get_forward_context
    from vllm.models.deepseek_v41.nvidia import sm80_attention
    from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core

    _installed = True
    cls = sm80_attention.DeepseekV41Sm80Attention
    original = cls._sm80_attention

    @functools.wraps(original)
    def attention(self, qr_kv, kv_score, index_weights, positions, out):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return original(self, qr_kv, kv_score, index_weights, positions, out)
        # The control is captured, the new body is replayed offline. This also
        # retains the independent score/latent boundaries, not a preselected
        # compressed list masquerading as full pipeline input.
        if self._sm80_attn_pipeline_enabled:
            raise ValueError('coupled capture requires the qualified control arm')
        if qr_kv.shape[0] > self._sm80_pipeline_max_rows:
            return original(self, qr_kv, kv_score, index_weights, positions, out)
        if not self._sm80_is_draft_layer and sm80_attention.prefill_step(self._sm80_max_decode_query_len):
            return original(self, qr_kv, kv_score, index_weights, positions, out)
        rec = capture.begin('attn', 'attn_pipeline', self.prefix + '.pipeline',
                            qr_kv.shape[0], cfg=dataclasses.asdict(self.sm80_cfg),
                            layer_prefix=self.prefix, drafter=self._sm80_is_draft_layer)
        if rec is None:
            return original(self, qr_kv, kv_score, index_weights, positions, out)
        batch = dataclasses.replace(core.pad_batch(self._sm80_batch(metadata, positions), qr_kv.shape[0]))
        caches = self._sm80_caches()
        weights = self._sm80_weights(qr_kv.device)
        projections = {'wq_b': self.wq_b, 'wo_a': self.wo_a}
        if self.indexer is not None:
            projections['index_wq_b'] = self.indexer.wq_b
            if self.indexer.owns_k:
                projections['wk'] = self.indexer.wk
        facts = {}
        for name, module in projections.items():
            fact = getattr(module, '_hybrid_capture', None)
            if fact is None:
                raise ValueError(f'coupled attention capture lacks loaded projection facts: {name}')
            facts[name] = fact
        rec.meta['projections'] = facts
        rec.meta['fused_control'] = self._sm80_fused_attn_call(qr_kv.shape[0])
        rec.put(qr_kv=qr_kv, kv_score=kv_score, index_weights=index_weights,
                topk=caches.topk[:batch.num_tokens],
                candidates=None if caches.candidates is None else caches.candidates[:batch.num_tokens])
        maps = {}
        swa, maps['swa'], _, batch.swa_slots, batch.swa_indices = snapshot_cache(
            caches.swa, slots=batch.swa_slots, indices=batch.swa_indices)
        comp, maps['comp'], batch.comp_block_table, batch.comp_slots, _ = snapshot_cache(
            caches.comp, table=batch.comp_block_table, slots=batch.comp_slots)
        idxk, maps['idxk'], batch.idxk_block_table, batch.idxk_slots, _ = snapshot_cache(
            caches.idxk, table=batch.idxk_block_table, slots=batch.idxk_slots)
        ring, maps['ring'], _, batch.ring_slots, _ = snapshot_cache(caches.ring, slots=batch.ring_slots)
        rec.put(swa=swa, comp=comp, idxk=idxk, ring=ring)
        rec.put(**{f'batch.{f.name}': getattr(batch, f.name) for f in dataclasses.fields(batch)})
        for name in ('q_norm', 'kv_norm', 'comp_norm', 'k_norm', 'freqs', 'attn_sink'):
            rec.put(**{name: getattr(weights, name)})
        for name, ids in maps.items():
            rec.put(**{f'blocks.{name}': ids})
        token = _current.set(rec)
        try:
            result = original(self, qr_kv, kv_score, index_weights, positions, out)
            rec.put(out=out, **{'post.topk': caches.topk[:batch.num_tokens],
                'post.candidates': None if caches.candidates is None else caches.candidates[:batch.num_tokens]})
            for name, ids in maps.items():
                rec.put(**{f'post.{name}': None if ids is None else getattr(caches, name)[ids]})
            rec.done()
            return result
        finally:
            _current.reset(token)

    cls._sm80_attention = attention

    def wrap(name, outputs, module=core):
        fn = getattr(module, name)
        @functools.wraps(fn)
        def record(*args, **kwargs):
            result = fn(*args, **kwargs)
            rec = _current.get()
            if rec is not None:
                values = result if isinstance(result, tuple) else (result,)
                rec.put(**dict(zip(outputs, values)))
            return result
        setattr(module, name, record)

    wrap('compress_rows', ('closes', 'latent'))
    wrap('index_query', ('index_query', 'scaled_index_weights'))
    wrap('index_key_rows', ('index_keys',))
    wrap('index_scores', ('index_scores_unmasked',))
    wrap('prepare_layer', ('query',))
    from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_fused
    wrap('prologue', ('qr',), sm80_attn_fused)
