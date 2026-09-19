# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression guard: the two multi-stream overlaps must not share event handles.

``DeepseekV4Attention`` overlaps the input GEMM (compressor kv_score,
indexer.weights_proj) with ``fused_wqa_wkv``, then overlaps a second time around
the compressed-KV insert / indexer prep. If both overlaps use one set of event
handles, the second recording overwrites the first generation before the first
join's waits have executed, so the sparse-MLA read can observe a half-written
cache. Under CUDA graph capture the same handle is additionally baked into more
than one graph, and a replay can then satisfy a wait from a stale generation.

These tests assert the separation at the real call sites, with no GPU and no
model build: the lease must hand back capture-private handles, and
``_prepare_and_attn`` must actually use them for both post-GEMM overlaps.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.models.deepseek_v41 import attention as attention_module
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.utils.multi_stream_utils import (
    CUDAGraphCaptureEventPool,
    vllm_cudagraph_capture_scope,
)


class _ConcreteAttention(DeepseekV4Attention):
    """Concrete subclass so ``object.__new__`` can bypass ``__init__``."""

    def get_padded_num_q_heads(cls, num_heads: int) -> int:  # type: ignore[override]
        return num_heads

    def forward_mqa(self, q, kv, positions, output) -> None:
        return None

    def _o_proj(self, o, positions):
        return o


def _bare_layer(**attrs) -> DeepseekV4Attention:
    layer = object.__new__(_ConcreteAttention)
    for name, value in attrs.items():
        setattr(layer, name, value)
    return layer


def test_post_gemm_lease_is_distinct_from_the_gemm_event_set() -> None:
    ln_events = [torch.cuda.Event() for _ in range(4)]
    pool = CUDAGraphCaptureEventPool(2)
    layer = _bare_layer(
        ln_events=ln_events,
        attn_event_pool=pool,
        attn_events=pool.default_events,
    )

    with layer._post_gemm_event_lease() as leased:
        # The post-GEMM overlap must never reuse the input GEMM's handles.
        assert set(map(id, leased)).isdisjoint(map(id, ln_events))
        # Eager (not capturing) private_eager lease: a fresh generation.
        assert leased is not pool.default_events
        assert len(leased) == 2


def test_post_gemm_lease_hands_back_capture_private_handles() -> None:
    pool = CUDAGraphCaptureEventPool(2)
    layer = _bare_layer(
        ln_events=[torch.cuda.Event() for _ in range(4)],
        attn_event_pool=pool,
        attn_events=pool.default_events,
    )

    with layer._post_gemm_event_lease() as first:
        pass
    with layer._post_gemm_event_lease() as second:
        pass
    # Two calls must not alias: the second recording would otherwise overwrite
    # the generation the first join is still waiting on.
    assert set(map(id, first)).isdisjoint(map(id, second))

    # A capture scope must retain the set so the graph owns those handles, and
    # must never fall back to the module-wide default set.
    with vllm_cudagraph_capture_scope(), layer._post_gemm_event_lease() as cap:
        assert len(cap) == 2
    assert cap in pool._captured_event_sets
    assert set(map(id, cap)).isdisjoint(map(id, pool.default_events))


def test_prepare_and_attn_uses_leased_events_for_both_overlaps() -> None:
    """The real call sites must pass the leased handles, not ln_events."""
    ln_events = [torch.cuda.Event() for _ in range(4)]
    pool = CUDAGraphCaptureEventPool(2)
    layer = _bare_layer(
        ln_events=ln_events,
        attn_event_pool=pool,
        attn_events=pool.default_events,
        indexer=None,
        compressor=object(),
        aux_stream_list=[object()],
        rotary_emb=object(),
        indexer_rotary_emb=object(),
        _wq_b_proj=lambda qr, scale: qr,
        _fused_qnorm_rope_kv_insert=lambda q, kv, pos, md: q,
        _sparse_indexer_and_attn=lambda *a, **k: None,
    )
    seen: list[tuple] = []
    calls = {"n": 0}

    def fake_maybe(fn0, fn1, event0, event1, aux_stream=None):
        seen.append((event0, event1))
        calls["n"] += 1
        if calls["n"] == 1:
            # Site 1 unpacks (q, latent).
            return (torch.empty(1), None)
        # Site 2 unpacks (indexer_result, _); the indexer result is itself the
        # 3-tuple that prepare_indexer() returns.
        return (fn0(), None)

    with (
        patch.object(attention_module, "maybe_execute_in_parallel", fake_maybe),
        patch.object(
            attention_module,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=None),
        ),
    ):
        layer._prepare_and_attn(
            torch.empty(1),  # hidden_states
            torch.empty(1),  # qr
            torch.empty(1),  # kv
            None,  # qr_scale
            torch.empty(1),  # kv_score
            torch.empty(1),  # indexer_weights
            torch.empty(1, dtype=torch.int64),  # positions
            torch.empty(1),  # o_padded
        )

    # Two post-GEMM overlaps, each on its own leased generation.
    assert len(seen) == 2
    for event0, event1 in seen:
        assert event0 not in ln_events
        assert event1 not in ln_events
    assert seen[0][0] is not seen[1][0]
    assert seen[0][1] is not seen[1][1]


def test_lease_falls_back_without_a_pool() -> None:
    """Layers without a pool must still yield usable handles."""
    attn_events = [torch.cuda.Event(), torch.cuda.Event()]
    with_pool_attr = _bare_layer(attn_events=attn_events)
    with with_pool_attr._post_gemm_event_lease() as events:
        assert events is attn_events

    ln_events = [torch.cuda.Event() for _ in range(4)]
    without_attn_events = _bare_layer(ln_events=ln_events)
    with without_attn_events._post_gemm_event_lease() as events:
        assert events == ln_events[:2]
