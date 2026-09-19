# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-size selection for the DeepSeek V4.1 Engram warmup.

The Engram row gather is JIT'ed on its first call and that call happens inside
the captured decode graph, so the warmup has to cover exactly the token sizes
serving can present -- no more (each size is a compile) and no fewer (a missing
size compiles during serving).
"""

import pytest

from vllm.model_executor.warmup.deepseek_v4_engram_warmup import (
    _select_engram_warmup_token_sizes,
)


def test_covers_capture_sizes_and_the_batch_budget():
    sizes = _select_engram_warmup_token_sizes(
        max_tokens=4096, cudagraph_capture_sizes=[1, 2, 4, 8, 96]
    )
    assert set(sizes) <= set(range(1, 4097))
    for expected in (1, 2, 4, 8, 96, 4096):
        assert expected in sizes


def test_is_sorted_and_deduplicated():
    sizes = _select_engram_warmup_token_sizes(
        max_tokens=512, cudagraph_capture_sizes=[4, 4, 8, 1]
    )
    assert sizes == sorted(set(sizes))


def test_ignores_sizes_above_the_budget():
    sizes = _select_engram_warmup_token_sizes(
        max_tokens=64, cudagraph_capture_sizes=[1, 128, 1024]
    )
    assert max(sizes) <= 64
    assert 128 not in sizes and 1024 not in sizes


def test_caps_the_auto_covered_range():
    """A large --max-num-batched-tokens must not turn warmup into a compile storm."""
    sizes = _select_engram_warmup_token_sizes(
        max_tokens=1 << 20, cudagraph_capture_sizes=[]
    )
    assert max(sizes) <= 16_384


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_no_sizes_for_a_nonpositive_budget(max_tokens):
    assert (
        _select_engram_warmup_token_sizes(
            max_tokens=max_tokens, cudagraph_capture_sizes=[1, 2]
        )
        == []
    )
