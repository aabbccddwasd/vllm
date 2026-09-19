# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``compute_global_topk_indices_and_lens`` must accept caller-owned buffers.

The fused local->global top-k translation runs inside the captured decode
region. Allocating its two outputs there puts them in the graph's private pool,
so the addresses are frozen per graph and every layer of a step re-derives them
from scratch. Callers can instead pass persistent buffers and get bit-identical
results, which is what the graph-persistent path relies on.

These tests pin the contract: same values as the allocating path, the caller's
storage is what gets written, and rows past ``num_tokens`` are left untouched so
an oversized persistent buffer is safe to reuse across steps.
"""

import pytest
import torch

from vllm.models.deepseek_v41.common.ops.cache_utils import (
    compute_global_topk_indices_and_lens,
)


def _case(num_tokens=6, topk=8, block_size=4, num_blocks=16, device="cuda"):
    torch.manual_seed(0)
    topk_indices = torch.randint(
        0,
        num_blocks * block_size,
        (num_tokens, topk),
        dtype=torch.int32,
        device=device,
    )
    # A sentinel past the block table must not be translated.
    topk_indices[0, topk // 2 :] = 10_000
    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, -1)
    is_valid = torch.ones(num_tokens, dtype=torch.bool, device=device)
    is_valid[-1] = False  # padding row must report length 0
    return topk_indices, token_to_req, block_table, is_valid, block_size


def test_out_buffers_reproduce_the_allocating_path() -> None:
    args = _case()
    topk_indices, token_to_req, block_table, is_valid, block_size = args

    ref_i, ref_l = compute_global_topk_indices_and_lens(
        topk_indices, token_to_req, block_table, block_size, is_valid
    )
    buf_i = torch.full_like(topk_indices, -7)
    buf_l = torch.full((topk_indices.shape[0],), -7, dtype=torch.int32, device="cuda")
    got_i, got_l = compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req,
        block_table,
        block_size,
        is_valid,
        out_indices=buf_i,
        out_lens=buf_l,
    )
    torch.cuda.synchronize()

    assert torch.equal(got_i, ref_i)
    assert torch.equal(got_l, ref_l)
    # The caller's own storage is what the kernel wrote.
    assert torch.equal(buf_i, ref_i)
    assert torch.equal(buf_l, ref_l)
    # Padding rows report zero valid entries.
    assert int(ref_l[-1]) == 0


def test_oversized_persistent_buffer_keeps_trailing_rows_intact() -> None:
    """A step-sized slice of a max-token buffer must not disturb the tail."""
    topk_indices, token_to_req, block_table, is_valid, block_size = _case()
    num_tokens = topk_indices.shape[0]
    spare = 4
    big_i = torch.full(
        (num_tokens + spare, topk_indices.shape[-1]),
        -7,
        dtype=torch.int32,
        device="cuda",
    )
    big_l = torch.full((num_tokens + spare,), -7, dtype=torch.int32, device="cuda")

    got_i, got_l = compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req,
        block_table,
        block_size,
        is_valid,
        out_indices=big_i,
        out_lens=big_l,
    )
    torch.cuda.synchronize()

    assert got_i.shape[0] == num_tokens
    assert got_l.shape[0] == num_tokens
    assert bool((big_l[num_tokens:] == -7).all())
    assert bool((big_i[num_tokens:] == -7).all())


def test_out_buffer_shape_mismatch_is_rejected() -> None:
    topk_indices, token_to_req, block_table, is_valid, block_size = _case()
    num_tokens = topk_indices.shape[0]

    with pytest.raises(AssertionError):
        compute_global_topk_indices_and_lens(
            topk_indices,
            token_to_req,
            block_table,
            block_size,
            is_valid,
            out_indices=torch.empty(
                num_tokens, topk_indices.shape[-1] + 4, dtype=torch.int32, device="cuda"
            ),
        )
    with pytest.raises(AssertionError):
        compute_global_topk_indices_and_lens(
            topk_indices,
            token_to_req,
            block_table,
            block_size,
            is_valid,
            out_lens=torch.empty(num_tokens - 1, dtype=torch.int32, device="cuda"),
        )
