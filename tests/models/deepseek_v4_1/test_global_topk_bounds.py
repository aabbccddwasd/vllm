# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local→global top-k translation must not index the block table out of bounds.

The top-k producers only guarantee that *valid* entries are non-negative: a slot
they never wrote keeps whatever the buffer held. Translating such an entry names
a block the request's block table cannot hold, and the sparse-MLA kernels then
gather from a slot outside the KV cache (observed as a Warp MMU fault in the
SM120 dual-cache prefill kernel with DSpark enabled, and as NaN attention output
on decode).
"""

import pytest
import torch

from vllm.models.deepseek_v41.common.ops import (
    compute_global_topk_indices_and_lens,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the translation kernel is a Triton kernel"
)

BLOCK_SIZE = 64


def _run(topk_indices: torch.Tensor, num_blocks_per_req: int):
    num_tokens, topk = topk_indices.shape
    # Two requests, block tables holding distinct, easily recognisable slots.
    block_table = torch.zeros((1, num_blocks_per_req), dtype=torch.int32, device="cuda")
    block_table[0] = torch.arange(100, 100 + num_blocks_per_req, device="cuda")
    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
    is_valid_token = torch.ones(num_tokens, dtype=torch.bool, device="cuda")
    return compute_global_topk_indices_and_lens(
        topk_indices, token_to_req, block_table, BLOCK_SIZE, is_valid_token
    )


def test_in_range_indices_translate_and_count():
    """The bound must not disturb indices a request can actually hold."""
    num_blocks = 2
    topk = torch.tensor([[0, BLOCK_SIZE, -1, -1]], dtype=torch.int32, device="cuda")
    indices, lens = _run(topk, num_blocks)
    assert lens.tolist() == [2]
    assert indices[0, 0].item() == 100 * BLOCK_SIZE
    assert indices[0, 1].item() == 101 * BLOCK_SIZE
    assert indices[0, 2].item() == -1 and indices[0, 3].item() == -1


def test_index_past_the_block_table_is_masked_not_translated():
    """An unwritten slot (garbage index) must be dropped, not translated."""
    num_blocks = 2
    width = num_blocks * BLOCK_SIZE
    topk = torch.tensor(
        [[0, width, width + 1, 2**30]], dtype=torch.int32, device="cuda"
    )
    indices, lens = _run(topk, num_blocks)
    assert lens.tolist() == [1]
    assert indices[0, 0].item() == 100 * BLOCK_SIZE
    assert (indices[0, 1:] == -1).all()
