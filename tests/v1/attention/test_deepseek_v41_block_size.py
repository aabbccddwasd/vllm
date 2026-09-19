# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel block-size selection for DeepSeek-V4.1 sparse MLA on Blackwell client.

FlashInfer's SM120 sparse-MLA decode kernels only ship 64-token pages, and the
DeepGEMM paged-MQA logits indexer only serves a 64-state page. V4.1 mixes
ratio-1 and ratio-2 layers and the kernels read
``num_states = block_size // tokens_per_state``, so on family 12 the cache block
is sized by the compression ratio (``64 * ratio``) to keep every ratio on that
page instead of splitting the manager block.

These tests patch the platform's reported capability, so they need no client
Blackwell GPU.
"""

from unittest.mock import patch

import pytest
import torch

from vllm.models.deepseek_v41.attention import (
    _sm12x_sparse_page_tokens,
    _swa_cache_block_size,
)
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLASparseBackend,
)
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4SparseMLABackend,
    FlashMLAMegaAttnBackend,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
from vllm.v1.kv_cache_interface import MLAAttentionSpec

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="DeepSeek V4.1 sparse MLA backends require a CUDA build of vLLM",
)

# Family 12 = sm120 (RTX PRO 6000 / RTX 50) and sm121 (GB10).
FAMILY_12 = [
    pytest.param(12, 0, id="sm120-blackwell-client"),
    pytest.param(12, 1, id="sm121-gb10"),
]
NOT_FAMILY_12 = [
    pytest.param(9, 0, id="sm90-hopper"),
    pytest.param(10, 0, id="sm100-blackwell-datacenter"),
]


def _capability(major: int, minor: int):
    """Patch the platform's reported device capability for one test."""
    return patch.object(
        type(current_platform),
        "get_device_capability",
        return_value=DeviceCapability(major, minor),
    )


@pytest.mark.parametrize(("major", "minor"), FAMILY_12)
def test_swa_cache_page_is_64_on_family_12(major: int, minor: int):
    with _capability(major, minor):
        assert _swa_cache_block_size() == 64


@pytest.mark.parametrize(("major", "minor"), NOT_FAMILY_12)
def test_swa_cache_page_stays_32_elsewhere(major: int, minor: int):
    with _capability(major, minor):
        assert _swa_cache_block_size() == 32


@pytest.mark.parametrize(
    ("major", "minor"), FAMILY_12 + [pytest.param(9, 0, id="sm90-hopper")]
)
def test_sparse_mla_backend_declares_the_64_page(major: int, minor: int):
    with _capability(major, minor):
        assert DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes() == [64]


def test_sparse_mla_backend_declares_the_128_page_on_sm100():
    with _capability(10, 0):
        assert DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes() == [128]


@pytest.mark.parametrize(("major", "minor"), FAMILY_12)
def test_flashinfer_sparse_backend_accepts_every_ratio_page(major: int, minor: int):
    with _capability(major, minor):
        sizes = DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes()
        assert sizes == [64, 128]


@pytest.mark.parametrize(("major", "minor"), NOT_FAMILY_12)
def test_flashinfer_sparse_backend_is_unchanged_elsewhere(major: int, minor: int):
    with _capability(major, minor):
        sizes = DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes()
        assert sizes == [128]


@pytest.mark.parametrize(("major", "minor"), FAMILY_12)
def test_indexer_backend_accepts_every_ratio_page_on_family_12(major: int, minor: int):
    with _capability(major, minor):
        assert DeepseekV41IndexerBackend.get_supported_kernel_block_sizes() == [64, 128]


def test_indexer_backend_is_unchanged_on_hopper_and_datacenter():
    with _capability(9, 0):
        assert DeepseekV41IndexerBackend.get_supported_kernel_block_sizes() == [64]
    with _capability(10, 0):
        assert DeepseekV41IndexerBackend.get_supported_kernel_block_sizes() == [128]


def test_mega_attn_backend_stays_datacenter_only():
    """FlashMLAMegaAttnBackend has no family-12 instantiation, so it keeps 128."""
    for major, minor in ((10, 0), (12, 0), (12, 1)):
        with _capability(major, minor):
            assert FlashMLAMegaAttnBackend.get_supported_kernel_block_sizes() == [128]


def test_flashmla_backend_declaration_is_unchanged():
    with _capability(10, 0):
        assert DeepseekV4FlashMLABackend.get_supported_kernel_block_sizes() == [128]


@pytest.mark.parametrize(
    ("compress_ratio", "expected_block_size"),
    [(1, 64), (2, 128)],
)
def test_sm12x_page_keeps_num_states_at_64(
    compress_ratio: int, expected_block_size: int
):
    """Every ratio has to reach the kernel as a 64-state page."""
    block_size = _sm12x_sparse_page_tokens(compress_ratio)
    assert block_size == expected_block_size
    spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        tokens_per_state=compress_ratio,
    )
    assert spec.num_states == 64


def test_sm12x_page_never_collapses_to_zero():
    """A sliding-window-only layer (ratio 0) must not ask for a 0-token block."""
    assert _sm12x_sparse_page_tokens(0) == 64
