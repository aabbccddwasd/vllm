# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the DeepSeek V4.1 candidate-block Triton kernels before serving.

Both the prefill and the decode path launch these kernels with a constexpr
tuple that varies by call site (whether row starts are present, and the
speculative row repeat), so the first request otherwise JITs them mid-step and
spikes latency. Gating is intrinsic: a model without candidate-block indexers
returns early.
"""

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    warmup_candidate_block_kernels,
)
from vllm.tracing import instrument

logger = init_logger(__name__)


def _find_candidate_block_owners(model: torch.nn.Module) -> list[Any]:
    """Collect indexers that publish or consume candidate blocks, by shape."""
    owners: list[Any] = []
    seen: set[int] = set()
    for module in model.modules():
        candidates = (module, getattr(module, "indexer_op", None))
        for owner in candidates:
            if owner is None or id(owner) in seen:
                continue
            if getattr(owner, "candidate_blocks", None) is None:
                continue
            block_size = getattr(owner, "candidate_block_size", 0)
            width = getattr(owner, "max_model_len", None) or getattr(
                owner, "max_total_seq_len", 0
            )
            if not block_size or not width:
                continue
            seen.add(id(owner))
            owners.append(owner)
    return owners


@instrument(span_name="DeepSeek V4.1 candidate-block warmup")
def deepseek_v4_candidate_blocks_warmup(
    model: torch.nn.Module,
    *,
    spec_token_counts: tuple[int, ...] = (0,),
) -> None:
    """Warm the candidate-block kernels for the signatures serving can reach."""
    owners = _find_candidate_block_owners(model)
    if not owners:
        return

    # The decode path shares one row bound per `1 + num_speculative_tokens`
    # rows; the prefill path uses one bound per row.
    row_repeats = sorted({1, *(count + 1 for count in spec_token_counts)})
    signatures = {
        (
            owner.candidate_blocks.device,
            int(
                getattr(owner, "max_model_len", None)
                or getattr(owner, "max_total_seq_len", 0)
            ),
            int(owner.candidate_blocks.shape[1]),
            int(owner.candidate_block_size),
        )
        for owner in owners
    }
    for device, width, candidates_width, block_size in signatures:
        warmup_candidate_block_kernels(
            device=device,
            width=width,
            candidates_width=candidates_width,
            block_size=block_size,
            row_repeats=row_repeats,
        )
    logger.info_once(
        "Warmed up DeepSeek V4.1 candidate-block kernels for row repeats %s.",
        tuple(row_repeats),
    )
