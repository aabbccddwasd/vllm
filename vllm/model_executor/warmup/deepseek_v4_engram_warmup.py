# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the DeepSeek V4.1 Engram Triton kernels before serving.

The Engram row gather is JIT'ed on its first call, and that call happens inside
the captured decode graph. A cold compile during serving therefore both spikes
latency and compiles a shape the graph was already captured around. Warm the
lookup across the token sizes the scheduler and the capture sizes can produce.

Gating is intrinsic: a model without Engram modules returns early.
"""

import time
from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.tracing import instrument

logger = init_logger(__name__)

_AUTO_WARMUP_MAX_TOKENS = 16_384
_DEFAULT_TOKEN_SIZE_CANDIDATES = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16_384,
)


def _select_engram_warmup_token_sizes(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int],
) -> list[int]:
    if max_tokens <= 0:
        return []

    max_auto_tokens = min(max_tokens, _AUTO_WARMUP_MAX_TOKENS)
    candidates: Iterable[int] = (
        *_DEFAULT_TOKEN_SIZE_CANDIDATES,
        *cudagraph_capture_sizes,
        max_auto_tokens,
    )
    return sorted({size for size in candidates if 1 <= size <= max_auto_tokens})


def _find_engram_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Collect the Engram modules by shape, so no model class is imported."""
    return [
        module
        for module in model.modules()
        if hasattr(module, "prepare_embeddings") and hasattr(module, "embed_tokens")
    ]


@instrument(span_name="DeepSeek V4.1 Engram warmup")
def deepseek_v4_engram_warmup(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    engram_modules = _find_engram_modules(model)
    if not engram_modules:
        return

    token_sizes = _select_engram_warmup_token_sizes(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
    )
    if not token_sizes:
        return

    started = time.perf_counter()
    logger.info(
        "Warming up DeepSeek V4.1 Engram kernels for token sizes: %s", token_sizes
    )
    with torch.inference_mode():
        for engram in engram_modules:
            embedding = engram.embed_tokens
            device = torch.device("cuda", torch.accelerator.current_device_index())
            for num_tokens in token_sizes:
                if num_tokens > engram.staged_rows.shape[0]:
                    continue
                # `prepare_embeddings` is the gather that runs inside the graph.
                # Hash ids are int32 [num_tokens, n_hash_cols] -- the *full* head
                # width, as `DeepseekV4Attention` passes them: the lookup kernel
                # indexes this tensor with the global head id
                # (`HEAD_START + row % LOCAL_HEADS`), so each rank reads its own
                # `part_n_hash_cols` columns out of the full-width rows. A
                # narrower tensor makes every rank past the first read past its
                # end.
                hash_ids = torch.zeros(
                    (num_tokens, embedding.n_hash_cols),
                    dtype=torch.int32,
                    device=device,
                )
                engram.prepare_embeddings(hash_ids)
        torch.accelerator.synchronize()
    logger.info(
        "DeepSeek V4.1 Engram warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )
