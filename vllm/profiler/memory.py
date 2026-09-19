# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute the per-rank non-KV footprint, once per rank at startup.

The KV cache budget is what is left after everything else, so the question worth
answering is where everything else went: parameters by group, torch's own
allocation, and the non-torch remainder. Reported once, at the point the budget
is decided, when ``VLLM_MEM_LOG=1``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.profiler.instrumentation import enabled

logger = init_logger(__name__)


def _group(name: str) -> str:
    if "engram" in name:
        return "engram"
    if "shared_experts" in name:
        return "shared expert"
    if "experts" in name:
        return "MoE experts"
    if ".attn." in name or "compressor" in name or "indexer" in name:
        return "attention/compressor/indexer"
    if "mtp." in name:
        return "MTP/DSpark"
    if "embed" in name:
        return "embedding"
    if "head" in name:
        return "lm_head"
    if "vision" in name or "aligner" in name:
        return "vision/aligner"
    return "other"


def log_memory_breakdown(
    model: Any,
    device: torch.device,
    *,
    consumed: int,
    peak_activation: int,
    cudagraph: int,
) -> None:
    """Log the rank's memory split; no-op unless VLLM_MEM_LOG=1."""
    if not enabled("mem"):
        return
    try:
        dev, host = Counter(), Counter()
        for name, param in model.named_parameters():
            (dev if param.device.type == "cuda" else host)[_group(name)] += param.nbytes
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        free, total = torch.cuda.mem_get_info(device)
        logger.info(
            "MEMLOG param_cuda=%.2f param_host=%.2f torch_alloc=%.2f "
            "torch_reserved=%.2f non_torch=%.2f consumed(w+non_torch)=%.2f "
            "peak_act=%.2f cudagraph=%.2f | total=%.2f free=%.2f",
            sum(dev.values()) / 2**30,
            sum(host.values()) / 2**30,
            allocated / 2**30,
            reserved / 2**30,
            (total - free - reserved) / 2**30,
            consumed / 2**30,
            peak_activation / 2**30,
            cudagraph / 2**30,
            total / 2**30,
            free / 2**30,
        )
        for group, nbytes in dev.most_common():
            logger.info("MEMLOG dev  %-30s %8.2f GiB", group, nbytes / 2**30)
        for group, nbytes in host.most_common():
            logger.info("MEMLOG host %-30s %8.2f GiB", group, nbytes / 2**30)
    except Exception as exc:  # noqa: BLE001 - diagnostics are never fatal
        logger.info("MEMLOG failed: %r", exc)
