# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Log the (M, N, K) of every GEMM call site, once per unique shape.

The shapes a kernel is asked for cannot be read back from a profile: CUTLASS and
DeepGEMM kernel names carry tile sizes, not problem sizes. Feeding a kernel
microbenchmark the shapes the server actually uses therefore needs one
instrumented run. Enable with ``VLLM_GEMM_SHAPE_LOG=1`` (optionally
``VLLM_GEMM_SHAPE_LOG_PATH=/path/file`` to collect across ranks); each unique
``(tag, M, N, K, extra)`` is reported once.

Usage::

    note_gemm("mxfp8.flashinfer_cutlass", M=x.shape[0], N=N, K=K)
"""

from __future__ import annotations

import threading

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = envs.VLLM_GEMM_SHAPE_LOG
_PATH = envs.VLLM_GEMM_SHAPE_LOG_PATH
_seen: set[tuple] = set()
_lock = threading.Lock()


def note_gemm(tag: str, M: int, N: int, K: int, **extra) -> None:
    """Record one GEMM shape; no-op unless VLLM_GEMM_SHAPE_LOG=1."""
    if not _ENABLED:
        return
    try:
        key = (tag, int(M), int(N), int(K), tuple(sorted(extra.items())))
        with _lock:
            if key in _seen:
                return
            _seen.add(key)
        parts = [f"tag={tag}", f"M={int(M)}", f"N={int(N)}", f"K={int(K)}"]
        parts += [f"{k}={v}" for k, v in extra.items()]
        line = " ".join(parts)
        logger.info("GEMMSHAPE %s", line)
        if _PATH:
            with open(_PATH, "a") as fh:
                fh.write(f"pid={__import__('os').getpid()} {line}\n")
    except Exception:  # noqa: BLE001 - diagnostic only, never fatal
        pass
