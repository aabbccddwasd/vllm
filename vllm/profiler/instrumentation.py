# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared machinery for the opt-in profiling switches.

Every switch is read once at import, is inert unless its environment variable is
set, and may only record -- none of them changes a decision, so a profiling run
serves exactly what the default configuration serves. A record is logged and,
when ``VLLM_PROFILE_DIR`` is set, appended to ``<kind>.log`` under it, which is
what makes a four-rank run collectable: the ranks share a working directory but
not a stream.

Each distinct observation is reported once, keyed by its tag and fields, so a
served step does not repeat itself thousands of times.

Usage::

    from vllm.profiler.instrumentation import note_once

    note_once("gemm", "mxfp8.b12x_tuned", M=x.shape[0], N=N, K=K)
"""

from __future__ import annotations

import os
import threading
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Profiling kind -> the switch that enables it.
_SWITCHES = {
    "gemm": "VLLM_GEMM_SHAPE_LOG",
    "comm": "VLLM_COMM_SHAPE_LOG",
    "moe_pad": "VLLM_MOE_PAD_LOG",
    "kv_accounting": "VLLM_KV_ACCOUNTING_LOG",
    "mem": "VLLM_MEM_LOG",
}
_ENABLED = {kind: getattr(envs, name) for kind, name in _SWITCHES.items()}
_DIR = envs.VLLM_PROFILE_DIR

_seen: dict[str, set[tuple]] = {}
_lock = threading.Lock()


def enabled(kind: str) -> bool:
    """Whether the profiling kind was switched on when this module was imported."""
    return _ENABLED.get(kind, False)


def _key(fields: dict[str, Any]) -> tuple:
    return tuple((name, repr(value)) for name, value in sorted(fields.items()))


def _append(kind: str, line: str) -> None:
    if not _DIR:
        return
    os.makedirs(_DIR, exist_ok=True)
    with open(os.path.join(_DIR, f"{kind}.log"), "a") as handle:
        handle.write(f"pid={os.getpid()} {line}\n")


def note_once(kind: str, tag: str = "", **fields: Any) -> None:
    """Record one observation per unique ``(tag, fields)``.

    No-op unless the kind's switch is on. Never fatal: a diagnostic that raises
    would turn a measurement run into a failed serve.
    """
    if not enabled(kind):
        return
    try:
        key = (tag, _key(fields))
        with _lock:
            if key in _seen.setdefault(kind, set()):
                return
            _seen[kind].add(key)
        line = " ".join(
            [tag] + [f"{name}={value}" for name, value in fields.items()]
        ).strip()
        logger.info("%s %s", kind.upper(), line)
        _append(kind, line)
    except Exception as exc:  # noqa: BLE001 - diagnostics are never fatal
        logger.debug("Profiling record for %s failed: %s", kind, exc)


def note_gemm(tag: str, M: int, N: int, K: int, **extra: Any) -> None:
    """Record one GEMM call site's shape; no-op unless VLLM_GEMM_SHAPE_LOG=1."""
    note_once("gemm", tag, M=int(M), N=int(N), K=int(K), **extra)
