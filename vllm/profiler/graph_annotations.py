# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute CUDA graph kernels to the model component that created them.

A decode step is one CUDA graph replay: no host code runs, so a profiler trace
of the replay shows thousands of kernels with no indication of which module
asked for them. ``torch.cuda._graph_annotations`` solves this at *capture* time
- ``mark_kernels`` records the graph nodes created inside its scope and tags
them by CUPTI ``toolsId``, which survives into every replay's trace as
``args["graph node id"]``. Merging the recorded annotations back into a trace
(``python -m torch.cuda._annotate_cuda_graph_trace``) then adds the component
name to each kernel event's args.

The scopes are only meaningful while a graph is being captured, and they are
compiled out entirely unless ``VLLM_GRAPH_KERNEL_ANNOTATIONS=1``, so the
default path is untouched.

Usage::

    with graph_kernel_scope({"component": "moe.gemm1"}):
        ...  # kernels created here are attributed to moe.gemm1
"""

from __future__ import annotations

import contextlib
import os
import pickle
import time
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = envs.VLLM_GRAPH_KERNEL_ANNOTATIONS
_available: bool | None = None


def _annotations_module():
    """Import the torch API lazily, or return None when it cannot work."""
    global _available
    if _available is not None:
        return _annotations_module._mod if _available else None
    mod = None
    if _ENABLED:
        try:
            from torch.cuda import _graph_annotations as mod  # type: ignore[no-redef]

            if not mod._is_tools_id_unavailable():
                mod.enable_annotations()
            else:
                logger.warning(
                    "VLLM_GRAPH_KERNEL_ANNOTATIONS=1 but cudaGraphNodeGetToolsId "
                    "is unavailable; kernel annotations are inert."
                )
                mod = None
        except Exception as exc:  # noqa: BLE001 - profiling aid, never fatal
            logger.warning("CUDA graph kernel annotations unavailable: %s", exc)
            mod = None
    _annotations_module._mod = mod
    _available = mod is not None
    return mod


_annotations_module._mod = None


def graph_kernel_scope(annotation: dict[str, Any]):
    """Context manager naming the graph nodes created inside it.

    No-op unless ``VLLM_GRAPH_KERNEL_ANNOTATIONS=1`` and the driver supports
    ``cudaGraphNodeGetToolsId``. Must be entered while a capture is active on
    the current stream (torch's ``mark_kernels`` snapshots the capturing
    stream's frontier on entry).
    """
    mod = _annotations_module()
    if mod is None:
        return contextlib.nullcontext()
    return mod.mark_kernels(annotation)


def resolve_capture() -> None:
    """Resolve the scopes of the capture in progress into node annotations.

    Call inside the ``torch.cuda.graph(...)`` block, after the last scope exits
    (the toolsIds only exist while the capture is live).
    """
    mod = _annotations_module()
    if mod is None:
        return
    try:
        mod.resolve_pending_annotations()
    except Exception as exc:  # noqa: BLE001 - profiling aid, never fatal
        logger.warning("Failed to resolve CUDA graph annotations: %s", exc)


def finish_capture(graph: Any, rank: int | None = None) -> None:
    """Remap the captured graph's annotations and persist them.

    Call once per captured graph, after the capture context exits. The pickle
    lands in ``VLLM_GRAPH_KERNEL_ANNOTATION_DIR`` (or the working directory) as
    ``kernel_annotations_rank{rank}_{ts}.pkl``, which is what
    ``torch.cuda._annotate_cuda_graph_trace`` expects.
    """
    mod = _annotations_module()
    if mod is None:
        return
    try:
        if rank is None:
            import torch.distributed as dist

            rank = (
                dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            )
        mod.remap_to_exec_graph(graph)
        annotations = dict(mod.get_kernel_annotations())
        logger.info(
            "CUDA graph kernel annotations: %d nodes on rank %d", len(annotations), rank
        )
        directory = envs.VLLM_GRAPH_KERNEL_ANNOTATION_DIR or os.getcwd()
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(
            directory, f"kernel_annotations_rank{rank}_{int(time.time() * 1000)}.pkl"
        )
        with open(path, "wb") as fh:
            pickle.dump(annotations, fh)
        logger.info_once("Persisted CUDA graph kernel annotations -> %s", path)
    except Exception as exc:  # noqa: BLE001 - profiling aid, never fatal
        logger.warning("Failed to persist CUDA graph annotations: %s", exc)
