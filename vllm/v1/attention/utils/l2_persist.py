"""L2 cache persistence control via CUDA Runtime API (ctypes).

Protect a GPU memory range from L2 eviction by interleaved operations
(MoE, norms, etc.) across attention layers.

Usage::

    from vllm.v1.attention.utils.l2_persist import (
        l2_persist_init, set_l2_persist_window, clear_l2_persist_window,
    )
    l2_persist_init(16 * 1024 * 1024)  # once at startup

    stream = torch.cuda.current_stream()
    set_l2_persist_window(stream.cuda_stream,
                          workspace.data_ptr(), workspace.nbytes)
    ...  # attention kernels
    clear_l2_persist_window(stream.cuda_stream)

References:
    https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaAccessPolicyWindow.html
"""

import ctypes
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# CUDA Runtime API — enums
# ---------------------------------------------------------------------------
_cudaLimitPersistingL2CacheSize = 0x06
_cudaStreamAttributeAccessPolicyWindow = 1
_cudaAccessPropertyNormal = 0
_cudaAccessPropertyStreaming = 1
_cudaAccessPropertyPersisting = 2


# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------
class _cudaAccessPolicyWindow(ctypes.Structure):
    _fields_ = [
        ("base_ptr", ctypes.c_void_p),
        ("num_bytes", ctypes.c_size_t),
        ("hitRatio", ctypes.c_float),
        ("hitProp", ctypes.c_int),
        ("missProp", ctypes.c_int),
    ]


class _cudaStreamAttrValue(ctypes.Structure):
    _fields_ = [("accessPolicyWindow", _cudaAccessPolicyWindow)]


# ---------------------------------------------------------------------------
# Load library once
# ---------------------------------------------------------------------------
def _load_cudart() -> Any:
    """Load libcudart and set function signatures."""
    # torch already loads CUDA; use CDLL to grab the runtime.
    lib = ctypes.CDLL("libcudart.so")
    lib.cudaDeviceSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
    lib.cudaDeviceSetLimit.restype = ctypes.c_int
    lib.cudaStreamSetAttribute.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.cudaStreamSetAttribute.restype = ctypes.c_int
    lib.cudaCtxResetPersistingL2Cache.restype = ctypes.c_int
    return lib


_cudart = None


def _check(err: int, msg: str = "") -> None:
    if err != 0:
        logger.warning("L2 persist API call failed (%s): cudaError_t=%d", msg, err)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def l2_persist_init(persist_bytes: int = 16 * 1024 * 1024) -> None:
    """Reserve *persist_bytes* of L2 for persisting accesses (one-time)."""
    global _cudart
    if _cudart is None:
        _cudart = _load_cudart()
    err = _cudart.cudaDeviceSetLimit(_cudaLimitPersistingL2CacheSize,
                                      persist_bytes)
    if err == 0:
        logger.info(
            "L2 persist set-aside reserved: %d MB (cudaDeviceSetLimit OK)",
            persist_bytes // (1024 * 1024),
        )
    else:
        _check(err, f"cudaDeviceSetLimit(persist={persist_bytes})")


def set_l2_persist_window(
    stream_ptr: int,
    base_ptr: int,
    num_bytes: int,
    hitRatio: float = 1.0,
) -> None:
    """Mark *base_ptr .. base_ptr+num_bytes* as persisting on *stream*."""
    global _cudart
    if _cudart is None:
        _cudart = _load_cudart()
    logger.info(
        "L2 persist window: stream=0x%x ptr=0x%x bytes=%d hitRatio=%.1f",
        stream_ptr, base_ptr, num_bytes, hitRatio,
    )
    window = _cudaAccessPolicyWindow()
    window.base_ptr = ctypes.c_void_p(base_ptr)
    window.num_bytes = num_bytes
    window.hitRatio = hitRatio
    window.hitProp = _cudaAccessPropertyPersisting
    window.missProp = _cudaAccessPropertyStreaming
    attr = _cudaStreamAttrValue()
    attr.accessPolicyWindow = window
    err = _cudart.cudaStreamSetAttribute(
        ctypes.c_void_p(stream_ptr),
        _cudaStreamAttributeAccessPolicyWindow,
        ctypes.byref(attr),
    )
    if err == 0:
        logger.info("L2 persist window SET successfully")
    else:
        _check(err, "cudaStreamSetAttribute(persist)")


def clear_l2_persist_window(stream_ptr: int) -> None:
    """Reset persistence on *stream* to normal caching behaviour."""
    global _cudart
    if _cudart is None:
        return
    window = _cudaAccessPolicyWindow()
    window.hitProp = _cudaAccessPropertyNormal
    window.missProp = _cudaAccessPropertyNormal
    attr = _cudaStreamAttrValue()
    attr.accessPolicyWindow = window
    _cudart.cudaStreamSetAttribute(
        ctypes.c_void_p(stream_ptr),
        _cudaStreamAttributeAccessPolicyWindow,
        ctypes.byref(attr),
    )


def l2_persist_reset() -> None:
    """Reset **all** L2 persistence state (for testing / teardown)."""
    global _cudart
    if _cudart is None:
        _cudart = _load_cudart()
    _check(_cudart.cudaCtxResetPersistingL2Cache(),
           "cudaCtxResetPersistingL2Cache")
