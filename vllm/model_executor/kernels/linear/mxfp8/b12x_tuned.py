# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 W8A8 with the b12x SM120 kernel driven by its own scheduler knobs.

FlashInfer's b12x MXFP8 runner
(``flashinfer/gemm/gemm_mm_mxfp8_cute_dsl.py``) builds
``Sm120B12xBlockScaledDenseGemmKernel`` with ``tile_k=128`` and cluster (1,1) and
leaves two of its parameters at their defaults:

    split_k_slices = 1
    direct_one_m_tile_scheduler = False   (and the single_work_tile_per_cta it needs)

Its tactic space varies only ``mma_tiler_mn`` x ``use_prefetch``. Decode is
exactly the case those two knobs exist for. The grid of the plain path is
``ceil(N/128)`` CTAs -- 9 of 188 SMs for N=1152 -- and ``split_k_slices=S``
changes it to ``(1, S, n_tiles)`` while giving each split a contiguous run of K
tiles, which is the textbook fix for a bandwidth-bound GEMV-shaped GEMM.

Measured on this box (``tools/tune_b12x_mxfp8.py``, in-graph with a flushed L2,
M=6), against the same kernel with ``split_k_slices=1``:

    N=1152 K=5120   18.43 -> 12.29 us  (1.50x)
    N=1792 K=5120   20.48 -> 13.33 us  (1.54x)
    N=5120 K=2048   15.34 -> 14.34 us  (1.07x)
    N=5120 K=15360  67.44 -> 61.42 us  (1.10x)
    N=25600 K=6144  124.91 -> 124.91 us (1.00x)

Narrower ``mma_tiler_mn`` tiles, ``use_prefetch`` and ``direct_one_m_tile_scheduler``
bought nothing on top of that.

Precision: split-K has to reduce the per-split partials, and the kernel's stock
reduction rounds each partial to bf16 first (``scatter_add_bf16x2``, measured
max|abs| 6e-2..2e-1 at operand scales of 10..25). So the kernel gained an
``split_k_atomic_f32`` mode (``patches/P-b12x-fp32-splitk-atomic.patch``) that
reduces in fp32 into an fp32 C instead. On this part that is **free**:

    N=1152 K=5120  split=4   bf16 12.29 us vs fp32 12.29 us  (rel err 3.6e-3 -> 2.1e-3)
    N=1792 K=5120  split=4   bf16 14.34 us vs fp32 14.34 us  (rel err 3.8e-3 -> 2.2e-3)
    N=5120 K=15360 split=4   bf16 61.44 us vs fp32 60.27 us  (rel err 3.8e-3 -> 2.1e-3)

and it halves the discrepancy: what is left (2.1e-3 ~ one bf16 ULP of the
output) is just the final rounding of a differently-ordered sum, not a lossy
reduction. The fp32 staging is still **zeroed** before every launch.

Opt in with ``VLLM_MXFP8_BACKEND_POLICY=b12x_tuned``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import (
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.profiler.instrumentation import note_gemm

from .flashinfer import FlashInferCutlassMxfp8LinearKernel

logger = init_logger(__name__)

_CLUSTER = (1, 1)
_CACHE: dict[tuple, object] = {}
_ALPHA: dict[str, torch.Tensor] = {}
# fp32 split-K staging, one per (M, N). The compiled wrapper type-checks
# mC.shape[0] == mA.shape[0], so the staging has to match the live M exactly.
_STAGING: dict[tuple, torch.Tensor] = {}


@triton.jit
def _cast_and_zero(src, dst, n_elem, BLOCK: tl.constexpr):
    """Dst = (bf16)src and src = 0, in one kernel.

    Reading and re-zeroing the same buffer is what keeps the staging invariant
    without a second kernel: the fp32 atomics need a zeroed C, and the layer
    still owes its caller a bf16 result. Each (staging, out) pair serves exactly
    one GEMM->cast sequence at a time and nothing else writes the staging in
    between, so writing the zeros back here is safe.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elem
    v = tl.load(src + offs, mask=m)
    tl.store(dst + offs, v.to(dst.dtype.element_ty), mask=m)
    tl.store(src + offs, tl.zeros_like(v), mask=m)


def _fused_cast_zero(src: torch.Tensor, dst: torch.Tensor) -> None:
    n = src.numel()
    _cast_and_zero[(triton.cdiv(n, 4096),)](src, dst, n, BLOCK=4096, num_warps=4)


# Split-K slices per served weight shape, measured cold-L2 with
# tools/tune_b12x_mxfp8.py (M=6): the optimum tracks how many CTAs the plain
# grid already has -- ceil(N/128) -- so split-K is applied where that is small
# and skipped where the grid is already wide.
#
#   N=1152 K=5120   9 CTAs   19.45 -> 7.99us @8   (2.43x)
#   N=1792 K=5120  14 CTAs   23.09 -> 10.33us @10 (2.23x)
#   N=5120 K=2048  40 CTAs   16.05 -> 11.59us @4  (1.38x)
#   N=8192 K=1280  64 CTAs   15.84 -> 12.55us @10 (1.26x)
#   N=25600 K=6144 200 CTAs  124.19us @1 -- grid already wide, split-K would
#                            only add the staging kernels
_SPLIT_BY_N = ((2048, 8), (8192, 8))


def _split_for(n: int, k: int) -> int:
    """Largest split <= the per-N target that divides the K tile count.

    ``k_tile_iter_cnt = k_tile_cnt // split`` and each CTA covers one contiguous
    run of that length, so a split that does not divide ``k // tile_k`` silently
    drops the leftover K tiles -- K=1280 has 10 tiles, and split=8 computed only
    8 of them (measured 50% error). Take the largest divisor instead.
    """
    want = 1
    for limit, split in _SPLIT_BY_N:
        if n <= limit:
            want = split
            break
    if want <= 1:
        return 1
    k_tiles = k // 128
    for d in range(want, 1, -1):
        if k_tiles % d == 0:
            return d
    return 1


def _plan(n: int, k: int, m: int) -> tuple[int | None, int]:
    """Return ``(tile_m, split_k)``, or ``(None, 1)`` to use the stock path.

    Three hard constraints, all measured on this box:

    * **``m <= tile_m``.** The split-K launch covers exactly one M tile -- its
      grid is ``(1, split_k, n_tiles)`` with no M dimension (the kernel's
      ``_compute_grid``), which is why ``direct_one_m_tile_scheduler`` exists.
      Confirmed at the boundary: with the (16, 128) tile M=16 is correct and
      M=17 comes back 100% wrong. 1+5 speculative decode needs 16 rows,
      4-way concurrency needs 24, so the tile grows to 32 rows for those.
    * **``k % tile_k == 0``** (tile_k is 128). Each split covers
      ``k_tiles // split`` contiguous K tiles, which silently drops the
      leftover tiles otherwise -- K=576 keeps the stock path.
    * **the plain grid must be narrow**, or split-K only adds the staging
      kernels (see ``_SPLIT_BY_N``).
    """
    if m > 32 or k % 128:
        return None, 1
    split = _split_for(n, k)
    if split <= 1:
        return None, 1
    return (16 if m <= 16 else 32), split


class B12xTunedMxfp8LinearKernel(FlashInferCutlassMxfp8LinearKernel):
    """MXFP8 W8A8 on the b12x SM120 CuTe-DSL kernel, with split-K selectable."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if envs.VLLM_MXFP8_BACKEND_POLICY != "b12x_tuned":
            return False, "set VLLM_MXFP8_BACKEND_POLICY=b12x_tuned"
        return super().is_supported(compute_capability)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        N, K = weight.shape
        M = x.shape[0] if hasattr(x, "shape") else -1
        tile_m, split_k = _plan(N, K, M)
        note_gemm(
            "mxfp8.b12x_tuned",
            M=M,
            N=N,
            K=K,
            layer=type(layer).__name__,
            prefix=getattr(layer, "prefix", ""),
            split_k=split_k,
            tile_m=tile_m or 0,
        )

        # The compiled kernel bakes sf_m into its signature, so it only serves
        # one 128-row block of M. Prefill (M up to the chunk size) would need a
        # fresh compile per block; decode is all M <= 128, so hand anything
        # larger to the stock path instead of growing a compile farm.
        if tile_m is None or x.dtype != torch.bfloat16:
            return super().apply_weights(layer, x, bias)

        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            input_mxfp8, input_scale = qa.data, qa.scale
            out_dtype, input_shape = qa.orig_dtype, qa.orig_shape
        else:
            out_dtype, input_shape = x.dtype, x.shape
            input_mxfp8, input_scale = mxfp8_e4m3_quantize(
                x.view(-1, K), is_sf_swizzled_layout=True
            )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        compiled = self._compile(N, K, split_k, tile_m)
        # The split-K epilogue reduces with fp32 atomics, so C is an fp32 staging
        # that has to be zeroed before every launch, and the layer still owes its
        # caller a bf16 result.
        key = (M, N, str(weight.device))
        staging = _STAGING.get(key)
        if staging is None:
            # Starts at zero, and _fused_cast_zero puts it back to zero after
            # every use, so no per-call zeroing kernel is needed.
            staging = torch.zeros(M, N, dtype=torch.float32, device=weight.device)
            _STAGING[key] = staging

        alpha = _ALPHA.get(str(weight.device))
        if alpha is None:
            alpha = torch.tensor([1.0], dtype=torch.float32, device=weight.device)
            _ALPHA[str(weight.device)] = alpha
        compiled(
            input_mxfp8,
            weight,
            staging,
            (M + 127) // 128,
            (N + 127) // 128,
            (K // 32 + 3) // 4,
            input_scale.data_ptr(),
            weight_scale.data_ptr(),
            alpha,
        )
        out = torch.empty(M, N, dtype=out_dtype, device=weight.device)
        _fused_cast_zero(staging, out)

        if bias is not None:
            out = out + bias
        return out.view(*input_shape[:-1], N)

    def _compile(self, n: int, k: int, split_k: int, tile_m: int):
        import cutlass
        from flashinfer.cute_dsl.utils import torch_to_cutlass_dtype
        from flashinfer.gemm.gemm_mm_fp4_cute_dsl import _compile_block_scaled_gemm
        from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import (
            Sm120B12xBlockScaledDenseGemmKernel,
        )

        tile = (tile_m, 128)
        key = (tile, split_k, n, k)
        compiled = _CACHE.get(key)
        if compiled is not None:
            return compiled

        def make():
            return Sm120B12xBlockScaledDenseGemmKernel(
                32,
                tile,
                _CLUSTER,
                mma_k=32,
                tile_k=128,
                use_prefetch=False,
                enable_pdl=True,
                swap_ab=False,
                load_path="tma",
                split_k_slices=split_k,
                split_k_atomic_bf16=False,
                split_k_atomic_f32=split_k > 1,
                direct_one_m_tile_scheduler=False,
            )

        compiled, _ = _compile_block_scaled_gemm(
            {},
            key,
            make,
            ab_cutlass_dtype=cutlass.Float8E4M3FN,
            sf_dtype=cutlass.Float8E8M0FNU,
            # fp32 C: the split-K reduction is exact, so the only rounding left
            # is the (M, N) bs16 cast below.
            c_cutlass_dtype=torch_to_cutlass_dtype(torch.float32),
            ab_assumed_align=32,
            cluster_shape_mn=_CLUSTER,
            swap_ab=False,
            sf_m=1,
            sf_n=(n + 127) // 128,
            sf_k=(k // 32 + 3) // 4,
            batch_size=1,
        )
        _CACHE[key] = compiled
        logger.info(
            "b12x_tuned MXFP8 kernel compiled: N=%d K=%d tile_m=%d split_k=%d "
            "(fp32 reduce)",
            n,
            k,
            tile_m,
            split_k,
        )
        return compiled
