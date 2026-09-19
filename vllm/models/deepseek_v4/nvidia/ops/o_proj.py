# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum

# 分组 wo_a 的 torch.bmm 会踩到 cuBLAS batched 启发式坏档的区间（实测见
# deep_gemm_fp8_o_proj）：group 数 = o_groups / tp_size，token 数为该 step 的 M。
_SMALL_BATCH_BMM_MAX_GROUPS = 4
_SMALL_BATCH_BMM_MAX_TOKENS = 64


def _use_grouped_mm(n_groups: int, num_tokens: int) -> bool:
    return (
        envs.VLLM_O_PROJ_GROUPED_MM
        and 2 <= num_tokens <= _SMALL_BATCH_BMM_MAX_TOKENS
        and n_groups <= _SMALL_BATCH_BMM_MAX_GROUPS
    )


def compute_fp8_einsum_recipe(
    block_size: int = 128,
) -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90 keeps block-row FP32 scales. SM100 uses packed per-row E8M0 scales.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, block_size)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + grouped wo_a + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. The attention
    layer selects the recipe at initialization.
    """
    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    o_proj_input, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=einsum_recipe[2],
        tma_aligned_scales=tma_aligned_scales,
        quantize=use_fp8,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    if use_fp8:
        weight_scale = (
            wo_a.weight_scale
            if hasattr(wo_a, "weight_scale")
            else wo_a.weight_scale_inv
        )
        # goal-6624d2f8 修复：`bhr,hdr->bhd` 的 `hdr` 需要三维权重
        # (n_groups, o_lora_rank, r)，而 `wo_a.weight` 是二维
        # (n_groups*o_lora_rank, r)。ds4 侧之所以不用这个 view，是因为它拿到的
        # wo_a.weight 本来就是三维；我们这边是二维（`is_bmm` 在 emulation 路径下
        # 被 process_weights_after_loading 拍平）。einsum 的 scale 也要跟着 view
        # 成同构三维，否则 deep_gemm 会报 layout 断言。
        grouped_weight = wo_a.weight.view(n_groups, o_lora_rank, -1)
        # deep_gemm 的 block scale 只接受 float32 或 int32（layout.hpp:103）；
        # int32 那条还要 `sf.stride(-3) == sf.stride(-1)*sf.size(-1)`（layout.hpp:113）
        # 且 `sf.size(-1) == ceil_div(k, gran_k*4)`。对所有三维排列联立，
        # **只有 (n_groups*o_lora_rank, 1, r/32/4) 同时满足**：
        #   stride(-3)=Kp = stride(-1)*size(-1)=1*Kp ✓ 且 size(-1)=Kp=32 ✓。
        # float32 分支能跑通但会逼 fp8_einsum 每步重新调
        # transpose_and_pack_fp32_into_ue8m0（实测 0.212 ms/step，把收益吃光），
        # 所以这里优先解析出 int32 打包布局并缓存在 wo_a 上（warmup 期一次性完成）。
        if weight_scale.dtype == torch.uint8:
            cached = getattr(wo_a, "_o_proj_scale_packed", None)
            if cached is None:
                packed = weight_scale.detach().contiguous().view(torch.int32)
                kp = packed.shape[-1]
                n_rows = n_groups * o_lora_rank
                cands = {}
                if packed.shape[0] == n_rows and packed.numel() == n_rows * kp:
                    # 扫描结论（layout.hpp:106 + :113 联立）：
                    #   size(-3) == num_groups (= n_groups) 且
                    #   stride(-3) == stride(-1)*size(-1) = Kp
                    # ⇒ 形状 (H, D, Kp) 且 **h 维步长必须是 Kp**，即把 (H,D)
                    #   块转置后连续化（这就是 `transpose_and_pack` 里的 transpose）。
                    # 四条断言联立（:106/:113/:115 + size(-1)==Kp）：
                    #   size(-3)=H, stride(-3)=stride(-1)*Kp, stride(-2)=1
                    # ⇒ 内存序必须是 (h, kp, d)，即 strides (D*Kp, 1, D)。
                    cands["(H,D,Kp) h/kp/d"] = (
                        packed.view(n_groups, o_lora_rank, kp)
                        .permute(0, 2, 1)
                        .contiguous()
                        .permute(0, 2, 1)
                    )
                    cands["(H,D,Kp) h-inner"] = (
                        packed.view(n_groups, o_lora_rank, kp)
                        .transpose(0, 1)
                        .contiguous()
                        .transpose(0, 1)
                    )
                    cands["(rows,1,Kp)"] = packed.view(n_rows, 1, kp)
                    cands["(H,D,Kp)"] = packed.view(n_groups, o_lora_rank, kp)
                notes = []
                winner = None
                for cname, sf in cands.items():
                    try:
                        fp8_einsum(
                            "bhr,hdr->bhd",
                            (o_proj_input, o_scale),
                            (grouped_weight, sf),
                            torch.empty_like(z),
                            recipe=einsum_recipe,
                        )
                        torch.cuda.synchronize()
                        notes.append(f"{cname}=OK")
                        if winner is None:
                            winner = sf
                    except Exception as exc:  # noqa: BLE001
                        notes.append(f"{cname}:{str(exc)[-90:]}")
                from vllm.logger import init_logger

                init_logger(__name__).info_once(
                    "O_PROJ_SCALE candidates: %s", " || ".join(notes)
                )
                if winner is not None:
                    cached = ("int32", winner)
                    init_logger(__name__).info_once(
                        "O_PROJ_SCALE resolved int32 packed layout"
                    )
            if cached is not None:
                object.__setattr__(wo_a, "_o_proj_scale_packed", cached)
            if cached is None:
                from vllm.logger import init_logger

                init_logger(__name__).info_once(
                    "O_PROJ_SCALE int32 packed layout rejected; falling "
                    "back to float32 (per-step repack)"
                )
                cached = (
                    "float32",
                    torch.pow(2.0, weight_scale.to(torch.float32) - 127.0).contiguous(),
                )
                object.__setattr__(wo_a, "_o_proj_scale_packed", cached)
            kind, weight_scale = cached
            if kind == "int32":
                fp8_einsum(
                    "bhr,hdr->bhd",
                    (o_proj_input, o_scale),
                    (grouped_weight, weight_scale),
                    z,
                    recipe=einsum_recipe,
                )
                return wo_b(z.flatten(1))
        try:
            grouped_scale = weight_scale.view(n_groups, o_lora_rank, -1)
        except RuntimeError:
            grouped_scale = weight_scale
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_proj_input, o_scale),
            (grouped_weight, grouped_scale),
            z,
            recipe=einsum_recipe,
        )
    else:
        grouped_weight = wo_a.weight.view(n_groups, o_lora_rank, -1)
        if _use_grouped_mm(n_groups, z.shape[0]):
            # cuBLAS 的 batched 启发式在 batch 很小时会落到 sm_80 的
            # `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32` 泛用内核上，
            # 只有 ~810 GB/s；逐组 torch.mm 走非 batched 路径，稳定在 ~1400 GB/s。
            # SM120 (RTX PRO 6000) 实测每层权重 16.8 MB：
            #   每 token 数 T:  bmm us/层 -> 逐组 mm us/层
            #     T=1  12.3 -> 11.1 ; T=2..48  20.6 -> 11.6 (-0.4 ms/step over 43 层)
            #     T=64 17.3 -> 12.3 ; T>=72    13.3 -> 13.1 (cuBLAS 换到了好内核)
            #   每 group 数 G:  G=2 是 TP4 的常见配置（本仓库交付态）；
            #     G=1/4 中性，G=8 时 bmm 反而更好 (41.6 -> 58.2)。
            # in-situ（served A/B，组件 attn.o_proj）：conc=4 -0.12 ms/step、
            # conc=16 -0.22 ms/step；conc=1 (T=1) 略负，故下界从 T=2 起。
            # 直接写进 z[:, g, :]：那是 z 的一个合法二维 strided 视图
            # (stride = (n_groups*o_lora_rank, 1))，cuBLAS 吃 ldc，所以不需要
            # 中间缓冲 + copy_。原先的 per_group + transpose(0,1) copy 每层
            # 多一个 kernel（实测 served 0.077 ms/step）。
            # VLLM_O_PROJ_DIRECT_WRITE=0 走旧路径，供 served A/B 背靠背对比。
            if envs.VLLM_O_PROJ_DIRECT_WRITE:
                for g in range(n_groups):
                    torch.mm(
                        o_proj_input[:, g, :],
                        grouped_weight[g].t(),
                        out=z[:, g, :],
                    )
            else:
                per_group = torch.empty(
                    (n_groups, z.shape[0], o_lora_rank),
                    device=z.device,
                    dtype=z.dtype,
                )
                for g in range(n_groups):
                    torch.mm(
                        o_proj_input[:, g, :], grouped_weight[g].t(), out=per_group[g]
                    )
                z.copy_(per_group.transpose(0, 1))
        else:
            torch.bmm(
                o_proj_input.transpose(0, 1),
                grouped_weight.transpose(1, 2),
                out=z.transpose(0, 1),
            )
    return wo_b(z.flatten(1))
