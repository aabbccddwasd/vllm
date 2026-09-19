#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One decode MoE layer at DeepSeek-V4.1 serving shapes, per backend.

Motivation
----------
End-to-end step time mixes attention, the all-reduce, the draft model and the
MoE.  To decide anything about MoE kernel selection (which decode band runs,
how much padding a backend pays, how much expert weight traffic it moves) the
routed experts have to be measured on their own, at the shapes the server
actually runs, with the production quant configuration.

Served geometry (DeepSeek-V4.1-Flash, TP=4, one rank)
----------------------------------------------------
    checkpoint          per-expert on disk        per-rank after TP sharding
    experts.w1.weight   I8 [2304, 2560]           uint8 [1152, 2560]  (2I_tp, H/2)
    experts.w1.scale    F8_E8M0 [2304, 160]       uint8 [1152, 160]   (2I_tp, H/32)
    experts.w3.weight   I8 [2304, 2560]           (fused into w13)
    experts.w2.weight   I8 [5120, 1152]           uint8 [5120, 288]   (H, I_tp/2)
    experts.w2.scale    F8_E8M0 [5120, 72]        uint8 [5120, 18]    (H, I_tp/32)

E=384 local experts, hidden 5120, moe_intermediate 2304, topk 6, and
``expert_dtype=fp4`` with ``scale_fmt=ue8m0``, so both backends consume MXFP4
weights with e8m0 block-32 scales.  Everything backend specific is taken from
the production oracle rather than guessed:

* activation format  -> ``make_mxfp4_moe_quant_config``
* per-rank I_tp      -> ``mxfp4_round_up_hidden_size_and_intermediate_size``
  (DeepGEMM needs 128 alignment, so it runs I=640; b12x runs the exact 576)
* decode band        -> ``b12x.moe.fused_moe._impl._resolve_workspace_layout``

Weights are synthesized (random FP4 codes, unit e8m0 scales) because kernel
cost is data independent; ``--real-weights`` quantizes a bf16 source with
flashinfer instead.

Usage
-----
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
        python benchmarks/kernels/benchmark_decode_moe_sm120.py --m 1,2,4,6,8,12

One launch per case, for ncu resource counters:

    ncu --launch-count 1 --metrics launch__registers_per_thread,\\
launch__shared_mem_per_block_allocated,\\
sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes.sum \\
        python benchmarks/kernels/benchmark_decode_moe_sm120.py --ncu-one 6

requires ``PYTHONPATH`` to point at an install of the external ``b12x`` package
for the ``b12x`` arm.
"""

from __future__ import annotations

import argparse
import os
import statistics
from types import SimpleNamespace

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    make_mxfp4_moe_quant_config,
    mxfp4_round_up_hidden_size_and_intermediate_size,
)

# DeepSeek-V4.1-Flash routed experts, per TP rank (tensor_parallel_size=4).
E_DEFAULT = 384
H_DEFAULT = 5120
I_DEFAULT = 576  # 2304 // 4
TOPK_DEFAULT = 6
MXFP4_BLOCK = 32
E8M0_ONE = 0x7F  # e8m0 encoding of 2**0


def parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.replace(",", " ").split()]


def make_moe_config(
    num_experts: int,
    experts_per_token: int,
    hidden_dim: int,
    intermediate_size: int,
    activation: MoEActivation,
) -> FusedMoEConfig:
    from vllm.platforms import current_platform

    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=activation,
        in_dtype=torch.bfloat16,
        device=current_platform.device_type,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=4096,
    )


def build_weights(
    num_experts: int,
    hidden: int,
    inter: int,
    *,
    device: str,
    seed: int,
    real_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MXFP4 weights (packed uint8) and e8m0 block scales (uint8, as served)."""
    generator = torch.Generator(device=device).manual_seed(seed)
    w13 = torch.randint(
        0,
        256,
        (num_experts, 2 * inter, hidden // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden, inter // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w13_scale = torch.full(
        (num_experts, 2 * inter, hidden // MXFP4_BLOCK),
        E8M0_ONE,
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.full(
        (num_experts, hidden, inter // MXFP4_BLOCK),
        E8M0_ONE,
        dtype=torch.uint8,
        device=device,
    )
    if not real_weights:
        return w13, w2, w13_scale, w2_scale

    from flashinfer import mxfp4_quantize

    src13 = torch.randn(
        num_experts,
        2 * inter,
        hidden,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(hidden**-0.5)
    src2 = torch.randn(
        num_experts,
        hidden,
        inter,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(inter**-0.5)
    for expert in range(num_experts):
        # flashinfer pads the e8m0 scale plane to a multiple of 4 columns; the
        # checkpoint (and every consumer) uses exactly dim // 32.
        packed, scale = mxfp4_quantize(src13[expert])
        w13[expert] = packed.view(torch.uint8)
        w13_scale[expert] = scale.view(torch.uint8)[:, : w13_scale.shape[2]]
        packed, scale = mxfp4_quantize(src2[expert])
        w2[expert] = packed.view(torch.uint8)
        w2_scale[expert] = scale.view(torch.uint8)[:, : w2_scale.shape[2]]
    return w13, w2, w13_scale, w2_scale


def build_routes(
    tokens: int,
    num_experts: int,
    topk: int,
    *,
    device: str,
    mode: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(tokens, num_experts, generator=generator, device=device)
    if mode == "same":
        logits = logits[0:1].expand(tokens, -1).contiguous()
    elif mode == "hot":
        logits[:, : num_experts // 8] += 8.0
    weights, ids = torch.topk(logits, topk, dim=-1)
    return torch.softmax(weights, dim=-1).float(), ids.to(torch.int32)


def apply_gate(mode: str | None) -> None:
    """Drive the b12x decode-band gate.

    The served weight format is MXFP4 with MXFP8 activations (``w4a8_mx``), and
    that path never consults ``_MICRO_MAX_TOKENS``: it runs the tiny-decode
    (micro) kernel when ``_tiny_decode_supports`` accepts the shape, which caps
    ``num_tokens`` at 4.  Modes target either knob:

    ``stock``    leave b12x alone
    ``off``      disable the tiny-decode band (everything goes dynamic)
    ``tiny:N``   accept the tiny-decode band up to N tokens
    ``micro:N``  raise ``_MICRO_MAX_TOKENS`` to N as well
    """
    import b12x.moe.fused_moe._impl as impl

    if not hasattr(impl, "_ORIGINAL_TINY_DECODE_SUPPORTS"):
        impl._ORIGINAL_TINY_DECODE_SUPPORTS = impl._tiny_decode_supports
    impl._tiny_decode_supports = impl._ORIGINAL_TINY_DECODE_SUPPORTS
    # The kill switch is an env var read on every selection, so a previous
    # ``off`` arm would otherwise keep suppressing the band for later arms.
    os.environ.pop("B12X_W4A8_TINY_DECODE", None)
    if mode is None or mode == "stock":
        return
    if mode.startswith("micro:"):
        impl._MICRO_MAX_TOKENS = int(mode.split(":", 1)[1])
        return
    if mode == "off":
        os.environ["B12X_W4A8_TINY_DECODE"] = "0"
        return
    if mode.startswith("tiny:"):
        limit = int(mode.split(":", 1)[1])
        original = impl._ORIGINAL_TINY_DECODE_SUPPORTS

        def widened(*, num_tokens: int, k: int, n: int, activation: str) -> bool:
            # Keep the shape/activation gates of the original predicate but
            # drop its 1..4 token bound, clamped so the caller's limit holds.
            if int(num_tokens) > limit:
                return False
            return original(num_tokens=4, k=k, n=n, activation=activation)

        impl._tiny_decode_supports = widened
        return
    raise ValueError(f"unknown gate mode {mode!r}")


def clear_b12x_caches() -> None:
    from b12x.moe.fused_moe import clear_caches

    clear_caches()


def resolve_band(
    tokens: int, hidden: int, inter: int, quant_mode: str, num_experts: int
) -> str:
    """Ask b12x which kernel band it selects for this shape."""
    import b12x.moe.fused_moe._impl as impl

    try:
        return impl._resolve_workspace_layout(
            num_tokens=tokens,
            weight_E=num_experts,
            num_topk=0,
            k=hidden,
            n=inter,
            activation="silu",
            quant_mode=quant_mode,
        )[0]
    except Exception:  # noqa: BLE001
        return "?"


def effective_inter(backend: str, inter: int) -> int:
    """Per-rank intermediate size, as the production oracle rounds it."""
    backend_enum = (
        Mxfp4MoeBackend.B12X_MXFP4_MXFP8
        if backend == "b12x"
        else Mxfp4MoeBackend.DEEPGEMM_MXFP4
    )
    _, rounded = mxfp4_round_up_hidden_size_and_intermediate_size(
        backend_enum, H_DEFAULT, inter
    )
    return rounded


def pad_weights_to(weights: tuple, inter: int, inter_pad: int) -> tuple:
    """Zero-pad MXFP4 weights from ``inter`` to ``inter_pad`` output rows.

    DeepGEMM runs a 128-aligned intermediate (576 -> 640 at TP=4) while b12x
    runs the exact model dimension.  Deriving the padded tensors from one
    source keeps the two arms comparable and mirrors the loader: the gate half
    occupies rows ``[0, inter)`` and the up half rows
    ``[inter_pad, inter_pad + inter)`` of a ``2 * inter_pad`` w13, so the
    unused rows are the two gaps, not the tail (see ``_load_w13``).
    """
    if inter_pad == inter:
        return weights
    w13, w2, w13_scale, w2_scale = weights
    experts = w13.shape[0]
    pad2 = inter_pad - inter

    padded13 = torch.zeros(
        experts, 2 * inter_pad, w13.shape[2], dtype=w13.dtype, device=w13.device
    )
    padded13[:, :inter] = w13[:, :inter]
    padded13[:, inter_pad : inter_pad + inter] = w13[:, inter:]
    padded13_scale = torch.zeros(
        experts,
        2 * inter_pad,
        w13_scale.shape[2],
        dtype=w13_scale.dtype,
        device=w13_scale.device,
    )
    padded13_scale[:, :inter] = w13_scale[:, :inter]
    padded13_scale[:, inter_pad : inter_pad + inter] = w13_scale[:, inter:]

    padded2 = torch.zeros(
        experts, w2.shape[1], w2.shape[2] + pad2 // 2, dtype=w2.dtype, device=w2.device
    )
    padded2[:, :, : w2.shape[2]] = w2
    padded2_scale = torch.zeros(
        experts,
        w2_scale.shape[1],
        w2_scale.shape[2] + pad2 // MXFP4_BLOCK,
        dtype=w2_scale.dtype,
        device=w2_scale.device,
    )
    padded2_scale[:, :, : w2_scale.shape[2]] = w2_scale
    return padded13, padded2, padded13_scale, padded2_scale


def backend_scales(backend: str, weights: tuple[torch.Tensor, ...]) -> tuple:
    """Apply the kernel-format conversion each backend gets in production.

    ``convert_gpt_oss_weight_to_mxfp4_moe_kernel_format`` returns the raw uint8
    e8m0 block scales for the b12x backends (b12x repacks them itself) and
    DeepGEMM-packed scales for ``DEEPGEMM_MXFP4`` -- DeepGEMM asserts on a scale
    dtype of float32/int32, so the raw uint8 plane is rejected.
    """
    if backend != "deepgemm":
        return weights
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        _pack_deepgemm_mxfp4_scales,
    )

    w13_scale, w2_scale = _pack_deepgemm_mxfp4_scales(
        weights[0], weights[1], weights[2], weights[3]
    )
    return (weights[0], weights[1], w13_scale, w2_scale)


def build_kernel(
    backend: str,
    moe_config: FusedMoEConfig,
    quant_config,
    weights: tuple[torch.Tensor, ...],
    activation: MoEActivation,
) -> mk.FusedMoEKernel:
    w13, w2, w13_scale, w2_scale = weights
    if backend == "b12x":
        from vllm.model_executor.layers.fused_moe.b12x import B12xExperts

        experts = B12xExperts(moe_config, quant_config)
        layer = SimpleNamespace(
            w13_weight=w13,
            w2_weight=w2,
            w13_weight_scale=w13_scale,
            w2_weight_scale=w2_scale,
            activation=activation,
            apply_router_weight_on_input=False,
        )
        experts.process_weights_after_loading(layer)
    elif backend == "deepgemm":
        from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            DeepGemmFP4Experts,
        )

        experts = DeepGemmFP4Experts(moe_config, quant_config)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    return mk.FusedMoEKernel(
        maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        experts,
    )


def make_runner(kernel: mk.FusedMoEKernel, tensors, activation: MoEActivation):
    w13, w2, topk_weights, topk_ids, num_experts = tensors

    def run(hidden: torch.Tensor) -> torch.Tensor:
        return kernel.apply(
            hidden_states=hidden,
            w1=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=num_experts,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

    return run


def l2_flush_buffer(device: str) -> torch.Tensor | None:
    """2x L2 (256 MB) scratch used to force cold-cache replays.

    Decode MoE touches only the active experts' weights (~4.4 MB per expert),
    which fits in this device's 126 MB L2 at small m, so back-to-back replays
    would otherwise report warm-cache numbers that serving never sees.
    """
    try:
        from flashinfer.testing import get_l2_cache_size

        size = get_l2_cache_size(device)
    except Exception:  # noqa: BLE001
        size = 128 * 1024 * 1024
    return torch.empty(size * 2, dtype=torch.int8, device=device)


def time_events(
    run,
    hidden,
    *,
    use_graph: bool,
    flush: torch.Tensor | None,
    iters: int,
    warmup: int = 10,
) -> tuple[float, float, float, float]:
    """Median/min/max GPU span of one call, in milliseconds.

    Timings come from CUDA events recorded on the same stream as the call, so
    the flush that precedes the start event cannot leak into the interval.
    flashinfer's CUPTI wrapper is not usable here: with ``use_cuda_graph=True``
    it reports the L2 flush inside the span (10x inflation at these shapes,
    while its own per-kernel totals stay correct).
    """
    graph = None
    if use_graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run(hidden)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(hidden)

    def call() -> None:
        if graph is not None:
            graph.replay()
        else:
            run(hidden)

    for _ in range(warmup):
        if flush is not None:
            flush.zero_()
        call()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        if flush is not None:
            flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return (
        statistics.median(samples),
        samples[0],
        samples[-1],
        sum(samples) / len(samples),
    )


def time_cupti(run, hidden, *, use_graph: bool, cold_l2: bool, repeat_ms: int) -> float:
    """Flashinfer CUPTI span; kept only for cross-checking the event timer."""
    from flashinfer.testing import bench_gpu_time_with_cupti

    times = bench_gpu_time_with_cupti(
        run,
        input_args=(hidden,),
        use_cuda_graph=use_graph,
        cold_l2_cache=cold_l2,
        dry_run_time_ms=50,
        repeat_time_ms=repeat_ms,
    )
    return statistics.median(times)


def profile_kernels(run, hidden, iters: int = 3) -> list[tuple[str, float, int]]:
    from torch.profiler import ProfilerActivity, profile

    for _ in range(2):
        run(hidden)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run(hidden)
        torch.cuda.synchronize()
    rows = []
    for event in prof.key_averages():
        if event.self_device_time_total > 0:
            rows.append(
                (
                    event.key,
                    event.self_device_time_total / iters / 1000.0,
                    event.count // iters,
                )
            )
    rows.sort(key=lambda row: -row[1])
    return rows


def expert_weight_bytes(num_experts: int, hidden: int, inter: int) -> float:
    per_expert = (2 * inter * hidden / 2) + (hidden * inter / 2)
    return num_experts * per_expert / 1e6


def report_checks(outputs) -> None:
    by_tokens: dict[int, list] = {}
    for key in outputs:
        by_tokens.setdefault(key[1], []).append(key)
    for tokens, group in sorted(by_tokens.items()):
        group.sort()
        if len(group) < 2:
            continue
        ref_key = group[0]
        ref = outputs[ref_key]
        scale = ref.abs().max().item() or 1.0
        for other_key in group[1:]:
            diff = (ref.float() - outputs[other_key].float()).abs().max().item()
            delta = diff / scale
            print(f"  [check] m={tokens} {ref_key[0]} vs {other_key[0]}: {delta:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", default="1,2,3,4,6,8,12,16,24,32,48")
    parser.add_argument("--experts", type=int, default=E_DEFAULT)
    parser.add_argument("--hidden", type=int, default=H_DEFAULT)
    parser.add_argument("--inter", type=int, default=I_DEFAULT)
    parser.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    parser.add_argument("--layers", type=int, default=40, help="for ms/step scaling")
    parser.add_argument("--backends", default="deepgemm,b12x")
    parser.add_argument(
        "--gates",
        default="stock",
        help="comma list of b12x band gates: stock | off | tiny:N | micro:N",
    )
    parser.add_argument(
        "--route-mode", default="random", choices=("random", "same", "hot")
    )
    parser.add_argument("--real-weights", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--warm-l2", action="store_true")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--timer", default="event", choices=("event", "cupti", "both")
    )
    parser.add_argument("--repeat-ms", type=int, default=200)
    parser.add_argument("--kernel-profile", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="cross-backend output check"
    )
    parser.add_argument("--ncu-one", type=int, default=0, help="launch one case only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device(args.device))

    import b12x.moe.fused_moe._impl as impl

    if not hasattr(impl, "_ORIGINAL_TINY_DECODE_SUPPORTS"):
        impl._ORIGINAL_TINY_DECODE_SUPPORTS = impl._tiny_decode_supports

    torch.manual_seed(args.seed)
    activation = MoEActivation.SILU
    token_counts = [args.ncu_one] if args.ncu_one else parse_ints(args.m)
    flush = None if args.warm_l2 else l2_flush_buffer(args.device)

    records: list[dict] = []
    outputs: dict[tuple[str, int], torch.Tensor] = {}
    weight_cache: dict[tuple[str, int], tuple] = {}
    for backend in args.backends.split(","):
        gates = args.gates.split(",") if backend == "b12x" else ["stock"]
        for gate in gates:
            inter = effective_inter(backend, args.inter)
            key = (backend, inter)
            if key not in weight_cache:
                if inter == args.inter:
                    weight_cache[key] = build_weights(
                        args.experts,
                        args.hidden,
                        inter,
                        device=args.device,
                        seed=args.seed,
                        real_weights=args.real_weights,
                    )
                else:
                    base_key = ("base", args.inter)
                    if base_key not in weight_cache:
                        weight_cache[base_key] = build_weights(
                            args.experts,
                            args.hidden,
                            args.inter,
                            device=args.device,
                            seed=args.seed,
                            real_weights=args.real_weights,
                        )
                    weight_cache[key] = pad_weights_to(
                        weight_cache[base_key], args.inter, inter
                    )
            weights = weight_cache[key]
            kernel_weights = backend_scales(backend, weights)
            moe_config = make_moe_config(
                args.experts, args.topk, args.hidden, inter, activation
            )
            quant_backend = (
                Mxfp4MoeBackend.B12X_MXFP4_MXFP8
                if backend == "b12x"
                else Mxfp4MoeBackend.DEEPGEMM_MXFP4
            )
            label = f"{backend}[I={inter}]" + ("" if gate == "stock" else f"/{gate}")
            apply_gate(gate if backend == "b12x" else None)
            if backend == "b12x":
                clear_b12x_caches()
            quant_config = make_mxfp4_moe_quant_config(
                mxfp4_backend=quant_backend,
                w1_scale=kernel_weights[2],
                w2_scale=kernel_weights[3],
            )
            kernel = build_kernel(
                backend, moe_config, quant_config, kernel_weights, activation
            )
            quant_mode = (
                str(getattr(kernel.fused_experts, "_quant_mode", ""))
                if backend == "b12x"
                else ""
            )
            print(
                f"\n[arm] {label} quant_mode={quant_mode or '-'} "
                f"w13={tuple(weights[0].shape)} w2={tuple(weights[1].shape)} "
                "weights="
                f"{expert_weight_bytes(args.experts, args.hidden, inter):.0f} MB"
            )
            for tokens in token_counts:
                topk_weights, topk_ids = build_routes(
                    tokens,
                    args.experts,
                    args.topk,
                    device=args.device,
                    mode=args.route_mode,
                    seed=args.seed + tokens,
                )
                hidden = torch.randn(
                    tokens, args.hidden, device=args.device, dtype=torch.bfloat16
                )
                runner = make_runner(
                    kernel,
                    (weights[0], weights[1], topk_weights, topk_ids, args.experts),
                    activation,
                )
                band = (
                    resolve_band(tokens, args.hidden, inter, quant_mode, args.experts)
                    if backend == "b12x"
                    else "-"
                )
                if args.ncu_one:
                    runner(hidden)
                    torch.cuda.synchronize()
                    print(f"  [ncu] one call launched: {label} m={tokens} band={band}")
                    continue
                try:
                    median_ms, min_ms, max_ms, mean_ms = time_events(
                        runner,
                        hidden,
                        use_graph=not args.no_graph,
                        flush=None if args.warm_l2 else flush,
                        iters=args.iters,
                    )
                    if args.timer in ("cupti", "both"):
                        cupti_ms = time_cupti(
                            runner,
                            hidden,
                            use_graph=not args.no_graph,
                            cold_l2=not args.warm_l2,
                            repeat_ms=args.repeat_ms,
                        )
                        print(f"        [cupti cross-check] {cupti_ms:8.3f} ms")
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  {label:28s} m={tokens:<4d} FAILED "
                        f"{type(exc).__name__}: {str(exc)[:110]}"
                    )
                    records.append(
                        dict(
                            backend=label,
                            tokens=tokens,
                            band=band,
                            ms=float("nan"),
                            error=f"{type(exc).__name__}: {str(exc)[:300]}",
                        )
                    )
                    continue
                kernel_ms = None
                if args.kernel_profile:
                    rows = profile_kernels(runner, hidden)
                    kernel_ms = sum(row[1] for row in rows)
                    for name, row_ms, count in rows[:4]:
                        print(f"        {row_ms:8.3f} ms x{count:<3d} {name[:92]}")
                records.append(
                    dict(
                        backend=label,
                        tokens=tokens,
                        band=band,
                        ms=median_ms,
                        ms_min=min_ms,
                        ms_max=max_ms,
                        ms_per_step=median_ms * args.layers,
                        kernel_sum_ms=kernel_ms,
                    )
                )
                print(
                    f"  {label:28s} m={tokens:<4d} band={band:8s} "
                    f"{median_ms:8.3f} ms/layer [{min_ms:.3f},{max_ms:.3f}] "
                    f"{median_ms * args.layers:7.2f} ms/step({args.layers})"
                    + (f" kernelsum={kernel_ms:7.3f}" if kernel_ms else "")
                )
                if args.check:
                    outputs[(label, tokens)] = runner(hidden).clone()

    if args.check and not args.ncu_one:
        report_checks(outputs)

    if args.json_out:
        import json

        with open(args.json_out, "w") as handle:
            json.dump(records, handle, indent=1)
        print(f"[out] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
