# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark: FlashInfer PCIe CUDA-IPC all-reduce vs PyNCCL.

Question
--------
On a 4-GPU PCIe-only node (no NVLink, no symmetric-memory support) is
FlashInfer's PCIe IPC all-reduce faster than the PyNCCL all-reduce vLLM uses
today, at the shapes DeepSeek-V4.1 issues for tensor parallelism
(hidden = 5120, bf16, one collective per layer)?

Method
------
nccl-tests style: event timing around a single collective, with a device-side
rendezvous **outside** the timed interval so a fast rank cannot start its clock
before a slow rank arrives. Every rank is measured and each sample is reduced
with MAX; the reported number is the median of those per-sample maxima -- a
rank-local event time is not a distributed latency.

Both eager launches and CUDA-graph replays are measured. Production decode
issues this collective from inside a captured FULL graph, so the graph number
is the decision-relevant one; the eager number shows what launch overhead
costs at these small payloads.

Candidates are rotated within each sample so clock and rank drift are shared
rather than attributed to one backend. L2 is *not* flushed between iterations:
the payloads in question stay resident in production, so flushing would
measure a different machine.

``CUDA_DEVICE_ORDER=PCI_BUS_ID`` is required, not cosmetic: without it the
visible-device ordinals follow the runtime's own "fastest first" order, so on a
mixed box ``CUDA_VISIBLE_DEVICES=0,1,3,4`` can select the wrong four GPUs. The
CUDA IPC workspace then fails with "peer access is not supported between these
two devices", which looks like a hardware limitation but is a device-selection
mistake. The script checks the peer matrix up front and says so.

Run (the PCIe IPC workspace only exists in FlashInfer >= 0.7.0):

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,3,4 \
    VLLM_NCCL_SO_PATH=$HOME/.local/lib/nccl/libnccl.so.2.30.4 \
    PYTHONPATH=/tmp/fi-new torchrun --standalone --nproc-per-node=4 \
        benchmarks/bench_flashinfer_pcie_ipc_allreduce.py \
        --tokens 1,2,4,6,8,12,16,24,32,48 --samples 100
"""

import argparse
import csv
import os
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import find_nccl_library
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)

DTYPE = torch.bfloat16

# bf16 accumulation of `world_size` copies of one value is not exact, and the
# two backends are free to sum in different orders. 2e-2 relative is a bit over
# two bf16 ulps, which is what a differing summation order can produce here.
RTOL = 2e-2
ATOL = 0.0


def parse_ints(raw: str) -> list[int]:
    return [int(part) for part in raw.replace(",", " ").split()]


def peer_matrix() -> tuple[list[str], list[list[bool]]]:
    """Per-pair CUDA peer access, and the device names in visible order."""
    import ctypes

    lib = ctypes.CDLL("libcudart.so")
    count = torch.cuda.device_count()
    # Leave the current device exactly as it was found: a later collective on a
    # process group binds to this process's device, and leaving the probe's last
    # device selected makes every subsequent NCCL call fail with "invalid usage".
    original = torch.cuda.current_device()
    names, matrix = [], []
    try:
        for row_device in range(count):
            torch.cuda.set_device(row_device)
            names.append(torch.cuda.get_device_name(row_device))
            row = []
            for column_device in range(count):
                can = ctypes.c_int(0)
                lib.cudaDeviceCanAccessPeer(
                    ctypes.byref(can),
                    ctypes.c_int(row_device),
                    ctypes.c_int(column_device),
                )
                row.append(bool(can.value))
            matrix.append(row)
    finally:
        torch.cuda.set_device(original)
    return names, matrix


def check_topology(require_peer_access: bool) -> None:
    """Report the GPU set, and refuse to mis-measure a peer-dependent backend."""
    names, matrix = peer_matrix()
    count = len(names)
    print(f"visible devices ({os.environ.get('CUDA_DEVICE_ORDER', 'order UNSET')}):")
    for index, name in enumerate(names):
        print(
            f"  vis{index} {name:<34} "
            + " ".join("Y" if cell else "." for cell in matrix[index])
        )
    if not require_peer_access:
        return
    blocked = [
        (i, j)
        for i in range(count)
        for j in range(count)
        if i != j and not matrix[i][j]
    ]
    if not blocked:
        return
    raise SystemExit(
        "\n".join(
            [
                (
                    f"peer access is missing for {len(blocked)} device pair(s): "
                    f"{blocked[:6]}."
                ),
                (
                    "The PCIe IPC all-reduce needs every pair, so this run would "
                    "fail inside cudaIpcOpenMemHandle with 'peer access is not "
                    "supported'."
                ),
                (
                    "Check that CUDA_DEVICE_ORDER=PCI_BUS_ID is set: without it, "
                    "CUDA_VISIBLE_DEVICES ordinals follow the runtime's own "
                    "fastest-first order and can select a different GPU set, for "
                    "example one that includes a GPU which cannot peer with the "
                    "others."
                ),
                "Use --backends nccl to benchmark NCCL only.",
            ]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument(
        "--tokens",
        default="1,2,4,6,8,12,16,24,32,48",
        help="Comma or space separated row counts to all-reduce. Production "
        "uses the CUDA-graph buckets ([6,12,24,48] with spec, "
        "[1,2,4,8,16,24,32] without).",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--backends", default="nccl,pcie_ipc", help="nccl, nccl_stock, pcie_ipc"
    )
    parser.add_argument("--modes", default="eager,graph", help="eager and/or graph")
    parser.add_argument(
        "--tune-cache",
        default=str(Path.home() / ".cache/vllm/pcie_ipc_allreduce_bench.json"),
        help="Tuning cache path, shared by all ranks.",
    )
    parser.add_argument(
        "--skip-tune",
        action="store_true",
        help="Reuse the cache only; fail rather than profile a miss.",
    )
    parser.add_argument("--csvexport", default="", help="Write the table here.")
    return parser.parse_args()


def make_pcie_workspace(
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
    hidden: int,
    tokens: list[int],
    tune_cache: str,
    skip_tune: bool,
):
    """Build the FlashInfer PCIe IPC workspace exactly as serving does.

    ``max_numel`` is sized to the largest collective that will be issued: the
    library warns that an oversized workspace spreads its epoch double buffer
    (``world_size * max_numel`` elements apart) and costs measurable time at
    small batch. So the workspace covers the requested buckets and nothing more.
    """
    from flashinfer.comm import PcieIpcAllReduceWorkspace

    workspace = PcieIpcAllReduceWorkspace(
        group=device_group,
        max_numel=max(tokens) * hidden,
        dtype=DTYPE,
        tune_batches=tuple(sorted(set(tokens))),
        tune_cache=tune_cache,
    )
    torch.accelerator.synchronize()
    workspace.rebind_stream()
    cached = Path(tune_cache).exists()
    if skip_tune and not cached:
        raise SystemExit(
            f"--skip-tune was requested but {tune_cache} does not exist; "
            "the untuned seed policy is not representative."
        )
    if not skip_tune:
        # Tuning is what replaces the workable-but-not-fast seed policy with
        # measurements from this machine; skipping it would benchmark the seed.
        workspace.tune([hidden], dtype=DTYPE, tune_group=cpu_group)
    # `tune` clears the in-process cache, and a shape whose configuration is
    # resolved lazily cannot be captured, so resolve every captured shape here.
    workspace.prepare([(t, hidden) for t in tokens], dtype=DTYPE)
    torch.accelerator.synchronize()
    workspace.rebind_stream()
    return workspace


def bench_tokens(
    args: argparse.Namespace,
    tokens: int,
    hidden: int,
    device: torch.device,
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
    comms: dict[str, PyNcclCommunicator],
    workspace,
    device_barrier: Callable[[], None],
) -> list[dict]:
    numel = tokens * hidden
    nbytes = numel * DTYPE.itemsize
    # Identical input on every rank, so the reference is `world_size * x`: a
    # closed form both backends can be checked against without either one
    # defining what "correct" means.
    torch.manual_seed(1234)
    x = torch.randn(tokens, hidden, dtype=DTYPE, device=device)
    expected = x.to(torch.float32) * dist.get_world_size(device_group)

    names = list(comms)
    inputs: dict[str, torch.Tensor] = {}
    outputs: dict[str, torch.Tensor] = {}
    for name, comm in comms.items():
        inputs[name] = x.clone()
        outputs[name] = torch.empty_like(x)
        # Warm the collective once so the timed region contains no allocation,
        # no communicator setup and no first-call configuration lookup.
        comm.all_reduce(inputs[name], outputs[name])
    unsupported = False
    if workspace is not None:
        # The workspace refuses shapes it cannot serve; serving falls back to
        # NCCL there, so record the miss instead of timing a fallback.
        if workspace.supports(x):
            names.append("pcie_ipc")
            inputs["pcie_ipc"] = x.clone()
            outputs["pcie_ipc"] = torch.empty_like(x)
            workspace.all_reduce(inputs["pcie_ipc"], out=outputs["pcie_ipc"])
        else:
            unsupported = True
    torch.accelerator.synchronize()

    rows = []
    for name in names:
        got = outputs[name].to(torch.float32)
        rel = ((got - expected).abs() / expected.abs().clamp_min(1e-3)).max().item()
        torch.testing.assert_close(
            outputs[name], expected.to(DTYPE), rtol=RTOL, atol=ATOL
        )
        if "nccl" in outputs and name != "nccl":
            torch.testing.assert_close(
                outputs[name], outputs["nccl"], rtol=RTOL, atol=ATOL
            )
        rows.append({"backend": name, "tokens": tokens, "max_rel_err": rel})

    eager_ops = {
        name: (lambda n=name: comms[n].all_reduce(inputs[n], outputs[n]))
        if name in comms
        else (lambda: workspace.all_reduce(inputs["pcie_ipc"], out=outputs["pcie_ipc"]))
        for name in names
    }
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    if "graph" in args.modes:
        for name in names:
            graphs[name] = capture_graph(
                eager_ops[name], workspace if name == "pcie_ipc" else None, cpu_group
            )

    times: dict[str, float] = {}
    if "eager" in args.modes:
        times.update(
            time_ops(
                eager_ops,
                names,
                args.samples,
                args.warmup,
                device_barrier,
                device,
                device_group,
                "eager",
            )
        )
    if "graph" in args.modes:
        times.update(
            time_ops(
                {name: (lambda g=g: g.replay()) for name, g in graphs.items()},
                names,
                args.samples,
                args.warmup,
                device_barrier,
                device,
                device_group,
                "graph",
            )
        )

    out_rows = []
    if unsupported:
        rows.append(
            {
                "backend": "pcie_ipc",
                "tokens": tokens,
                "max_rel_err": float("nan"),
                "note": "workspace does not support this shape",
            }
        )
    for name in names:
        check = next(r for r in rows if r["backend"] == name)
        for mode in args.modes.split(","):
            micros = times[f"{name}:{mode}"]
            out_rows.append(
                {
                    "tokens": tokens,
                    "numel": numel,
                    "backend": name,
                    "mode": mode,
                    "median_us": micros,
                    "algbw_GBps": nbytes / (micros * 1e3),
                    "busbw_GBps": nbytes
                    * 2
                    * (dist.get_world_size(device_group) - 1)
                    / dist.get_world_size(device_group)
                    / (micros * 1e3),
                    "max_rel_err": check["max_rel_err"],
                }
            )
    return out_rows


def capture_graph(
    run: Callable[[], None],
    workspace,
    cpu_group: dist.ProcessGroup,
) -> torch.cuda.CUDAGraph:
    """Capture one collective on a side stream, with ranks kept in step."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    dist.barrier(group=cpu_group)
    if workspace is not None:
        # The workspace asserts a single stream and exempts capture; sync and
        # rebind make the handoff to the capture stream explicit. Capture
        # warmups really execute, so they must run on the capture stream.
        torch.accelerator.synchronize()
        workspace.rebind_stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    stream.synchronize()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    dist.barrier(group=cpu_group)
    if workspace is not None:
        torch.accelerator.synchronize()
        workspace.rebind_stream()
    return graph


def time_ops(
    ops: dict[str, Callable[[], None]],
    names: list[str],
    samples: int,
    warmup: int,
    device_barrier: Callable[[], None],
    device: torch.device,
    device_group: dist.ProcessGroup,
    mode: str,
) -> dict[str, float]:
    """Median over samples of the per-sample MAX across ranks."""
    for index in range(warmup):
        device_barrier()
        ops[names[index % len(names)]]()
    torch.accelerator.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings: dict[str, list[float]] = {name: [] for name in names}
    for index in range(samples):
        name = names[index % len(names)]
        device_barrier()
        start.record()
        ops[name]()
        end.record()
        end.synchronize()
        elapsed_us = torch.tensor(
            start.elapsed_time(end) * 1000, dtype=torch.float64, device=device
        )
        dist.all_reduce(elapsed_us, op=dist.ReduceOp.MAX, group=device_group)
        timings[name].append(elapsed_us.item())
    return {f"{name}:{mode}": statistics.median(v) for name, v in timings.items()}


def main() -> None:
    args = parse_args()
    tokens = sorted({int(t) for t in parse_ints(args.tokens) if int(t) > 0})
    assert tokens and args.hidden > 0
    backends = [b for b in args.backends.split(",") if b]

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    init_distributed_environment()
    world_size = dist.get_world_size()
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=world_size)

    check_topology(require_peer_access="pcie_ipc" in backends)

    tp_group = get_tp_group()
    device_group = tp_group.device_group
    cpu_group = tp_group.cpu_group
    rank = tp_group.rank_in_group
    device = torch.device("cuda", local_rank)

    comms: dict[str, PyNcclCommunicator] = {}
    if "nccl" in backends:
        comm = tp_group.device_communicator.pynccl_comm
        assert comm is not None, "PyNCCL communicator was not built"
        comms["nccl"] = comm
    if "nccl_stock" in backends:
        # find_nccl_library() honours VLLM_NCCL_SO_PATH, which is what the
        # patched-NCCL baseline already uses; resolve the stock library with the
        # variable hidden so the two baselines really are different libraries.
        saved = os.environ.pop("VLLM_NCCL_SO_PATH", None)
        try:
            stock_library = find_nccl_library()
        finally:
            if saved is not None:
                os.environ["VLLM_NCCL_SO_PATH"] = saved
        # PyNcclCommunicator builds its own NCCL communicator over the group's
        # store, so it must be attached to a non-NCCL group -- same as vLLM does
        # for its own communicator.
        comms["nccl_stock"] = PyNcclCommunicator(
            group=cpu_group, device=device, library_path=stock_library
        )
    workspace = None
    if "pcie_ipc" in backends:
        workspace = make_pcie_workspace(
            device_group,
            cpu_group,
            args.hidden,
            tokens,
            args.tune_cache,
            args.skip_tune,
        )

    # A cheap device-side rendezvous for the timed region.
    sync_in = torch.zeros(1, device=device)
    sync_out = torch.empty_like(sync_in)
    barrier_comm = comms["nccl"] if "nccl" in comms else next(iter(comms.values()))

    def device_barrier() -> None:
        barrier_comm.all_reduce(sync_in, sync_out)

    results: list[dict] = []
    for count in tokens:
        results.extend(
            bench_tokens(
                args,
                count,
                args.hidden,
                device,
                device_group,
                cpu_group,
                comms,
                workspace,
                device_barrier,
            )
        )
        if rank == 0:
            print(f"[tokens={count}] done", flush=True)

    if rank == 0:
        metadata = {
            "world_size": world_size,
            "gpu": torch.cuda.get_device_name(local_rank),
            "hidden": args.hidden,
            "dtype": str(DTYPE),
            "backends": ",".join(backends),
            "pynccl_lib": env_lib(),
            "fi_version": _flashinfer_version(),
            "samples_per_backend": args.samples // max(len(backends), 1),
            "timing": "event, median of per-sample MAX across ranks; device "
            "barrier outside the timed interval; L2 kept warm",
            "clocks": "unpinned (no privilege); candidates rotated per sample, "
            "warmup before timing, medians reported",
        }
        for key, value in metadata.items():
            print(f"{key:<22}{value}")
        print()
        print_table(results)
        speedups(results, backends[0])
        if args.csvexport:
            with open(args.csvexport, "w", newline="") as handle:
                fields = sorted({k for row in results for k in row})
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(results)
            print(f"\nwrote {args.csvexport}")

    if workspace is not None:
        workspace.destroy()
    dist.barrier(group=cpu_group)
    cleanup_dist_env_and_memory()


def print_table(results: list[dict]) -> None:
    """Plain-text table with per-backend speedups against the first backend."""
    columns = [
        "tokens",
        "numel",
        "backend",
        "mode",
        "median_us",
        "algbw_GBps",
        "busbw_GBps",
        "max_rel_err",
    ]
    header = [
        "tokens",
        "numel",
        "backend",
        "mode",
        "median_us",
        "algbw_GB/s",
        "busbw_GB/s",
        "max_rel_err",
    ]
    body = []
    for row in results:
        if row.get("note"):
            body.append(
                [
                    str(row["tokens"]),
                    str(row.get("numel", "")),
                    row["backend"],
                    "-",
                    row["note"],
                    "",
                    "",
                    "",
                ]
            )
            continue
        body.append(
            [
                str(row["tokens"]),
                str(row["numel"]),
                row["backend"],
                row["mode"],
                f"{row['median_us']:.2f}",
                f"{row['algbw_GBps']:.2f}",
                f"{row['busbw_GBps']:.2f}",
                f"{row['max_rel_err']:.2e}",
            ]
        )
    widths = [
        max(len(header[i]), *(len(r[i]) for r in body)) if body else len(header[i])
        for i in range(len(header))
    ]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    for row in body:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    del columns


def speedups(results: list[dict], baseline: str) -> None:
    """Median-latency ratio of every backend against `baseline`, per (tokens, mode)."""
    index = {
        (r["tokens"], r["mode"], r["backend"]): r["median_us"]
        for r in results
        if "median_us" in r
    }
    print()
    print(f"speedup vs {baseline} (baseline_us / backend_us, >1 means faster)")
    print(
        f"{'tokens':>7}  {'mode':<6}  "
        + "  ".join(
            f"{name:>12}" for name in sorted({r["backend"] for r in results}) if True
        )
    )
    for tokens in sorted({r["tokens"] for r in results}):
        for mode in sorted({r["mode"] for r in results if "mode" in r}):
            base = index.get((tokens, mode, baseline))
            if base is None:
                continue
            cells = []
            for name in sorted({r["backend"] for r in results}):
                value = index.get((tokens, mode, name))
                cells.append(f"{base / value:>12.3f}" if value else f"{'-':>12}")
            print(f"{tokens:>7}  {mode:<6}  " + "  ".join(cells))


def env_lib() -> str:
    """Which PyNCCL library the `nccl` baseline resolved."""
    return os.environ.get("VLLM_NCCL_SO_PATH", "(default)")


def _flashinfer_version() -> str:
    try:
        import flashinfer

        return flashinfer.__version__
    except Exception:  # noqa: BLE001
        return "(absent)"


if __name__ == "__main__":
    main()
