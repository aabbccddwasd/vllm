"""GPU Performance Monitor for DeepSeek V4 on SM120.

Uses torch.cuda.Event pairs for low-overhead GPU timing with zero CPU
synchronization during the hot path.  Events are recorded during the forward
pass and resolved asynchronously after model() returns, avoiding GPU stalls.

Thread-local singleton pattern ensures the monitor is accessible from model
code without plumbing through forward_context.

Usage:
    monitor = get_perf_monitor()
    with monitor.timing("attn.phase_a"):
        ...
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

import torch

_local = threading.local()


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ComponentStats:
    """Accumulated stats for one named component."""

    name: str
    total_time_us: float = 0.0
    count: int = 0
    # Per-iteration times (resolved after each iter, kept for percentile calc).
    times_us: list[float] = field(default_factory=list)

    def record(self, elapsed_us: float) -> None:
        self.total_time_us += elapsed_us
        self.count += 1
        self.times_us.append(elapsed_us)

    @property
    def avg_time_us(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_time_us / self.count

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_time_us": self.total_time_us,
            "count": self.count,
            "avg_time_us": self.avg_time_us,
        }


@dataclass
class CpuTiming:
    """Accumulated CPU wall-clock timing for one named component.

    Unlike ComponentStats which relies on torch.cuda.Event pairs,
    this measures CPU-side wall-clock time using time.perf_counter().
    Useful for diagnostics like IPC/RPC latency, event synchronization
    wait time, and serialization overhead that happen on the CPU.
    """

    name: str
    total_time_us: float = 0.0
    count: int = 0
    times_us: list[float] = field(default_factory=list)

    def record(self, elapsed_us: float) -> None:
        self.total_time_us += elapsed_us
        self.count += 1
        self.times_us.append(elapsed_us)

    @property
    def avg_time_us(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_time_us / self.count

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_time_us": self.total_time_us,
            "count": self.count,
            "avg_time_us": self.avg_time_us,
        }


@dataclass
class PhaseStats:
    """Prefill or decode stats for all components."""

    name: str  # "prefill" or "decode"
    num_iterations: int = 0
    num_tokens: int = 0
    components: dict[str, ComponentStats] = field(default_factory=dict)

    def ensure_component(self, name: str) -> ComponentStats:
        if name not in self.components:
            self.components[name] = ComponentStats(name=name)
        return self.components[name]

    def record(self, comp_name: str, elapsed_us: float) -> None:
        self.ensure_component(comp_name).record(elapsed_us)


# ---------------------------------------------------------------------------
# Timing Context Manager
# ---------------------------------------------------------------------------


class _TimingScope(AbstractContextManager):
    """Records a torch.cuda.Event pair for a named component.

    Automatically degrades to a no-op when:
    - Inside a CUDA graph capture (cuEventCreate corrupts the graph).
    - Inside a ``torch.compile`` trace (non-Tensor ops are rejected).
    - ``full_mode`` is disabled on the monitor.

    Timing is collected from:
    - Prefill passes (never use CUDA graphs, not compiled with -O3).
    - Warmup iterations before CUDA graph capture.
    - Eager mode decode (``--enforce-eager``).
    """

    __slots__ = ("_monitor", "_name", "_start", "_end", "_skip")

    def __init__(self, monitor: "DeepSeekV4PerfMonitor", name: str) -> None:
        self._monitor = monitor
        self._name = name
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None
        self._skip: bool = False

    def __enter__(self) -> "_TimingScope":
        if not self._monitor._full_mode:
            self._skip = True
            return self
        # torch.compile traces the model forward; CUDA event ops return
        # non-Tensor values which Dynamo rejects.
        if torch.compiler.is_compiling():
            self._skip = True
            return self
        # cuEventCreate is NOT safe inside a CUDA graph capture.
        if torch.cuda.is_current_stream_capturing():
            self._skip = True
            return self
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self._start = start
        self._end = end
        return self

    def __exit__(self, *args: object) -> None:
        if self._skip:
            return
        assert self._end is not None
        self._end.record()
        self._monitor._push_event(self._name, self._start, self._end)  # type: ignore[arg-type]
        self._start = None
        self._end = None


class _CpuTimingScope(AbstractContextManager):
    """Records CPU wall-clock time for a named component.

    Uses ``time.perf_counter()`` — no GPU interaction required.
    Safe to use from engine core process or any non-GPU context.
    """

    __slots__ = ("_monitor", "_name", "_t0", "_skip")

    def __init__(self, monitor: "DeepSeekV4PerfMonitor", name: str) -> None:
        self._monitor = monitor
        self._name = name
        self._t0: float = 0.0
        self._skip: bool = False

    def __enter__(self) -> "_CpuTimingScope":
        if not self._monitor._enabled:
            self._skip = True
            return self
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        if self._skip:
            return
        elapsed_us = (time.perf_counter() - self._t0) * 1_000_000
        self._monitor.record_cpu_time(self._name, elapsed_us)


class _GraphTimingScope(AbstractContextManager):
    """Records a torch.cuda.Event pair for timing CUDA graph REPLAY.

    Unlike _TimingScope, this does NOT check for CUDA graph capture or
    torch.compile tracing.  It is the caller's responsibility to ensure
    this scope is only entered OUTSIDE of a CUDA graph capture (i.e.
    during warmup or replay, never during capture).

    This enables per-component GPU timing even when the model forward
    itself is executed via ``cudagraph.replay()``.
    """

    __slots__ = ("_monitor", "_name", "_start", "_end", "_skip")

    def __init__(self, monitor: "DeepSeekV4PerfMonitor", name: str) -> None:
        self._monitor = monitor
        self._name = name
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None
        self._skip: bool = False

    def __enter__(self) -> "_GraphTimingScope":
        if not self._monitor._full_mode:
            self._skip = True
            return self
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self._start = start
        self._end = end
        return self

    def __exit__(self, *args: object) -> None:
        if self._skip:
            return
        assert self._end is not None
        self._end.record()
        self._monitor._push_event(self._name, self._start, self._end)  # type: ignore[arg-type]
        self._start = None
        self._end = None


# ---------------------------------------------------------------------------
# Global-singleton access (thread-local, pattern borrowed from vllm/utils.py)
# ---------------------------------------------------------------------------

class _NullTimingScope(AbstractContextManager):
    """No-op fallback for disabled monitor."""

    __slots__ = ()

    def __enter__(self) -> "_NullTimingScope":
        return self

    def __exit__(self, *args: object) -> None:
        pass


class DeepSeekV4PerfMonitor:
    """Low-overhead per-component GPU timer for DeepSeek V4.

    Thread-local singleton — access via ``get_perf_monitor()``.
    When ``VLLM_DEEPSEEK_V4_PERF_MONITOR`` is unset the singleton is a
    ``_NullMonitor`` whose ``timing()`` is a no-op (zero overhead).
    """

    def __init__(self, full_mode: bool = True) -> None:
        self._enabled = True
        self._full_mode = full_mode  # component-level timing
        self._started_at: float | None = None

        # ---- iteration-level state ----
        # Pending events from the just-completed forward pass.
        self._pending_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        # Current iteration phase (set by runner before forward).
        self._current_phase: str = "unknown"
        self._current_num_prefill_tokens: int = 0
        self._current_num_decode_tokens: int = 0

        # ---- accumulated stats ----
        self._prefill: PhaseStats = PhaseStats(name="prefill")
        self._decode: PhaseStats = PhaseStats(name="decode")
        self._mixed: PhaseStats = PhaseStats(name="mixed")
        self._model_total_stats = ComponentStats(name="model.total")
        self._num_iterations: int = 0
        # Iterations in which Python code actually ran inside the model
        # (CUDA graph capture, prefill, or eager mode).  CDGA graph replay
        # passes do NOT count here because no TimingScope runs.
        self._num_python_iterations: int = 0
        self._wall_time_ms: list[float] = []
        self._cpu_timings: dict[str, CpuTiming] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def full_mode(self) -> bool:
        return self._full_mode

    def start(self) -> None:
        self._started_at = time.monotonic()
        self.reset()
        # Register atexit handler to dump report on exit.
        atexit.register(self._on_exit)

    def stop(self) -> str:
        self._enabled = False
        return self.get_report()

    def reset(self) -> None:
        self._pending_events.clear()
        self._prefill = PhaseStats(name="prefill")
        self._decode = PhaseStats(name="decode")
        self._mixed = PhaseStats(name="mixed")
        self._model_total_stats = ComponentStats(name="model.total")
        self._num_iterations = 0
        self._num_python_iterations = 0
        self._wall_time_ms.clear()
        self._cpu_timings.clear()

    def timing(self, name: str) -> AbstractContextManager:
        """Return a context manager that times GPU work in ``name``.

        When the monitor is disabled or in graph-safe (non-full) mode the
        returned context manager is a no-op.
        """
        if not self._enabled or not self._full_mode:
            return _NullTimingScope()
        return _TimingScope(self, name)

    def record_cpu_time(self, name: str, elapsed_us: float) -> None:
        """Record a CPU wall-clock timing measurement in microseconds.

        Unlike ``timing()`` which uses CUDA events for GPU work, this
        records purely CPU-side duration (e.g. IPC/RPC latency, event
        synchronization wait time, serialization overhead).

        Safe to call from any thread or process (no GPU required).
        """
        if not self._enabled:
            return
        if name not in self._cpu_timings:
            self._cpu_timings[name] = CpuTiming(name=name)
        self._cpu_timings[name].record(elapsed_us)

    def cpu_timing(self, name: str) -> AbstractContextManager:
        """Return a context manager that measures CPU wall-clock time.

        Uses ``time.perf_counter()`` for high-resolution CPU timing.
        When the monitor is disabled the returned context manager is a no-op.
        """
        if not self._enabled:
            return _NullTimingScope()
        return _CpuTimingScope(self, name)

    def record_graph_timing(self, name: str) -> AbstractContextManager:
        """Return a context manager that times GPU work around a CUDA graph
        REPLAY call (or eager model forward).

        Unlike ``timing()`` this does NOT check for CUDA graph capture or
        torch.compile tracing.  It is safe to use during graph *replay*
        or eager execution, but MUST NOT be used during graph *capture*
        (creating CUDA events inside capture corrupts the graph).

        When the monitor is disabled the returned context manager is a no-op.
        """
        if not self._enabled or not self._full_mode:
            return _NullTimingScope()
        return _GraphTimingScope(self, name)

    # ------------------------------------------------------------------
    # Internal: called by TimingScope / GPU model runner
    # ------------------------------------------------------------------

    def _push_event(
        self,
        name: str,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
    ) -> None:
        """Push a pending event pair (called from TimingScope.__exit__)."""
        if not self._enabled:
            return
        self._pending_events.append((name, start, end))

    def begin_iteration(
        self,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        wall_ms: float = 0.0,
    ) -> None:
        """Called at the start of each model execution step."""
        if not self._enabled:
            return
        self._current_num_prefill_tokens = num_prefill_tokens
        self._current_num_decode_tokens = num_decode_tokens
        self._current_phase = _resolve_phase(num_prefill_tokens, num_decode_tokens)
        self._pending_events.clear()
        self._num_iterations += 1
        if wall_ms > 0:
            self._wall_time_ms.append(wall_ms)

    def end_iteration(self) -> None:
        """Called after model() returns; resolves all pending event pairs.

        When CUDA graphs are active, decode replay passes only have
        model.total events (recorded outside the graph in
        _model_forward). Phase-level component events are only
        recorded when Python code actually runs inside the model
        (CUDA graph capture, prefill, or eager mode).
        """
        if not self._enabled:
            return
        # Synchronize once to drain all pending CUDA work.
        torch.cuda.synchronize()
        had_python = False
        for name, start, end in self._pending_events:
            elapsed_us = start.elapsed_time(end) * 1000  # ms → us
            if name == "model.total":
                self._model_total_stats.record(elapsed_us)
            else:
                phase_stats = self._get_phase_stats(self._current_phase)
                phase_stats.record(name, elapsed_us)
                had_python = True
        if had_python:
            self._num_python_iterations += 1
            self._update_token_counts()
        self._pending_events.clear()
        # Flush report to file every 8 total iterations so it survives
        # regardless of shutdown path.  model.total is always available
        # even when torch.compile elides component scopes.
        if self._num_iterations > 0 and self._num_iterations % 8 == 0:
            self._save_to_file()

    def get_report(self) -> str:
        """Return a human-readable ASCII report."""
        return format_report(self)

    def export_json(self, path: str) -> None:
        """Export all stats as JSON."""
        data = _to_json_dict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _save_to_file(self) -> str:
        """Write current report to a temp file; return the path."""
        import os as _os

        report = self.get_report()
        path = f"/tmp/dsv4_perf_monitor_{_os.getpid()}.txt"
        with open(path, "w") as f:
            f.write(report)
            f.write("\n")
        return path

    def _on_exit(self) -> None:
        """atexit handler — flush pending events and save final report.

        Writes to a temp file because worker processes' stderr may be
        piped through the multiprocessing executor which is already
        shutting down when atexit runs.
        """
        if not self._enabled or self._num_iterations == 0:
            return
        # Sync any remaining pending events.
        if self._pending_events:
            try:
                torch.cuda.synchronize()
                for name, start, end in self._pending_events:
                    elapsed_us = start.elapsed_time(end) * 1000
                    if name == "model.total":
                        self._model_total_stats.record(elapsed_us)
                    else:
                        phase_stats = self._get_phase_stats(self._current_phase)
                        phase_stats.record(name, elapsed_us)
                if self._pending_events:
                    self._update_token_counts()
            except Exception:
                pass
        try:
            path = self._save_to_file()
            if sys.stderr is not None:
                print(f"\n[DSv4PerfMonitor] Report: {path}", file=sys.stderr)
        except Exception:
            pass

    def _get_phase_stats(self, phase: str) -> PhaseStats:
        if phase == "prefill":
            return self._prefill
        elif phase == "decode":
            return self._decode
        else:
            return self._mixed

    def _update_token_counts(self) -> None:
        phase_stats = self._get_phase_stats(self._current_phase)
        phase_stats.num_iterations += 1
        phase_stats.num_tokens += (
            self._current_num_prefill_tokens + self._current_num_decode_tokens
        )

    @property
    def prefill(self) -> PhaseStats:
        return self._prefill

    @property
    def decode(self) -> PhaseStats:
        return self._decode

    @property
    def mixed(self) -> PhaseStats:
        return self._mixed

    @property
    def model_total(self) -> ComponentStats:
        return self._model_total_stats

    @property
    def num_iterations(self) -> int:
        return self._num_iterations

    @property
    def started_at(self) -> float | None:
        return self._started_at

    @property
    def wall_time_ms(self) -> list[float]:
        return self._wall_time_ms

    @property
    def cpu_timings(self) -> dict[str, CpuTiming]:
        return self._cpu_timings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_phase(num_prefill: int, num_decode: int) -> str:
    if num_prefill > 0 and num_decode == 0:
        return "prefill"
    elif num_decode > 0 and num_prefill == 0:
        return "decode"
    elif num_prefill > 0 and num_decode > 0:
        return "mixed"
    return "unknown"


# ---------------------------------------------------------------------------
# Thread-local singleton
# ---------------------------------------------------------------------------

_NullMonitor = DeepSeekV4PerfMonitor(full_mode=False)
_NullMonitor._enabled = False


def _create_monitor() -> DeepSeekV4PerfMonitor:
    import os

    if os.environ.get("VLLM_DEEPSEEK_V4_PERF_MONITOR", "").strip() in (
        "1",
        "true",
        "True",
    ):
        return DeepSeekV4PerfMonitor(full_mode=True)
    return _NullMonitor


def get_perf_monitor() -> DeepSeekV4PerfMonitor:
    """Get or create the thread-local performance monitor.

    Returns a no-op monitor when VLLM_DEEPSEEK_V4_PERF_MONITOR is unset.
    """
    monitor = getattr(_local, "perf_monitor", None)
    if monitor is None:
        monitor = _create_monitor()
        _local.perf_monitor = monitor
    return monitor


def set_perf_monitor(monitor: DeepSeekV4PerfMonitor) -> None:
    """Replace the thread-local monitor (used in tests)."""
    _local.perf_monitor = monitor


def reset_perf_monitor() -> None:
    """Remove the thread-local monitor."""
    _local.perf_monitor = None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_report(monitor: DeepSeekV4PerfMonitor) -> str:
    """Format accumulated stats as an ASCII table."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  DeepSeek V4 SM120 Perf Monitor 报告")
    lines.append("=" * 78)
    lines.append(
        f"  总迭代数: {monitor.num_iterations} "
        f"(其中 Python 执行: {monitor._num_python_iterations})"
    )
    lines.append(
        f"  阶段分布: prefill {monitor.prefill.num_iterations}, "
        f"decode {monitor.decode.num_iterations}, "
        f"mixed {monitor.mixed.num_iterations}"
    )

    # Per-phase tables
    for phase in (
        monitor.prefill,
        monitor.decode,
        monitor.mixed,
    ):
        if phase.num_iterations == 0 or not phase.components:
            continue
        _format_phase_table(lines, phase)

    # CPU wall-clock timing section
    if monitor.cpu_timings:
        lines.append("")
        lines.append("  ── CPU WALL-CLOCK TIMINGS ──")
        header = (f"  {'Component':<42s} {'Avg(us)':>10s} "
                  f"{'Total':>12s} {'Count':>6s}")
        lines.append(header)
        lines.append("  " + "-" * 72)
        for name, t in sorted(monitor.cpu_timings.items(),
                              key=lambda x: x[1].total_time_us,
                              reverse=True):
            lines.append(
                f"  {name:<42s} {t.avg_time_us:>10.1f} "
                f"{t.total_time_us:>12,.1f} {t.count:>6d}"
            )
        lines.append("  " + "-" * 72)

    # Model total row
    if monitor.model_total.count > 0:
        lines.append("")
        total_avg = monitor.model_total.avg_time_us
        lines.append(
            f"  model.total: {total_avg:,.1f} us avg "
            f"({monitor.model_total.count} iters)"
        )

    if monitor.started_at is not None:
        elapsed = time.monotonic() - monitor.started_at
        lines.append(f"  采集耗时: {elapsed:.1f}s")

    lines.append("=" * 78)
    return "\n".join(lines)


def _format_phase_table(lines: list[str], phase: PhaseStats) -> None:
    """Format a single phase table, showing exclusive (non-overlapping) time.

    Timing scopes nest naturally: ``attn`` wraps all ``attn.phase_*``
    children.  We compute *exclusive* time for each component by subtracting
    the total of its direct children, so the reported total equals the
    outermost scope without double-counting.
    """
    components = sorted(
        phase.components.values(), key=lambda c: c.total_time_us, reverse=True
    )
    if not components:
        return

    iters = phase.num_iterations
    tokens = phase.num_tokens

    # ---- compute exclusive times ----
    # exclusive(c) = total(c) - sum{total(d) | d is direct child of c}
    # d is a direct child of c if d.name.startswith(c.name + ".") and no
    # other component sits between them in the prefix chain.
    comp_map = {c.name: c for c in components}
    exclusive: dict[str, float] = {}
    for c in components:
        exc = c.total_time_us
        prefix = c.name + "."
        for d in components:
            if d.name.startswith(prefix):
                # d is a child; check it's a *direct* child (no deeper
                # prefix between c and d).
                suffix = d.name[len(prefix):]
                if "." not in suffix:
                    exc -= d.total_time_us
        exclusive[c.name] = max(exc, 0.0)

    # ---- filter & sort by exclusive time ----
    # Show only components with meaningful exclusive time (> 0.1% of max).
    max_exc = max(exclusive.values()) if exclusive else 1.0
    threshold = max_exc * 0.001
    significant = [
        (name, exc)
        for name, exc in exclusive.items()
        if exc >= threshold
    ]
    significant.sort(key=lambda x: x[1], reverse=True)

    total_exclusive = sum(e for _, e in significant)
    if total_exclusive == 0:
        return

    avg_iter_exc = total_exclusive / iters if iters > 0 else 0.0
    header_extra = ""
    if tokens > 0:
        header_extra = f", {total_exclusive / tokens:.1f} us/token"

    lines.append("")
    lines.append(
        f"  ── {phase.name.upper()} "
        f"({iters} 次 Python 执行, "
        f"{tokens} tokens{header_extra}) ──"
    )
    header = (f"  {'Component':<42s} {'Avg/iter(us)':>14s} "
              f"{'%':>8s} {'Count':>6s}")
    lines.append(header)
    lines.append("  " + "-" * 72)
    for name, exc in significant:
        pct = exc / total_exclusive * 100 if total_exclusive > 0 else 0
        comp = comp_map[name]
        avg_exc = exc / comp.count if comp.count > 0 else 0.0
        # Convert to per-iteration: avg_exc * comp.count / iters
        avg_iter = exc / iters if iters > 0 else 0.0
        lines.append(
            f"  {name:<42s} {avg_iter:>14.1f} "
            f"{pct:>7.1f}% {comp.count:>6d}"
        )
    lines.append("  " + "-" * 72)
    lines.append(
        f"  {'TOTAL':<42s} {avg_iter_exc:>14.1f} {'100.0%':>8s}"
    )


def _to_json_dict(monitor: DeepSeekV4PerfMonitor) -> dict[str, Any]:
    """Serialize monitor stats to a JSON-serializable dict."""
    result: dict[str, Any] = {
        "num_iterations": monitor.num_iterations,
        "num_python_iterations": monitor._num_python_iterations,
    }
    for phase in (monitor.prefill, monitor.decode, monitor.mixed):
        if phase.num_iterations > 0:
            result[phase.name] = {
                "num_iterations": phase.num_iterations,
                "num_tokens": phase.num_tokens,
                "components": {
                    n: c.to_dict() for n, c in phase.components.items()
                },
            }
    if monitor.model_total.count > 0:
        result["model_total"] = monitor.model_total.to_dict()
        result["model_total"]["count"] = monitor.model_total.count
    return result


# Patch total_component_time_us onto PhaseStats (avoid circular import).
def _phase_total(self: PhaseStats) -> float:
    return sum(c.total_time_us for c in self.components.values())


PhaseStats.total_component_time_us = _phase_total  # type: ignore[method-assign]
