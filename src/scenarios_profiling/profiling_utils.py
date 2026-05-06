"""PyTorch Profiler utilities for scenario-generation pipeline profiling.
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

_log = logging.getLogger(__name__)

# Try importing torch and related profiler APIs.  If torch is not installed, we will fall back to no-op stubs.
try:
    import torch
    import torch.profiler as _torch_profiler

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    _log.warning(
        "torch is not installed – profiling stubs will be used.  "
        "Install PyTorch to enable real profiling."
    )


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProfilerConfig:
    """Configuration for the PyTorch profiler.

    Attributes
    ----------
    profile_dir:
        Directory where profiler trace files will be written.
        Created automatically if it does not exist.
    record_shapes:
        Record tensor shapes in profiler events.  Useful when profiling
        embedding or tokenisation stages; mostly a no-op for LLM I/O work.
    with_stack:
        Capture Python call stacks for each operator.  Adds overhead but
        gives richer flame-graph output.
    with_flops:
        Estimate FLOPs for compute-heavy ops (matmuls, convolutions).
    profile_memory:
        Track tensor memory allocation/deallocation.
    export_chrome_trace:
        Write a ``chrome_trace.json`` that can be loaded in
        ``chrome://tracing`` or Perfetto.
    export_stacks:
        Write a ``stacks.txt`` flamegraph-compatible file (requires
        ``with_stack=True``).
    print_summary:
        Print a human-readable key-averages table to stdout when the
        profiler context exits.
    wait_steps:
        How many ``prof.step()`` calls to skip before profiling begins.
    warmup_steps:
        How many ``prof.step()`` calls are treated as warmup.
    active_steps:
        How many ``prof.step()`` calls are actively profiled.
    repeat:
        How many schedule cycles to run (0 = repeat indefinitely).
    """

    profile_dir: str = "profiling_output"
    record_shapes: bool = False
    with_stack: bool = False
    with_flops: bool = False
    profile_memory: bool = False
    export_chrome_trace: bool = True
    export_stacks: bool = False
    print_summary: bool = True
    # Schedule knobs
    wait_steps: int = 0
    warmup_steps: int = 0
    active_steps: int = 1
    repeat: int = 1


# ---------------------------------------------------------------------------
# No-op stubs used when torch is not available
# ---------------------------------------------------------------------------


class _NoOpProfiler:
    """A no-op context manager that mimics the torch.profiler.profile API."""

    def __enter__(self) -> "_NoOpProfiler":
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def step(self) -> None:  # noqa: D102
        pass


@contextlib.contextmanager
def _noop_record(name: str) -> Generator[None, None, None]:  # pragma: no cover
    """No-op record_function stub."""
    _log.debug("[profiling stub] record_function: %s", name)
    yield


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_profiler(cfg: ProfilerConfig | None = None) -> object:
    """Build and return a ``torch.profiler.profile`` context manager.

    If PyTorch is not installed, returns a no-op profiler so callers do not
    need to guard every use site.

    Parameters
    ----------
    cfg:
        ``ProfilerConfig`` instance.  Uses default values when ``None``.

    Returns
    -------
    A context manager whose ``__enter__`` value exposes ``.step()``.
    """
    if cfg is None:
        cfg = ProfilerConfig()

    if not _TORCH_AVAILABLE:
        _log.warning("torch not available – using no-op profiler.")
        return _NoOpProfiler()

    out_dir = Path(cfg.profile_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chrome_trace_path = str(out_dir / "chrome_trace.json")
    stacks_path = str(out_dir / "stacks.txt")

    activities = [_torch_profiler.ProfilerActivity.CPU]

    schedule = _torch_profiler.schedule(
        wait=cfg.wait_steps,
        warmup=cfg.warmup_steps,
        active=cfg.active_steps,
        repeat=cfg.repeat,
    )

    def _on_trace_ready(prof: object) -> None:
        if cfg.export_chrome_trace:
            prof.export_chrome_trace(chrome_trace_path)
            _log.info("Profiler chrome trace written to: %s", chrome_trace_path)
        if cfg.export_stacks and cfg.with_stack:
            prof.export_stacks(stacks_path, metric="self_cpu_time_total")
            _log.info("Profiler stacks written to: %s", stacks_path)
        if cfg.print_summary:
            print("\n" + "=" * 80)
            print("PyTorch Profiler — Key Averages (sorted by CPU time)")
            print("=" * 80)
            print(
                prof.key_averages().table(
                    sort_by="self_cpu_time_total",
                    row_limit=30,
                )
            )

    profiler = _torch_profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=_on_trace_ready,
        record_shapes=cfg.record_shapes,
        with_stack=cfg.with_stack,
        with_flops=cfg.with_flops,
        profile_memory=cfg.profile_memory,
    )
    return profiler


@contextlib.contextmanager
def record(name: str) -> Generator[None, None, None]:
    """Context manager that wraps a code block in a ``record_function`` span.

    Works as a no-op when torch is not installed.

    Parameters
    ----------
    name:
        Human-readable label shown in profiler traces and tables.
        By convention use ``snake_case`` with a phase prefix, e.g.
        ``"phase_1_grounding"`` or ``"phase_3_generate_iot"``.
    """
    if not _TORCH_AVAILABLE:
        yield
        return

    with _torch_profiler.record_function(name):
        yield


__all__ = [
    "ProfilerConfig",
    "build_profiler",
    "record",
]
