"""W&B logging utilities for the scenario generation pipeline.

Design
------
All W&B logging is deferred to ``log_run_summary()``, which is called from
the main asyncio thread just before the pipeline finishes.  This avoids the
silent metric-drop problem that occurs when ``wandb.log()`` is called from
background executor threads.

Accumulated records (LLM calls, batch results) are stored in thread-safe
lists and replayed in ``log_run_summary()`` as a sequence of ``wandb.log()``
calls, followed by a single ``wandb.summary.update()`` for scalar aggregates.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm.base import LLMResult

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WandbConfig:
    project: str = "assetopsbench-scenario-gen"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    enabled: bool = True


# ---------------------------------------------------------------------------
# Internal data records
# ---------------------------------------------------------------------------


@dataclass
class LLMCallRecord:
    phase: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    wall_seconds: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class GenerationBatchRecord:
    focus: str
    attempt: int
    generated: int
    valid: int
    repaired: int
    rejected: int


# ---------------------------------------------------------------------------
# Main logger class
# ---------------------------------------------------------------------------


class WandbRunLogger:
    def __init__(
        self,
        config: WandbConfig,
        run_config: dict[str, Any] | None = None,
    ) -> None:
        self._config = config
        self._run: Any = None
        self._lock = threading.Lock()
        self._llm_calls: list[LLMCallRecord] = []
        self._batch_records: list[GenerationBatchRecord] = []

        if not config.enabled:
            return

        try:
            import wandb  # type: ignore[import]
        except ImportError:
            _log.warning("wandb not installed — W&B logging disabled.")
            self._config = WandbConfig(enabled=False)
            return

        try:
            self._run = wandb.init(
                project=config.project,
                entity=config.entity,
                name=config.run_name,
                tags=config.tags or None,
                notes=config.notes,
                config=run_config or {},
            )
            _log.info("W&B run started: %s", self._run.url)
        except Exception as exc:  # noqa: BLE001
            _log.warning("wandb.init failed: %s — W&B logging disabled.", exc)
            self._config = WandbConfig(enabled=False)

    @property
    def enabled(self) -> bool:
        return self._config.enabled and self._run is not None

    # ------------------------------------------------------------------
    # Accumulate from threads (no wandb.log here)
    # ------------------------------------------------------------------

    def log_llm_call(
        self,
        phase: str,
        result: "LLMResult | None",
        wall_seconds: float = 0.0,
    ) -> None:
        """Accumulate one LLM call record. Called from executor threads."""
        record = LLMCallRecord(
            phase=phase,
            prompt_tokens=result.input_tokens if result else None,
            completion_tokens=result.output_tokens if result else None,
            total_tokens=result.total_tokens if result else None,
            wall_seconds=wall_seconds,
        )
        with self._lock:
            self._llm_calls.append(record)

    def log_batch_result(
        self,
        focus: str,
        attempt: int,
        generated: int,
        valid: int,
        repaired: int,
        rejected: int,
    ) -> None:
        """Accumulate one batch result record. Called from executor threads."""
        with self._lock:
            self._batch_records.append(
                GenerationBatchRecord(
                    focus=focus,
                    attempt=attempt,
                    generated=generated,
                    valid=valid,
                    repaired=repaired,
                    rejected=rejected,
                )
            )

    # ------------------------------------------------------------------
    # Flush everything to W&B from the main thread
    # ------------------------------------------------------------------

    def log_run_summary(
        self,
        timing_registry: Any,
        cache_stats: dict[str, Any],
        total_scenarios: int,
        asset_name: str = "",
        generation_mode: str = "",
    ) -> None:
        """Replay all accumulated records and log summary.
        Must be called from the main thread."""
        if not self.enabled:
            return

        import wandb  # type: ignore[import]

        # ---- 1. Per-LLM-call metrics (one wandb.log per call) ----
        for rec in self._llm_calls:
            wandb.log({
                "llm/phase":             rec.phase,
                "llm/wall_seconds":      round(rec.wall_seconds, 3),
                "llm/prompt_tokens":     rec.prompt_tokens     or 0,
                "llm/completion_tokens": rec.completion_tokens or 0,
                "llm/total_tokens":      rec.total_tokens      or 0,
            })

        # ---- 2. Tables ----
        token_table = wandb.Table(
            columns=["phase", "prompt_tokens", "completion_tokens", "total_tokens", "wall_s"],
            data=[
                [r.phase, r.prompt_tokens or 0, r.completion_tokens or 0,
                 r.total_tokens or 0, round(r.wall_seconds, 3)]
                for r in self._llm_calls
            ],
        )
        batch_table = wandb.Table(
            columns=["focus", "attempt", "generated", "valid", "repaired", "rejected", "pass_rate"],
            data=[
                [b.focus, b.attempt, b.generated, b.valid, b.repaired, b.rejected,
                 round((b.valid + b.repaired) / b.generated, 3) if b.generated > 0 else 0.0]
                for b in self._batch_records
            ],
        )
        wandb.log({
            "llm_calls_table":     token_table,
            "batch_results_table": batch_table,
        })

        # ---- 4. Scalar summary (run-level aggregates) ----
        total_prompt     = sum(r.prompt_tokens     or 0 for r in self._llm_calls)
        total_completion = sum(r.completion_tokens or 0 for r in self._llm_calls)
        total_tokens     = sum(r.total_tokens      or 0 for r in self._llm_calls)

        phase_tokens: dict[str, dict[str, int]] = {}
        for rec in self._llm_calls:
            e = phase_tokens.setdefault(rec.phase, {"prompt": 0, "completion": 0, "total": 0})
            e["prompt"]     += rec.prompt_tokens     or 0
            e["completion"] += rec.completion_tokens or 0
            e["total"]      += rec.total_tokens      or 0

        focus_totals: dict[str, dict[str, int]] = {}
        for br in self._batch_records:
            e = focus_totals.setdefault(br.focus, {"generated": 0, "valid": 0, "repaired": 0, "rejected": 0, "attempts": 0})
            e["generated"] += br.generated
            e["valid"]     += br.valid
            e["repaired"]  += br.repaired
            e["rejected"]  += br.rejected
            e["attempts"]  += 1

        summary: dict[str, Any] = {
            "run/asset_name":          asset_name,
            "run/generation_mode":     generation_mode,
            "run/total_scenarios":     total_scenarios,
            "run/total_llm_calls":     len(self._llm_calls),
            "tokens/total_prompt":     total_prompt,
            "tokens/total_completion": total_completion,
            "tokens/total":            total_tokens,
        }

        for phase, tok in phase_tokens.items():
            safe = phase.replace(" ", "_")
            summary[f"tokens/{safe}/prompt"]     = tok["prompt"]
            summary[f"tokens/{safe}/completion"] = tok["completion"]
            summary[f"tokens/{safe}/total"]      = tok["total"]

        for row in timing_registry.to_dict():
            safe = row["name"].replace(" ", "_")
            summary[f"timing/{safe}/wall_s"] = row["wall_seconds"]
            summary[f"timing/{safe}/cpu_s"]  = row["cpu_seconds"]

        profile_l1 = cache_stats.get("profile_cache", {}).get("l1", {})
        l1_total   = profile_l1.get("hits", 0) + profile_l1.get("misses", 0)
        summary["cache/profile_l1_hits"]     = profile_l1.get("hits", 0)
        summary["cache/profile_l1_misses"]   = profile_l1.get("misses", 0)
        summary["cache/profile_l1_hit_rate"] = round(profile_l1.get("hits", 0) / l1_total, 3) if l1_total > 0 else 0.0
        fewshot = cache_stats.get("fewshot_cache", {})
        summary["cache/fewshot_hits"]   = fewshot.get("hits", 0)
        summary["cache/fewshot_misses"] = fewshot.get("misses", 0)

        for focus, totals in focus_totals.items():
            gen      = totals["generated"]
            accepted = totals["valid"] + totals["repaired"]
            summary[f"batch_summary/{focus}/total_generated"]   = gen
            summary[f"batch_summary/{focus}/total_valid"]       = totals["valid"]
            summary[f"batch_summary/{focus}/total_repaired"]    = totals["repaired"]
            summary[f"batch_summary/{focus}/total_rejected"]    = totals["rejected"]
            summary[f"batch_summary/{focus}/total_attempts"]    = totals["attempts"]
            summary[f"batch_summary/{focus}/overall_pass_rate"] = round(accepted / gen, 3) if gen > 0 else 0.0

        self._run.summary.update(summary)
        _log.info("W&B summary updated (%d keys).", len(summary))

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    def finish(self) -> None:
        if not self.enabled:
            return
        self._run.finish()
        _log.info("W&B run finished.")


# ---------------------------------------------------------------------------
# Null logger — always a no-op, used as default when --wandb is not passed
# ---------------------------------------------------------------------------


class NullWandbLogger(WandbRunLogger):
    def __init__(self) -> None:
        self._config = WandbConfig(enabled=False)
        self._run = None
        self._lock = threading.Lock()
        self._llm_calls = []
        self._batch_records = []
