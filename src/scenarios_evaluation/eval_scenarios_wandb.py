"""
Scenario quality evaluator — with Weights & Biases logging.

Identical to eval_scenarios.py but logs metrics to a W&B run after evaluation.

Logged metrics
--------------
Per-scenario (wandb.log, one call per scenario):
  eval/scenario_id            — scenario id string
  eval/type                   — scenario type
  eval/category               — scenario category
  eval/static_score           — static weighted score (/20)
  eval/llm_score              — LLM judge weighted score (/30), or None
  eval/dry_run_score          — dry-run weighted score (/50), or None
  eval/quality_score          — composite score (/100)

Per-LLM-call timeline (wandb.log, one call per generate() invocation):
  llm_call/phase              — "judge" or "dry_run"
  llm_call/wall_seconds       — wall-clock time for the call
  llm_call/prompt_tokens      — prompt token count
  llm_call/completion_tokens  — completion token count
  llm_call/total_tokens       — total tokens

Summary scalars (run.summary):
  summary/total_scenarios
  summary/static/mean_score
  summary/llm/evaluated_count
  summary/llm/mean_score
  summary/llm/dim/<dimension>_mean   — per LLM dimension mean score
  summary/dry_run/executed_count
  summary/dry_run/mean_score
  summary/dry_run/dim/<dimension>_mean — per dry-run dimension mean score
  summary/quality/mean_score
  summary/quality/min_score
  summary/quality/max_score
  summary/type_distribution/<type>        — scenario count per type
  summary/category_distribution/<cat>     — scenario count per category
  summary/llm_calls/total_count           — total LLM generate() calls
  summary/llm_calls/total_tokens          — total tokens across all calls
  summary/llm_calls/<phase>/count         — calls per phase
  summary/llm_calls/<phase>/total_tokens  — tokens per phase

Rich tables (wandb.log):
  eval_report_table    — one row per scenario with all scores
  llm_calls_table      — one row per LLM generate() call

Usage:
    uv run python src/scenarios_evaluation/eval_scenarios_wandb.py \\
        --wandb --wandb-project assetopsbench --wandb-run-name transformer-eval-50
    uv run python src/scenarios_evaluation/eval_scenarios_wandb.py --limit 10 --wandb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Re-use everything from the base evaluator
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR.parent))

from scenarios_evaluation.eval_scenarios import (  # noqa: E402
    _DEFAULT_MODEL,
    _DEFAULT_OUTPUT,
    _DEFAULT_SCENARIOS,
    _DRY_RUN_MAX,
    _DRY_RUN_WEIGHTS,
    _LLM_MAX,
    _LLM_WEIGHTS,
    _SCORE_MAX,
    _STATIC_MAX,
    _STATIC_WEIGHTS,
    _load_live_tool_schemas,
    _live_tool_schemas,
    _print_summary,
    _run_dry_run,
    _run_llm_judge,
    _run_static_checks,
)


# ---------------------------------------------------------------------------
# LLM call instrumentation
# ---------------------------------------------------------------------------


@dataclass
class _LLMCallRecord:
    phase: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    wall_seconds: float


_llm_call_records: list[_LLMCallRecord] = []


@contextmanager
def _instrument_llm(phase: str):
    """Patch LiteLLMBackend so every generate_with_usage() call is recorded.

    Works because _run_llm_judge / _run_dry_run import LiteLLMBackend
    inside their function bodies, so they pick up the patched class.
    """
    import llm.litellm as _llm_mod  # noqa: PLC0415

    _Orig = _llm_mod.LiteLLMBackend
    _records = _llm_call_records
    _phase = phase

    class _Recording(_Orig):  # type: ignore[valid-type]
        def generate_with_usage(
            self,
            prompt: str,
            temperature: float = 0.0,
            max_tokens: int | None = None,
        ) -> Any:
            t0 = time.perf_counter()
            result = super().generate_with_usage(prompt, temperature, max_tokens)
            elapsed = time.perf_counter() - t0
            _records.append(
                _LLMCallRecord(
                    phase=_phase,
                    prompt_tokens=result.input_tokens,
                    completion_tokens=result.output_tokens,
                    total_tokens=result.total_tokens,
                    wall_seconds=elapsed,
                )
            )
            return result

    _llm_mod.LiteLLMBackend = _Recording
    try:
        yield
    finally:
        _llm_mod.LiteLLMBackend = _Orig


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------


def _log_to_wandb(
    run: object,
    scenarios: list[dict],
    static_results: list[dict],
    llm_results: list[dict] | None,
    dry_run_results: list[dict] | None,
    report: list[dict],
) -> None:
    """Log all evaluation metrics to the provided W&B run."""
    import wandb  # type: ignore[import]

    llm_by_id = {r["id"]: r for r in (llm_results or [])}
    dry_run_by_id = {r["id"]: r for r in (dry_run_results or [])}

    # ---- 1. Per-scenario metrics (one wandb.log per scenario) ----
    for entry in report:
        sid = entry["id"]
        lr = llm_by_id.get(sid, {})
        dr = dry_run_by_id.get(sid, {})

        row: dict = {
            "eval/scenario_id":  sid,
            "eval/type":         entry.get("type", ""),
            "eval/category":     entry.get("category", ""),
            "eval/static_score": entry.get("static_weighted_score") or 0,
            "eval/quality_score": entry.get("quality_score") or 0,
        }
        if lr.get("llm_weighted_score") is not None:
            row["eval/llm_score"] = lr["llm_weighted_score"]
        if dr.get("dry_run_weighted_score") is not None:
            row["eval/dry_run_score"] = dr["dry_run_weighted_score"]

        wandb.log(row)

    # ---- 2. Per-LLM-call timeline ----
    for rec in _llm_call_records:
        wandb.log({
            "llm_call/phase":             rec.phase,
            "llm_call/wall_seconds":      round(rec.wall_seconds, 3),
            "llm_call/prompt_tokens":     rec.prompt_tokens     or 0,
            "llm_call/completion_tokens": rec.completion_tokens or 0,
            "llm_call/total_tokens":      rec.total_tokens      or 0,
        })

    # ---- 3. LLM calls table ----  (must be logged before summary)
    wandb.log({
        "llm_calls_table": wandb.Table(
            columns=["phase", "prompt_tokens", "completion_tokens", "total_tokens", "wall_s"],
            data=[
                [r.phase, r.prompt_tokens or 0, r.completion_tokens or 0,
                 r.total_tokens or 0, round(r.wall_seconds, 3)]
                for r in _llm_call_records
            ],
        )
    })

    # ---- 4. Eval report table ----
    table_cols = [
        "id", "type", "category",
        "static_score", "llm_score", "dry_run_score", "quality_score",
        "static_issues", "llm_suggestion", "dry_run_issues",
    ]
    table_data = []
    for entry in report:
        table_data.append([
            entry.get("id", ""),
            entry.get("type", ""),
            entry.get("category", ""),
            entry.get("static_weighted_score") or 0,
            entry.get("llm_weighted_score") or 0,
            entry.get("dry_run_weighted_score") or 0,
            entry.get("quality_score") or 0,
            "; ".join(entry.get("static_issues") or []) or "none",
            entry.get("llm_suggestion") or "none",
            entry.get("dry_run_issues") or "none",
        ])
    wandb.log({"eval_report_table": wandb.Table(columns=table_cols, data=table_data)})

    # ---- 5. Summary scalars ----
    total = len(scenarios)
    summary: dict = {"summary/total_scenarios": total}

    # Static
    mean_static = sum(r.get("static_weighted_score", 0) or 0 for r in static_results) / total
    summary["summary/static/mean_score"] = round(mean_static, 2)

    # LLM judge
    if llm_results:
        judged = [r for r in llm_results if r.get("llm_scores") is not None]
        mean_llm = (
            sum(r.get("llm_weighted_score", 0) or 0 for r in judged) / len(judged)
            if judged else 0
        )
        summary["summary/llm/evaluated_count"] = len(judged)
        summary["summary/llm/mean_score"]       = round(mean_llm, 2)
        for dim in ("clarity", "answerability", "difficulty", "tool_usability", "characteristic_quality"):
            dim_scores = [
                r["llm_scores"].get(dim, 0)
                for r in judged if r.get("llm_scores")
            ]
            summary[f"summary/llm/dim/{dim}_mean"] = round(
                sum(dim_scores) / len(dim_scores), 2
            ) if dim_scores else 0.0

    # Dry-run
    if dry_run_results:
        executed = [r for r in dry_run_results if r.get("dry_run_error") is None]
        mean_dr = (
            sum(r.get("dry_run_weighted_score", 0) or 0 for r in executed) / len(executed)
            if executed else 0
        )
        summary["summary/dry_run/executed_count"] = len(executed)
        summary["summary/dry_run/mean_score"]       = round(mean_dr, 2)
        for dim in ("plausible_response", "used_correct_tools", "no_obvious_errors"):
            dim_scores = [
                r["dry_run_scores"].get(dim, 0)
                for r in executed
                if r.get("dry_run_scores")
            ]
            summary[f"summary/dry_run/dim/{dim}_mean"] = round(
                sum(dim_scores) / len(dim_scores), 3
            ) if dim_scores else 0.0

    # Quality score aggregates
    quality_scores = [e.get("quality_score") or 0 for e in report]
    if quality_scores:
        summary["summary/quality/mean_score"] = round(sum(quality_scores) / len(quality_scores), 2)
        summary["summary/quality/min_score"]  = min(quality_scores)
        summary["summary/quality/max_score"]  = max(quality_scores)

    # LLM call aggregates
    if _llm_call_records:
        summary["summary/llm_calls/total_count"]  = len(_llm_call_records)
        summary["summary/llm_calls/total_tokens"] = sum(r.total_tokens or 0 for r in _llm_call_records)
        phase_groups: dict[str, list[_LLMCallRecord]] = defaultdict(list)
        for rec in _llm_call_records:
            phase_groups[rec.phase].append(rec)
        for ph, recs in phase_groups.items():
            summary[f"summary/llm_calls/{ph}/count"]        = len(recs)
            summary[f"summary/llm_calls/{ph}/total_tokens"] = sum(r.total_tokens or 0 for r in recs)
            summary[f"summary/llm_calls/{ph}/mean_wall_s"]  = round(
                sum(r.wall_seconds for r in recs) / len(recs), 3
            )

    # Type and category distributions
    type_counts = Counter(s.get("type", "unknown") for s in scenarios)
    for t, n in type_counts.items():
        summary[f"summary/type_distribution/{t}"] = n

    cat_counts = Counter(s.get("category", "unknown") for s in scenarios)
    for c, n in cat_counts.items():
        safe_cat = c.replace(" ", "_")
        summary[f"summary/category_distribution/{safe_cat}"] = n

    run.summary.update(summary)
    print(f"W&B summary updated ({len(summary)} keys).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate scenario quality and log results to W&B"
    )
    parser.add_argument("--scenarios", default=str(_DEFAULT_SCENARIOS), metavar="PATH")
    parser.add_argument("--output",    default=str(_DEFAULT_OUTPUT),    metavar="PATH")
    parser.add_argument("--model-id",  default=_DEFAULT_MODEL,          metavar="MODEL_ID")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Only evaluate the first N scenarios")
    # W&B args
    parser.add_argument("--wandb",           action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project",   default="assetopsbench", metavar="PROJECT")
    parser.add_argument("--wandb-entity",    default=None,             metavar="ENTITY")
    parser.add_argument("--wandb-run-name",  default=None,             metavar="NAME")
    parser.add_argument("--wandb-tags",      nargs="*",                metavar="TAG")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    scenarios_path = Path(args.scenarios)
    if not scenarios_path.is_file():
        print(f"error: file not found: {scenarios_path}", file=sys.stderr)
        sys.exit(1)

    scenarios: list[dict] = json.loads(scenarios_path.read_text(encoding="utf-8"))
    if args.limit:
        scenarios = scenarios[: args.limit]
    print(f"Loaded {len(scenarios)} scenario(s) from {scenarios_path}")

    # Optionally initialise W&B run before evaluation starts
    wandb_run = None
    if args.wandb:
        try:
            import wandb  # type: ignore[import]
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                tags=args.wandb_tags or None,
                config={
                    "scenarios_file": str(scenarios_path),
                    "model_id": args.model_id,
                    "num_scenarios": len(scenarios),
                },
            )
            print(f"W&B run started: {wandb_run.url}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: W&B init failed ({exc}) — continuing without logging.")
            wandb_run = None

    print("Querying MCP servers for live tool schemas ...")
    from agent.plan_execute.executor import DEFAULT_SERVER_PATHS
    _load_live_tool_schemas(DEFAULT_SERVER_PATHS)
    print(f"  Loaded schemas for {len(_live_tool_schemas)} tool(s): {sorted(_live_tool_schemas)}")

    print("Running static checks ...")
    static_results = _run_static_checks(scenarios)

    print(f"Running LLM judge ({args.model_id}) ...")
    with _instrument_llm("judge"):
        llm_results = _run_llm_judge(scenarios, model_id=args.model_id)

    print(f"Running dry-run execution + plausibility judge ({args.model_id}) ...")
    with _instrument_llm("dry_run"):
        dry_run_results = _run_dry_run(scenarios, model_id=args.model_id)

    _print_summary(scenarios, static_results, llm_results, dry_run=dry_run_results)

    # Build merged report (same logic as eval_scenarios.py)
    llm_by_id     = {r["id"]: r for r in (llm_results or [])}
    dry_run_by_id = {r["id"]: r for r in (dry_run_results or [])}
    report = []
    for sr in static_results:
        entry = dict(sr)
        lr = llm_by_id.get(sr["id"])
        if lr:
            entry["llm_scores"]         = lr["llm_scores"]
            entry["llm_weighted_score"] = lr["llm_weighted_score"]
            entry["llm_suggestion"]     = lr["llm_suggestion"]
        dr = dry_run_by_id.get(sr["id"])
        if dr:
            entry["dry_run_scores"]              = dr["dry_run_scores"]
            entry["dry_run_weighted_score"]      = dr.get("dry_run_weighted_score")
            entry["dry_run_output"]              = dr["dry_run_output"]
            entry["dry_run_plan_summary"]        = dr["dry_run_plan_summary"]
            entry["dry_run_trajectory_summary"]  = dr["dry_run_trajectory_summary"]
            entry["dry_run_issues"]              = dr["dry_run_issues"]
            entry["dry_run_error"]               = dr["dry_run_error"]

        s_pts  = entry.get("static_weighted_score")  or 0
        l_pts  = entry.get("llm_weighted_score")     or 0
        dr_pts = entry.get("dry_run_weighted_score") or 0
        entry["quality_score"] = round(100 * (s_pts + l_pts + dr_pts) / _SCORE_MAX, 1)
        report.append(entry)

    # Write JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {output_path}")

    # Log to W&B
    if wandb_run is not None:
        print("Logging metrics to W&B ...")
        _log_to_wandb(wandb_run, scenarios, static_results, llm_results, dry_run_results, report)
        wandb_run.finish()
        print("W&B run finished.")


if __name__ == "__main__":
    main()
