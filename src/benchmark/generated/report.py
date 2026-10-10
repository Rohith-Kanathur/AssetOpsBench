"""Small, reproducible README reports from execution and judge records."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from math import isfinite
import os
from pathlib import Path
import re

from .judge import CRITERIA, JUDGE_MODEL, evidence_fingerprint

DIMENSION_LABELS = (
    "Task\ncompletion ↑",
    "Data retrieval\naccuracy ↑",
    "Result\nverification ↑",
    "Agent sequence\n& order ↑",
    "Clarity &\njustification ↑",
    "Hallucination\nrate ↓",
)


def load_case(case_dir: Path) -> dict:
    """Combine saved records for publication; missing grades remain visible."""
    case_dir = Path(case_dir)
    row = json.loads((case_dir / "result.json").read_text())
    path = case_dir / "judge.json"
    grade = json.loads(path.read_text()) if path.exists() else {"status": "pending"}
    if grade.get("fingerprint"):
        scenario = json.loads((case_dir / "scenario.json").read_text())
        try:
            matches = grade["fingerprint"] == evidence_fingerprint(scenario, row, grade["model"], case_dir)
        except (OSError, ValueError):
            matches = False
        if not matches:
            grade = {"status": "pending", "error": "Saved grade belongs to different execution evidence"}
    row["grading"] = grade
    return row


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.0f}%)" if denominator else "—"


def _execution_outcome(case: dict) -> str:
    if case.get("status") == "completed":
        return "completed"
    if case.get("timed_out") is True or re.fullmatch(r"Execution timed out after [\d.]+ seconds", str(case.get("error", ""))):
        return "timed_out"
    return "execution_error" if case.get("status") in {"error", "failed"} else case.get("status", "pending")


def write_report(root: Path, cases: list[dict] | None = None, *,
                 title: str = "Generated scenario results", chart: bool = True) -> Path:
    """Write README.md/cases.csv; chart is optional and requires matplotlib.

    Each case is an execution result with an optional ``grading`` judge record.
    When omitted, cases are discovered below root from result.json files.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if cases is None:
        cases = [load_case(path.parent) for path in sorted((root / "cases").glob("*/result.json"))]
    snapshot = root / "snapshot.json"
    snapshot_data = json.loads(snapshot.read_text()) if snapshot.exists() else {}
    mode = snapshot_data.get("request", {}).get("generation_mode", "")
    rows = []
    groups = defaultdict(list)
    identities = set()
    for case in cases:
        grade = case.get("grading") or {"status": "pending"}
        score = grade.get("score") or {}
        details = score.get("details") or {}
        judged = (case.get("status") == "completed" and grade.get("status") == "completed"
                  and all(type(details.get(key)) is bool for key in CRITERIA))
        row = {key: case.get(key, "") for key in
               ("scenario_id", "domain", "runner", "model", "status", "duration_seconds", "elapsed_seconds",
                "api_calls", "api_prompt_tokens", "api_output_tokens", "cache_read_tokens", "cost_usd", "cost_source")}
        row["execution_outcome"] = _execution_outcome(case)
        row["generation_mode"] = case.get("generation_mode", mode)
        identity = tuple(row[key] for key in ("generation_mode", "runner", "model", "scenario_id"))
        if identity in identities:
            raise ValueError(f"Duplicate execution in report: {identity}")
        identities.add(identity)
        row.update(judge_status=grade.get("status", "pending") if judged or grade.get("status") != "completed" else "invalid",
                   strict_pass=(all(details[key] for key in CRITERIA[:5]) and not details["hallucinations"]) if judged else "",
                   error=case.get("error") or grade.get("error") or "",
                   rationale=score.get("rationale", ""))
        row.update({key: details.get(key, "") if judged else "" for key in CRITERIA})
        rows.append(row)
        groups[(str(row["runner"]), str(row["model"]))].append(row)
    fields = ["scenario_id", "generation_mode", "domain", "runner", "model", "status", "execution_outcome", "elapsed_seconds", "duration_seconds",
              "api_calls", "api_prompt_tokens", "api_output_tokens", "cache_read_tokens", "cost_usd", "cost_source",
              "judge_status", "strict_pass", *CRITERIA, "error", "rationale"]
    with (root / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    scenario_count = len({(row["generation_mode"], row["scenario_id"]) for row in rows})
    lines = [f"# {title}", "", f"{scenario_count} scenarios · {len(cases)} executions planned · Fable 5.1 judge · 1 repetition.", "",
             "| Runner / model | Executed | Judged | Strict pass / planned |",
             "|---|---:|---:|---:|"]
    summaries = []
    for (runner, model), group in groups.items():
        total = len(group)
        judged = [row for row in group if row["judge_status"] == "completed"]
        passed = sum(row["strict_pass"] is True for row in group)
        lines.append(f"| {_cell(runner)} / {_cell(model)} | "
                     f"{sum(row['status'] == 'completed' for row in group)}/{total} | {len(judged)}/{total} | "
                     f"{_fraction(passed, total)} |")
        summaries.append({"label": f"{runner}\n{model}", "judged": len(judged), "total": total,
                          "strict_rates": [passed / total if total else None],
                          "rates": [sum(row[key] is True for row in judged) / len(judged)
                                    if judged else None for key in CRITERIA]})
    lines += ["", "Strict pass requires the first five rubric criteria and no hallucinations. "
              "Success is assessed against each scenario’s characteristic form, including any supported limitation.",
              "", "Pass rates use every planned execution; failed executions and missing grades contribute no passes. "
              "Criterion rates below use judged executions only."]
    if chart and any(item["judged"] for item in summaries):
        if _chart(root, summaries, strict=True):
            lines += ["", "![Strict pass rates](strict_pass.png)"]
        if _chart(root, summaries):
            lines += ["", "![All six evaluation dimensions](criteria.png)"]
    lines += ["", "| Dimension | " + " | ".join(_cell(item["label"]) for item in summaries) + " |",
              "|---|" + "---:|" * len(summaries)]
    for index, label in enumerate(DIMENSION_LABELS):
        rates = [f"{100 * item['rates'][index]:.0f}%" if item["rates"][index] is not None else "—"
                 for item in summaries]
        lines.append(f"| {_cell(label)} | {' | '.join(rates)} |")
    timed_groups = []
    for (runner, model), group in groups.items():
        timed = [row for row in group if isinstance(row["elapsed_seconds"], (int, float))
                 and not isinstance(row["elapsed_seconds"], bool)
                 and isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0]
        completed = [row["elapsed_seconds"] for row in timed if row["execution_outcome"] == "completed"]
        timeouts = [row["elapsed_seconds"] for row in timed if row["execution_outcome"] == "timed_out"]
        timed_groups.append({"label": f"{runner}\n{model}", "completed": completed, "timeouts": timeouts})
    if chart and any(item["completed"] or item["timeouts"] for item in timed_groups) and _time_chart(root, timed_groups):
        lines += ["", "Agent execution time excludes environment setup and judging; crosses mark timeouts.",
                  "", "![Execution time distribution; crosses mark timeouts](execution_time.png)"]
    lines += ["", f"Judge: `{JUDGE_MODEL}` in a separate read-only Claude Code session per execution. "
              "Uses the existing AssetOpsBench six-criterion rubric with access to full saved traces and artifacts.",
              "", "[Per-scenario results](cases.csv)", ""]
    if (root / "scenarios.json").exists():
        lines += ["[Evaluated scenarios and rubrics](scenarios.json)", ""]
    generation = Path(snapshot_data["generation"]) if snapshot_data.get("generation") else None
    if generation is not None and (generation / "scenarios.md").exists():
        relative = Path(os.path.relpath(generation / "scenarios.md", root)).as_posix()
        lines += [f"[Read the scenario requests]({relative})", ""]
    path = root / "README.md"
    path.write_text("\n".join(lines))
    return path


def _chart(root: Path, summaries: list[dict], *, strict: bool = False) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    colors = ["#4785ad", "#cab5d6", "#a28bcc", "#348fb0", "#929292"]
    labels = ("Strict pass ↑",) if strict else DIMENSION_LABELS
    figure, axes = plt.subplots(figsize=(8 if strict else 11, 3.8))
    width = min(.18, .75 / len(summaries))
    for index, item in enumerate(summaries):
        if not strict and not item["judged"]:
            continue
        rates = item["strict_rates"] if strict else item["rates"]
        positions = [x + width * (index - (len(summaries) - 1) / 2)
                     for x, rate in enumerate(rates) if rate is not None]
        bars = axes.bar(positions, [100 * rate for rate in rates if rate is not None], width,
                        color=colors[index % len(colors)], edgecolor="#555555", linewidth=.6,
                        label=item["label"].replace("\n", " / "))
        axes.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=7, rotation=90)
    axes.set_xticks(range(len(labels)), labels)
    axes.set_ylim(0, 113)
    axes.set_yticks([])
    axes.spines[:].set_visible(False)
    axes.tick_params(labelsize=9, length=0)
    axes.legend(loc="upper center", bbox_to_anchor=(.5, 1.25), frameon=False,
                fontsize=8, title="Models", title_fontsize=8, ncol=min(3, len(summaries)))
    figure.tight_layout()
    figure.savefig(root / ("strict_pass.png" if strict else "criteria.png"), dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def _time_chart(root: Path, groups: list[dict]) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    colors = ["#4785ad", "#cab5d6", "#a28bcc", "#348fb0", "#929292"]
    figure, axes = plt.subplots(figsize=(11, 3.8))
    for index, item in enumerate(groups):
        values = [seconds / 60 for seconds in item["completed"]]
        color = colors[index % len(colors)]
        if values:
            axes.boxplot([values], positions=[index], widths=.45, patch_artist=True,
                         showfliers=False, boxprops={"facecolor": color, "alpha": .4},
                         medianprops={"color": "#333333"})
            positions = [index + .22 * (i / max(1, len(values) - 1) - .5) for i in range(len(values))]
            axes.scatter(positions, values, s=16, color=color, edgecolors="#555555", linewidths=.4, zorder=3)
        if item["timeouts"]:
            positions = [index + .12 * (i - (len(item["timeouts"]) - 1) / 2) for i in range(len(item["timeouts"]))]
            axes.scatter(positions, [seconds / 60 for seconds in item["timeouts"]],
                         marker="x", color="#b34b3b", s=50, label="Timed out" if not axes.get_legend_handles_labels()[0] else None, zorder=4)
    axes.set_xticks(range(len(groups)), [item["label"] for item in groups], fontsize=8)
    axes.set_ylabel("Agent execution time (minutes)", fontsize=9)
    axes.set_ylim(bottom=0)
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", alpha=.15)
    if axes.get_legend_handles_labels()[0]:
        axes.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "execution_time.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True
