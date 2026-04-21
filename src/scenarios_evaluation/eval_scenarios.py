"""
Scenario quality evaluator for scenarios.json

Runs the below three evaluation modes on every scenario:
  1. STATIC    — schema, category alignment, length, diversity, duplicates
  2. LLM JUDGE — clarity, answerability, difficulty, tool_usability, characteristic_quality
  3. DRY-RUN   — execute via PlanExecuteRunner, judge plausibility

Usage:
    uv run python src/scenarios_evaluation/eval_scenarios.py
    uv run python src/scenarios_evaluation/eval_scenarios.py --limit 10
    uv run python src/scenarios_evaluation/eval_scenarios.py --output my_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_DIR = Path(__file__).parent
_DEFAULT_SCENARIOS = _DIR / "scenarios.json"
_DEFAULT_OUTPUT = _DIR / "eval_scenarios_report.json"
_DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"


# Populated at startup by _load_live_tool_schemas().
# Maps tool_name -> {"required": set[str], "all_params": set[str]}.
# requires is the subset of all_params that are required (not optional) for a tool call.
_live_tool_schemas: dict[str, dict[str, set[str]]] = {}

# Known valid categories for scenario classification.
_CATEGORY_GROUPS: set[str] = {
    "data query",
    "knowledge query",
    "condition analysis",
    "health assessment",
    "data query and analysis",
    "diagnostic assessment",
    "prediction",
    "predictive maintenance",
    "recommendation",
    "decision support",
    "condition assessment",
}

# Fields that every scenario object must have (schema completeness check)
_REQUIRED_FIELDS = ["id", "text", "type", "category", "characteristic_form"]

# Heuristic text-length bounds for the scenario task description
_MIN_TEXT_LEN = 15  # below this → too vague to be useful
_MAX_TEXT_LEN = 400  # above this → likely too complex or malformed


# -- Prompts -----------------------------------------------------------------

_LLM_JUDGE_PROMPT = """\
You are an expert evaluator for an industrial asset management AI benchmark.

A "scenario" is a natural-language task that an AI agent must answer using specific tools.
You will be given one scenario. Score it on five dimensions (true = passes):

1. clarity:               Is the task clearly stated? Can a domain expert immediately understand what is being asked?
2. answerability:         Can this task realistically be answered at all?
3. difficulty:            Is the difficulty appropriate? (false if trivially simple with no reasoning, or impossibly complex)
4. tool_usability:        Given the EXACT tool signatures below, is it actually possible to answer this task
                          using those tools? Consider whether the required parameters can be supplied
                          and whether the tools return the kind of data the task needs.
5. characteristic_quality: Is the grading rubric (characteristic_form) specific and measurable enough to fairly
                          grade an agent's response? (false if it is vague, incomplete, or untestable)

Also write a one-sentence suggestion (or "None" if no issues).

Reply with ONLY valid JSON — no prose, no markdown fences:
{{
  "clarity": true or false,
  "answerability": true or false,
  "difficulty": true or false,
  "tool_usability": true or false,
  "characteristic_quality": true or false,
  "suggestion": "..."
}}

---
SCENARIO TYPE: {stype}
SCENARIO CATEGORY: {category}
TASK TEXT: {text}
CHARACTERISTIC (grading rubric): {characteristic}

AVAILABLE TOOLS FOR THIS TYPE:
{tool_schemas}
"""

_DRY_RUN_JUDGE_PROMPT = """\
You are an expert evaluator for an industrial asset management AI benchmark.

A "scenario" is a natural-language task that an AI agent answers using specific MCP tools.
You have been given:
1. The task the agent was asked to perform
2. The plan the agent produced (list of tool-call steps)
3. The trajectory (actual tool calls + responses)
4. The final answer the agent gave

There is NO gold answer.  Judge only whether the execution looks plausible and coherent given the task:

1. plausible_response  (true/false): The final answer addresses the task (not an error message, not completely unrelated, not empty)
2. used_correct_tools  (0.0-1.0):   Fraction of trajectory steps that used appropriate real tools for this task type.
                                    A step is WRONG if: server is "none", tool is "none", or the step result is ERROR(…).
                                    Example: 3 of 4 steps used correct tools gets a score of 0.75
3. no_obvious_errors   (0.0-1.0):   Fraction of trajectory steps that completed without any error.
                                    A step counts as an error if: server/tool is "none" OR the result contains ERROR(…).
                                    Example: 1 error out of 4 steps gets a score of 0.75

Reply with ONLY valid JSON — no prose, no markdown fences:
{{
  "plausible_response": true or false,
  "used_correct_tools": 0.0 to 1.0,
  "no_obvious_errors": 0.0 to 1.0,
  "issues": "list every problem found (including any none-server steps), or 'None' only if the trajectory is completely clean"
}}

---
SCENARIO TYPE: {stype}
TASK: {text}
CHARACTERISTIC (grading rubric): {characteristic}

PLAN (intended steps):
{plan}

TRAJECTORY (actual tool calls):
{trajectory}

FINAL ANSWER:
{answer}
"""


# -- Scoring weights --------------------------------------------------------
#
# Each check is worth a different number of points reflecting its importance.
# Totals: static=20, llm=30, dry_run=50 gives quality_score directly out of 100
#
# Rationale:
#   static  (20) — mechanical format checks; easy to pass with a broken scenario, but still important to weed out garbage
#   llm     (30) — semantic quality; harder to fake
#   dry_run (50) — end-to-end execution; strongest real-world signal

_STATIC_WEIGHTS: dict[str, int] = {
    "schema": 6,  # Missing fields means the scenario is fundamentally broken
    "type_diversity": 4,  # A scenario type should not dominate the dataset
    "category_alignment": 4,  # Wrong category corrupts benchmark grouping
    "text_length": 2,  # Quality signal but not a hard blocker
    "no_duplicate_id": 2,  # Operational hygiene
    "no_duplicate_text": 2,  # Duplicate scenario adds no benchmark value
}
_STATIC_MAX = sum(_STATIC_WEIGHTS.values())  # 20

_LLM_WEIGHTS: dict[str, int] = {
    "answerability": 10,  # Unanswerable scenarios are useless
    "tool_usability": 8,  # Solvable only if the right tools exist
    "difficulty": 6,  # Trivial scenarios inflate agent scores in a benchmark
    "characteristic_quality": 4,  # Vague rubric makes downstream benchmark evaluation unfair
    "clarity": 2,  # Clarity failures are usually caught by the dry-run too
}
_LLM_MAX = sum(_LLM_WEIGHTS.values())  # 30

_DRY_RUN_WEIGHTS: dict[str, int] = {
    "plausible_response": 20,  # Did the agent produce any meaningful answer
    "used_correct_tools": 15,  # Wrong tools + plausible response likely means hallucination
    "no_obvious_errors": 15,  # Mostly-failing trajectories means broken scenario
}
_DRY_RUN_MAX = sum(_DRY_RUN_WEIGHTS.values())  # 50

_SCORE_MAX = _STATIC_MAX + _LLM_MAX + _DRY_RUN_MAX  # 100 → quality_score is direct


def _weighted_score(scores: dict[str, int], weights: dict[str, int]) -> int:
    """Sum all numeric score values (each is already 0 or its weight)."""
    return sum(scores.values())


async def _fetch_server_schemas(
    server_paths: dict[str, object],
) -> dict[str, dict[str, set[str]]]:
    """Query every MCP server and return tool_name -> {required, all_params}.

    asset_name is excluded from every parameter set (universal context arg).
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agent.plan_execute.executor import _list_tools  # noqa: PLC0415

    schemas: dict[str, dict[str, set[str]]] = {}
    for server_name, server_path in server_paths.items():
        print(f"  Querying '{server_name}' ...", end=" ", flush=True)
        try:
            tools = await _list_tools(server_path)
            for tool in tools:
                all_params = {
                    p["name"] for p in tool["parameters"] if p["name"] != "asset_name"
                }
                required = {
                    p["name"]
                    for p in tool["parameters"]
                    if p["required"] and p["name"] != "asset_name"
                }
                schemas[tool["name"]] = {
                    "server": server_name,
                    "required": required,
                    "all_params": all_params,
                }
            print(f"OK ({len(tools)} tool(s))")
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP ({exc})")
    return schemas


def _load_live_tool_schemas(server_paths: dict[str, object]) -> None:
    """Sync wrapper around _fetch_server_schemas; no-op if already loaded."""
    global _live_tool_schemas  # noqa: PLW0603
    if _live_tool_schemas:
        return
    _live_tool_schemas = asyncio.run(_fetch_server_schemas(server_paths))


def _tools_for_type(stype: str) -> set[str]:
    """Return the set of tool names belonging to the server matching `stype`."""
    return {name for name, s in _live_tool_schemas.items() if s["server"] == stype}


# -- Static checks -----------------------------------------------------------


def _check_schema(scenario: dict) -> list[str]:
    """Return issues for any required field that is missing or empty."""
    issues = []
    for field in _REQUIRED_FIELDS:
        val = scenario.get(field)
        if val is None:
            issues.append(f"missing field '{field}'")
        elif not str(val).strip():
            issues.append(f"empty field '{field}'")
    return issues


def _check_text_length(scenario: dict) -> list[str]:
    """Flag text that is too short (< 15 chars) or too long (> 400 chars)."""
    text = scenario.get("text", "")
    issues = []
    if len(text) < _MIN_TEXT_LEN:
        issues.append(f"text too short ({len(text)} chars < {_MIN_TEXT_LEN})")
    if len(text) > _MAX_TEXT_LEN:
        issues.append(f"text very long ({len(text)} chars > {_MAX_TEXT_LEN})")
    return issues


def _check_category_alignment(scenario: dict) -> list[str]:
    """Verify the category maps to a recognised problem group."""
    category = (scenario.get("category") or "").strip().lower()
    if category not in _CATEGORY_GROUPS:
        return [
            f"unrecognised category '{scenario.get('category')}' "
            f"(known: {sorted(_CATEGORY_GROUPS)})"
        ]
    return []


# Maximum share any single type may hold before the dataset is considered
# poorly diversified (e.g. 0.5 means one type should not exceed 50% of all scenarios).
_TYPE_DIVERSITY_THRESHOLD = 0.5


def _compute_type_diversity_flags(scenarios: list[dict]) -> dict[str, bool]:
    """Return a per-scenario flag: True if this scenario's type does not dominate the dataset."""
    total = len(scenarios)
    if total == 0:
        return {}
    type_counts = Counter(s.get("type", "").lower() for s in scenarios)
    dominant_types = {
        t for t, n in type_counts.items() if n / total > _TYPE_DIVERSITY_THRESHOLD
    }
    return {
        (s.get("id") or f"<{i}>"): (s.get("type", "").lower() not in dominant_types)
        for i, s in enumerate(scenarios)
    }


def _run_static_checks(scenarios: list[dict]) -> list[dict]:
    """Run all static checks; return one result dict per scenario."""
    seen_ids: dict[str, int] = {}
    seen_texts: dict[str, int] = {}
    diversity_flags = _compute_type_diversity_flags(scenarios)
    results = []

    for idx, scenario in enumerate(scenarios):
        sid = scenario.get("id", f"<missing_id_{idx}>")

        schema_issues = _check_schema(scenario)
        length_issues = _check_text_length(scenario)
        category_issues = _check_category_alignment(scenario)

        duplicate_id_issues: list[str] = []
        if sid in seen_ids:
            duplicate_id_issues.append(f"duplicate id (also at index {seen_ids[sid]})")
        else:
            seen_ids[sid] = idx

        duplicate_text_issues: list[str] = []
        text = scenario.get("text", "")
        if text and text in seen_texts:
            duplicate_text_issues.append(
                f"duplicate text (also at index {seen_texts[text]})"
            )
        elif text:
            seen_texts[text] = idx

        type_diversity_pass = diversity_flags.get(sid, True)
        type_diversity_issues: list[str] = (
            []
            if type_diversity_pass
            else [
                f"type '{scenario.get('type')}' exceeds {int(_TYPE_DIVERSITY_THRESHOLD*100)}% of dataset"
            ]
        )

        issues = (
            schema_issues
            + length_issues
            + category_issues
            + duplicate_id_issues
            + duplicate_text_issues
            + type_diversity_issues
        )

        static_scores = {
            "schema": _STATIC_WEIGHTS["schema"] if len(schema_issues) == 0 else 0,
            "text_length": _STATIC_WEIGHTS["text_length"] if len(length_issues) == 0 else 0,
            "category_alignment": _STATIC_WEIGHTS["category_alignment"] if len(category_issues) == 0 else 0,
            "type_diversity": _STATIC_WEIGHTS["type_diversity"] if type_diversity_pass else 0,
            "no_duplicate_id": _STATIC_WEIGHTS["no_duplicate_id"] if len(duplicate_id_issues) == 0 else 0,
            "no_duplicate_text": _STATIC_WEIGHTS["no_duplicate_text"] if len(duplicate_text_issues) == 0 else 0,
        }
        results.append(
            {
                "id": sid,
                "scenario_text": scenario.get("text", ""),
                "characteristic_form": scenario.get("characteristic_form", ""),
                "type": scenario.get("type", ""),
                "category": scenario.get("category", ""),
                "static_scores": static_scores,
                "static_weighted_score": _weighted_score(
                    static_scores, _STATIC_WEIGHTS
                ),
                "static_issues": issues,
            }
        )

    return results


# -- LLM judge ---------------------------------------------------------------


def _run_llm_judge(scenarios: list[dict], model_id: str) -> list[dict]:
    """Score each scenario on clarity, answerability, and difficulty."""
    from llm.litellm import LiteLLMBackend

    llm = LiteLLMBackend(model_id=model_id)
    results = []
    total = len(scenarios)

    for idx, scenario in enumerate(scenarios, start=1):
        sid = scenario.get("id", f"<{idx}>")
        stype = (scenario.get("type") or "").lower()

        # Build a schema summary: tool_name(param1, *required_param2, ...)
        type_tools = _tools_for_type(stype)
        schema_lines = []
        for tool_name in sorted(type_tools):
            s = _live_tool_schemas.get(tool_name, {})
            params = sorted(s.get("all_params", []))
            required = s.get("required", set())
            param_str = ", ".join(f"*{p}" if p in required else p for p in params)
            schema_lines.append(f"  {tool_name}({param_str})")
        tool_schemas = (
            "\n".join(schema_lines)
            or "  (no tool schemas loaded — use --no-servers was set)"
        )

        prompt = _LLM_JUDGE_PROMPT.format(
            stype=stype,
            category=scenario.get("category", ""),
            text=scenario.get("text", ""),
            characteristic=scenario.get("characteristic_form", ""),
            tool_schemas=tool_schemas,
        )

        print(f"  [{idx}/{total}] LLM judging {sid} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            response = llm.generate(prompt, temperature=0.0)
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    l for l in raw.splitlines() if not l.startswith("```")
                ).strip()
            parsed = json.loads(raw)
            elapsed = time.perf_counter() - t0
            llm_pass = (
                parsed.get("clarity")
                and parsed.get("answerability")
                and parsed.get("difficulty")
                and parsed.get("tool_usability")
                and parsed.get("characteristic_quality")
            )
            print(f"{'PASS' if llm_pass else 'FAIL'} ({elapsed:.1f}s)")
            llm_scores = {
                "clarity": _LLM_WEIGHTS["clarity"] if parsed.get("clarity") else 0,
                "answerability": _LLM_WEIGHTS["answerability"] if parsed.get("answerability") else 0,
                "difficulty": _LLM_WEIGHTS["difficulty"] if parsed.get("difficulty") else 0,
                "tool_usability": _LLM_WEIGHTS["tool_usability"] if parsed.get("tool_usability") else 0,
                "characteristic_quality": _LLM_WEIGHTS["characteristic_quality"] if parsed.get("characteristic_quality") else 0,
            }
            results.append(
                {
                    "id": sid,
                    "llm_scores": llm_scores,
                    "llm_weighted_score": _weighted_score(llm_scores, _LLM_WEIGHTS),
                    "llm_suggestion": str(parsed.get("suggestion", "")).strip()
                    or "None",
                }
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            print(f"ERROR ({elapsed:.1f}s) — {exc}")
            results.append(
                {
                    "id": sid,
                    "llm_scores": None,
                    "llm_weighted_score": None,
                    "llm_suggestion": None,
                }
            )

    return results


# -- Dry-run executor + plausibility judge -----------------------------------


async def _run_single_dry_run(
    scenario: dict,
    runner: object,
    llm: object,
    idx: int,
    total: int,
) -> dict:
    """Execute one scenario and judge whether the output looks plausible."""
    sid = scenario.get("id", f"<{idx}>")
    task = str(scenario.get("text", "")).strip()
    stype = (scenario.get("type") or "").lower()
    characteristic = scenario.get("characteristic_form", "")

    base: dict = {
        "id": sid,
        "dry_run_output": None,
        "dry_run_plan_summary": None,
        "dry_run_trajectory_summary": None,
        "dry_run_scores": None,
        "dry_run_issues": None,
        "dry_run_error": None,
    }

    if not task:
        base["dry_run_error"] = "empty text field"
        return base

    print(f"  [{idx}/{total}] Executing {sid} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        result = await runner.run(task)
        exec_elapsed = time.perf_counter() - t0
        print(f"done ({exec_elapsed:.1f}s)", end=" ", flush=True)
    except Exception as exc:  # noqa: BLE001
        exec_elapsed = time.perf_counter() - t0
        print(f"EXEC-ERROR ({exec_elapsed:.1f}s) — {exc}")
        base["dry_run_error"] = f"execution failed: {exc}"
        return base

    plan_text = (
        "\n".join(
            f"  step {s.step_number}: [{s.server}.{s.tool}] {s.task}"
            for s in result.plan.steps
        )
        or "(no plan)"
    )
    traj_text = (
        "\n".join(
            f"  step {r.step_number}: [{r.server}.{r.tool}]"
            f"  args={json.dumps(r.tool_args or {})}  → {'OK' if r.success else f'ERROR({r.error})'}"
            for r in result.trajectory
        )
        or "(no trajectory)"
    )
    answer_text = str(result.answer or "").strip() or "(no answer)"

    base["dry_run_output"] = answer_text
    base["dry_run_plan_summary"] = plan_text
    base["dry_run_trajectory_summary"] = traj_text

    prompt = _DRY_RUN_JUDGE_PROMPT.format(
        stype=stype,
        text=task,
        characteristic=characteristic,
        plan=plan_text,
        trajectory=traj_text,
        answer=answer_text,
    )
    try:
        response = llm.generate(prompt, temperature=0.0)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = "\n".join(
                line for line in raw.splitlines() if not line.startswith("```")
            ).strip()
        parsed = json.loads(raw)
        judge_elapsed = time.perf_counter() - t0 - exec_elapsed

        correct_ratio = max(0.0, min(1.0, float(parsed.get("used_correct_tools", 0))))
        errors_ratio = max(0.0, min(1.0, float(parsed.get("no_obvious_errors", 0))))
        scores = {
            "plausible_response": _DRY_RUN_WEIGHTS["plausible_response"] if parsed.get("plausible_response") else 0,
            "used_correct_tools": round(_DRY_RUN_WEIGHTS["used_correct_tools"] * correct_ratio, 1),
            "no_obvious_errors": round(_DRY_RUN_WEIGHTS["no_obvious_errors"] * errors_ratio, 1),
        }
        dry_run_pass = all(v > 0 for v in scores.values())
        print(f"→ {'PASS' if dry_run_pass else 'FAIL'} (judge {judge_elapsed:.1f}s)")

        base["dry_run_scores"] = scores
        base["dry_run_weighted_score"] = round(sum(scores.values()), 1)
        base["dry_run_issues"] = str(parsed.get("issues", "")).strip() or "None"
    except Exception as exc:  # noqa: BLE001
        print(f"→ JUDGE-ERROR — {exc}")
        base["dry_run_error"] = f"judge failed: {exc}"

    return base


def _run_dry_run(scenarios: list[dict], model_id: str) -> list[dict]:
    """Execute every scenario and judge plausibility; return dry_run_* result dicts."""
    from agent.plan_execute.runner import PlanExecuteRunner
    from llm.litellm import LiteLLMBackend

    judge_llm = LiteLLMBackend(model_id=model_id)
    runner_llm = LiteLLMBackend(model_id=model_id)
    runner = PlanExecuteRunner(llm=runner_llm)

    total = len(scenarios)

    async def _run_all() -> list[dict]:
        results = []
        for idx, scenario in enumerate(scenarios, start=1):
            result = await _run_single_dry_run(scenario, runner, judge_llm, idx, total)
            results.append(result)
        return results

    return asyncio.run(_run_all())


# -- Reporting ---------------------------------------------------------------


def _print_summary(
    scenarios: list[dict],
    static: list[dict],
    llm: list[dict] | None,
    dry_run: list[dict] | None = None,
) -> None:
    """Print a human-readable quality summary to stdout."""
    total = len(scenarios)
    static_pass = sum(1 for r in static if all(v > 0 for v in r["static_scores"].values()))
    print(f"\n{'='*65}")
    print(f"SCENARIO QUALITY REPORT  ({total} scenarios)")
    print(f"{'='*65}")
    avg_static_w = sum(r.get("static_weighted_score", 0) for r in static) / total
    print(f"\n[STATIC CHECKS]")
    print(f"  Pass: {static_pass}/{total}  ({100*static_pass/total:.1f}%)")
    print(
        f"  Avg weighted score: {avg_static_w:.1f}/{_STATIC_MAX}  "
        f"(weights: schema={_STATIC_WEIGHTS['schema']}, "
        f"category={_STATIC_WEIGHTS['category_alignment']}, "
        f"length={_STATIC_WEIGHTS['text_length']}, "
        f"type_diversity={_STATIC_WEIGHTS['type_diversity']}, "
        f"dup_id={_STATIC_WEIGHTS['no_duplicate_id']})"
    )

    issue_counts: Counter = Counter()
    for r in static:
        for issue in r["static_issues"]:
            issue_counts[issue.split("(")[0].strip()] += 1
    if issue_counts:
        print("\n  Top issues:")
        for issue, count in issue_counts.most_common(10):
            print(f"    {count:3d}x  {issue}")

    type_counts = Counter(s.get("type", "unknown") for s in scenarios)
    print(f"\n  Type distribution:")
    for t, n in sorted(type_counts.items()):
        print(f"    {t:<15} {n}")

    # Category distribution
    cat_counts = Counter(s.get("category", "unknown") for s in scenarios)
    print(f"\n  Category distribution:")
    for c, n in sorted(cat_counts.items()):
        print(f"    {c:<35} {n}")

    if llm:
        judged = [r for r in llm if r["llm_scores"] is not None]
        llm_pass = sum(1 for r in judged if r["llm_scores"] and all(v > 0 for v in r["llm_scores"].values()))
        avg_llm_w = (
            sum(r.get("llm_weighted_score", 0) for r in judged) / len(judged)
            if judged
            else 0
        )
        print(f"\n[LLM JUDGE]  ({len(judged)} evaluated)")
        print(
            f"  Pass: {llm_pass}/{len(judged)}  ({100*llm_pass/len(judged):.1f}% if judged else 0)"
        )
        print(
            f"  Avg weighted score: {avg_llm_w:.1f}/{_LLM_MAX}  "
            f"(weights: answerability={_LLM_WEIGHTS['answerability']}, "
            f"tool_usability={_LLM_WEIGHTS['tool_usability']}, "
            f"difficulty={_LLM_WEIGHTS['difficulty']}, "
            f"characteristic_quality={_LLM_WEIGHTS['characteristic_quality']}, "
            f"clarity={_LLM_WEIGHTS['clarity']})"
        )
        for dim in (
            "clarity",
            "answerability",
            "difficulty",
            "tool_usability",
            "characteristic_quality",
        ):
            w = _LLM_WEIGHTS[dim]
            count = sum(
                1 for r in judged if r["llm_scores"] and r["llm_scores"].get(dim, 0) > 0
            )
            print(f"  {dim:<20} {count}/{len(judged)}  (weight {w})")

    if dry_run is not None:
        executed = [r for r in dry_run if r["dry_run_error"] is None]
        dr_pass = sum(1 for r in executed if r["dry_run_scores"] and all(v > 0 for v in r["dry_run_scores"].values()))
        avg_dr_w = (
            sum(r.get("dry_run_weighted_score", 0) for r in executed) / len(executed)
            if executed
            else 0
        )
        print(f"\n[DRY-RUN]  ({len(dry_run)} attempted, {len(executed)} completed)")
        print(
            f"  Plausible: {dr_pass}/{len(executed)}"
            f"  ({100*dr_pass/len(executed):.1f}%)"
            if executed
            else "  Plausible: 0/0"
        )
        print(
            f"  Avg weighted score: {avg_dr_w:.1f}/{_DRY_RUN_MAX}  "
            f"(weights: plausible={_DRY_RUN_WEIGHTS['plausible_response']}, "
            f"no_errors={_DRY_RUN_WEIGHTS['no_obvious_errors']}, "
            f"correct_tools={_DRY_RUN_WEIGHTS['used_correct_tools']})"
        )
        for dim in ("plausible_response", "used_correct_tools", "no_obvious_errors"):
            w = _DRY_RUN_WEIGHTS[dim]
            count = sum(
                1
                for r in executed
                if r["dry_run_scores"] and r["dry_run_scores"].get(dim, 0) > 0
            )
            print(f"  {dim:<25} {count}/{len(executed)}  (weight {w})")
        exec_errors = [r for r in dry_run if r["dry_run_error"]]
        if exec_errors:
            print(f"  Execution errors: {len(exec_errors)}")
            for r in exec_errors[:5]:
                print(f"    {r['id']}: {r['dry_run_error']}")

    # Overall quality score summary (computed post-merge; approximate here)
    all_w = []
    for sr, lr, dr in zip(
        static,
        llm or [{}] * total,
        dry_run or [{}] * total,
    ):
        s_pts = sr.get("static_weighted_score", 0) or 0
        l_pts = (lr.get("llm_weighted_score") or 0) if lr else 0
        dr_pts = (dr.get("dry_run_weighted_score") or 0) if dr else 0
        all_w.append(round(100 * (s_pts + l_pts + dr_pts) / _SCORE_MAX, 1))
    if all_w:
        static_pts = [sr.get("static_weighted_score", 0) or 0 for sr in static]
        llm_pts = [(lr.get("llm_weighted_score") or 0) for lr in (llm or [{}] * total)]
        dr_pts = [(dr.get("dry_run_weighted_score") or 0) for dr in (dry_run or [{}] * total)]

        mean_static = sum(static_pts) / len(static_pts) if static_pts else 0
        mean_llm = sum(llm_pts) / len(llm_pts) if llm_pts else 0
        mean_dr = sum(dr_pts) / len(dr_pts) if dr_pts else 0
        mean_quality = sum(all_w) / len(all_w)

        print(f"\n[QUALITY SCORE]  (weighted, 0-100)")
        print(f"  Mean static score:   {mean_static:.1f} / {_STATIC_MAX}")
        print(f"  Mean LLM score:      {mean_llm:.1f} / {_LLM_MAX}")
        print(f"  Mean dry-run score:  {mean_dr:.1f} / {_DRY_RUN_MAX}")
        print(f"  {'─'*38}")
        print(f"  Mean quality score:  {mean_quality:.1f} / {_SCORE_MAX}   "
              f"(= {mean_static:.1f} + {mean_llm:.1f} + {mean_dr:.1f})")
        print(f"  Min: {min(all_w):.1f}   Max: {max(all_w):.1f}")

    print(f"\n{'='*65}\n")


# -- Parse args and main entry point ------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Parse command-line args."""
    parser = argparse.ArgumentParser(
        description="Evaluate quality of generated scenarios"
    )
    parser.add_argument("--scenarios", default=str(_DEFAULT_SCENARIOS), metavar="PATH")
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT), metavar="PATH")
    parser.add_argument("--model-id", default=_DEFAULT_MODEL, metavar="MODEL_ID")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only evaluate the first N scenarios",
    )
    return parser


def main() -> None:
    """Main entry point for scenario evaluation."""
    args = _build_parser().parse_args()
    scenarios_path = Path(args.scenarios)
    if not scenarios_path.is_file():
        print(f"error: file not found: {scenarios_path}", file=sys.stderr)
        sys.exit(1)

    scenarios: list[dict] = json.loads(scenarios_path.read_text(encoding="utf-8"))
    if args.limit:
        scenarios = scenarios[: args.limit]
    print(f"Loaded {len(scenarios)} scenario(s) from {scenarios_path}")

    print("Querying MCP servers for live tool schemas ...")
    from agent.plan_execute.executor import DEFAULT_SERVER_PATHS

    _load_live_tool_schemas(DEFAULT_SERVER_PATHS)
    print(
        f"  Loaded schemas for {len(_live_tool_schemas)} tool(s): {sorted(_live_tool_schemas)}"
    )

    print("Running static checks ...")
    static_results = _run_static_checks(scenarios)

    print(f"Running LLM judge ({args.model_id}) ...")
    llm_results = _run_llm_judge(scenarios, model_id=args.model_id)

    print(f"Running dry-run execution + plausibility judge ({args.model_id}) ...")
    dry_run_results = _run_dry_run(scenarios, model_id=args.model_id)

    _print_summary(scenarios, static_results, llm_results, dry_run=dry_run_results)

    llm_by_id = {r["id"]: r for r in (llm_results or [])}
    dry_run_by_id = {r["id"]: r for r in (dry_run_results or [])}
    report = []
    for sr in static_results:
        entry = dict(sr)
        lr = llm_by_id.get(sr["id"])
        if lr:
            entry["llm_scores"] = lr["llm_scores"]
            entry["llm_weighted_score"] = lr["llm_weighted_score"]
            entry["llm_suggestion"] = lr["llm_suggestion"]
        dr = dry_run_by_id.get(sr["id"])
        if dr:
            entry["dry_run_scores"] = dr["dry_run_scores"]
            entry["dry_run_weighted_score"] = dr.get("dry_run_weighted_score")
            entry["dry_run_output"] = dr["dry_run_output"]
            entry["dry_run_plan_summary"] = dr["dry_run_plan_summary"]
            entry["dry_run_trajectory_summary"] = dr["dry_run_trajectory_summary"]
            entry["dry_run_issues"] = dr["dry_run_issues"]
            entry["dry_run_error"] = dr["dry_run_error"]

        # Roll up all weighted points into a single 0-100 quality score.
        # Missing modes (error/None) contribute 0 to their portion.
        s_pts = entry.get("static_weighted_score") or 0
        l_pts = entry.get("llm_weighted_score") or 0
        dr_pts = entry.get("dry_run_weighted_score") or 0
        entry["quality_score"] = round(100 * (s_pts + l_pts + dr_pts) / _SCORE_MAX, 1)
        report.append(entry)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
