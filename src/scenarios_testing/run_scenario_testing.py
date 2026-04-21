"""Run every scenario in scenarios.json through the plan-execute pipeline.

For each scenario the "text" field is used as the question.
Results are written to scenario_testing_output.json as a list of:
    {
        "id":              scenario id (if present),
        "scenario_task":   the text field used as the question,
        "scenario_output": the answer returned by the runner,
        "success":         true/false,
        "error":           null or error message string
    }

Usage:
    uv run python run_scenario_testing.py
    uv run python run_scenario_testing.py --input path/to/scenarios.json
    uv run python run_scenario_testing.py --output path/to/output.json
    uv run python run_scenario_testing.py --model-id watsonx/ibm/granite-3-3-8b-instruct
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_INPUT = Path(__file__).parent / "scenarios.json"
_DEFAULT_OUTPUT = Path(__file__).parent / "scenario_testing_output.json"
_DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
_log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all scenarios from a JSON file through the plan-execute pipeline."
    )
    parser.add_argument(
        "--input",
        default=str(_DEFAULT_INPUT),
        metavar="PATH",
        help=f"Path to the scenarios JSON file (default: {_DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        metavar="PATH",
        help=f"Path to write the results JSON (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--model-id",
        default=_DEFAULT_MODEL,
        metavar="MODEL_ID",
        help=f"LiteLLM model string (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only run the first N scenarios (useful for quick tests).",
    )
    return parser


async def run_all(
    scenarios: list[dict],
    model_id: str,
) -> list[dict]:
    from llm.litellm import LiteLLMBackend
    from agent.plan_execute.runner import PlanExecuteRunner

    llm = LiteLLMBackend(model_id=model_id)
    runner = PlanExecuteRunner(llm=llm)

    results: list[dict] = []
    total = len(scenarios)

    for index, scenario in enumerate(scenarios, start=1):
        scenario_id = scenario.get("id", f"scenario_{index:03d}")
        task = str(scenario.get("text", "")).strip()

        if not task:
            print(f"[{index}/{total}] SKIP  {scenario_id} — empty text field")
            results.append(
                {
                    "id": scenario_id,
                    "scenario_task": task,
                    "scenario_output": None,
                    "success": False,
                    "error": "empty text field",
                }
            )
            continue

        print(f"[{index}/{total}] Running  {scenario_id} ...")
        t0 = time.perf_counter()

        try:
            result = await runner.run(task)
            elapsed = time.perf_counter() - t0
            print(f"[{index}/{total}] OK       {scenario_id}  ({elapsed:.1f}s)")
            results.append(
                {
                    "id": scenario_id,
                    "scenario_task": task,
                    "scenario_output": result.answer,
                    "plan": [
                        {
                            "step": s.step_number,
                            "task": s.task,
                            "server": s.server,
                            "tool": s.tool,
                            "tool_args": s.tool_args,
                            "dependencies": s.dependencies,
                            "expected_output": s.expected_output,
                        }
                        for s in result.plan.steps
                    ],
                    "trajectory": [
                        {
                            "step": r.step_number,
                            "task": r.task,
                            "server": r.server,
                            "tool": r.tool,
                            "tool_args": r.tool_args,
                            "response": r.response,
                            "error": r.error,
                            "success": r.success,
                        }
                        for r in result.trajectory
                    ],
                    "success": True,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            print(f"[{index}/{total}] FAILED   {scenario_id}  ({elapsed:.1f}s) — {exc}")
            results.append(
                {
                    "id": scenario_id,
                    "scenario_task": task,
                    "scenario_output": None,
                    "success": False,
                    "error": str(exc),
                }
            )

    return results


def main() -> None:
    args = _build_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    scenarios: list[dict] = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(scenarios, list):
        print("error: scenarios.json must be a JSON array", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        scenarios = scenarios[: args.limit]

    print(f"Running {len(scenarios)} scenario(s) from {input_path}")
    print(f"Model : {args.model_id}")
    print(f"Output: {output_path}\n")

    results = asyncio.run(run_all(scenarios, model_id=args.model_id))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    passed = sum(1 for r in results if r["success"])
    print(f"\nDone. {passed}/{len(results)} succeeded. Results → {output_path}")


if __name__ == "__main__":
    main()
