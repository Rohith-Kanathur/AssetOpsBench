"""CLI entry for `python -m scenarios.generator`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv

from agent.cli import _DEFAULT_MODEL

from .agent import ScenarioGeneratorAgent
from .prompt_helpers import default_scenario_output_path


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Scenario Generator for AssetOpsBench")
    parser.add_argument("asset_name", help="Asset class name (for example 'Chiller' or 'Wind Turbine')")
    parser.add_argument("--model-id", default=_DEFAULT_MODEL, help="Model ID for LiteLLM")
    parser.add_argument("--num-scenarios", type=int, default=50, help="Total number of scenarios to generate")
    parser.add_argument("--show-workflow", action="store_true", help="Show intermediate pipeline steps in the console")
    parser.add_argument("--log", action="store_true", help="Dump raw prompts and results to a log directory")
    parser.add_argument(
        "--data-in-couchdb",
        action="store_true",
        help="Use grounded open-form generation when matching live CouchDB-backed asset data is available",
    )

    args = parser.parse_args()
    output_path = default_scenario_output_path(args.asset_name)

    # With --show-workflow alone, keep the root logger quiet (WARNING). When
    # --data-in-couchdb is also set, Phase 1 runs FMSR mapping (per-pair LLM work
    # inside the server); use INFO so `scenarios.grounding` can log progress.
    if args.show_workflow and not args.data_in_couchdb:
        level = logging.WARNING
    else:
        level = logging.INFO
    # Imported MCP server modules (e.g. servers.fmsr.main) call basicConfig(WARNING)
    # at import time; without force=True, this CLI's basicConfig would be a no-op.
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    if level == logging.INFO:
        for noisy in ("httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    log_dir = None
    if args.log:
        log_dir = str(output_path.parent / "logs")
        print(f"Logging session to: {log_dir}")
        os.makedirs(log_dir, exist_ok=True)

    agent = ScenarioGeneratorAgent(model_id=args.model_id, show_workflow=args.show_workflow, log_dir=log_dir)

    try:
        final_scenarios = asyncio.run(
            agent.run(
                args.asset_name,
                num_scenarios=args.num_scenarios,
                data_in_couchdb=args.data_in_couchdb,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n[FATAL ERROR] {exc}")
        sys.exit(1)

    if not final_scenarios:
        print("\n[WARNING] No scenarios were successfully generated and validated.")
        sys.exit(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump([scenario.to_dict() for scenario in final_scenarios], handle, indent=2)

    if not args.show_workflow:
        print(f"Success! Generated {len(final_scenarios)} scenarios at {output_path}")
    else:
        print(f"Scenarios saved to {output_path}")
