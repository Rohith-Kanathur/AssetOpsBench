"""CLI entry for ``python -m scenarios_optimization.generator``.

Provides the same interface as ``scenarios_profiling.generator.cli`` with
additional flags to control the optimization parameters:

--batch-size INT
    Number of scenarios to request from the LLM in a single call.
    Larger batches reduce round-trips but may reduce per-scenario quality.
    Default: 10.

--max-concurrent INT
    Maximum number of focuses that run in parallel via asyncio.gather.
    Default: 3.

--no-disk-cache
    Disable the disk-based (L2) asset profile / research digest cache.
    The in-memory (L1) LRU cache remains active.

--cache-dir PATH
    Root directory for disk cache files.
    Default: .cache/scenarios_optimization

--cache-ttl-hours FLOAT
    How long (in hours) disk-cached profiles are considered fresh.
    Default: 24.0.  Pass 0 to effectively disable TTL.

--timing-report
    Print a detailed wall/CPU timing breakdown at the end of the run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv

from agent.cli import _DEFAULT_MODEL

from .agent import OptimizedScenarioGeneratorAgent
from .prompt_helpers import default_scenario_output_path
from ..optimization_utils import BatchConfig, GLOBAL_TIMING, ProfilerConfig
from ..wandb_logger import WandbConfig, WandbRunLogger


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Optimized Scenario Generator for AssetOpsBench",
    )
    parser.add_argument("asset_name", help="Asset class name (e.g. 'Chiller' or 'Wind Turbine')")
    parser.add_argument("--model-id", default=_DEFAULT_MODEL, help="Model ID for LiteLLM")
    parser.add_argument("--num-scenarios", type=int, default=50, help="Total number of scenarios to generate")
    parser.add_argument("--show-workflow", action="store_true", help="Show intermediate pipeline steps in the console")
    parser.add_argument(
        "--log",
        action="store_true",
        help=(
            "Write prompts and pipeline artifacts under the run directory "
            "(generated/scenarios_optimized/.../logs/). "
            "Does not change console output; use --show-workflow for step-by-step output."
        ),
    )
    parser.add_argument(
        "--data-in-couchdb",
        action="store_true",
        help="Use grounded open-form generation when matching live CouchDB-backed asset data is available",
    )
    parser.add_argument(
        "--retriever",
        choices=("arxiv", "semantic_scholar"),
        default="arxiv",
        help="Academic search backend for Phase 1 evidence retrieval. Default: arxiv.",
    )
    parser.add_argument(
        "--research-digest",
        metavar="PATH",
        default=None,
        help=(
            "Optional. Path to a precomputed research digest (Markdown). When set and the file exists, "
            "skips academic retrieval and digest LLM steps; loads this text for asset profile construction."
        ),
    )

    # -----------------------------------------------------------------------
    # Profiling flags  (identical to scenarios_profiling.generator.cli for
    # fair comparison — same ProfilerConfig, same chrome trace format)
    # -----------------------------------------------------------------------
    prof_group = parser.add_argument_group("PyTorch profiling options")
    prof_group.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Run the optimized pipeline under the PyTorch profiler. "
            "Writes chrome_trace.json and a key-averages table to --profile-dir. "
            "Requires torch to be installed."
        ),
    )
    prof_group.add_argument(
        "--profile-dir",
        metavar="DIR",
        default=None,
        help=(
            "Directory for profiler output (chrome trace, stacks, etc.). "
            "Defaults to profiling_output/<asset_slug>_optimized/ when --profile is set."
        ),
    )
    prof_group.add_argument(
        "--profile-memory",
        action="store_true",
        default=False,
        help="Track tensor memory allocation/deallocation in the profiler.",
    )
    prof_group.add_argument(
        "--profile-with-stack",
        action="store_true",
        default=False,
        help="Capture Python call stacks for richer flame-graph output.",
    )

    # ---- Optimization flags ----
    opt_group = parser.add_argument_group("Optimization options")
    opt_group.add_argument(
        "--batch-size",
        type=int,
        default=10,
        metavar="INT",
        help="Number of scenarios to request from the LLM per generation call. Default: 10.",
    )
    opt_group.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        metavar="INT",
        help="Maximum number of parallel focus-group generation tasks. Default: 3.",
    )
    opt_group.add_argument(
        "--max-workers",
        type=int,
        default=4,
        metavar="INT",
        help="Thread pool size for blocking I/O offloading. Default: 4.",
    )
    opt_group.add_argument(
        "--no-disk-cache",
        action="store_true",
        help="Disable the disk-level (L2) cache. The in-memory (L1) LRU cache remains active.",
    )
    opt_group.add_argument(
        "--cache-dir",
        metavar="PATH",
        default=".cache/scenarios_optimization",
        help="Root directory for disk cache files. Default: .cache/scenarios_optimization",
    )
    opt_group.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=24.0,
        metavar="FLOAT",
        help="Disk cache TTL in hours. Pass 0 to use no expiry. Default: 24.0",
    )
    opt_group.add_argument(
        "--timing-report",
        action="store_true",
        help="Print a detailed wall/CPU phase timing report at the end of the run.",
    )
    opt_group.add_argument(
        "--invalidate-cache",
        action="store_true",
        help="Invalidate any cached asset profile and research digest for the specified asset before running.",
    )

    # ---- W&B flags ----
    wandb_group = parser.add_argument_group("Weights & Biases logging")
    wandb_group.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging for this run.",
    )
    wandb_group.add_argument(
        "--wandb-project",
        metavar="PROJECT",
        default="assetopsbench-scenario-gen",
        help="W&B project name. Default: assetopsbench-scenario-gen",
    )
    wandb_group.add_argument(
        "--wandb-entity",
        metavar="ENTITY",
        default=None,
        help="W&B entity (user or team). Defaults to the entity set in the W&B environment.",
    )
    wandb_group.add_argument(
        "--wandb-run-name",
        metavar="NAME",
        default=None,
        help="Human-readable W&B run name. Auto-generated by W&B when omitted.",
    )
    wandb_group.add_argument(
        "--wandb-tags",
        metavar="TAG",
        nargs="+",
        default=[],
        help="One or more string tags to attach to the W&B run (e.g. --wandb-tags optimized arxiv).",
    )

    args = parser.parse_args()
    output_path = default_scenario_output_path(args.asset_name)

    if args.show_workflow:
        level = logging.WARNING
    elif args.data_in_couchdb:
        level = logging.INFO
    else:
        level = logging.WARNING

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
        os.makedirs(log_dir, exist_ok=True)

    batch_config = BatchConfig(
        generation_batch_size=args.batch_size,
        validation_batch_size=args.batch_size,
        max_concurrent_batches=args.max_concurrent,
        max_workers=args.max_workers,
    )

    cache_dir = None if args.no_disk_cache else args.cache_dir
    ttl_hours = None if args.cache_ttl_hours == 0 else args.cache_ttl_hours

    # ---- W&B setup ----
    wandb_logger: WandbRunLogger | None = None
    if args.wandb:
        run_config = {
            "model_id": args.model_id,
            "asset_name": args.asset_name,
            "num_scenarios": args.num_scenarios,
            "retriever": args.retriever,
            "batch_size": args.batch_size,
            "max_concurrent": args.max_concurrent,
            "disk_cache_enabled": not args.no_disk_cache,
            "data_in_couchdb": args.data_in_couchdb,
        }
        wandb_config = WandbConfig(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name,
            tags=args.wandb_tags or [],
            enabled=True,
        )
        wandb_logger = WandbRunLogger(wandb_config, run_config=run_config)

    agent = OptimizedScenarioGeneratorAgent(
        model_id=args.model_id,
        show_workflow=args.show_workflow,
        log_dir=log_dir,
        retriever=args.retriever,
        research_digest_path=args.research_digest,
        batch_config=batch_config,
        profile_cache_dir=cache_dir,
        profile_cache_ttl_hours=ttl_hours,
        timing_registry=GLOBAL_TIMING,
        wandb_logger=wandb_logger,
    )

    if args.invalidate_cache:
        agent.invalidate_profile_cache(args.asset_name, retriever=args.retriever)
        if not args.show_workflow:
            print(f"[INFO] Cache invalidated for '{args.asset_name}' ({args.retriever})")

    try:
        if args.profile:
            profiler_config = ProfilerConfig(
                profile_dir=(
                    args.profile_dir
                    or f"profiling_output/{args.asset_name.lower().replace(' ', '_')}_optimized"
                ),
                profile_memory=args.profile_memory,
                with_stack=args.profile_with_stack,
                export_chrome_trace=True,
                print_summary=True,
            )
            final_scenarios = asyncio.run(
                agent.run_with_profiling(
                    args.asset_name,
                    num_scenarios=args.num_scenarios,
                    data_in_couchdb=args.data_in_couchdb,
                    profiler_config=profiler_config,
                )
            )
        else:
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

    # Always write timing_report.json next to scenarios.json
    timing_path = output_path.parent / "timing_report.json"
    with timing_path.open("w") as handle:
        json.dump(GLOBAL_TIMING.to_dict(), handle, indent=2)

    if not args.show_workflow:
        print(f"Success! Generated {len(final_scenarios)} scenarios at {output_path}")
        print(f"Timing report saved to {timing_path}")
    else:
        print(f"Scenarios saved to {output_path}")
        print(f"Timing report saved to {timing_path}")

    if args.timing_report:
        print("\n" + GLOBAL_TIMING.summary())

    # Print cache stats when verbose
    if args.show_workflow:
        stats = agent.cache_stats
        print("\n[Cache Stats]")
        for level_name, level_stats in stats.items():
            print(f"  {level_name}: {level_stats}")
