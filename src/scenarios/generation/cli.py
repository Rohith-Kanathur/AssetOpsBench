"""Generate grounded asset scenarios with a native agent harness."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from . import runtime
from .harnesses import DEFAULT_MODEL, DEFAULT_REASONING, DEFAULT_TIER, HARNESSES
from .budget import DEFAULT_COUNT, parse_budget
from .modes import DEFAULT_MODE
from .environment import DEFAULT_POLICY, POLICIES, policy
from .workspace import audit_baseline, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("action", nargs="?", default="run", choices=("run", "check", "inspect", "watch", "stop", "build"))
    parser.add_argument("directory", type=Path, nargs="?", help="Saved generation directory")
    parser.add_argument("--asset", help="Asset class to generate scenarios for")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--count", type=int,
                        help=f"Total scenarios to allocate across domains (default: {DEFAULT_COUNT})")
    budget.add_argument("--plan", metavar="JSON",
                        help='Scenario counts per domain, e.g. {"iot":10,"multiagent":15}; omitted domains request zero')
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Environment source checkout")
    parser.add_argument("--ref", default="HEAD", help="Committed environment revision")
    parser.add_argument("--seed", type=Path,
                        help="Prepared snapshot supplying database and input files; code still comes from --repo/--ref")
    parser.add_argument("--environment", choices=POLICIES,
                        help="Preparation policy: extend data/tools (default) or preserve the starting data and tools")
    parser.add_argument("--harness", choices=HARNESSES, default="codex", help="Generation harness")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Generation model")
    parser.add_argument("--reasoning", default=DEFAULT_REASONING, help="Model reasoning effort")
    parser.add_argument("--tier", default=DEFAULT_TIER, help="Generation service tier")
    parser.add_argument("--followup", help="Continue a saved generation session")
    args = parser.parse_args(argv)
    if args.seed is not None and args.action != "run":
        parser.error("--seed only applies to run")
    if args.action == "build":
        runtime.build()
        return
    if args.action == "run" and not args.directory:
        if not args.asset or not args.asset.strip():
            parser.error("--asset is required for a new generation")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        args.directory = Path.home() / ".cache/assetopsbench/generation" / f"{stamp}-{uuid.uuid4().hex[:8]}"
    if args.directory is None:
        parser.error("directory is required for this action")
    destination = args.directory.expanduser().resolve()
    if args.action == "run":
        if destination.exists():
            if not args.followup:
                parser.error("directory already exists; use --followup to continue it")
            request = json.loads((destination / "workspace/request.json").read_text())
            saved_mode = request.get("generation_mode")
            if saved_mode != DEFAULT_MODE:
                parser.error("saved run does not use general execution; start a new generation")
            if args.seed is not None:
                from .seed import describe
                try:
                    seed = describe(args.seed)
                except (OSError, ValueError) as exc:
                    parser.error(str(exc))
                if request.get("seed") != {"sha256": seed["sha256"]}:
                    parser.error("follow-up seed must match the saved request")
            if args.environment is not None and args.environment != policy(request):
                parser.error("follow-up environment policy must match the saved request")
            if args.count is not None or args.plan is not None:
                try:
                    supplied = parse_budget(args.count, args.plan)
                except ValueError as exc:
                    parser.error(str(exc))
                if any(request.get(key) != value for key, value in supplied.items()):
                    parser.error("follow-up budget must match the saved request")
            if args.asset and args.asset != request["asset_class"]:
                parser.error("follow-up asset class must match the saved request")
        else:
            if args.followup:
                parser.error("--followup requires an existing generation directory")
            if not args.asset or not args.asset.strip():
                parser.error("--asset is required for a new generation")
            try:
                budget = parse_budget(args.count, args.plan)
            except ValueError as exc:
                parser.error(str(exc))
            seed = None
            if args.seed is not None:
                from .seed import describe
                try:
                    seed = describe(args.seed)
                except (OSError, ValueError) as exc:
                    parser.error(str(exc))
            prepare(args.repo.resolve(), destination, ref=args.ref)
            errors = audit_baseline(destination)
            if errors:
                raise RuntimeError("\n".join(errors))
            request = {"asset_class": args.asset.strip(),
                       "generation_mode": DEFAULT_MODE,
                       "environment_policy": args.environment or DEFAULT_POLICY, **budget}
            if seed is not None:
                from .seed import prepare as prepare_seed
                prepare_seed(args.seed, destination, seed)
                request["seed"] = {"sha256": seed["sha256"]}
            (destination / "workspace/request.json").write_text(json.dumps(request, indent=2) + "\n")
        print(f"Generation directory: {destination}", flush=True)
        settings = dict(harness=args.harness, reasoning_effort=args.reasoning,
                        service_tier=args.tier, env_file=args.repo.expanduser().resolve() / '.env')
        if os.environ.get('CODEX_HOME'):
            # Explicit isolated login remains available for small manual smoke tests.
            runtime.configure(destination, Path(os.environ['CODEX_HOME']).expanduser(), Path.home() / '.kaggle')
            runtime.run(destination, args.model, args.followup, **settings)
        else:
            runtime.run_with_pool(destination, args.model, args.followup, **settings)
    elif args.action == "check":
        from .progress import write_index
        report = runtime.check(destination)
        write_index(destination)
        print(f"Saved live checks to {destination / 'review.json'}")
        raise SystemExit(bool(report["errors"]))
    elif args.action in {"inspect", "watch"}:
        from .progress import inspect_run, watch_run
        if args.action == "inspect":
            print(inspect_run(destination))
        else:
            watch_run(destination)
    else:
        runtime.stop(destination)
