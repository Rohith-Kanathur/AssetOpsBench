"""Generate grounded asset scenarios with a native agent harness."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from . import runtime
from .harnesses import DEFAULT_MODEL, DEFAULT_REASONING, DEFAULT_TIER, HARNESSES
from .budget import parse_budget
from .workspace import audit_baseline, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("action", nargs="?", default="run", choices=("run", "check", "inspect", "watch", "stop", "build"))
    parser.add_argument("directory", type=Path, nargs="?", help="Saved generation directory")
    parser.add_argument("--asset-class", help="Asset class to generate scenarios for")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--scenario-counts", metavar="JSON",
                        help='Positive/negative totals to allocate (default: {"positive":50,"negative":2})')
    budget.add_argument("--scenario-plan", metavar="JSON",
                        help="Positive/negative counts per domain; omitted entries request zero")
    parser.add_argument("--repository", type=Path, default=Path.cwd(), help="Environment source checkout")
    parser.add_argument("--ref", default="HEAD", help="Committed environment revision")
    parser.add_argument("--harness", choices=HARNESSES, default="codex", help="Generation harness")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Generation model")
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING, help="Model reasoning effort")
    parser.add_argument("--service-tier", default=DEFAULT_TIER, help="Generation service tier")
    parser.add_argument("--followup", help="Continue a saved generation session")
    args = parser.parse_args(argv)
    if args.action == "build":
        runtime.build()
        return
    if args.action == "run" and not args.directory:
        if not args.asset_class or not args.asset_class.strip():
            parser.error("--asset-class is required for a new generation")
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
            if args.scenario_counts is not None or args.scenario_plan is not None:
                try:
                    supplied = parse_budget(args.scenario_counts, args.scenario_plan)
                except ValueError as exc:
                    parser.error(str(exc))
                if any(request.get(key) != value for key, value in supplied.items()):
                    parser.error("follow-up budget must match the saved request")
            if args.asset_class and args.asset_class != request["asset_class"]:
                parser.error("follow-up asset class must match the saved request")
        else:
            if args.followup:
                parser.error("--followup requires an existing generation directory")
            if not args.asset_class or not args.asset_class.strip():
                parser.error("--asset-class is required for a new generation")
            try:
                budget = parse_budget(args.scenario_counts, args.scenario_plan)
            except ValueError as exc:
                parser.error(str(exc))
            prepare(args.repository.resolve(), destination, ref=args.ref)
            errors = audit_baseline(destination)
            if errors:
                raise RuntimeError("\n".join(errors))
            request = {"asset_class": args.asset_class.strip(), **budget}
            (destination / "workspace/request.json").write_text(json.dumps(request, indent=2) + "\n")
        print(f"Generation directory: {destination}", flush=True)
        runtime.configure(destination, Path.home() / ".codex", Path.home() / ".kaggle")
        runtime.run(destination, args.model, args.followup, harness=args.harness,
                    reasoning_effort=args.reasoning_effort, service_tier=args.service_tier,
                    env_file=args.repository.expanduser().resolve() / ".env")
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
