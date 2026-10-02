"""Generate grounded asset scenarios with a native agent harness."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from . import runtime
from .harnesses import DEFAULT_MODEL, DEFAULT_REASONING, DEFAULT_TIER, HARNESSES
from .review import DOMAINS
from .workspace import audit_baseline, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("action", nargs="?", default="run", choices=("run", "check", "stop", "build"))
    parser.add_argument("directory", type=Path, nargs="?", help="Saved generation directory")
    parser.add_argument("--asset-class", help="Asset class to generate scenarios for")
    parser.add_argument("--count", type=int, default=5, help="Number of scenarios")
    parser.add_argument("--domains", nargs="+", choices=sorted(DOMAINS), default=sorted(DOMAINS))
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
            if args.asset_class and args.asset_class != request["asset_class"]:
                parser.error("follow-up asset class must match the saved request")
        else:
            if args.followup:
                parser.error("--followup requires an existing generation directory")
            if not args.asset_class or not args.asset_class.strip():
                parser.error("--asset-class is required for a new generation")
            domains = list(dict.fromkeys(args.domains))
            if args.count < len(domains):
                parser.error("--count must allow at least one scenario per requested domain")
            prepare(args.repository.resolve(), destination, ref=args.ref)
            errors = audit_baseline(destination)
            if errors:
                raise RuntimeError("\n".join(errors))
            request = {"asset_class": args.asset_class, "count": args.count, "domains": domains}
            (destination / "workspace/request.json").write_text(json.dumps(request, indent=2) + "\n")
        print(f"Generation directory: {destination}", flush=True)
        runtime.configure(destination, Path.home() / ".codex", Path.home() / ".kaggle")
        runtime.run(destination, args.model, args.followup, harness=args.harness,
                    reasoning_effort=args.reasoning_effort, service_tier=args.service_tier)
    elif args.action == "check":
        review = Path(__file__).with_name("review.py").resolve()
        runtime.start(destination)
        with (destination / "review.json").open("w") as out:
            runtime.compose(destination, "run", "--rm", "-T", "--volume",
                            f"{review}:/opt/generation/review.py:ro", "agent", "python",
                            "/opt/generation/review.py", stdout=out, timeout=300)
        print(f"Saved live checks to {destination / 'review.json'}")
    else:
        runtime.stop(destination)
