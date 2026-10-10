"""CLI for one scenario in an already isolated execution container."""

import argparse
import json
from pathlib import Path

from .runner import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", choices=("codex", "claude", "zcode"), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--mcp-config", type=Path, required=True)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--service-tier", default="fast")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--auth-home", type=Path)
    args = parser.parse_args()
    if not args.model and args.harness != "zcode":
        parser.error("--model is required for Codex and Claude Code")
    servers = json.loads(args.mcp_config.read_text())
    result = run(args.question_file.read_text(), harness=args.harness,
                 workspace=args.workspace, mcp_servers=servers.get("mcpServers", servers),
                 output_dir=args.output_dir, model=args.model or "GLM-5.3",
                 reasoning_effort=args.reasoning_effort or ("high" if args.harness == "zcode" else "xhigh"),
                 service_tier=args.service_tier,
                 timeout=args.timeout, auth_home=args.auth_home)
    print(json.dumps({"status": result["status"], "harness": args.harness,
                      "result": str(args.output_dir / "result.json")}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
