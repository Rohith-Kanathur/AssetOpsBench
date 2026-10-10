"""Run an existing SDK agent with only the final environment's MCP tools."""

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time


async def run(args):
    servers = json.loads(args.mcp_config.read_text())["mcpServers"]
    if args.runner == "openai":
        from agent.openai_agent.runner import OpenAIAgentRunner
        runner = OpenAIAgentRunner(model=args.model, server_paths=servers, max_turns=40)
    else:
        from agent.claude_agent.runner import ClaudeAgentRunner
        runner = ClaudeAgentRunner(model=args.model, server_paths=servers,
                                   max_turns=40, mcp_only=True, cli_path=shutil.which("claude"))
    started = time.monotonic()
    try:
        result = await asyncio.wait_for(runner.run(args.question_file.read_text()), args.timeout)
        if not result.answer or not result.answer.strip():
            raise ValueError("Agent returned no final answer")
        record = {"status": "completed", "answer": result.answer,
                  "trajectory": asdict(result.trajectory)}
    except Exception as exc:
        # API exception strings can include request headers; persist only the type.
        record = {"status": "error", "answer": "", "trajectory": {},
                  "error": type(exc).__name__}
    record.update(model=args.model, runner=args.runner,
                  elapsed_seconds=round(time.monotonic() - started, 3))
    args.output.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return 0 if record["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("openai", "claude"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mcp-config", type=Path, required=True)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
