"""Run one native CLI inside a caller-provided isolated scenario environment."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from .commands import PROMPT, prepare, private_write
from .trajectory import parse

_SECRET = re.compile(r"token|password|secret|api.?key|authorization|credential", re.I)


def _secrets(value, key="") -> set[str]:
    if isinstance(value, dict):
        return set().union(*(_secrets(v, str(k)) for k, v in value.items()), set())
    if isinstance(value, list):
        return set().union(*(_secrets(v, key) for v in value), set())
    return {value} if isinstance(value, str) and len(value) >= 8 and _SECRET.search(key) else set()


def run(question: str, *, harness: str, workspace: Path, mcp_servers: dict,
        output_dir: Path, model: str, reasoning_effort: str | None = None,
        service_tier: str = "fast", timeout: float = 900,
        auth_home: Path | None = None, environment: dict[str, str] | None = None) -> dict:
    """Capture an execution, not a score; the caller owns data isolation and judging."""
    env = dict(os.environ if environment is None else environment)
    reasoning_effort = reasoning_effort or ("high" if harness == "zcode" else "xhigh")
    if env.get("ASSETOPS_EXECUTION_ISOLATED") != "1":
        raise ValueError("Run inside an isolated container with ASSETOPS_EXECUTION_ISOLATED=1")
    workspace, output_dir = workspace.resolve(), output_dir.resolve()
    if not workspace.is_dir() or timeout <= 0 or not question.strip():
        raise ValueError("An existing workspace, positive timeout and question are required")
    for parent in (workspace, *workspace.parents):
        if any((parent / name).exists() for name in
               ("CLAUDE.md", "CLAUDE.local.md", ".claude", ".codex", ".agents")
               + (("AGENTS.md", ".zcode") if harness == "zcode" else ())):
            raise ValueError("Workspace must not inherit repository instructions or CLI settings")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("result.json", "stdout.jsonl", "stderr.log")):
        raise ValueError("Output directory already contains an execution")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="assetops-agent-") as temporary:
        home = Path(temporary)
        command, child_env = prepare(harness, home, workspace, mcp_servers, model=model,
                                     reasoning_effort=reasoning_effort, service_tier=service_tier,
                                     auth_home=auth_home, environment=env)
        prompt = PROMPT + "\nRequest:\n" + question
        if harness == "zcode":
            command += ["--prompt", prompt]
        secrets = _secrets(env) | _secrets(mcp_servers)
        for path in (home / ".codex" / "auth.json", home / ".claude" / ".credentials.json"):
            if path.is_file():
                secrets |= _secrets(json.loads(path.read_text()))

        def redact(text: str) -> str:
            for secret in sorted(secrets, key=len, reverse=True):
                text = text.replace(secret, "[REDACTED]")
            return text

        error, stdout, stderr, code = None, "", "", None
        try:
            process = subprocess.Popen(command, cwd=workspace, env=child_env,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                stdout, stderr = process.communicate(None if harness == "zcode" else prompt,
                                                     timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                error = f"Execution timed out after {timeout:g} seconds"
            code = process.returncode
        except OSError as exc:
            error = f"Could not start {harness}: {exc.strerror}"
        stdout, stderr = redact(stdout), redact(stderr)
        private_write(output_dir / "stdout.jsonl", stdout)
        private_write(output_dir / "stderr.log", stderr)
        native_artifacts = []
        if harness == "zcode":
            artifacts = home / ".zcode/state/cli/artifacts"
            for path in sorted(artifacts.rglob("*")):
                if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(artifacts.resolve()):
                    continue
                relative = str(path.relative_to(artifacts))
                content = redact(path.read_text())
                target = output_dir / "tool-results" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                private_write(target, content)
                native_artifacts.append({"path": "tool-results/" + relative, "content": content})
        parsed = parse(stdout, harness)
        native_error = parsed.pop("error")
        error = error or native_error
        completed = parsed.pop("completed")
        if not error and code != 0:
            error = f"{harness} exited with status {code}; see stderr.log"
        if not error and (not completed or not parsed["answer"].strip()):
            error = "Native CLI did not report a completed turn with an answer"
        result = {"harness": harness, "model": model, "question": redact(question),
                  "reasoning_effort": reasoning_effort,
                  "service_tier": service_tier if harness == "codex" else None,
                  "status": "error" if error else "completed", "exit_code": code,
                  "elapsed_seconds": round(time.monotonic() - started, 3),
                  "error": error, **parsed}
        if native_artifacts:
            result["native_tool_artifacts"] = native_artifacts
        private_write(output_dir / "result.json", redact(json.dumps(result, indent=2)) + "\n")
    return result
