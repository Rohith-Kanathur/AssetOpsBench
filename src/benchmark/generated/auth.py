"""Prepare subscription authentication without copying personal agent settings."""

import json
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time


def private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pending = path.with_name(path.name + ".tmp")
    pending.touch(mode=0o600)
    pending.chmod(0o600)
    pending.write_text(json.dumps(value) + "\n")
    pending.replace(path)


def prepare_auth(directory, harness):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if harness == "codex":
        source = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
        if not source.is_file():
            raise ValueError("Run codex login before starting subscription evaluations")
        target = directory / ".codex/auth.json"
        target.parent.mkdir(mode=0o700, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o600)
        return
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        # Caller-provided access tokens are passed as environment values.
        return
    credentials = _claude_credentials()
    if _needs_refresh(credentials):
        # Refresh on the host, where Claude itself safely persists rotated
        # credentials. Never let a disposable container own a refresh token.
        lock = Path.home() / ".cache/assetopsbench/claude-auth-refresh.lock"
        lock.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with lock.open("a") as handle:
            lock.chmod(0o600)
            fcntl.flock(handle, fcntl.LOCK_EX)
            credentials = _claude_credentials()
            if _needs_refresh(credentials):
                environment = {key: value for key, value in os.environ.items()
                               if key not in {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"}}
                result = subprocess.run([
                    "claude", "--print", "--model", "claude-fable-5-1",
                    "--setting-sources", "", "--no-session-persistence",
                    "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                    "--tools", "", "Reply with OK only."],
                    capture_output=True, text=True, timeout=120, env=environment)
                credentials = _claude_credentials()
                if result.returncode or _needs_refresh(credentials):
                    raise ValueError("Claude subscription authentication expired; run claude auth login on the host")
    # A valid access token is sufficient for the bounded isolated run. Keeping
    # refresh authority on the host prevents rotation being lost on cleanup.
    credentials = json.loads(json.dumps(credentials))
    credentials.get("claudeAiOauth", {}).pop("refreshToken", None)
    private_json(directory / ".claude/.credentials.json", credentials)
    private_json(directory / ".claude.json", {"hasCompletedOnboarding": True})


def _needs_refresh(credentials):
    expires = credentials.get("claudeAiOauth", {}).get("expiresAt")
    return isinstance(expires, (int, float)) and expires / 1000 <= time.time() + 900


def _claude_credentials():
    config = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
    source = config / ".credentials.json"
    if source.is_file():
        return json.loads(source.read_text())
    if sys_platform() == "darwin":
        result = subprocess.run(["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                                capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            return json.loads(result.stdout)
    raise ValueError("Claude login credentials unavailable; run claude auth login")


def sys_platform():
    import sys
    return sys.platform
