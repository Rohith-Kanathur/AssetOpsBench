"""Prepare subscription authentication without copying personal agent settings."""

import json
import os
from pathlib import Path
import shutil
import subprocess


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
    config = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
    source = config / ".credentials.json"
    if source.is_file():
        credentials = json.loads(source.read_text())
    elif os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        # Passed as an environment value by the caller, never as a CLI argument.
        return
    elif sys_platform() == "darwin":
        result = subprocess.run(["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                                capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise ValueError("Claude login credentials unavailable; run claude auth login")
        credentials = json.loads(result.stdout)
    else:
        raise ValueError("Claude login credentials unavailable; run claude auth login")
    private_json(directory / ".claude/.credentials.json", credentials)
    private_json(directory / ".claude.json", {"hasCompletedOnboarding": True})


def sys_platform():
    import sys
    return sys.platform
