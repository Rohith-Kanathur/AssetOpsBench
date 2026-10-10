"""Ephemeral native CLI configuration; authentication is copied without settings."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

PROMPT = """You are an industrial asset operations assistant.
Answer the request using the available MCP servers for operational facts.
You can run code and create files in the current workspace. Return requested
artifacts with their paths and describe evidence limitations accurately.
Keep your work within this workspace; never inspect credentials or hidden
evaluation materials. Complete the request without asking follow-up questions.
"""


def _toml(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)} = {_toml(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    if isinstance(value, (str, bool, int, float)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    raise ValueError("Unsupported MCP configuration value")


def private_write(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o600)


def prepare(harness: str, home: Path, workspace: Path, servers: dict, *,
            model: str, reasoning_effort: str, service_tier: str,
            auth_home: Path | None, environment: dict[str, str]) -> tuple[list[str], dict]:
    env = dict(environment)
    original_home = Path(env.get("HOME", str(Path.home())))
    env["HOME"] = str(home)
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_SIMPLE", None)
    env.pop("CLAUDE_CODE_SAFE_MODE", None)
    if harness == "zcode":
        from .zcode import prepare as prepare_zcode
        return prepare_zcode(home, workspace, servers, model, reasoning_effort, env)
    zai = harness == "claude" and model.startswith("zai/")
    if zai:
        key = env.get("ZAI_API_KEY")
        if not key:
            raise ValueError("ZAI_API_KEY is required for a zai/ model")
        model = model.removeprefix("zai/")
        if not model:
            raise ValueError("A model name must follow zai/")
        for name in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
            env.pop(name, None)
        env.update(ANTHROPIC_AUTH_TOKEN=key, ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic",
                   ANTHROPIC_DEFAULT_OPUS_MODEL=model, ANTHROPIC_DEFAULT_SONNET_MODEL=model,
                   ANTHROPIC_DEFAULT_HAIKU_MODEL=model, CLAUDE_CODE_SUBAGENT_MODEL=model)
    for name in (".codex", ".claude"):
        (home / name).mkdir(mode=0o700)
    if harness == "codex":
        source = auth_home or Path(environment.get("CODEX_HOME", original_home / ".codex"))
        target = home / ".codex"
        credential = "auth.json"
        env["CODEX_HOME"] = str(target)
    elif harness == "claude":
        source = auth_home or Path(environment.get("CLAUDE_CONFIG_DIR", original_home / ".claude"))
        target = home / ".claude"
        credential = ".credentials.json"
        env["CLAUDE_CONFIG_DIR"] = str(target)
        env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
        env["ENABLE_CLAUDEAI_MCP_SERVERS"] = "false"
        private_write(home / ".claude.json", '{"hasCompletedOnboarding":true}')
    else:
        raise ValueError("Harness must be codex, claude or zcode")
    if not zai and (source / credential).is_file():
        shutil.copyfile(source / credential, target / credential)
        (target / credential).chmod(0o600)

    if harness == "codex":
        entries = []
        for name, spec in servers.items():
            spec = dict(spec)
            transport = spec.pop("type", "stdio")
            if transport not in {"http", "stdio"}:
                raise ValueError("Codex MCP transport must be http or stdio")
            if "headers" in spec:
                spec["http_headers"] = spec.pop("headers")
            spec.setdefault("startup_timeout_sec", 60)
            spec.setdefault("tool_timeout_sec", 120)
            entries.append(f"mcp_servers.{json.dumps(name)} = {_toml(spec)}")
        entries += ["project_doc_max_bytes = 0", "agents.enabled = false",
                    'web_search = "disabled"', 'approval_policy = "never"',
                    f"projects.{json.dumps(str(workspace))}.trust_level = \"untrusted\"",
                    f"model_reasoning_effort = {json.dumps(reasoning_effort)}",
                    f"service_tier = {json.dumps(service_tier)}"]
        private_write(target / "config.toml", "\n".join(entries) + "\n")
        command = ["codex", "exec", "--ephemeral", "--ignore-rules", "--skip-git-repo-check",
                   "--json", "--color", "never", "--sandbox", "danger-full-access",
                   "--model", model, "--cd", str(workspace), "-"]
    else:
        config = home / "mcp.json"
        private_write(config, json.dumps({"mcpServers": servers}))
        command = ["claude", "--print", "--verbose", "--output-format", "stream-json",
                   "--no-session-persistence", "--setting-sources", "",
                   "--strict-mcp-config", "--mcp-config", str(config),
                   "--disable-slash-commands", "--tools", "Bash,Read,Write,Edit,Glob,Grep",
                   "--permission-mode", "bypassPermissions", "--model", model,
                   "--effort", reasoning_effort]
    return command, env
