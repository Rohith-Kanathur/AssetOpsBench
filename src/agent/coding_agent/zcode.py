"""First-party ZCode configuration and native evidence normalization."""

import json
from pathlib import Path

NODE = "/opt/zcode-node/bin/node"
CLI = "/opt/zcode/apps/zcode-cli/packages/cli/dist/zcode.cjs"
BUILTIN = "/opt/zcode/config/provider/zcode-builtin.json"
DISALLOWED = ("WebSearch,WebFetch,Agent,Task,Skill,AskUserQuestion,EnterPlanMode,"
              "ExitPlanMode,TodoRead,TodoWrite,ReadSessionContext,TaskOutput,TaskStop,"
              "SendMessage,RespondToCoordinator,Workflow,js")


def prepare(home: Path, workspace: Path, servers: dict, model: str,
            effort: str, environment: dict) -> tuple[list[str], dict]:
    from .commands import private_write

    key = environment.get("ZAI_API_KEY")
    if not key:
        raise ValueError("ZAI_API_KEY is required for ZCode Coding Plan authentication")
    if effort not in {"low", "high", "max"}:
        raise ValueError("ZCode reasoning effort must be low, high or max")
    env = {name: value for name, value in environment.items()
           if not name.startswith(("ZCODE_", "CODEX_", "CLAUDE_", "ANTHROPIC_", "OPENAI_"))}
    env.pop("ZAI_API_KEY", None)
    directory = home / ".zcode/cli"
    directory.mkdir(parents=True, mode=0o700)
    provider = home / ".zcode/provider.json"
    private_write(provider, json.dumps({"schemaVersion": 1, "config": {
        "providerConfigRules": {"providerRules": [{"providerId": "zai-api",
            "templateId": "zai-api", "enabled": True, "config": {"group": "standard-personal",
            "access": {"type": "zhipu-coding-plan-api-key", "apiKey": key}}}]},
        "modelConfigRules": {"providerModelRules": [], "manualProviderModelRules": []},
        "defaultModelSelection": {"providerId": "zai-api", "modelId": model,
                                  "options": {"reasoningLevel": effort}}}}))
    private_write(directory / "config.json", json.dumps({
        "plugins": {"enabled": False}, "skills": {"enabled": False, "includeInstructions": False},
        "features": {"memory": False, "skill": False, "subagent": False},
        "memory": {"use": False}, "hooks": {"enabled": False}, "mcp": {"servers": servers}}))
    env.update(HOME=str(home), ZCODE_DATA_BASE_DIR=str(home), ZCODE_STORAGE_DIR=str(home / ".zcode/state"),
               ZCODE_BFS_BINARY="/opt/zcode/runtime-tools/bfs", ZCODE_UGREP_BINARY="/opt/zcode/runtime-tools/ugrep",
               ZCODE_RG_BINARY="/opt/zcode/runtime-tools/rg",
               ZCODE_BUILTIN_PROVIDER_CONFIG_FILE=BUILTIN,
               ZCODE_PERSONAL_PROVIDER_CONFIG_FILE=str(provider))
    return [NODE, CLI, "--cwd", str(workspace), "--mode", "yolo", "--output-format",
            "stream-json", "--no-color", "--disallowed-tools", DISALLOWED], env


def parse(text: str) -> dict:
    answer, usage, error, completed = "", {}, None, False
    turns, calls, messages = [], {}, {}
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        kind, payload = event.get("type"), event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        if kind == "tool.updated":
            sid, status = payload.get("toolCallId"), payload.get("kind")
            if status == "scheduled":
                calls[sid] = {"id": sid, "name": payload.get("toolName"), "input": payload.get("input"),
                              "output": None, "status": status}
                turns.append({"text": "", "tool_calls": [calls[sid]]})
            elif sid in calls:
                calls[sid].update(status=status)
                if status in {"result", "error"}:
                    calls[sid]["output"] = payload.get(status)
        elif kind == "model.streaming" and payload.get("kind") == "text_delta":
            sid = payload.get("assistantMessageId")
            if sid not in messages:
                messages[sid] = {"text": "", "tool_calls": []}
                turns.append(messages[sid])
            messages[sid]["text"] += payload.get("delta", "")
        elif kind == "result":
            answer, usage = event.get("response", ""), event.get("usage", {})
            completed = event.get("projection", {}).get("status") in {"idle", "completed"}
            if not completed:
                error = error or "ZCode did not finish the session: " + str(event.get("projection", {}).get("status"))
            if answer and (not turns or turns[-1]["text"] != answer):
                turns.append({"text": answer, "tool_calls": []})
        elif kind in {"error", "turn.failed"}:
            error = event.get("message") or payload.get("error") or payload.get("message") or "ZCode execution failed"
    return {"answer": answer, "trajectory": {"turns": turns}, "usage": usage,
            "completed": completed, "error": error}
