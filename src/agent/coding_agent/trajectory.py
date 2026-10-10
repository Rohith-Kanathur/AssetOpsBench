"""Normalize native event streams without implementing an agent loop."""

from __future__ import annotations

import json
from typing import Any


def parse(text: str, harness: str) -> dict[str, Any]:
    if harness == "zcode":
        from .zcode import parse as parse_zcode
        return parse_zcode(text)
    answer, error, usage = "", None, {}
    turns, calls = [], {}
    completed = False
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if harness == "codex":
            if kind == "turn.completed":
                usage = event.get("usage", {})
                completed = True
            elif kind in {"error", "turn.failed"}:
                error = event.get("message") or event.get("error") or "Codex turn failed"
            item = event.get("item", {})
            if kind != "item.completed" or not isinstance(item, dict):
                continue
            if item.get("type") == "agent_message":
                answer = item.get("text", "")
                turns.append({"text": answer, "tool_calls": []})
            elif item.get("type") in {"mcp_tool_call", "command_execution", "file_change"}:
                name = item.get("tool") or item["type"]
                if item.get("server"):
                    name = f"{item['server']}.{name}"
                call = {"id": item.get("id", ""), "name": name,
                        "input": item.get("arguments", item.get("command", item.get("changes"))),
                        "output": item.get("result", item.get("aggregated_output", item.get("error"))),
                        "status": item.get("status")}
                turns.append({"text": "", "tool_calls": [call]})
        else:
            if kind == "result":
                completed = not event.get("is_error", False)
                answer = event.get("result", answer)
                usage = event.get("usage", {})
                if event.get("is_error"):
                    error = event.get("errors") or event.get("result") or event.get("subtype")
            message = event.get("message", {})
            content = message.get("content", []) if isinstance(message, dict) else []
            if not isinstance(content, list):
                continue
            if kind == "assistant":
                turn = {"text": "", "tool_calls": []}
                for block in content:
                    if block.get("type") == "text":
                        turn["text"] += block.get("text", "")
                    elif block.get("type") == "tool_use":
                        call = {"id": block.get("id", ""), "name": block.get("name", ""),
                                "input": block.get("input", {}), "output": None}
                        calls[call["id"]] = call
                        turn["tool_calls"].append(call)
                turns.append(turn)
                if turn["text"]:
                    answer = turn["text"]
            elif kind == "user":
                for block in content:
                    if block.get("type") == "tool_result" and block.get("tool_use_id") in calls:
                        calls[block["tool_use_id"]]["output"] = block.get("content")
    return {"answer": answer, "trajectory": {"turns": turns}, "usage": usage,
            "completed": completed, "error": error}
