"""Readable views of native execution and saved evidence; never an agent controller."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import re
import time


ACTIVE = {"running", "checking", "repairing"}
STAGES = {
    "Inventory": ("baseline.json", "initialized.json", "workspace/request.json",
                  "workspace/data/traces/*inventory*.json", "workspace/data/traces/*environment*.json"),
    "Research": ("logs/research.jsonl", "workspace/output/sources.json",
                 "workspace/data/research/search-*.json", "workspace/data/raw/download.json"),
    "Profile": ("workspace/output/profile.json",),
    "Environment": ("workspace/output/environment.md", "workspace/output/requirements.txt",
                    "workspace/data/prepared/manifest.json", "workspace/scripts/*.py"),
    "Scenarios": ("workspace/output/allocation.json", "workspace/output/scenarios.json",
                  "workspace/output/README.md"),
    "Checks": ("review.json", "review.stderr", "workspace/output/tool_checks.json",
               "workspace/data/traces/*validation*.json"),
}


def _safe(value, limit=220):
    text = str(value)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = re.sub(r"(?i)[\"']?\b[\w-]*(?:password|token|secret|api[-_]key|credential)[\w-]*[\"']?\s*[:=]\s*"
                  r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)", "[credential redacted]", text)
    text = re.sub(r"(?i)--[\w-]*(?:password|token|secret|api[-_]key|credential)[\w-]*\s+"
                  r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)", "[credential flag redacted]", text)
    text = re.sub(r"(?i)(?:authorization\s*[:=]\s*)?basic\s+[^\s\"',;]+", "[authorization redacted]", text)
    text = re.sub(r"(?i)(?:authorization\s*[:=]\s*)?bearer\s+[^\s\"',;]+", "[authorization redacted]", text)
    text = re.sub(r"(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[credentials redacted]@", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}", "[key redacted]", text)
    text = " ".join("".join(c for c in text if c.isprintable() or c.isspace()).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError):
        return None


def _records(path):
    try:
        lines = path.read_bytes().splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
        except (ValueError, UnicodeError):
            pass
    return records


def _number(path):
    match = re.search(r"-(\d+)\.", path.name)
    return int(match[1]) if match else 0


def _state(destination):
    paths = sorted((destination / "logs").glob("run-*.json"), key=_number, reverse=True)
    latest = next(((path, value) for path in paths if isinstance(value := _json(path), dict)), None)
    metadata = latest[1] if latest else {}
    status = _json(destination / "status.json")
    status = status if isinstance(status, dict) else {}
    if not status.get("status"):
        process, validation = metadata.get("process_status"), metadata.get("validation_status")
        fallback = {"running": "running", "failed": "failed", "succeeded": "incomplete"}.get(process, "unknown")
        if process == "succeeded" and validation == "passed":
            fallback = "complete"
        elif process == "succeeded" and validation == "pending":
            fallback = "checking"
        status = {"status": fallback, "attempt": _number(latest[0]) if latest else None}
    return status, metadata


def _event(record, prefix):
    kind = record.get("type", "")
    item = record.get("item", {})
    if not isinstance(item, dict):
        return None
    label = f"{prefix}:{item.get('id', kind)}"
    phase = kind.removeprefix("item.")
    item_type = item.get("type")
    if item_type == "agent_message" and kind == "item.completed":
        message = f"message: {_safe(item.get('text', ''))}"
    elif item_type == "command_execution":
        # Show the command, never its captured stdout, environment or result payload.
        command = str(item.get("command", "")).split("\n", 1)[0]
        code = f" (exit {item['exit_code']})" if item.get("exit_code") is not None else ""
        message = f"command {phase}{code}: {_safe(command)}"
    elif item_type in {"mcp_tool_call", "tool_call"}:
        message = f"tool {phase}: {_safe(item.get('server', ''))}.{_safe(item.get('tool', item.get('name', '')))}"
    elif item_type == "web_search":
        action = item.get("action", {})
        query = item.get("query", "")
        if not query and isinstance(action, dict):
            query = action.get("query") or action.get("queries", [])
        message = f"web search {phase}: {_safe(query)}"
    elif item_type == "file_change" and kind == "item.completed":
        message = "files changed: " + _safe(", ".join(str(c.get("path", "")) for c in item.get("changes", []) if isinstance(c, dict)))
    elif kind in {"turn.completed", "turn.failed", "error"}:
        message = kind + (": " + _safe(record.get("message", record.get("error", ""))) if kind != "turn.completed" else " (process evidence; validation separate)")
    elif "query" in record and not kind:
        message = f"research {_safe(record.get('status', 'recorded'))}: {_safe(record['query'])}"
        if record.get("response_file"):
            message += " → " + _safe(record["response_file"])
    else:
        return None
    return f"{_safe(label, 80)} · {message}"


def _link(path, base):
    target = os.path.relpath(path, base) if base else str(path.resolve())
    return f"[{path.name}](<{target}>)"


def _render(destination, base=None):
    status, metadata = _state(destination)
    lines = [f"Generation: {_safe(status['status'])} · attempt {_safe(status.get('attempt', '?'))}",
             f"Directory: {destination.resolve()}"]
    if metadata:
        lines.append(f"Native: {_safe(metadata.get('harness', 'codex'))} / {_safe(metadata.get('requested_model', '?'))}"
                     f" · process {_safe(metadata.get('process_status', 'not recorded'))}"
                     f" · validation {_safe(metadata.get('validation_status', 'not recorded'))}")
    if status.get("counts"):
        lines.append("Counts: " + _safe(json.dumps(status["counts"], sort_keys=True)))
    logs = sorted((destination / "logs").glob("codex-*.jsonl"), key=_number)
    counts, recent = Counter(), []
    for path in [*logs, destination / "logs/research.jsonl"]:
        for record in _records(path):
            item = record.get("item")
            if record.get("type") == "item.completed" and isinstance(item, dict):
                counts[_safe(item.get("type", "unknown"), 50)] += 1
            if event := _event(record, path.stem):
                recent.append(event)
    if counts:
        lines.append("Completed native items: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    lines.extend(["", "Artifact views (recorded means present, not validated):"])
    for stage, patterns in STAGES.items():
        artifacts = sorted({p for pattern in patterns for p in destination.glob(pattern) if p.is_file()})
        links = ", ".join(_link(path, base) for path in artifacts[:6])
        more = f" (+{len(artifacts) - 6} files)" if len(artifacts) > 6 else ""
        lines.append(f"- {stage}: recorded — {links}{more}" if artifacts else f"- {stage}: not recorded")
    traces = [p for p in sorted((destination / "logs").glob("*"))
              if p.is_file() and p.name != "index.md" and p.suffix in {".json", ".jsonl", ".stderr"}]
    if traces:
        lines.extend(["", "Native logs and provenance: " + ", ".join(_link(p, base) for p in traces)])
    if recent:
        lines.extend(["", "Latest recorded events:", *("- " + event for event in recent[-6:])])
    return "\n".join(lines) + "\n"


def inspect_run(destination: Path) -> str:
    """Return a compact snapshot, including incomplete/partially written runs."""
    return _render(Path(destination))


def write_index(destination: Path) -> Path:
    """Refresh only the human index, preserving every native log byte."""
    destination = Path(destination)
    logs = destination / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "index.md"
    temporary = logs / ".index.md.tmp"
    temporary.write_text("# Generation evidence\n\n" + _render(destination, logs))
    temporary.replace(path)
    return path


def watch_run(destination: Path):
    """Follow new complete native events; Ctrl-C stops only this viewer."""
    destination = Path(destination)
    print(inspect_run(destination), flush=True)
    cursors = {}
    for path in (destination / "logs").glob("*.jsonl"):
        try:
            data, stat = path.read_bytes(), path.stat()
            cursors[path] = (stat.st_ino, data.rfind(b"\n") + 1)
        except OSError:
            pass
    previous = None
    try:
        while True:
            for path in sorted((destination / "logs").glob("*.jsonl")):
                try:
                    stat = path.stat()
                    inode, offset = cursors.get(path, (stat.st_ino, 0))
                    if inode != stat.st_ino or stat.st_size < offset:
                        offset = 0
                    with path.open("rb") as stream:
                        stream.seek(offset)
                        chunk = stream.read()
                    end = chunk.rfind(b"\n") + 1
                    for line in chunk[:end].splitlines():
                        try:
                            record = json.loads(line)
                            event = _event(record, path.stem) if isinstance(record, dict) else None
                            if event:
                                print(event, flush=True)
                        except (ValueError, UnicodeError):
                            pass
                    cursors[path] = (stat.st_ino, offset + end)
                except OSError:
                    continue
            state = _state(destination)[0]["status"]
            if previous in ACTIVE and (destination / "status.json").exists() and _json(destination / "status.json") is None:
                state = previous  # A partial status rewrite must not stop the viewer.
            if state != previous:
                print(f"Generation status: {_safe(state)}", flush=True)
                previous = state
            if state not in ACTIVE:
                return
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped watching; generation continues independently.", flush=True)
