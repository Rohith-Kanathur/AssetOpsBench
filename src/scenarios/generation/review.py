"""Check saved evidence and resolve scenario references through the live MCP tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable


DOMAINS = {"IoT", "FMSR", "TSFM", "WO", "Vibration"}


def local_file(workspace: Path, name: str) -> Path:
    path = (workspace / name).resolve()
    if not path.is_relative_to(workspace.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-workspace evidence file: {name}")
    return path


def read_json(workspace: Path, name: str):
    return json.loads(local_file(workspace, name).read_text())


def check_contract(workspace: Path) -> dict:
    errors = []
    for name in ("output/README.md", "output/profile.json", "output/environment.md",
                 "output/requirements.txt"):
        local_file(workspace, name)
    request = read_json(workspace, "request.json")
    scenarios = read_json(workspace, "output/scenarios.json")
    sources = read_json(workspace, "output/sources.json")
    checks = read_json(workspace, "output/tool_checks.json")
    source_ids = {s["id"] for s in sources}
    if len(source_ids) != len(sources):
        errors.append("Duplicate source IDs")
    for source in sources:
        if source.get("kind") != "synthetic" and not source.get("url", "").startswith(("https://", "http://", "repository:")):
            errors.append(f"Missing source URL: {source['id']}")
        files = source.get("files", [])
        if isinstance(files, dict):
            files = [{"path": k, "sha256": v} for k, v in files.items()]
        for evidence in files:
            path = local_file(workspace, evidence["path"])
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != evidence["sha256"]:
                errors.append(f"Source checksum mismatch: {evidence['path']}")
    if len(scenarios) != request["count"] or {s["type"] for s in scenarios} != set(request["domains"]):
        errors.append("Scenario count or domains do not match request.json")
    if len({s["id"] for s in scenarios}) != len(scenarios):
        errors.append("Duplicate scenario IDs")
    check_ids = {item["scenario_id"] for item in checks}
    for scenario in scenarios:
        sid = scenario["id"]
        for key in ("text", "category", "characteristic_form"):
            if not isinstance(scenario.get(key), str) or not scenario[key].strip():
                errors.append(f"Scenario {sid}: missing {key}")
        if not isinstance(scenario.get("positive"), bool):
            errors.append(f"Scenario {sid}: positive must be a boolean")
        if not scenario.get("source_ids") or not set(scenario["source_ids"]) <= source_ids:
            errors.append(f"Scenario {sid}: unresolved source IDs")
        if sid not in check_ids:
            errors.append(f"Scenario {sid}: missing tool checks")
        if scenario.get("positive") is False and not scenario.get("missing_evidence"):
            errors.append(f"Scenario {sid}: missing insufficient-data explanation")
    for check in checks:
        if not check.get("calls"):
            errors.append(f"Scenario {check['scenario_id']}: empty tool checks")
        for call in check.get("calls", []):
            read_json(workspace, call["response_file"])
            if "." not in call["tool"] or not isinstance(call["arguments"], dict):
                errors.append(f"Invalid tool call for scenario {check['scenario_id']}")
    return {"errors": errors, "scenarios": scenarios, "checks": checks,
            "positive": sum(s.get("positive") is True for s in scenarios),
            "negative": sum(s.get("positive") is False for s in scenarios)}


def check_grounding(scenarios: list[dict], invoke: Callable) -> tuple[list[str], list[dict]]:
    errors, evidence = [], []

    def call(sid, tool, arguments):
        result = invoke(tool, arguments)
        # FastMCP wraps typed return models; tools returning dicts are already flat.
        if isinstance(result, dict) and set(result) == {"result"}:
            result = result["result"]
        evidence.append({"scenario_id": sid, "tool": tool,
                         "arguments": arguments, "result": result})
        if not isinstance(result, dict) or result.get("error"):
            errors.append(f"Scenario {sid}: {tool} failed")
            return {}
        return result

    for scenario in scenarios:
        if not scenario.get("positive"):
            continue
        sid = scenario["id"]
        ground = scenario.get("grounding", {})
        site, asset = ground.get("site"), ground.get("asset_id")
        if not site or not asset:
            errors.append(f"Scenario {sid}: missing asset/site grounding")
            continue
        args = {"site_name": site, "asset_id": asset}
        detail = call(sid, "iot.asset_detail", args)
        if detail.get("asset_id") != asset or detail.get("site_name") != site:
            errors.append(f"Scenario {sid}: unresolved asset/site")
        if scenario["type"] == "Vibration":
            result = call(sid, "vibration.list_vibration_sensors", args)
            if not result.get("sensors"):
                errors.append(f"Scenario {sid}: no vibration sensors")
            continue
        sensors = ground.get("sensors", [])
        installed = call(sid, "iot.installed_sensors", args) if sensors else {}
        for sensor in sensors:
            if sensor not in installed.get("sensors", []):
                errors.append(f"Scenario {sid}: unknown sensor {sensor}")
            window = {k: ground[k] for k in ("start", "end") if ground.get(k)}
            extent = call(sid, "iot.stream_extent", {**args, "sensor": sensor, **window})
            if extent.get("total_records", 0) < 1:
                errors.append(f"Scenario {sid}: no readings for {sensor} in requested interval")
        for wonum in ground.get("workorder_ids", []):
            result = call(sid, "wo.get_workorder", {"site_id": site, "wonum": wonum})
            if result.get("work_order", {}).get("assetnum") != asset:
                errors.append(f"Scenario {sid}: work order {wonum} belongs to another asset")
    return errors, evidence


def main():
    import sys
    from mcphub import ToolUniverse

    workspace = Path("/workspace")
    report = check_contract(workspace)
    servers = {n: [sys.executable, "-m", f"servers.{n}.main"]
               for n in ("iot", "fmsr", "tsfm", "wo", "vibration")}
    with ToolUniverse(servers=servers) as client:
        client.load_tools()
        report["tools"] = client.list_tools()
        errors, evidence = check_grounding(report["scenarios"], client.run)
        report["errors"].extend(errors)
        report["grounding_checks"] = evidence
        for item in report["checks"]:
            for call in item["calls"]:
                if call["tool"] not in report["tools"]:
                    report["errors"].append(f"Unavailable tool: {call['tool']}")
    report["scope"] = "Evidence files, MCP discovery and live entity checks; human review still required"
    print(json.dumps(report, indent=2))
    return bool(report["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
