"""Completion gates for saved contracts, discovered tools and safe live references."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

from .budget import validate_budget
from .contracts import TOOL, string, strings, validate_profile, validate_scenarios
from .data_grounding import validate_data_references, validate_data_sources

SCOPE = ("Validates artifact structure, budgets, declared data lineage, evidence links/checksums, discovered tool names "
         "and read-only live entity/coverage checks. Saved response files are declarations, "
         "not independent proof of execution or diagnosis. Data relevance, transformation fidelity "
         "and semantic/operator review are still required.")


def local_file(workspace: Path, name: str) -> Path:
    if not string(name):
        raise ValueError("Evidence path must be a nonempty string")
    if Path(name).is_absolute():
        raise ValueError(f"Evidence path must be workspace-relative: {name}")
    path = (workspace / name).resolve()
    if not path.is_relative_to(workspace.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-workspace evidence file: {name}")
    return path


def read_json(workspace: Path, name: str):
    return json.loads(local_file(workspace, name).read_text())


def check_contract(workspace: Path, stage="all", tools=None) -> dict:
    report = {"errors": [], "warnings": [], "scope": SCOPE, "stage": stage}
    errors = report["errors"]

    def load(name, kind):
        try:
            value = read_json(workspace, name)
            if not isinstance(value, kind):
                raise ValueError(f"{name} must contain a {kind.__name__}")
            return value
        except (OSError, ValueError, TypeError) as exc:
            errors.append(str(exc))
            return kind()

    request = load("request.json", dict)
    profile = load("output/profile.json", dict)
    sources = load("output/sources.json", list)
    source_ids = set()
    for source in sources:
        if not isinstance(source, dict) or not string(source.get("id")):
            errors.append("Sources must be objects with string IDs")
            continue
        sid = source["id"]
        if sid in source_ids:
            errors.append("Duplicate source IDs")
        source_ids.add(sid)
        url = source.get("url")
        if not string(source.get("kind")) or source["kind"] not in {"observed", "simulated", "derived", "synthetic"}:
            errors.append(f"Source {sid}: invalid kind")
        if source.get("kind") != "synthetic" and (not string(url) or not url.startswith(("https://", "http://", "repository:"))):
            errors.append(f"Missing source URL: {sid}")
        files = source.get("files")
        if isinstance(files, dict):
            files = [{"path": k, "sha256": v} for k, v in files.items()]
        if not isinstance(files, list) or not files:
            errors.append(f"Source {sid}: evidence files are required")
            continue
        for evidence in files:
            try:
                if not isinstance(evidence, dict) or not string(evidence.get("sha256")):
                    raise ValueError(f"Source {sid}: evidence path and sha256 are required")
                path = local_file(workspace, evidence.get("path"))
                if hashlib.sha256(path.read_bytes()).hexdigest() != evidence["sha256"]:
                    errors.append(f"Source checksum mismatch: {evidence['path']}")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(str(exc))
    data_errors, grounded = validate_data_sources(sources)
    errors.extend(data_errors)
    errors.extend(validate_data_references(profile, grounded))
    profile_errors, warnings = validate_profile(profile, source_ids, tools)
    errors.extend(profile_errors)
    report["warnings"].extend(warnings)
    if string(request.get("asset_class")) and string(profile.get("asset_class")) and request["asset_class"].casefold() != profile["asset_class"].casefold():
        errors.append("Profile asset_class does not match request")
    report["profile"] = profile
    if stage == "profile":
        return report
    for name in ("output/README.md", "output/environment.md", "output/requirements.txt"):
        try:
            local_file(workspace, name)
        except (OSError, ValueError, TypeError) as exc:
            errors.append(str(exc))
    raw_scenarios = load("output/scenarios.json", list)
    scenarios = [s for s in raw_scenarios if isinstance(s, dict)]
    if len(scenarios) != len(raw_scenarios):
        errors.append("Scenarios must be objects")
    errors.extend(validate_data_references({}, grounded, scenarios))
    raw_checks = load("output/tool_checks.json", list)
    checks = []
    for item in raw_checks:
        if not isinstance(item, dict) or type(item.get("scenario_id")) not in (str, int) or not isinstance(item.get("calls"), list):
            errors.append("Tool checks require scenario_id and a calls list")
            continue
        checks.append(item)
        if not item["calls"]:
            errors.append(f"Scenario {item['scenario_id']}: empty tool checks")
        for call in item["calls"]:
            if not isinstance(call, dict):
                errors.append(f"Scenario {item['scenario_id']}: invalid tool receipt")
                continue
            tool = call.get("tool")
            if not string(tool) or not TOOL.fullmatch(tool) or not isinstance(call.get("arguments"), dict):
                errors.append(f"Invalid tool call for scenario {item['scenario_id']}")
            elif tools is not None and tool not in tools:
                errors.append(f"Unavailable tool: {tool}")
            if string(tool) and tool.split(".")[-1].startswith(("create_", "add_", "update_", "delete_", "save_", "close_", "assign_", "approve_", "set_", "cancel_", "generate_work_order")):
                context = item.get("write_context", {})
                try:
                    if not isinstance(context, dict) or context.get("environment") != "disposable":
                        raise ValueError(f"Scenario {item['scenario_id']}: write receipt requires disposable write_context")
                    saved_context = read_json(workspace, context.get("evidence_file"))
                    if not isinstance(saved_context, dict) or not saved_context:
                        raise ValueError(f"Scenario {item['scenario_id']}: empty write-context evidence")
                except (OSError, ValueError, TypeError) as exc:
                    errors.append(str(exc))
            try:
                call_result = read_json(workspace, call.get("response_file"))
                if call_result is None:
                    errors.append(f"Scenario {item['scenario_id']}: null response is not a retained tool result")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(str(exc))
    valid_ids = [s.get("id") for s in scenarios if type(s.get("id")) in (str, int)]
    for check in checks:
        if check["scenario_id"] not in valid_ids:
            errors.append(f"Tool check references unknown scenario {check['scenario_id']}")
    errors.extend(validate_budget(request, scenarios, load("output/allocation.json", dict)))
    scenario_errors, warnings = validate_scenarios(scenarios, source_ids, profile, checks, tools)
    errors.extend(scenario_errors)
    report["warnings"].extend(warnings)
    for scenario in scenarios:
        ground = scenario.get("grounding", {})
        if not isinstance(ground, dict) or not strings(ground.get("output_workorder_ids", [])):
            continue
        for wonum in ground.get("output_workorder_ids", []):
            found = False
            for check in checks:
                for call in check["calls"]:
                    if check["scenario_id"] != scenario.get("id") or not isinstance(call, dict) or call.get("tool") != "wo.get_workorder" or not isinstance(call.get("arguments"), dict) or call["arguments"].get("wonum") != wonum:
                        continue
                    try:
                        result = read_json(workspace, call.get("response_file"))
                        if isinstance(result, dict):
                            result = result.get("result", result)
                            order = result.get("work_order", {}) if isinstance(result, dict) else {}
                            found |= isinstance(order, dict) and order.get("wonum") == wonum and order.get("assetnum") == ground.get("workorder_asset_id", ground.get("asset_id")) and order.get("siteid") == ground.get("site")
                    except (OSError, ValueError, TypeError):
                        pass  # Receipt loading errors are already reported above.
            if not found:
                errors.append(f"Scenario {scenario.get('id')}: output work order {wonum} lacks matching read-back evidence")
    report.update(scenarios=scenarios, checks=checks,
                  positive=sum(s.get("positive") is True for s in scenarios),
                  negative=sum(s.get("positive") is False for s in scenarios))
    return report


def check_grounding(scenarios: list[dict], invoke: Callable, tools=None) -> tuple[list[str], list[dict]]:
    """Recheck read-only references. Never replay receipts or output-creation calls."""
    errors, evidence = [], []

    def call(sid, tool, arguments):
        if tools is not None and tool not in tools:
            errors.append(f"Scenario {sid}: unavailable grounding tool {tool}")
            return {}
        try:
            result = invoke(tool, arguments)
            if isinstance(result, dict) and set(result) == {"result"}:
                result = result["result"]
            evidence.append({"scenario_id": sid, "tool": tool, "arguments": arguments, "result": result})
            if not isinstance(result, dict) or result.get("error"):
                errors.append(f"Scenario {sid}: {tool} failed")
                return {}
            return result
        except Exception as exc:
            errors.append(f"Scenario {sid}: {tool} raised {type(exc).__name__}")
            return {}

    for scenario in scenarios:
        if scenario.get("positive") is not True:
            continue  # Negative dependencies are checked via their declared receipts/gaps.
        sid, ground = scenario.get("id"), scenario.get("grounding", {})
        if not isinstance(ground, dict):
            errors.append(f"Scenario {sid}: missing grounding")
            continue
        if ground.get("scope") == "class":
            if scenario.get("type") == "fmsr":
                result = call(sid, "fmsr.get_failure_modes", {"asset_class": ground.get("asset_class")})
                if not result.get("failure_modes"):
                    errors.append(f"Scenario {sid}: no stored failure modes for class")
            continue
        site, asset = ground.get("site"), ground.get("asset_id")
        if not string(site) or not string(asset):
            errors.append(f"Scenario {sid}: missing asset/site grounding")
            continue
        args = {"site_name": site, "asset_id": asset}
        detail = call(sid, "iot.asset_detail", args)
        if detail.get("asset_id") != asset or detail.get("site_name") != site:
            errors.append(f"Scenario {sid}: unresolved asset/site")
        sensors = ground.get("sensors", [])
        if not strings(sensors):
            continue
        if str(scenario.get("type", "")).lower() == "vibration":
            result = call(sid, "vibration.list_vibration_sensors", args)
            if not result.get("sensors"):
                errors.append(f"Scenario {sid}: no vibration sensors")
            for sensor in sensors:
                if sensor not in result.get("sensors", []):
                    errors.append(f"Scenario {sid}: unknown vibration sensor {sensor}")
                if ground.get("start"):
                    data = call(sid, "vibration.get_vibration_data", {**args, "sensor_name": sensor,
                                "start": ground["start"], "final": ground.get("end")})
                    if not data.get("data_id"):
                        errors.append(f"Scenario {sid}: no vibration readings for {sensor} in requested interval")
        else:
            measured = call(sid, "iot.measured_sensors", args) if sensors else {}
            for sensor in sensors:
                if sensor not in measured.get("sensors", []):
                    errors.append(f"Scenario {sid}: unknown sensor {sensor}")
                window = {k: ground[k] for k in ("start", "end") if ground.get(k)}
                extent = call(sid, "iot.stream_extent", {**args, "sensor": sensor, **window})
                if not isinstance(extent.get("total_records"), (int, float)) or extent["total_records"] < 1:
                    errors.append(f"Scenario {sid}: no readings for {sensor} in requested interval")
        for wonum in ground.get("workorder_ids", []):
            result = call(sid, "wo.get_workorder", {"site_id": site, "wonum": wonum})
            if result.get("work_order", {}).get("assetnum") != ground.get("workorder_asset_id", asset):
                errors.append(f"Scenario {sid}: work order {wonum} belongs to another asset")
    return errors, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--stage", choices=("profile", "all"), default="all")
    args = parser.parse_args(argv)
    report = check_contract(args.workspace, args.stage)
    if not report["errors"]:
        try:
            from mcphub import ToolUniverse
            servers = {n: [sys.executable, "-m", f"servers.{n}.main"] for n in ("iot", "fmsr", "tsfm", "wo", "vibration", "utilities")}
            with contextlib.redirect_stdout(sys.stderr), ToolUniverse(servers=servers) as client:
                client.load_tools()
                tools = client.list_tools()
                report = check_contract(args.workspace, args.stage, tools)
                report["tools"] = tools
                if args.stage == "all" and not report["errors"]:
                    errors, evidence = check_grounding(report["scenarios"], client.run, tools)
                    report["errors"].extend(errors)
                    report["grounding_checks"] = evidence
        except Exception as exc:
            report["errors"].append(f"Live tool discovery/check failed: {type(exc).__name__}: {exc}")
    print(json.dumps(report, indent=2))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
