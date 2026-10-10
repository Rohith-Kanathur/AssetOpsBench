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
from .contracts import string, strings, validate_profile, validate_scenarios
from .data_grounding import validate_data_references, validate_data_sources
from .modes import MODES, validate_execution
from . import environment

SCOPE = ("Validates artifact structure, budgets, declared data lineage, source checksums, "
         "discovered tool names and read-only live entity/coverage checks. Execution is recorded "
         "by the harness; agent-written receipts are not used. These checks do not verify scenario "
         "execution, output creation or the correctness of a claimed limitation. Those require evaluation; "
         "data relevance, transformation fidelity and operator realism also require review.")


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
    report = {"errors": [], "warnings": [], "scope": SCOPE, "stage": stage,
              "scenario_execution": "not_verified"}
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
    baseline = None
    try:
        selected = environment.policy(request)
        if selected == "existing":
            baseline = environment.load_baseline()
            errors.extend(environment.audit_source(workspace, baseline))
    except (OSError, ValueError, KeyError) as exc:
        errors.append(str(exc))
        selected = request.get("environment_policy")
    report["environment_policy"] = selected
    mode = request.get("generation_mode")
    report["generation_mode"] = mode
    if mode is not None and (not string(mode) or mode not in MODES):
        errors.append("Request: invalid generation_mode")
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
        kinds = {"observed", "simulated", "derived", "synthetic"}
        if selected == "existing":
            kinds.add("fixture")
        if not string(source.get("kind")) or source["kind"] not in kinds:
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
    fixture_files = baseline["files"] if baseline is not None else {} if selected == "existing" else None
    data_errors, grounded = validate_data_sources(sources, fixture_files)
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
    mode_errors, mode_warnings = validate_execution(
        request, scenarios, lambda name: local_file(workspace, name), tools)
    errors.extend(mode_errors)
    report["warnings"].extend(mode_warnings)
    errors.extend(validate_budget(request, scenarios, load("output/allocation.json", dict)))
    scenario_errors, warnings = validate_scenarios(scenarios, source_ids, profile, tools)
    errors.extend(scenario_errors)
    report["warnings"].extend(warnings)
    report.update(scenarios=scenarios, scenario_count=len(scenarios),
                  counts={domain: sum(s.get("type") == domain for s in scenarios)
                          for domain in sorted({s.get("type") for s in scenarios if isinstance(s.get("type"), str)})})
    return report


def check_grounding(scenarios: list[dict], invoke: Callable, tools=None) -> tuple[list[str], list[dict]]:
    """Capture live evidence; the exercised reference determines its meaning.

    An empty result or an explicit not-found response is not automatically an
    invalid scenario. Infrastructure failures still block the check.
    """
    errors, evidence = [], []
    # Known absence responses from the read-only tools, not arbitrary failures.
    absence_prefixes = {
        "iot.asset_detail": ("unknown site ", "unknown asset_id "),
        "iot.measured_sensors": ("unknown site ", "unknown asset_id "),
        "iot.stream_extent": ("unknown site ", "no records for asset_id "),
        "vibration.list_vibration_sensors": ("no sensors found for asset ",),
        "vibration.get_vibration_data": ("no vibration data found for asset ",),
        "fmsr.get_failure_modes": ("no failure_mode record for asset_class ",),
        "wo.get_workorder": ("work order '",),
    }

    def call(sid, tool, arguments):
        if tools is not None and tool not in tools:
            errors.append(f"Scenario {sid}: unavailable grounding tool {tool}")
            return {}
        try:
            result = invoke(tool, arguments)
            if isinstance(result, dict) and set(result) == {"result"}:
                result = result["result"]
            evidence.append({"scenario_id": sid, "tool": tool, "arguments": arguments, "result": result})
            if not isinstance(result, dict):
                errors.append(f"Scenario {sid}: {tool} failed")
                return {}
            if result.get("error"):
                message = str(result["error"]).lower()
                absent = message.startswith(absence_prefixes.get(tool, ()))
                if tool == "wo.get_workorder":
                    absent = absent and "not found in site" in message
                if not absent:
                    errors.append(f"Scenario {sid}: {tool} failed")
            return result
        except Exception as exc:
            errors.append(f"Scenario {sid}: {tool} raised {type(exc).__name__}")
            return {}

    for scenario in scenarios:
        sid, ground = scenario.get("id"), scenario.get("grounding", {})
        if not isinstance(ground, dict):
            errors.append(f"Scenario {sid}: missing grounding")
            continue
        if ground.get("scope") == "class":
            if scenario.get("type") == "fmsr":
                call(sid, "fmsr.get_failure_modes", {"asset_class": ground.get("asset_class")})
            continue
        site, asset = ground.get("site"), ground.get("asset_id")
        if not string(site) or not string(asset):
            errors.append(f"Scenario {sid}: missing asset/site grounding")
            continue
        args = {"site_name": site, "asset_id": asset}
        call(sid, "iot.asset_detail", args)
        sensors = ground.get("sensors", [])
        if not strings(sensors):
            continue
        if str(scenario.get("type", "")).lower() == "vibration":
            call(sid, "vibration.list_vibration_sensors", args)
            for sensor in sensors:
                if ground.get("start"):
                    call(sid, "vibration.get_vibration_data", {**args, "sensor_name": sensor,
                         "start": ground["start"], "final": ground.get("end")})
        elif sensors:
            call(sid, "iot.measured_sensors", args)
            for sensor in sensors:
                window = {k: ground[k] for k in ("start", "end") if ground.get(k)}
                call(sid, "iot.stream_extent", {**args, "sensor": sensor, **window})
        for wonum in ground.get("workorder_ids", []):
            result = call(sid, "wo.get_workorder", {"site_id": site, "wonum": wonum})
            order_asset = result.get("work_order", {}).get("assetnum")
            if order_asset and order_asset != ground.get("workorder_asset_id", asset):
                errors.append(f"Scenario {sid}: work order {wonum} belongs to another asset")
    return errors, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--stage", choices=("profile", "all"), default="all")
    args = parser.parse_args(argv)
    report = check_contract(args.workspace, args.stage)
    if report.get("environment_policy") == "existing" and args.stage == "all":
        try:
            report["errors"].extend(environment.audit_database(environment.load_baseline(), environment.database_state()))
        except Exception as exc:
            report["errors"].append(f"Existing environment database check failed: {type(exc).__name__}: {exc}")
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
