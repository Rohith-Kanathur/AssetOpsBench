"""Evaluation capabilities, independent of the generator's own harness."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from .contracts import TOOL, string, strings

MODES = ("general-execution",)
DEFAULT_MODE = "general-execution"
GAP_KINDS = {"missing_data", "unknown_asset", "unknown_site", "missing_sensor",
             "missing_history", "unsupported_diagnosis", "unsupported_model",
             "insufficient_coverage", "conflicting_evidence"}


def write_guidance(workspace: Path) -> None:
    """Append the selected evaluation contract after copying the base prompts."""
    mode = json.loads((workspace / "request.json").read_text()).get("generation_mode")
    if mode not in MODES:
        raise ValueError("Generation mode is missing or invalid; start a new generation")
    guidance = (Path(__file__).parent / "prompts" / f"{mode}.md").read_text()
    for name in ("profile.md", "generate.md"):
        with (workspace / name).open("a") as handle:
            handle.write("\n" + guidance)


def validate_execution(request, scenarios, local_file, tools=None):
    """Validate required capabilities and file contracts, not claimed executions."""
    errors, warnings = [], []
    mode = request.get("generation_mode")
    if mode is None:
        return errors, ["Legacy run: no generation mode; evaluation capability compatibility is unverified"]
    if not string(mode) or mode not in MODES:
        return ["Request: invalid generation_mode"], warnings
    for row in scenarios:
        label = f"Scenario {row.get('id')}"
        execution = row.get("execution")
        if not isinstance(execution, dict):
            errors.append(f"{label}: execution capability and file declarations are required")
            continue
        requires = execution.get("requires")
        if not strings(requires, True) or not set(requires) <= {"mcp", "general-execution"}:
            errors.append(f"{label}: execution.requires must list mcp and/or general-execution")
            requires = []
        inputs = execution.get("input_files")
        if not strings(inputs):
            errors.append(f"{label}: execution.input_files must be a path list")
            inputs = []
        for path in inputs:
            try:
                local_file(path)
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"{label}: invalid input file: {exc}")
        outputs = execution.get("output_files")
        if not isinstance(outputs, list):
            errors.append(f"{label}: execution.output_files must be a list")
            outputs = []
        for output in outputs:
            if not isinstance(output, dict) or not string(output.get("path")) or not string(output.get("created_by")):
                errors.append(f"{label}: output files require path and created_by")
                continue
            path, producer = output["path"], output["created_by"]
            target = PurePosixPath(path)
            if target.is_absolute() or ".." in target.parts or str(target) == ".":
                errors.append(f"{label}: output paths must be workspace-relative files: {path}")
            if str(target) in {str(PurePosixPath(p)) for p in inputs}:
                errors.append(f"{label}: output file cannot also be a supplied input: {path}")
            if producer == "general-execution":
                if "general-execution" not in requires:
                    errors.append(f"{label}: output {path} requires general-execution capability")
            elif not TOOL.fullmatch(producer) or (tools is not None and producer not in tools):
                errors.append(f"{label}: unavailable MCP output producer {producer}")
            elif "mcp" not in requires:
                errors.append(f"{label}: output {path} requires mcp capability")
        if row.get("positive") is False:
            missing = row.get("missing_evidence", [])
            for item in missing if isinstance(missing, list) else []:
                if isinstance(item, dict) and (not string(item.get("kind")) or item["kind"] not in GAP_KINDS):
                    errors.append(f"{label}: negative evidence requires a domain gap kind; missing file execution is not a valid negative")
    return errors, warnings
