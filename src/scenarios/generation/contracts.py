"""Typed generation contracts; evidence linkage is distinct from execution proof."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .budget import DOMAINS

RESEARCH = ("diagnostics", "maintenance", "sensors", "failure_modes", "standards", "operational_tasks")
TOOL = re.compile(r"\b([a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)\b")


def string(value):
    return isinstance(value, str) and bool(value.strip())


def strings(value, nonempty=False):
    return isinstance(value, list) and (bool(value) or not nonempty) and all(string(x) for x in value)


def references(value, source_ids, label):
    errors = []
    if isinstance(value, dict):
        if "source_ids" in value and (not strings(value["source_ids"]) or
                                       not set(value["source_ids"]) <= source_ids):
            errors.append(f"{label}: unresolved source IDs")
        for key, item in value.items():
            errors.extend(references(item, source_ids, f"{label}.{key}"))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            errors.extend(references(item, source_ids, f"{label}[{i}]"))
    return errors


def tool_refs(text, tools=None):
    prefixes = set(DOMAINS) | {t.split(".")[0] for t in (tools or [])}
    return {t for t in TOOL.findall(text) if t.split(".")[0] in prefixes}


def validate_profile(profile, source_ids, tools=None):
    errors, warnings = [], []
    if not isinstance(profile, dict):
        return ["Profile must be an object"], warnings
    for key in ("asset_class", "description"):
        if not string(profile.get(key)):
            errors.append(f"Profile: {key} must be a nonempty string")
    for key in ("operator_tasks", "manager_tasks"):
        value = profile.get(key)
        if not isinstance(value, list) or not value or not all(string(x) or isinstance(x, dict) and string(x.get("description", x.get("task"))) for x in value):
            errors.append(f"Profile: {key} must contain described tasks")
    for key in ("assets", "failure_modes", "sensor_mapping", "gaps"):
        if not isinstance(profile.get(key), list):
            errors.append(f"Profile: {key} must be a list")
    for i, asset in enumerate(profile.get("assets", []) if isinstance(profile.get("assets"), list) else []):
        label = f"Profile asset {i}"
        if not isinstance(asset, dict) or not string(asset.get("site")) or not string(asset.get("asset_id")):
            errors.append(f"{label}: site and asset_id are required")
            continue
        coverage = [asset[k] for k in ("iot", "vibration") if k in asset] or [asset]
        for item in coverage:
            if not isinstance(item, dict) or not strings(item.get("sensors")):
                errors.append(f"{label}: coverage requires a sensors list")
                continue
            for key in ("start", "end"):
                if item.get(key) is not None and not string(item[key]):
                    errors.append(f"{label}: {key} must be a timestamp string or null")
            count = item.get("total_observations", item.get("rows"))
            if count is not None and (type(count) is not int or count < 0):
                errors.append(f"{label}: observation count must be nonnegative")
        if not strings(asset.get("source_ids"), True):
            errors.append(f"{label}: evidence source_ids are required")
    for key in ("failure_modes", "sensor_mapping"):
        for i, item in enumerate(profile.get(key, []) if isinstance(profile.get(key), list) else []):
            if not isinstance(item, dict) or not strings(item.get("source_ids"), True):
                errors.append(f"Profile {key}[{i}]: an object with evidence source_ids is required")
    caps = profile.get("available_capabilities")
    if not isinstance(caps, dict):
        errors.append("Profile: available_capabilities must be an object")
    else:
        for domain, entries in caps.items():
            if domain not in DOMAINS and not isinstance(entries, list):
                continue  # Counts and notes may accompany domain lists.
            if not isinstance(entries, list):
                errors.append(f"Profile capabilities {domain}: must be a list")
                continue
            for item in entries:
                name = item if isinstance(item, str) else item.get("tool") if isinstance(item, dict) else None
                if not string(name) or not TOOL.fullmatch(name):
                    errors.append(f"Profile capabilities {domain}: invalid tool")
                elif tools is not None and name not in tools:
                    errors.append(f"Unavailable profile tool: {name}")
    research = profile.get("research")
    if not isinstance(research, dict):
        errors.append("Profile: research must be an object")
    else:
        for key in RESEARCH:
            area = research.get(key)
            if not isinstance(area, dict) or not string(area.get("status")) or area["status"] not in {"supported", "gap", "not_applicable"} or not string(area.get("summary")) or not strings(area.get("source_ids")):
                errors.append(f"Profile research {key}: status, summary and source_ids are required")
            elif area["status"] == "supported" and not area["source_ids"]:
                errors.append(f"Profile research {key}: supported claims require evidence")
    errors.extend(references(profile, source_ids, "Profile"))
    return errors, warnings


def validate_scenarios(scenarios, source_ids, profile, checks, tools=None):
    errors, warnings, seen = [], [], set()
    gaps = {g.get("id") for g in profile.get("gaps", []) if isinstance(g, dict) and string(g.get("id"))} if isinstance(profile, dict) else set()
    texts = []
    for row in scenarios:
        sid = row.get("id")
        label = f"Scenario {sid}"
        if type(sid) not in (str, int) or isinstance(sid, str) and not sid.strip():
            errors.append("Scenario IDs must be integers or nonempty strings")
            sid = None
        elif sid in seen:
            errors.append("Duplicate scenario IDs")
        seen.add(sid)
        for key in ("text", "category", "characteristic_form"):
            if not string(row.get(key)):
                errors.append(f"{label}: {key} must be a nonempty string")
        text, rubric = row.get("text", ""), row.get("characteristic_form", "")
        if isinstance(text, str):
            if tool_refs(text, tools):
                errors.append(f"{label}: text contains qualified API references")
            normalized = re.sub(r"\W+", " ", text.casefold()).strip()
            for prior_id, prior in texts:
                if normalized == prior:
                    errors.append(f"{label}: duplicate text with scenario {prior_id}")
                elif len(normalized.split()) >= 8 and SequenceMatcher(None, prior, normalized).ratio() >= .94:
                    errors.append(f"{label}: near-duplicate text with scenario {prior_id}")
            texts.append((sid, normalized))
        if isinstance(rubric, str):
            names = set(TOOL.findall(rubric)) if tools is None else tool_refs(rubric, tools)
            if not names:
                errors.append(f"{label}: characteristic_form must name concrete qualified tools")
            if tools is not None:
                errors.extend(f"{label}: unavailable rubric tool {n}" for n in sorted(names - set(tools)))
        if type(row.get("positive")) is not bool:
            errors.append(f"{label}: positive must be a boolean")
        if not strings(row.get("source_ids"), True) or not set(row.get("source_ids", []) if strings(row.get("source_ids")) else []) <= source_ids:
            errors.append(f"{label}: unresolved source IDs")
        ground = row.get("grounding")
        if not isinstance(ground, dict):
            errors.append(f"{label}: grounding must be an object")
            ground = {}
        scope = ground.get("scope")
        if not string(scope) or scope not in {"asset", "class"}:
            errors.append(f"{label}: grounding.scope must be asset or class")
        if scope == "class":
            if not string(ground.get("asset_class")):
                errors.append(f"{label}: class grounding requires asset_class")
            if row.get("type") != "fmsr" and not string(ground.get("justification")):
                errors.append(f"{label}: class scope outside FMSR requires justification")
            if ground.get("sensors") or ground.get("workorder_ids") or ground.get("output_workorder_ids"):
                errors.append(f"{label}: class scope cannot hide asset-specific references")
        if scope == "asset" and (not string(ground.get("site")) or not string(ground.get("asset_id"))):
            errors.append(f"{label}: asset grounding requires site and asset_id")
        for key in ("sensors", "workorder_ids", "output_workorder_ids"):
            if key in ground and not strings(ground[key]):
                errors.append(f"{label}: grounding.{key} must be a string list")
        for key in ("start", "end"):
            if ground.get(key) is not None and not string(ground[key]):
                errors.append(f"{label}: grounding.{key} must be a string or null")
        if scope == "asset" and isinstance(text, str) and string(ground.get("asset_id")):
            if ground["asset_id"].casefold() not in text.casefold():
                warnings.append(f"{label}: asset identifier is not explicit in scenario text; review naming consistency")
            assets = profile.get("assets", []) if isinstance(profile, dict) else []
            other = [a.get("asset_id") for a in assets if isinstance(a, dict) and string(a.get("asset_id")) and a["asset_id"] != ground["asset_id"] and a["asset_id"].casefold() in text.casefold()]
            if other and ground["asset_id"].casefold() not in text.casefold():
                errors.append(f"{label}: text references {other} instead of grounded asset")
        receipts = [c for c in checks if c.get("scenario_id") == sid]
        calls = [c for receipt in receipts for c in receipt.get("calls", []) if isinstance(c, dict)]
        if not calls:
            errors.append(f"{label}: missing tool checks")
        if row.get("positive") is True and row.get("type") == "multiagent":
            servers = {c.get("tool", "").split(".")[0] for c in calls if isinstance(c.get("tool"), str)}
            if len(servers & (set(DOMAINS) - {"multiagent"})) < 2:
                errors.append(f"{label}: positive multiagent tool checks must span at least two domain servers")
        if row.get("positive") is False:
            missing = row.get("missing_evidence")
            if not isinstance(missing, list) or not missing:
                errors.append(f"{label}: missing_evidence must declare missing dependencies and receipts")
            else:
                response_files = {c.get("response_file") for c in calls if string(c.get("response_file"))}
                for item in missing:
                    if not isinstance(item, dict) or not string(item.get("dependency")) or not string(item.get("reason")):
                        errors.append(f"{label}: missing evidence requires dependency and reason")
                        continue
                    refs = item.get("response_files")
                    if not strings(refs, True) or not set(refs) <= response_files:
                        errors.append(f"{label}: missing evidence must cite this scenario's response files")
                    if "gap_ids" in item and (not strings(item["gap_ids"]) or not set(item["gap_ids"]) <= gaps):
                        errors.append(f"{label}: unresolved missing-evidence gap IDs")
    return errors, warnings
