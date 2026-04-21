"""Scenario-generation utilities shared across pipeline stages."""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

from .text import normalize_example_fingerprint

_log = logging.getLogger(__name__)

_SCENARIOS_ROOT = Path(__file__).resolve().parent

_HF = _SCENARIOS_ROOT / "huggingface"
_ALL_UTTERANCE = _HF / "scenarios" / "all_utterance.jsonl"
_FAILURE_MAPPING = _HF / "task" / "failure_mapping_senarios.jsonl"
_RULE_MONITORING = _HF / "task" / "rule_monitoring_scenarios.jsonl"
_COMPRESSOR = _HF / "asset" / "compressor_utterance.jsonl"
_HYDRAULIC_PUMP = _HF / "asset" / "hydrolicpump_utterance.jsonl"
_LOCAL_VIBRATION = _SCENARIOS_ROOT / "local" / "vibration_utterance.json"

_ASSET_KEYWORD_SUBSTRINGS = (
    "chiller",
    "ahu",
    "wind turbine",
    "equipment",
)
_ASSET_ID_PATTERN = re.compile(r"\b[A-Z]{2,6}\d{4,}\b")

_GENERIC_UTTERANCE_DENYLIST = (
    "what iot sites are available",
    "can you list the iot sites",
    "list the iot sites",
    "is lstm model supported",
    "lstm model supported in tsfm",
    "what sites are available",
)

_TSFM_ENTITY_ORDER = (
    "Chiller",
    "CRAC",
    "Boiler",
    "AHU",
    "Cooling Tower",
    "HXU",
    "Pump",
    "SVL",
)


def parse_llm_json(raw: str) -> tuple[Any, str | None]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)
        if text.lower().startswith("json"):
            text = text[4:]

    text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        start_obj = text.find("{")
        start_arr = text.find("[")
        if start_obj == -1 and start_arr == -1:
            return None, f"No JSON start character found. Error: {exc}"

        start = (
            start_obj
            if start_arr == -1 or (start_obj != -1 and start_obj < start_arr)
            else start_arr
        )
        end_char = "}" if start == start_obj else "]"
        end = text.rfind(end_char) + 1

        if start != -1 and end > 0:
            try:
                return json.loads(text[start:end]), None
            except json.JSONDecodeError as inner_exc:
                return None, f"Failed to parse inner JSON block. Error: {inner_exc}"
    return None, "Unknown parsing error."


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        _log.warning("Few-shot file missing: %s", path)
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            _log.warning("Skip bad JSONL line in %s: %s", path, exc)
    return rows


def _load_local_vibration_list() -> list[dict[str, Any]]:
    if not _LOCAL_VIBRATION.exists():
        return []
    try:
        raw = json.loads(_LOCAL_VIBRATION.read_text(encoding="utf-8"))
    except Exception as exc:
        _log.warning("Failed to load local vibration few-shot file: %s", exc)
        return []
    items = raw if isinstance(raw, list) else raw.get("train", [])
    return [dict(item) for item in items] if isinstance(items, list) else []


def _is_asset_specific_utterance(text: str) -> bool:
    tl = text.lower()
    for phrase in _GENERIC_UTTERANCE_DENYLIST:
        if phrase in tl:
            return False
    if any(k in tl for k in _ASSET_KEYWORD_SUBSTRINGS):
        return True
    if _ASSET_ID_PATTERN.search(text):
        return True
    return False


_FMSR_FEWSHOT_BUCKET_ORDER: tuple[str, ...] = (
    "list_all",
    "for_list_all",
    "what",
    "which",
    "for_which",
)


def _failure_mapping_bucket(text: str) -> str | None:
    s = text.strip()
    sl = s.lower()
    if sl.startswith("list all"):
        return "list_all"
    if sl.startswith("for ") and "list all" in sl:
        return "for_list_all"
    if sl.startswith("what "):
        return "what"
    if sl.startswith("which "):
        return "which"
    if sl.startswith("for ") and re.search(r"\bwhich\b", sl):
        return "for_which"
    return None


def _pick_failure_mapping_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {key: [] for key in _FMSR_FEWSHOT_BUCKET_ORDER}
    for row in rows:
        text = str(row.get("text", ""))
        b = _failure_mapping_bucket(text)
        if b and b in buckets:
            buckets[b].append(row)
    out: list[dict[str, Any]] = []
    for key in _FMSR_FEWSHOT_BUCKET_ORDER:
        if buckets[key]:
            out.append(random.choice(buckets[key]))
    return out


def _pick_rule_monitoring_diverse(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    used_ids: set[int | str] = set()

    for target in _TSFM_ENTITY_ORDER:
        candidates: list[dict[str, Any]] = []
        if target == "SVL":
            for row in rows:
                rid = row.get("id")
                if rid in used_ids:
                    continue
                t = str(row.get("text", ""))
                if re.search(r"\bSVL\b", t, re.IGNORECASE):
                    candidates.append(row)
        else:
            for row in rows:
                rid = row.get("id")
                if rid in used_ids:
                    continue
                if str(row.get("entity", "")).strip() == target:
                    candidates.append(row)
        if candidates:
            choice = random.choice(candidates)
            picked.append(choice)
            used_ids.add(choice.get("id"))
    return picked


def _normalize_fewshot_row(
    row: dict[str, Any], source_config: str
) -> dict[str, Any]:
    text = str(row.get("text", "")).strip()
    category = str(row.get("category", "")).strip() or "Knowledge Query"
    characteristic_form = str(row.get("characteristic_form", "")).strip()
    return {
        "id": row.get("id"),
        "text": text,
        "category": category,
        "characteristic_form": characteristic_form,
        "entity": str(row.get("entity", "") or "").strip(),
        "group": str(row.get("group", "") or "").strip(),
        "note": str(row.get("note", "") or "").strip(),
        "hint": str(row.get("hint", "") or "").strip(),
        "source_config": source_config,
        "source_type": str(row.get("type", "") or "").strip(),
    }


def _to_prompt_dict(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": example.get("text", ""),
        "category": example.get("category", ""),
        "characteristic_form": example.get("characteristic_form", ""),
        "source_config": example.get("source_config", ""),
    }


def _build_candidate_pool(
    focus: str,
) -> list[dict[str, Any]]:
    focus_lower = (focus or "").lower()
    if focus_lower == "iot":
        rows = _load_jsonl(_ALL_UTTERANCE)
        pool: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("type", "")).strip() != "IoT":
                continue
            text = str(row.get("text", ""))
            if not _is_asset_specific_utterance(text):
                continue
            pool.append(_normalize_fewshot_row(row, "all_utterance_iot"))
        return pool

    if focus_lower == "fmsr":
        rows = _load_jsonl(_FAILURE_MAPPING)
        picked = _pick_failure_mapping_examples(rows)
        return [_normalize_fewshot_row(r, "failure_mapping") for r in picked]

    if focus_lower == "tsfm":
        rows = _load_jsonl(_RULE_MONITORING)
        picked = _pick_rule_monitoring_diverse(rows)
        return [_normalize_fewshot_row(r, "rule_monitoring") for r in picked]

    if focus_lower == "wo":
        rows = _load_jsonl(_ALL_UTTERANCE)
        pool = []
        for row in rows:
            if str(row.get("type", "")).strip() != "Workorder":
                continue
            text = str(row.get("text", ""))
            if not _is_asset_specific_utterance(text):
                continue
            pool.append(_normalize_fewshot_row(row, "all_utterance_wo"))
        return pool

    if focus_lower == "vibration":
        items = _load_local_vibration_list()
        return [
            _normalize_fewshot_row(dict(item), "local_vibration") for item in items
        ]

    if focus_lower == "multiagent":
        pool = []
        for path, tag in (
            (_COMPRESSOR, "compressor_utterance"),
            (_HYDRAULIC_PUMP, "hydrolicpump_utterance"),
        ):
            for row in _load_jsonl(path):
                pool.append(_normalize_fewshot_row(row, tag))
        for row in _load_jsonl(_ALL_UTTERANCE):
            if str(row.get("type", "")).strip() != "multiagent":
                continue
            text = str(row.get("text", ""))
            if not _is_asset_specific_utterance(text):
                continue
            pool.append(_normalize_fewshot_row(row, "all_utterance_multiagent"))
        return pool

    _log.warning("Unknown focus %r for few-shot; returning empty pool.", focus)
    return []


def fetch_hf_fewshot(
    focus: str | None = None,
    max_examples: int = 6,
    seed: int | None = None,
) -> list[dict]:
    if max_examples <= 0:
        return []

    if seed is not None:
        random.seed(seed)

    pool = _build_candidate_pool(focus)
    if not pool:
        _log.info("No few-shot examples were available for focus %r.", focus)
        return []

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for example in pool:
        fp = normalize_example_fingerprint(example.get("text", ""))
        if not fp or fp in seen:
            continue
        seen.add(fp)
        unique.append(example)

    random.shuffle(unique)
    selected = unique[:max_examples]
    return [_to_prompt_dict(ex) for ex in selected]


__all__ = ["fetch_hf_fewshot", "parse_llm_json"]
