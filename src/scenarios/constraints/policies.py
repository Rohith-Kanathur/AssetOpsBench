"""Scenario focus policies and prompt-formatting helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json

from ..models import AssetProfile


def _format_bullet_list(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


# Shared across all focuses: injected into prompts via forbidden_patterns.
FORBIDDEN_MCP_NAMES_IN_SCENARIO_TEXT = (
    "Do not put MCP tool names, function names, or parenthetical examples like "
    '"e.g. get_failure_modes tool" in the scenario text; put those only in characteristic_form.'
)

FOCUS_ORDER: tuple[str, ...] = ("iot", "fmsr", "tsfm", "wo", "vibration", "multiagent")


@dataclass(frozen=True)
class FocusPolicy:
    """Prompt and validation guidance for one focus."""

    categories: tuple[str, ...]
    prompt_requirements: tuple[str, ...]
    forbidden_patterns: tuple[str, ...]

    def format_categories(self) -> str:
        return _format_bullet_list(self.categories)

    def format_requirements(self) -> str:
        return _format_bullet_list(self.prompt_requirements)

    def format_forbidden_patterns(self) -> str:
        return _format_bullet_list(self.forbidden_patterns)


SCENARIO_POLICIES: dict[str, FocusPolicy] = {
    "iot": FocusPolicy(
        categories=("Data Query", "Knowledge Query"),
        prompt_requirements=(
            "Keep the primary burden on telemetry retrieval, asset discovery, sensor discovery, or historical observations.",
            "Supporting FMSR, TSFM, vibration, or WO steps are allowed, but IoT must remain the main focus.",
        ),
        forbidden_patterns=(
            FORBIDDEN_MCP_NAMES_IN_SCENARIO_TEXT,
            "Avoid turning a primary IoT scenario into a generic asset essay with no telemetry task.",
        ),
    ),
    "fmsr": FocusPolicy(
        categories=("Knowledge Query", "Diagnostic Assessment", "Recommendation"),
        prompt_requirements=(
            "Keep the primary burden on failure modes, failure-to-sensor reasoning, DGA interpretation, or engineering assessment.",
            "Supporting IoT, TSFM, vibration, or WO steps are allowed, but FMSR must remain the main focus.",
        ),
        forbidden_patterns=(
            FORBIDDEN_MCP_NAMES_IN_SCENARIO_TEXT,
            "Avoid generic telemetry listing tasks with no failure, reliability, or engineering reasoning.",
        ),
    ),
    "tsfm": FocusPolicy(
        categories=(
            "Inference Query",
            "Anomaly Detection Query",
            "Tuning Query",
            "Complex Query",
        ),
        prompt_requirements=(
            "Keep the primary burden on time-series forecasting, anomaly detection, model evaluation, or tuning.",
            "Supporting IoT, vibration, FMSR, or WO steps are allowed, but TSFM must remain the main focus.",
        ),
        forbidden_patterns=(
            FORBIDDEN_MCP_NAMES_IN_SCENARIO_TEXT,
            "Avoid plain asset discovery or failure-list questions with no time-series analysis task.",
        ),
    ),
    "wo": FocusPolicy(
        categories=("Decision Support", "Prediction", "Knowledge Query"),
        prompt_requirements=(
            "Keep the primary burden on maintenance planning, alerts, events, work-order analysis, or maintenance decisions.",
            "Supporting IoT, FMSR, TSFM, or vibration steps are allowed, but WO must remain the main focus.",
        ),
        forbidden_patterns=(
            FORBIDDEN_MCP_NAMES_IN_SCENARIO_TEXT,
            "Avoid plain sensor-list or standalone forecasting tasks with no maintenance decision component.",
        ),
    ),
    "vibration": FocusPolicy(
        categories=(
            "Diagnostic Assessment",
            "Bearing Analysis",
            "Severity Assessment",
            "Knowledge Query",
        ),
        prompt_requirements=(
            "Keep the primary burden on vibration diagnostics, spectra, bearing reasoning, or severity assessment.",
            "Supporting IoT, FMSR, TSFM, or WO steps are allowed, but vibration must remain the main focus.",
        ),
        forbidden_patterns=(
            FORBIDDEN_MCP_NAMES_IN_SCENARIO_TEXT,
            "Avoid generic telemetry prompts that do not require vibration-specific tools or reasoning.",
        ),
    ),
    "multiagent": FocusPolicy(
        categories=("Knowledge Query", "Workflow Coordination"),
        prompt_requirements=(
            "Every multiagent scenario should require at least two distinct namespaces from iot, fmsr, tsfm, wo, and vibration.",
            "The characteristic_form should mention a realistic coordinated workflow, not just a list of tool calls.",
        ),
        forbidden_patterns=(
            FORBIDDEN_MCP_NAMES_IN_SCENARIO_TEXT,
            "Avoid collapsing multiagent scenarios into a single-focus task.",
        ),
    ),
}


def get_scenario_policy(focus: str) -> FocusPolicy:
    key = focus.lower()
    if key not in SCENARIO_POLICIES:
        raise KeyError(f"Unknown scenario policy for focus '{focus}'")
    return SCENARIO_POLICIES[key]


def format_categories_for_prompt(focus: str) -> str:
    return get_scenario_policy(focus).format_categories()


def format_requirements_for_prompt(focus: str) -> str:
    return get_scenario_policy(focus).format_requirements()


def format_forbidden_patterns_for_prompt(focus: str) -> str:
    return get_scenario_policy(focus).format_forbidden_patterns()


def format_accepted_scenarios_for_prompt(
    scenarios: Iterable[dict], limit: int = 12
) -> str:
    texts: list[str] = []
    for scenario in scenarios:
        text = str(scenario.get("text", "")).strip()
        if text:
            texts.append(text)
        if len(texts) >= limit:
            break
    return json.dumps(texts, indent=2) if texts else "[]"


def format_mode_requirements(
    profile: AssetProfile,
    focus: str,
    generation_mode: str,
) -> str:
    """Return prompt guidance for closed-form vs grounded open-form runs."""

    if generation_mode == "open_form":
        asset_ids = profile.grounded_asset_ids(focus)
        sensors = profile.grounded_sensor_names(focus)
        timestamps = profile.grounded_timestamps(focus)
        lines = [
            "Use only grounded identifiers from the Asset Profile.",
            f"Allowed sites: {', '.join(profile.grounded_sites()) or '(none)'}",
            f"Allowed asset ids: {', '.join(asset_ids) or '(none)'}",
            f"Allowed sensors: {', '.join(sensors[:20]) or '(none)'}",
            f"Allowed timestamps: {', '.join(timestamps[:12]) or '(none)'}",
        ]
        return _format_bullet_list(lines)

    return _format_bullet_list(
        (
            "Make the scenario self-contained. Do not assume hidden live site names, asset ids, sensor names, or timestamps.",
            "In the scenario text, list concrete sensor measurements: for each channel, give the sensor name (or clear label), a numeric value, and a unit (e.g. ppm, %, Hz, mm/s, kV). Prefer names that appear under iot_sensors and vibration_sensors in the Asset Profile where applicable.",
            'Use operator-style phrasing such as: "What is the <metric> of a <asset> with the following sensor readings: <name> <value> <unit>, ..."',
            "You may still embed rule text, short summaries, or dataset identifiers in the query when the task requires them.",
        )
    )
