"""Prompt-construction helpers for the scenarios_optimization generator.

This module re-exports all helpers from ``scenarios_profiling.generator.prompt_helpers``
and adds only the output-path helper so the optimized generator can write its
output to a distinct directory (``generated/scenarios_optimized/``).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scenarios_profiling.generator.prompt_helpers import (  # noqa: F401  (re-export)
    _MULTIAGENT_MAX_TOKENS,
    _MAX_FEWSHOT_EXAMPLES,
    _MAX_SCENARIO_ATTEMPTS,
    _PROFILE_MAX_TOKENS,
    _asset_profile_json,
    _few_shot_examples_section,
    _grounding_summary_for_prompt,
    _invert_failure_mapping,
    _label_desc_dict_from_list,
    _multiagent_budget_cap,
    _normalize_failure_sensor_mapping,
    _normalize_string_list,
    _ordered_descriptions,
    _print_live_step,
    _print_section,
    _print_step,
    _quiet_litellm_logging,
    _redact_logged_prompt,
    _require_nonempty_list,
    _require_nonempty_str,
    _tool_summary_for_prompt,
    _validation_tool_names_by_focus,
)
from scenarios_profiling.text import slugify_asset_name

# Output directory for optimized-run scenarios (separate from profiling runs)
DEFAULT_GENERATED_SCENARIOS_DIR = Path("generated/scenarios_optimized")


def _log_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_scenario_output_path(asset_name: str) -> Path:
    """Return the default output path for optimized scenario runs.

    Writes to ``generated/scenarios_optimized/<asset_slug>_<timestamp>/scenarios.json``
    so that optimized runs are clearly distinguished from profiling runs.
    """
    timestamp = _log_timestamp()
    run_dir = DEFAULT_GENERATED_SCENARIOS_DIR / f"{slugify_asset_name(asset_name)}_scenarios_{timestamp}"
    return run_dir / "scenarios.json"
