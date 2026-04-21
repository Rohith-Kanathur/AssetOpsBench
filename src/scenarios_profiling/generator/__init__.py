"""Scenario generation: orchestration, validation loop, and CLI (`python -m scenarios_profiling.generator`).

This package mirrors ``scenarios.generator`` with PyTorch profiler instrumentation
added to every major pipeline stage.  Use ``ScenarioGeneratorAgent.run_with_profiling``
to run the full pipeline under the profiler.
"""

from .agent import ScenarioGeneratorAgent
from .cli import main
from .prompt_helpers import DEFAULT_GENERATED_SCENARIOS_DIR, default_scenario_output_path

__all__ = [
    "DEFAULT_GENERATED_SCENARIOS_DIR",
    "ScenarioGeneratorAgent",
    "default_scenario_output_path",
    "main",
]
