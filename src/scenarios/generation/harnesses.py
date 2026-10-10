"""Native harness command builders; add another builder when its runtime is supported."""

import json
import math

DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_REASONING = "xhigh"
DEFAULT_TIER = "fast"
SERVERS = ("iot", "fmsr", "tsfm", "wo", "vibration", "utilities")


def codex_command(model: str = DEFAULT_MODEL, reasoning_effort: str = DEFAULT_REASONING,
                  service_tier: str = DEFAULT_TIER, *, temperature: float | None = None) -> list[str]:
    if temperature is not None:
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be a finite number between 0 and 2")
        raise ValueError(
            "The Codex CLI harness does not expose temperature control. "
            "Omit --temperature to use its native settings; setting temperature requires "
            "a harness and model that support it. No temperature override was applied."
        )
    command = ["codex", "exec", "--ignore-user-config", "--ephemeral",
               "--skip-git-repo-check", "--json", "--color", "never",
               # Docker provides isolation; no host tree or Docker socket is mounted.
               "--sandbox", "danger-full-access", "-c", 'approval_policy="never"',
               "-c", 'web_search="live"', "-c", "agents.enabled=false",
               "-o", "/workspace/output/final.md"]
    for name in SERVERS:
        prefix = f"mcp_servers.{name}"
        command.extend(["-c", f'{prefix}.command="python"', "-c",
                        f'{prefix}.args=["-m","servers.{name}.main"]',
                        "-c", f"{prefix}.startup_timeout_sec=60",
                        "-c", f"{prefix}.tool_timeout_sec=120"])
    command.extend(["-c", 'mcp_servers.research.command="python"', "-c",
                    'mcp_servers.research.args=["-m","scenarios.generation.research"]', "-c",
                    'mcp_servers.research.env_vars=["SEMANTIC_SCHOLAR_API_KEY","SCENARIO_RESEARCH_LOG"]',
                    "-c", "mcp_servers.research.tool_timeout_sec=120"])
    command.extend(["--model", model, "-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}",
                    "-c", f"service_tier={json.dumps(service_tier)}"])
    return command + ["-"]


HARNESSES = {"codex": codex_command}
