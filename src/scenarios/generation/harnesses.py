"""Native harness command builders; add another builder when its runtime is supported."""

import json

DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_REASONING = "xhigh"
DEFAULT_TIER = "fast"
SERVERS = ("iot", "fmsr", "tsfm", "wo", "vibration", "utilities")


def codex_command(model: str = DEFAULT_MODEL, reasoning_effort: str = DEFAULT_REASONING,
                  service_tier: str = DEFAULT_TIER) -> list[str]:
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
