"""MCP agent orchestration package, with runners loaded only when requested."""

from importlib import import_module

_EXPORTS = {
    **dict.fromkeys(("AgentResult", "ToolCall", "Trajectory", "TurnRecord"), ".models"),
    "AgentRunner": ".runner",
    "ClaudeAgentRunner": ".claude_agent.runner",
    "DeepAgentRunner": ".deep_agent.runner",
    "DirectLLMAgentRunner": ".direct_llm_agent.runner",
    "OpenAIAgentRunner": ".openai_agent.runner",
    **dict.fromkeys(("OrchestratorResult", "Plan", "PlanStep", "StepResult"), ".plan_execute.models"),
    "PlanExecuteRunner": ".plan_execute.runner",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "AgentRunner",
    "AgentResult",
    "ClaudeAgentRunner",
    "DeepAgentRunner",
    "DirectLLMAgentRunner",
    "OpenAIAgentRunner",
    "OrchestratorResult",
    "Plan",
    "PlanExecuteRunner",
    "PlanStep",
    "StepResult",
    "ToolCall",
    "Trajectory",
    "TurnRecord",
]
