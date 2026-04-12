"""Abstract LLM backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMUsage:
    """Token usage for a single completion (OpenAI-compatible fields)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMResult:
    """LLM output text plus optional provider usage metadata."""

    text: str
    usage: LLMUsage | None = None


class LLMBackend(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Generate text given a prompt; includes usage when the backend provides it."""
        ...
