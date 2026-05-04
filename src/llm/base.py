"""Abstract LLM backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResult:
    """Return type for :meth:`LLMBackend.generate_with_usage`.

    ``input_tokens`` / ``output_tokens`` / ``total_tokens`` are ``0`` when the
    backend cannot report usage (e.g. mocks in unit tests). ``total_tokens``
    should mirror the provider when available; otherwise callers may treat
    ``input_tokens + output_tokens`` as an estimate.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class LLMBackend(ABC):
    """Abstract interface for LLM backends.

    Subclasses implement :meth:`generate_with_usage`. :meth:`generate` returns
    only the text via ``generate_with_usage(...).text``.
    """

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text given a prompt."""
        return self.generate_with_usage(prompt, temperature, max_tokens).text

    @abstractmethod
    def generate_with_usage(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Generate text and report token usage when the backend can."""
        ...

    @property
    def model_id(self) -> str:
        """Return the backend's model identifier, or ``"unknown"``.

        Default impl reads ``self._model_id`` if present so existing
        subclasses work without modification.
        """
        return getattr(self, "_model_id", "unknown")
