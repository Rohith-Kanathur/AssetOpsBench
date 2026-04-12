"""Shared fixtures for plan_execute unit tests."""

import pytest

from llm import LLMBackend, LLMResult, LLMUsage


class MockLLM(LLMBackend):
    """Deterministic LLM that returns a canned response — no network calls."""

    def __init__(self, response: str = "") -> None:
        self._response = response

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        return LLMResult(
            text=self._response,
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class SequentialMockLLM(LLMBackend):
    """Returns responses in order across successive generate() calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        text = next(self._responses, "")
        return LLMResult(
            text=text,
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


@pytest.fixture
def mock_llm():
    """Factory fixture: MockLLM(response='')."""

    def _factory(response: str = "") -> MockLLM:
        return MockLLM(response)

    return _factory


@pytest.fixture
def sequential_llm():
    """Factory fixture: SequentialMockLLM(responses=[...])."""

    def _factory(responses: list[str]) -> SequentialMockLLM:
        return SequentialMockLLM(responses)

    return _factory
