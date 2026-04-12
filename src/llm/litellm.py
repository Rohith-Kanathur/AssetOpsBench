"""Unified LLM backend via the litellm library.

Supports any model string that litellm recognizes.  The provider is encoded
in the model-string prefix — no separate platform flag is needed:

    watsonx/meta-llama/llama-3-3-70b-instruct   → IBM WatsonX
    litellm_proxy/GCP/claude-4-sonnet            → LiteLLM proxy

Credentials are resolved from environment variables based on the prefix:

    watsonx/*  :  WATSONX_APIKEY, WATSONX_PROJECT_ID, WATSONX_URL (optional)
    otherwise  :  LITELLM_API_KEY, LITELLM_BASE_URL
"""

from __future__ import annotations

import logging
import os

from .base import LLMBackend, LLMResult, LLMUsage

_log = logging.getLogger(__name__)


class LiteLLMBackend(LLMBackend):
    """LLM backend using the litellm library.

    Args:
        model_id: litellm model string with provider prefix, e.g.:
                  ``"watsonx/meta-llama/llama-3-3-70b-instruct"``
                  ``"litellm_proxy/GCP/claude-4-sonnet"``
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        import litellm

        kwargs: dict = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens or 2048,
        }

        if self._model_id.startswith("watsonx/"):
            kwargs["api_key"] = os.environ["WATSONX_APIKEY"]
            kwargs["project_id"] = os.environ["WATSONX_PROJECT_ID"]
            if url := os.environ.get("WATSONX_URL"):
                kwargs["api_base"] = url
        else:
            kwargs["api_key"] = os.environ["LITELLM_API_KEY"]
            kwargs["api_base"] = os.environ["LITELLM_BASE_URL"]

        response = litellm.completion(**kwargs)
        text = response.choices[0].message.content or ""
        usage: LLMUsage | None = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = LLMUsage(
                prompt_tokens=getattr(raw_usage, "prompt_tokens", None),
                completion_tokens=getattr(raw_usage, "completion_tokens", None),
                total_tokens=getattr(raw_usage, "total_tokens", None),
            )
            _log.info(
                "LLM usage model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                self._model_id,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
        else:
            _log.info(
                "LLM completion model=%s (no usage object from provider)",
                self._model_id,
            )
        return LLMResult(text=text, usage=usage)
