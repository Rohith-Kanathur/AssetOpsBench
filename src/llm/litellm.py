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

from .base import LLMBackend, LLMResult

_log = logging.getLogger(__name__)

_WATSONX_PREFIX = "watsonx/"


class LiteLLMBackend(LLMBackend):
    """LLM backend using the litellm library.

    Args:
        model_id: litellm model string with provider prefix, e.g.:
                  ``"watsonx/meta-llama/llama-3-3-70b-instruct"``
                  ``"litellm_proxy/GCP/claude-4-sonnet"``
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    def generate_with_usage(
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
            "max_tokens": max_tokens if max_tokens is not None else 2048,
        }

        if self._model_id.startswith(_WATSONX_PREFIX):
            kwargs["api_key"] = os.environ["WATSONX_APIKEY"]
            kwargs["project_id"] = os.environ["WATSONX_PROJECT_ID"]
            if url := os.environ.get("WATSONX_URL"):
                kwargs["api_base"] = url
        else:
            kwargs["api_key"] = os.environ["LITELLM_API_KEY"]
            kwargs["api_base"] = os.environ["LITELLM_BASE_URL"]

        response = litellm.completion(**kwargs)
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        prompt_tok = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tok = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total_raw = getattr(usage, "total_tokens", None) if usage else None
        total_tok = (
            int(total_raw)
            if total_raw is not None
            else (prompt_tok + completion_tok)
        )

        if usage is not None:
            _log.info(
                "LLM usage model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                self._model_id,
                prompt_tok,
                completion_tok,
                total_tok,
            )
        else:
            _log.info(
                "LLM completion model=%s (no usage object from provider)",
                self._model_id,
            )

        return LLMResult(
            text=text,
            input_tokens=prompt_tok,
            output_tokens=completion_tok,
            total_tokens=total_tok,
        )
