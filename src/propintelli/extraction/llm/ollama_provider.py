"""Local LLM extraction via Ollama.

Talks to a local ``ollama serve`` instance over its HTTP chat API, requesting a
JSON-formatted response. Any transport, HTTP, or JSON-decoding failure is mapped
to :class:`~propintelli.errors.LlmError`, which the engine treats as a
recoverable downgrade to the deterministic layer.
"""

from __future__ import annotations

import json

import httpx

from propintelli.config import Settings
from propintelli.errors import LlmError
from propintelli.extraction.llm.base import LlmExtraction, parse_extraction
from propintelli.extraction.llm.prompts import build_messages


class OllamaProvider:
    """Extraction provider backed by a local Ollama model."""

    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        """Store the settings used to reach the Ollama server.

        Parameters
        ----------
        settings : Settings
            Provides the base URL, model name, timeout, and prompt variant.
        """
        self._settings = settings

    def extract(self, text: str) -> LlmExtraction:
        """Extract fields by prompting the configured Ollama model.

        Parameters
        ----------
        text : str
            Document text to extract from.

        Returns
        -------
        LlmExtraction
            The parsed extraction.

        Raises
        ------
        LlmError
            If the request fails or the response is not valid JSON.
        """
        system, user = build_messages(text, self._settings.llm_prompt_variant)
        payload = {
            "model": self._settings.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
            # Greedy decoding: extraction seeks the single most-probable reading,
            # not sampled variety. Temperature 0 makes the output reproducible
            # (so the evaluation/comparison is stable) and matches the OpenAI path.
            "options": {"temperature": 0},
        }
        try:
            response = httpx.post(
                f"{self._settings.ollama_base_url}/api/chat",
                json=payload,
                timeout=self._settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
            data = json.loads(content)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise LlmError(
                f"Ollama extraction failed ({self._settings.ollama_model}): {exc}",
                details={"provider": "ollama", "model": self._settings.ollama_model},
            ) from exc
        return parse_extraction(data)
