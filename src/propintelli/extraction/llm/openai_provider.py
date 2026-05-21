"""Cloud LLM extraction via OpenAI or Azure OpenAI.

A single provider serves both vendors because they share the OpenAI SDK surface;
the only differences are client construction and the model/deployment name. JSON
mode is requested so the response is a parseable object. Misconfiguration and
API failures are surfaced as :class:`~propintelli.errors.LlmError`, a recoverable
downgrade.
"""

from __future__ import annotations

import json

from propintelli.config import LlmProvider, Settings
from propintelli.errors import LlmError
from propintelli.extraction.llm.base import LlmExtraction, parse_extraction
from propintelli.extraction.llm.prompts import build_messages


class OpenAIProvider:
    """Extraction provider backed by OpenAI or Azure OpenAI chat completions."""

    def __init__(self, settings: Settings) -> None:
        """Validate configuration and build the appropriate client.

        Parameters
        ----------
        settings : Settings
            Provides credentials, the model/deployment name, and the prompt
            variant.

        Raises
        ------
        LlmError
            If required credentials for the selected vendor are missing.
        """
        self._settings = settings
        self.name = settings.llm_provider.value
        self._client, self._model = self._build_client(settings)

    @staticmethod
    def _build_client(settings: Settings) -> tuple[object, str]:
        """Construct the vendor client and resolve the model/deployment name."""
        from openai import AzureOpenAI, OpenAI

        if settings.llm_provider is LlmProvider.AZURE_OPENAI:
            if not (
                settings.azure_openai_endpoint
                and settings.azure_openai_api_key
                and settings.azure_openai_deployment
            ):
                raise LlmError(
                    "Azure OpenAI requires endpoint, API key, and deployment to be set",
                    details={"provider": "azure_openai"},
                )
            client: object = AzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
            )
            return client, settings.azure_openai_deployment

        if not settings.openai_api_key:
            raise LlmError(
                "OpenAI requires OPENAI_API_KEY to be set",
                details={"provider": "openai"},
            )
        return OpenAI(api_key=settings.openai_api_key), settings.openai_model

    def extract(self, text: str) -> LlmExtraction:
        """Extract fields by prompting the configured chat model.

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
            If the API call fails or the response is not valid JSON.
        """
        system, user = build_messages(text, self._settings.llm_prompt_variant)
        try:
            # The client is typed as ``object`` to avoid a hard dependency on the
            # SDK types at module scope; the call surface is stable across vendors.
            completion = self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                timeout=self._settings.llm_timeout_seconds,
            )
            content = completion.choices[0].message.content or "{}"
            data = json.loads(content)
        except LlmError:
            raise
        except Exception as exc:  # SDK raises a wide range of error types
            raise LlmError(
                f"OpenAI extraction failed ({self.name}): {exc}",
                details={"provider": self.name, "model": self._model},
            ) from exc
        return parse_extraction(data)
