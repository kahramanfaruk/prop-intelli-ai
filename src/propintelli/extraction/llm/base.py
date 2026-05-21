"""LLM provider abstraction and shared parsing.

The :class:`ExtractionProvider` protocol decouples the engine from any concrete
LLM vendor. :func:`build_provider` selects the implementation from configuration,
importing the concrete module lazily so optional dependencies (``httpx``,
``openai``) are only required when their provider is actually used.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from propintelli.config import LlmProvider, Settings
from propintelli.schemas.fields import field_names


class LlmExtraction(BaseModel):
    """Parsed output of an LLM extraction call.

    Attributes
    ----------
    fields : dict of str to object
        Field values keyed by canonical field name. Null/unknown values are
        dropped during parsing.
    field_confidences : dict of str to float
        Optional per-field confidence in ``[0, 1]`` (populated by the
        ``v3_reasoning`` variant).
    """

    fields: dict[str, Any] = Field(default_factory=dict)
    field_confidences: dict[str, float] = Field(default_factory=dict)


@runtime_checkable
class ExtractionProvider(Protocol):
    """A backend that turns document text into an :class:`LlmExtraction`."""

    name: str

    def extract(self, text: str) -> LlmExtraction:
        """Extract fields from ``text``; raise ``LlmError`` on backend failure."""
        ...


def parse_extraction(data: dict[str, Any]) -> LlmExtraction:
    """Parse a model's JSON response into an :class:`LlmExtraction`.

    The parser is defensive: it accepts both the ``{"fields": {...}}`` envelope
    and a flat object, restricts keys to the canonical field registry, and drops
    null values so the model cannot introduce unknown or empty fields.

    Parameters
    ----------
    data : dict
        The parsed JSON object returned by the model.

    Returns
    -------
    LlmExtraction
        The cleaned extraction.
    """
    known = set(field_names())
    raw_fields = data.get("fields", data)
    fields: dict[str, Any] = {}
    if isinstance(raw_fields, dict):
        fields = {
            key: value for key, value in raw_fields.items() if key in known and value is not None
        }

    confidences: dict[str, float] = {}
    raw_confidences = data.get("confidences", {})
    if isinstance(raw_confidences, dict):
        for key, value in raw_confidences.items():
            if key in known and isinstance(value, (int, float)):
                confidences[key] = max(0.0, min(1.0, float(value)))
    return LlmExtraction(fields=fields, field_confidences=confidences)


def build_provider(settings: Settings) -> ExtractionProvider:
    """Construct the configured extraction provider.

    Parameters
    ----------
    settings : Settings
        Settings whose ``llm_provider`` selects the backend.

    Returns
    -------
    ExtractionProvider
        A ready-to-use provider. The ``none`` provider is a no-op that keeps the
        pipeline offline.
    """
    if settings.llm_provider is LlmProvider.NONE:
        from propintelli.extraction.llm.none_provider import NoneProvider

        return NoneProvider()
    if settings.llm_provider is LlmProvider.OLLAMA:
        from propintelli.extraction.llm.ollama_provider import OllamaProvider

        return OllamaProvider(settings)
    from propintelli.extraction.llm.openai_provider import OpenAIProvider

    return OpenAIProvider(settings)
