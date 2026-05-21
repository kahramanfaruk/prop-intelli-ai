"""Optional LLM extraction layer (Layer B).

The package exposes a provider abstraction so the engine can call a language
model without depending on any concrete vendor. The default ``none`` provider
keeps the pipeline fully offline; Ollama, OpenAI, and Azure OpenAI providers are
selected via configuration.
"""

from __future__ import annotations

from propintelli.extraction.llm.base import (
    ExtractionProvider,
    LlmExtraction,
    build_provider,
    parse_extraction,
)

__all__ = [
    "ExtractionProvider",
    "LlmExtraction",
    "build_provider",
    "parse_extraction",
]
