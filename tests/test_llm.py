"""Tests for the LLM layer: prompts, parsing, and provider selection."""

from __future__ import annotations

import pytest

from propintelli.config import LlmProvider, PromptVariant, Settings
from propintelli.errors import LlmError
from propintelli.extraction.llm.base import build_provider, parse_extraction
from propintelli.extraction.llm.none_provider import NoneProvider
from propintelli.extraction.llm.prompts import build_messages


def test_none_provider_returns_empty() -> None:
    extraction = NoneProvider().extract("any text")
    assert extraction.fields == {}
    assert extraction.field_confidences == {}


@pytest.mark.parametrize("variant", list(PromptVariant))
def test_build_messages_returns_system_and_user(variant: PromptVariant) -> None:
    system, user = build_messages("Kaufpreis 100.000 €", variant)
    assert "extraction" in system.lower()
    assert "Kaufpreis 100.000 €" in user


def test_schema_variants_inject_field_schema() -> None:
    _, direct = build_messages("doc", PromptVariant.V1_DIRECT)
    _, schema = build_messages("doc", PromptVariant.V2_SCHEMA)
    _, reasoning = build_messages("doc", PromptVariant.V3_REASONING)
    assert "FIELDS:" not in direct
    assert "price_eur" in schema and "FIELDS:" in schema
    assert "confidences" in reasoning  # v3 asks for per-field confidence


def test_parse_extraction_filters_unknown_and_null_and_clamps() -> None:
    data = {
        "fields": {"price_eur": "100000", "unknown_field": "x", "city": None},
        "confidences": {"city": 2.0, "not_a_field": 0.5, "price_eur": 0.9},
    }
    extraction = parse_extraction(data)
    assert extraction.fields == {"price_eur": "100000"}
    assert extraction.field_confidences["price_eur"] == 0.9
    assert "not_a_field" not in extraction.field_confidences


def test_parse_extraction_tolerates_flat_object() -> None:
    extraction = parse_extraction({"city": "Berlin", "rooms": 3})
    assert extraction.fields == {"city": "Berlin", "rooms": 3}


def test_build_provider_selects_none_by_default() -> None:
    provider = build_provider(Settings(llm_provider=LlmProvider.NONE))
    assert isinstance(provider, NoneProvider)


def test_build_provider_openai_without_key_raises() -> None:
    with pytest.raises(LlmError):
        build_provider(Settings(llm_provider=LlmProvider.OPENAI, openai_api_key=None))
