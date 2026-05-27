"""Tests for deterministic extraction, reconciliation, and the engine."""

from __future__ import annotations

import pytest

from propintelli.config import LlmProvider, Settings
from propintelli.errors import LlmError
from propintelli.extraction import extract_deterministic, reconcile, run_extraction
from propintelli.extraction.llm.base import LlmExtraction
from propintelli.preprocessing import DocumentClass, TextSource
from propintelli.preprocessing.text_extractor import PreprocessedDocument
from propintelli.schemas.enums import Provenance
from propintelli.schemas.extraction import FieldValue

_SALE_TEXT = """Moderne 3-Zimmer-Eigentumswohnung mit Balkon in Nürnberg
Kaufpreis: 449.000 €
Wohnfläche: ca. 92 m²
Zimmer: 3
Baujahr: 1998
Adresse: Bucher Straße 42, 90408 Nürnberg
Energieeffizienzklasse: C
Heizung: Gas-Zentralheizung
Ausstattung: Balkon, Keller, Aufzug, Einbauküche
"""

_RENT_TEXT = """Helle 2-Zimmer-Mietwohnung in Nürnberg
Kaltmiete 980 €
Wohnfläche 58,5 m²
"""


def _doc(text: str) -> PreprocessedDocument:
    return PreprocessedDocument(
        document_id="doc-1",
        source_document="x.pdf",
        text=text,
        document_class=DocumentClass.DIGITAL,
        text_source=TextSource.DIGITAL,
        page_count=1,
        char_count=len(text),
    )


def test_deterministic_extracts_core_sale_fields() -> None:
    fields = extract_deterministic(_SALE_TEXT)
    assert fields["price_eur"].raw_value == "449.000"
    assert fields["price_kind"].raw_value == "purchase"
    assert fields["listing_type"].raw_value == "sale"
    assert fields["living_area_sqm"].raw_value == "92"
    assert fields["rooms"].raw_value == "3"
    assert fields["year_built"].raw_value == "1998"
    assert fields["postal_code"].raw_value == "90408"
    assert fields["city"].raw_value == "Nürnberg"
    assert fields["energy_class"].raw_value == "C"
    assert fields["heating_type"].raw_value == "gas"
    assert {"balcony", "cellar", "elevator", "fitted_kitchen"} <= set(fields)


def test_deterministic_detects_rent_listing() -> None:
    fields = extract_deterministic(_RENT_TEXT)
    assert fields["price_kind"].raw_value == "cold_rent"
    assert fields["listing_type"].raw_value == "rent"
    assert fields["price_eur"].raw_value == "980"


_NEGATION_TEXT = """Wohnung in Berlin
Kaufpreis: 300.000 EUR
Wohnfläche 70 m²
Adresse: Hauptstraße 1, 10115 Berlin
Kein Balkon, ohne Aufzug. Ein Keller ist vorhanden.
Kein Stellplatz, aber eine Tiefgarage steht zur Verfügung.
"""


def test_deterministic_records_negated_features_as_false() -> None:
    fields = extract_deterministic(_NEGATION_TEXT)
    # Explicit absence is recorded as False, exercising the tri-state boolean.
    assert fields["balcony"].raw_value == "false"
    assert fields["elevator"].raw_value == "false"
    # A non-negated mention in the same text still asserts presence.
    assert fields["cellar"].raw_value == "true"


def test_deterministic_positive_mention_overrides_earlier_negation() -> None:
    # "Kein Stellplatz, aber ... Tiefgarage": one synonym is negated, another is
    # present: a single positive mention is decisive evidence of presence.
    fields = extract_deterministic(_NEGATION_TEXT)
    assert fields["parking"].raw_value == "true"


_MULTI_AMOUNT_TEXT = """Eigentumswohnung in Nürnberg
Kaufpreis auf Anfrage. Hausgeld: 250 EUR monatlich.
Provision: 3.570 EUR inkl. MwSt. Der Kaufpreis beträgt 449.000 €.
Wohnfläche 92 m²
Adresse: Bucher Straße 42, 90408 Nürnberg
"""


def test_deterministic_price_ignores_ancillary_costs() -> None:
    # Hausgeld and Provision amounts precede the real price; the extractor must
    # not capture them, and must skip the "auf Anfrage" Kaufpreis with no number.
    fields = extract_deterministic(_MULTI_AMOUNT_TEXT)
    assert fields["price_eur"].raw_value == "449.000"
    assert fields["price_kind"].raw_value == "purchase"
    assert fields["listing_type"].raw_value == "sale"


def test_reconcile_passthrough_for_single_layer() -> None:
    layer_a = {"city": FieldValue(raw_value="Berlin", confidence=0.8)}
    merged, warnings = reconcile(layer_a, {})
    assert merged["city"].raw_value == "Berlin"
    assert warnings == []


def test_reconcile_agreement_boosts_confidence_across_formats() -> None:
    layer_a = {
        "price_eur": FieldValue(
            raw_value="449.000", confidence=0.82, provenance=Provenance.DETERMINISTIC
        )
    }
    layer_b = {
        "price_eur": FieldValue(raw_value="449000", confidence=0.7, provenance=Provenance.LLM)
    }
    merged, warnings = reconcile(layer_a, layer_b)
    assert merged["price_eur"].provenance is Provenance.RECONCILED
    assert merged["price_eur"].confidence == pytest.approx(0.92)
    assert warnings == []


def test_reconcile_disagreement_penalises_and_warns() -> None:
    layer_a = {
        "year_built": FieldValue(
            raw_value="1998", confidence=0.9, provenance=Provenance.DETERMINISTIC
        )
    }
    layer_b = {
        "year_built": FieldValue(raw_value="2001", confidence=0.6, provenance=Provenance.LLM)
    }
    merged, warnings = reconcile(layer_a, layer_b)
    assert merged["year_built"].raw_value == "1998"  # higher confidence wins
    assert merged["year_built"].confidence == pytest.approx(0.54)  # 0.9 * 0.6 penalty
    assert len(warnings) == 1
    assert "Disagreement" in warnings[0]


def test_engine_runs_offline_with_none_provider() -> None:
    result = run_extraction(_doc(_SALE_TEXT), Settings(llm_provider=LlmProvider.NONE))
    assert result.get("price_eur") is not None
    assert result.get("city").raw_value == "Nürnberg"  # type: ignore[union-attr]


def test_engine_downgrades_on_llm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingProvider:
        name = "ollama"

        def extract(self, text: str) -> LlmExtraction:
            raise LlmError("backend down", document_id="doc-1")

    monkeypatch.setattr(
        "propintelli.extraction.engine.build_provider", lambda _settings: _FailingProvider()
    )
    result = run_extraction(_doc(_SALE_TEXT), Settings(llm_provider=LlmProvider.OLLAMA))
    # Deterministic result still present; the failure is recorded as a warning.
    assert result.get("price_eur") is not None
    assert any("AI assistance" in warning for warning in result.warnings)


def test_engine_merges_llm_only_field(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ExtraFieldProvider:
        name = "ollama"

        def extract(self, text: str) -> LlmExtraction:
            return LlmExtraction(
                fields={"district": "Nordstadt"}, field_confidences={"district": 0.8}
            )

    monkeypatch.setattr(
        "propintelli.extraction.engine.build_provider", lambda _settings: _ExtraFieldProvider()
    )
    result = run_extraction(_doc(_SALE_TEXT), Settings(llm_provider=LlmProvider.OLLAMA))
    district = result.get("district")
    assert district is not None
    assert district.raw_value == "Nordstadt"
    assert district.provenance is Provenance.LLM
