"""Tests for parsing, normalisation, and record assembly."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from propintelli.schemas.enums import EnergyClass, Provenance
from propintelli.schemas.extraction import ExtractionResult, FieldValue
from propintelli.schemas.fields import get_field
from propintelli.schemas.property_record import QualityReport
from propintelli.transformation import assemble_record, normalize, normalize_value
from propintelli.transformation.normalize import NormalizedFields
from propintelli.transformation.parsing import parse_bool, parse_number


@pytest.mark.parametrize(
    ("raw", "german", "expected"),
    [
        ("449.000", True, 449000.0),
        ("124,5", True, 124.5),
        ("1.190.000", True, 1190000.0),
        ("€ 980", True, 980.0),
        ("449000", False, 449000.0),
        ("92.5", False, 92.5),
        ("1,234,567", False, 1234567.0),
        ("abc", True, None),
    ],
)
def test_parse_number(raw: str, german: bool, expected: float | None) -> None:
    assert parse_number(raw, german=german) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("ja", True), ("true", True), ("nein", False), ("false", False), ("maybe", None)],
)
def test_parse_bool(raw: str, expected: bool | None) -> None:
    assert parse_bool(raw) == expected


def test_normalize_value_decimal_from_german() -> None:
    spec = get_field("price_eur")
    value = FieldValue(raw_value="449.000", provenance=Provenance.DETERMINISTIC)
    assert normalize_value(spec, value) == Decimal("449000.00")


def test_normalize_value_reconciled_amount_uses_german_convention() -> None:
    # Reconciliation's agreement branch tags the value RECONCILED while keeping
    # the deterministic layer's German-formatted string, so it must parse as
    # German (449.000 -> 449000.00), not dot-decimal (which would yield 449.00).
    spec = get_field("price_eur")
    value = FieldValue(raw_value="449.000", provenance=Provenance.RECONCILED)
    assert normalize_value(spec, value) == Decimal("449000.00")


def test_normalize_value_date_both_formats() -> None:
    spec = get_field("availability_date")
    assert normalize_value(spec, FieldValue(raw_value="01.07.2026")) == date(2026, 7, 1)
    assert normalize_value(spec, FieldValue(raw_value="2026-07-01")) == date(2026, 7, 1)


def test_normalize_value_enum_case_insensitive() -> None:
    spec = get_field("energy_class")
    assert normalize_value(spec, FieldValue(raw_value="A+")) is EnergyClass.A_PLUS
    assert normalize_value(spec, FieldValue(raw_value="c")) is EnergyClass.C


def test_normalize_value_unparseable_returns_none() -> None:
    spec = get_field("year_built")
    assert normalize_value(spec, FieldValue(raw_value="not-a-year")) is None


def test_normalize_collects_typed_values_and_warns() -> None:
    extraction = ExtractionResult(
        document_id="d",
        source_document="x.pdf",
        fields={
            "price_eur": FieldValue(raw_value="449.000", confidence=0.9),
            "city": FieldValue(raw_value=" Nürnberg ", confidence=0.8),
            "year_built": FieldValue(raw_value="garbage", confidence=0.5),
        },
    )
    result = normalize(extraction)
    assert result.values["price_eur"] == Decimal("449000.00")
    assert result.values["city"] == "Nürnberg"
    assert "year_built" not in result.values
    assert any("year_built" in warning for warning in result.warnings)
    assert result.confidences["price_eur"] == 0.9


def test_assemble_record_routes_fields_into_nested_models() -> None:
    normalized = NormalizedFields(
        values={
            "price_eur": Decimal("450000.00"),
            "city": "Berlin",
            "balcony": True,
            "energy_class": EnergyClass.B,
        }
    )
    record = assemble_record(normalized, QualityReport(overall_confidence=0.8), "x.pdf")
    assert record.price_eur == Decimal("450000.00")
    assert record.location.city == "Berlin"
    assert record.features.balcony is True
    assert record.energy.energy_class is EnergyClass.B
