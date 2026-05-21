"""Tests for the schema layer: field registry and core models."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from propintelli.schemas import (
    FieldValue,
    PropertyRecord,
    Provenance,
    QualityReport,
    ReviewStatus,
    field_names,
    get_field,
    required_field_names,
)
from propintelli.schemas.fields import PROPERTY_FIELDS, FieldKind


def _minimal_quality() -> QualityReport:
    return QualityReport(overall_confidence=0.5)


def test_registry_required_fields_match_business_rules() -> None:
    required = set(required_field_names())
    assert required == {"price_eur", "living_area_sqm", "postal_code", "city"}


def test_registry_names_are_unique_and_paths_resolve() -> None:
    names = field_names()
    assert len(names) == len(set(names))
    # Every registered field declares a non-empty record path.
    assert all(PROPERTY_FIELDS[name].record_path for name in names)


def test_enum_fields_declare_enum_type() -> None:
    for spec in PROPERTY_FIELDS.values():
        if spec.kind is FieldKind.ENUM:
            assert spec.enum_type is not None
        else:
            assert spec.enum_type is None


def test_get_field_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_field("does_not_exist")


def test_field_value_presence_and_bounds() -> None:
    assert FieldValue(raw_value="120").is_present is True
    assert FieldValue(raw_value="   ").is_present is False
    assert FieldValue(raw_value=None).is_present is False
    with pytest.raises(ValidationError):
        FieldValue(raw_value="x", confidence=1.5)


def test_field_value_is_frozen() -> None:
    value = FieldValue(raw_value="120", provenance=Provenance.LLM)
    with pytest.raises(ValidationError):
        value.raw_value = "999"  # type: ignore[misc]


def test_price_per_sqm_is_computed() -> None:
    record = PropertyRecord(
        source_document="x.pdf",
        price_eur=Decimal("450000"),
        living_area_sqm=90.0,
        quality=_minimal_quality(),
    )
    assert record.price_per_sqm == Decimal("5000.00")


@pytest.mark.parametrize(
    ("price", "area"),
    [(None, 90.0), (Decimal("450000"), None), (Decimal("450000"), 0.0)],
)
def test_price_per_sqm_none_when_inputs_invalid(price: Decimal | None, area: float | None) -> None:
    record = PropertyRecord(
        source_document="x.pdf",
        price_eur=price,
        living_area_sqm=area,
        quality=_minimal_quality(),
    )
    assert record.price_per_sqm is None


def test_record_json_round_trip_serialises_enums_as_values() -> None:
    record = PropertyRecord(
        source_document="x.pdf",
        quality=QualityReport(
            overall_confidence=0.9,
            review_status=ReviewStatus.AUTO_APPROVED,
        ),
    )
    dumped = record.model_dump(mode="json")
    assert dumped["quality"]["review_status"] == "auto_approved"
    restored = PropertyRecord.model_validate(dumped)
    assert restored.quality.review_status is ReviewStatus.AUTO_APPROVED
    assert restored.property_id == record.property_id
