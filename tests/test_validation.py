"""Tests for the data-quality validation rules."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from propintelli.errors import ErrorSeverity
from propintelli.schemas.enums import ListingType
from propintelli.validation import validate


def _complete_sale() -> dict[str, object]:
    return {
        "price_eur": Decimal("450000"),
        "living_area_sqm": 90.0,
        "postal_code": "90408",
        "city": "Nürnberg",
        "listing_type": ListingType.SALE,
        "year_built": 1998,
        "rooms": 3.0,
        "floor": 2,
        "total_floors": 4,
    }


def test_complete_record_passes_all_rules() -> None:
    findings, pass_rate = validate(_complete_sale())
    assert findings == []
    assert pass_rate == 1.0


def test_missing_mandatory_fields_produce_errors() -> None:
    findings, pass_rate = validate({"city": "Berlin"})
    rule_ids = {finding.rule_id for finding in findings}
    assert "mandatory.price_eur" in rule_ids
    assert "mandatory.living_area_sqm" in rule_ids
    assert all(
        finding.severity is ErrorSeverity.ERROR
        for finding in findings
        if finding.rule_id.startswith("mandatory")
    )
    assert pass_rate < 1.0


def test_implausible_price_is_flagged() -> None:
    values = _complete_sale()
    values["price_eur"] = Decimal("-5")
    findings, _ = validate(values)
    assert any(finding.rule_id == "range.price_eur" for finding in findings)


def test_invalid_postal_code_is_flagged() -> None:
    values = _complete_sale()
    values["postal_code"] = "9040"
    findings, _ = validate(values)
    assert any(finding.rule_id == "format.postal_code" for finding in findings)


def test_year_in_future_is_flagged() -> None:
    values = _complete_sale()
    values["year_built"] = date.today().year + 5
    findings, _ = validate(values)
    assert any(finding.rule_id == "range.year_built" for finding in findings)


def test_floor_exceeding_total_is_flagged() -> None:
    values = _complete_sale()
    values["floor"] = 9
    values["total_floors"] = 4
    findings, _ = validate(values)
    assert any(finding.rule_id == "cross.floor_le_total" for finding in findings)


def test_price_per_sqm_plausibility_for_sale() -> None:
    values = _complete_sale()
    values["price_eur"] = Decimal("900")  # 10 €/m² is implausible for a sale
    findings, _ = validate(values)
    assert any(finding.rule_id == "plausibility.price_per_sqm" for finding in findings)
