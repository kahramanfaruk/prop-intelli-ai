"""Normalisation: raw extracted strings into typed, canonical values.

Each reconciled :class:`~propintelli.schemas.extraction.FieldValue` carries a raw
string and a provenance. Normalisation converts that string into the Python type
declared for the field in the registry (Decimal, float, int, date, bool, or a
canonical enum), parsing numbers under the convention implied by provenance. The
per-field confidence and provenance are carried through so downstream scoring and
the UI can reason about them.

This is the schema-enforcement / standardisation step: every field ends up with a
consistent Python type regardless of its source, the deterministic rules
(German-formatted tables), the LLM (free-form text), or any future backend.
Concretely:

* ``price_eur`` -> a :class:`~decimal.Decimal` with two decimal places;
* ``living_area_sqm`` -> a :class:`float`;
* ``rooms`` -> a :class:`float` (German listings quote half-rooms, e.g. 2.5);
* ``year_built`` -> an :class:`int`;
* ``availability_date`` -> a :class:`~datetime.date`;
* ``energy_class`` -> an :class:`~propintelli.schemas.enums.EnergyClass` member;
* ``balcony`` / ``cellar`` -> a :class:`bool`, or ``None`` when not stated.

So ``"450.000 EUR"``, ``"450000"``, and ``"450,000.00"`` all collapse to the same
typed value (``Decimal("450000.00")``).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from propintelli.schemas.enums import Provenance
from propintelli.schemas.extraction import ExtractionResult, FieldValue
from propintelli.schemas.fields import PROPERTY_FIELDS, FieldKind, FieldSpec
from propintelli.transformation.parsing import parse_bool, parse_number

_DATE_FORMATS = ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%y")


@dataclass(frozen=True, slots=True)
class NormalizedFields:
    """Typed field values plus their quality metadata.

    Attributes
    ----------
    values : dict of str to object
        Canonical field name to typed, normalised value.
    confidences : dict of str to float
        Per-field confidence carried over from extraction.
    provenance : dict of str to Provenance
        Per-field origin carried over from extraction.
    warnings : list of str
        Notes about values that could not be normalised.
    """

    values: dict[str, Any] = field(default_factory=dict)
    confidences: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Provenance] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _parse_date(raw: str) -> date | None:
    """Parse a date from the common German and ISO formats."""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _coerce_enum(spec: FieldSpec, raw: str) -> enum.StrEnum | None:
    """Coerce a raw token into the field's enum, case-insensitively."""
    enum_type = spec.enum_type
    if enum_type is None:
        return None
    candidate = raw.strip()
    try:
        return enum_type(candidate)
    except ValueError:
        lowered = candidate.lower()
        return next((member for member in enum_type if member.value.lower() == lowered), None)


def normalize_value(spec: FieldSpec, value: FieldValue) -> Any | None:
    """Normalise a single field value to its declared type.

    Parameters
    ----------
    spec : FieldSpec
        Registry metadata for the field.
    value : FieldValue
        The reconciled raw value.

    Returns
    -------
    object or None
        The typed value, or ``None`` if the raw value could not be parsed.
    """
    raw = value.raw_value
    if raw is None or not raw.strip():
        return None
    # Numbers are parsed under the locale implied by provenance. The deterministic
    # layer emits German-formatted numbers; the LLM is instructed to emit
    # dot-decimal ones. A reconciled value carries the deterministic layer's
    # German-formatted string (reconciliation's agreement branch preserves it),
    # so it must parse as German too, otherwise "449.000" would be read as 449.0.
    german = value.provenance in {Provenance.DETERMINISTIC, Provenance.RECONCILED}

    if spec.kind is FieldKind.STRING:
        return raw.strip()
    if spec.kind is FieldKind.BOOLEAN:
        return parse_bool(raw)
    if spec.kind is FieldKind.DATE:
        return _parse_date(raw)
    if spec.kind is FieldKind.ENUM:
        return _coerce_enum(spec, raw)

    number = parse_number(raw, german=german)
    if number is None:
        return None
    if spec.kind is FieldKind.INTEGER:
        return round(number)
    if spec.kind is FieldKind.FLOAT:
        return float(number)
    try:
        return Decimal(str(number)).quantize(Decimal("0.01"))
    except InvalidOperation:  # pragma: no cover - defensive
        return None


def normalize(extraction: ExtractionResult) -> NormalizedFields:
    """Normalise every extracted field into typed values.

    Parameters
    ----------
    extraction : ExtractionResult
        The reconciled extraction output.

    Returns
    -------
    NormalizedFields
        Typed values with carried-through confidence and provenance, plus any
        normalisation warnings.
    """
    result = NormalizedFields()
    for name, raw_value in extraction.fields.items():
        spec = PROPERTY_FIELDS.get(name)
        if spec is None or not raw_value.is_present:
            continue
        typed = normalize_value(spec, raw_value)
        if typed is None:
            result.warnings.append(f"Could not normalise {name} from {raw_value.raw_value!r}")
            continue
        result.values[name] = typed
        result.confidences[name] = raw_value.confidence
        result.provenance[name] = raw_value.provenance
    return result
