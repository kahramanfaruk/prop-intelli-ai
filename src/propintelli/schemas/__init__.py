"""Public schema surface for PropIntelli AI.

Re-exports the controlled vocabularies, the field registry, the extraction-time
models, and the structured target record so callers can import them from a single
namespace.
"""

from __future__ import annotations

from propintelli.schemas.enums import (
    EnergyClass,
    HeatingType,
    ListingType,
    PriceKind,
    PropertyCondition,
    Provenance,
    ReviewStatus,
)
from propintelli.schemas.extraction import ExtractionResult, FieldValue
from propintelli.schemas.fields import (
    PROPERTY_FIELDS,
    FieldKind,
    FieldSpec,
    field_names,
    get_field,
    required_field_names,
)
from propintelli.schemas.property_record import (
    EnergyProfile,
    Features,
    Location,
    PropertyRecord,
    QualityReport,
    ValidationFinding,
)

__all__ = [
    "PROPERTY_FIELDS",
    "EnergyClass",
    "EnergyProfile",
    "ExtractionResult",
    "Features",
    "FieldKind",
    "FieldSpec",
    "FieldValue",
    "HeatingType",
    "ListingType",
    "Location",
    "PriceKind",
    "PropertyCondition",
    "PropertyRecord",
    "Provenance",
    "QualityReport",
    "ReviewStatus",
    "ValidationFinding",
    "field_names",
    "get_field",
    "required_field_names",
]
