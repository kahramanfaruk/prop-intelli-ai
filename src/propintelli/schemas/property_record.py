"""The structured target schema (Silver record).

:class:`PropertyRecord` is the validated, normalised output of the pipeline. It
holds typed property attributes plus a :class:`QualityReport` that records
per-field confidence, provenance, validation findings, and the human-in-the-loop
routing decision.

Design choice: the schema is intentionally *permissive* on value ranges. An
implausible value (e.g. a negative price from a mis-parse) is kept and flagged by
the validation layer rather than rejected outright — failures are routed to human
review, not silently dropped. Type correctness is guaranteed upstream by the
normalisation step.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from propintelli.errors import ErrorSeverity
from propintelli.schemas.enums import (
    EnergyClass,
    HeatingType,
    ListingType,
    PriceKind,
    PropertyCondition,
    Provenance,
    ReviewStatus,
)


class Location(BaseModel):
    """Geographic location of the property."""

    street: str | None = None
    house_number: str | None = None
    postal_code: str | None = None
    city: str | None = None
    district: str | None = None
    country: str = "DE"


class Features(BaseModel):
    """Boolean equipment features (Ausstattung).

    Notes
    -----
    Each feature is tri-state: ``True`` (present), ``False`` (explicitly absent),
    or ``None`` (not stated in the document). Distinguishing "absent" from "not
    stated" prevents the pipeline from asserting facts the source never made.
    """

    balcony: bool | None = None
    terrace: bool | None = None
    garden: bool | None = None
    parking: bool | None = None
    cellar: bool | None = None
    elevator: bool | None = None
    fitted_kitchen: bool | None = None
    furnished: bool | None = None
    barrier_free: bool | None = None


class EnergyProfile(BaseModel):
    """Energy-related attributes (Energieausweis)."""

    energy_class: EnergyClass | None = None
    heating_type: HeatingType | None = None
    energy_demand_kwh: float | None = None
    energy_certificate_type: str | None = None


class ValidationFinding(BaseModel):
    """A single result from the validation rules engine.

    Attributes
    ----------
    rule_id : str
        Stable identifier of the rule that produced the finding.
    field : str or None
        Canonical field the finding relates to, or ``None`` for cross-field
        rules.
    severity : ErrorSeverity
        Severity of the finding.
    message : str
        Human-readable explanation.
    """

    rule_id: str
    field: str | None
    severity: ErrorSeverity
    message: str


class QualityReport(BaseModel):
    """Aggregated data-quality metadata attached to a record.

    Attributes
    ----------
    overall_confidence : float
        Weighted overall confidence in ``[0, 1]``.
    field_confidences : dict of str to float
        Confidence per canonical field.
    field_provenance : dict of str to Provenance
        Origin per canonical field.
    completeness : float
        Fraction of the *required* fields (price, living area, postal code,
        city) that were populated, in ``[0, 1]``. Optional fields are
        legitimately sparse and do not count against completeness.
    validation_pass_rate : float
        Fraction of executed validation rules that passed, in ``[0, 1]``.
    review_status : ReviewStatus
        The human-in-the-loop routing decision.
    findings : list of ValidationFinding
        Validation findings (warnings and errors).
    warnings : list of str
        Free-text processing warnings.
    """

    overall_confidence: float = Field(ge=0.0, le=1.0)
    field_confidences: dict[str, float] = Field(default_factory=dict)
    field_provenance: dict[str, Provenance] = Field(default_factory=dict)
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    validation_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    review_status: ReviewStatus = ReviewStatus.MANUAL_REQUIRED
    findings: list[ValidationFinding] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PropertyRecord(BaseModel):
    """A validated, normalised real-estate listing record.

    This is the canonical Silver-layer artifact and the unit persisted to
    storage and exported to the Gold layer.
    """

    model_config = ConfigDict(validate_assignment=True)

    schema_version: str = "1.0"
    property_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source_document: str
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    title: str | None = None
    listing_type: ListingType | None = None
    price_eur: Decimal | None = None
    price_kind: PriceKind | None = None
    living_area_sqm: float | None = None
    plot_area_sqm: float | None = None
    rooms: float | None = None
    floor: int | None = None
    total_floors: int | None = None
    year_built: int | None = None
    condition: PropertyCondition | None = None
    availability_date: date | None = None

    location: Location = Field(default_factory=Location)
    features: Features = Field(default_factory=Features)
    energy: EnergyProfile = Field(default_factory=EnergyProfile)

    quality: QualityReport

    @computed_field  # type: ignore[prop-decorator]
    @property
    def price_per_sqm(self) -> Decimal | None:
        """Derived purchase/rent price per square metre of living area.

        Returns
        -------
        Decimal or None
            ``price_eur / living_area_sqm`` rounded to two decimals, or ``None``
            when either input is missing or the area is non-positive.
        """
        if self.price_eur is None or self.living_area_sqm is None:
            return None
        if self.living_area_sqm <= 0:
            return None
        return (self.price_eur / Decimal(str(self.living_area_sqm))).quantize(Decimal("0.01"))
