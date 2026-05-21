"""Controlled vocabularies for the property domain.

Centralising the enumerations keeps extraction, normalisation, validation, and
storage aligned on a single set of canonical values. German source terms are
mapped onto these stable English tokens during transformation.
"""

from __future__ import annotations

import enum


class ListingType(enum.StrEnum):
    """Whether a listing offers a property for sale or for rent."""

    SALE = "sale"
    RENT = "rent"


class PriceKind(enum.StrEnum):
    """The kind of headline price quoted in the exposé."""

    PURCHASE = "purchase"
    COLD_RENT = "cold_rent"
    WARM_RENT = "warm_rent"


class EnergyClass(enum.StrEnum):
    """German energy-efficiency classes (Energieeffizienzklasse)."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"


class HeatingType(enum.StrEnum):
    """Canonical heating systems mapped from German source terms."""

    GAS = "gas"
    OIL = "oil"
    DISTRICT_HEATING = "district_heating"
    HEAT_PUMP = "heat_pump"
    ELECTRIC = "electric"
    PELLET = "pellet"
    SOLAR = "solar"
    UNDERFLOOR = "underfloor"
    OTHER = "other"


class PropertyCondition(enum.StrEnum):
    """Building condition (Objektzustand)."""

    NEW_BUILD = "new_build"
    FIRST_OCCUPANCY = "first_occupancy"
    MODERNISED = "modernised"
    RENOVATED = "renovated"
    WELL_KEPT = "well_kept"
    NEEDS_RENOVATION = "needs_renovation"


class ReviewStatus(enum.StrEnum):
    """Routing decision produced by the confidence model.

    Notes
    -----
    The three states implement the human-in-the-loop policy: high-confidence
    records are auto-approved, mid-confidence records are queued for review, and
    low-confidence records require manual correction before promotion.
    """

    AUTO_APPROVED = "auto_approved"
    NEEDS_REVIEW = "needs_review"
    MANUAL_REQUIRED = "manual_required"


class Provenance(enum.StrEnum):
    """Origin of a single extracted field value.

    Attributes
    ----------
    DETERMINISTIC
        Produced by the Layer-A regex/heuristic extractor.
    LLM
        Produced by the Layer-B language-model extractor.
    RECONCILED
        Produced by Layer-C agreement between Layer-A and Layer-B.
    MANUAL
        Supplied or corrected by a human reviewer (HITL).
    """

    DETERMINISTIC = "deterministic"
    LLM = "llm"
    RECONCILED = "reconciled"
    MANUAL = "manual"
