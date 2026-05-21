"""Canonical field registry.

A single declarative source of truth for every field the platform extracts. It
is consumed by the deterministic extractor, the LLM prompt builder, the
transformation/normalisation step, the validation rules, the completeness and
confidence scoring, the record assembler, the evaluation harness, and the UI.

Defining each field once — with its data kind, its location inside
:class:`~propintelli.schemas.property_record.PropertyRecord`, whether it is
mandatory, and its associated enum — eliminates the drift that otherwise creeps
in when the same field list is hand-maintained in many modules.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from propintelli.schemas.enums import (
    EnergyClass,
    HeatingType,
    ListingType,
    PriceKind,
    PropertyCondition,
)


class FieldKind(enum.StrEnum):
    """The data kind of a field, driving parsing and comparison."""

    STRING = "string"
    DECIMAL = "decimal"
    FLOAT = "float"
    INTEGER = "integer"
    DATE = "date"
    BOOLEAN = "boolean"
    ENUM = "enum"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Declarative metadata for a single extractable field.

    Attributes
    ----------
    name : str
        Canonical, flat field key (e.g. ``"postal_code"``). Used as the key in
        :class:`~propintelli.schemas.extraction.ExtractionResult.fields`.
    kind : FieldKind
        Data kind controlling normalisation and comparison.
    record_path : tuple of str
        Dotted path of the field inside ``PropertyRecord`` (e.g.
        ``("location", "postal_code")``), used by the record assembler.
    label : str
        Human-readable label for the UI.
    required : bool
        Whether the field is mandatory for a record to be considered complete.
    enum_type : type[enum.StrEnum] or None
        The enum the value must belong to, for ``FieldKind.ENUM`` fields.
    """

    name: str
    kind: FieldKind
    record_path: tuple[str, ...]
    label: str
    required: bool = False
    enum_type: type[enum.StrEnum] | None = None

    @property
    def is_numeric(self) -> bool:
        """Whether the field holds a numeric value (decimal/float/integer)."""
        return self.kind in {FieldKind.DECIMAL, FieldKind.FLOAT, FieldKind.INTEGER}


def _spec(
    name: str,
    kind: FieldKind,
    record_path: tuple[str, ...],
    label: str,
    *,
    required: bool = False,
    enum_type: type[enum.StrEnum] | None = None,
) -> tuple[str, FieldSpec]:
    """Build a ``(name, FieldSpec)`` pair for the registry mapping."""
    return name, FieldSpec(
        name=name,
        kind=kind,
        record_path=record_path,
        label=label,
        required=required,
        enum_type=enum_type,
    )


# The registry. Insertion order defines the canonical display/report order.
PROPERTY_FIELDS: dict[str, FieldSpec] = dict(
    [
        _spec("title", FieldKind.STRING, ("title",), "Title"),
        _spec(
            "listing_type",
            FieldKind.ENUM,
            ("listing_type",),
            "Listing type",
            enum_type=ListingType,
        ),
        _spec("price_eur", FieldKind.DECIMAL, ("price_eur",), "Price (EUR)", required=True),
        _spec(
            "price_kind",
            FieldKind.ENUM,
            ("price_kind",),
            "Price kind",
            enum_type=PriceKind,
        ),
        _spec(
            "living_area_sqm",
            FieldKind.FLOAT,
            ("living_area_sqm",),
            "Living area (m2)",
            required=True,
        ),
        _spec("plot_area_sqm", FieldKind.FLOAT, ("plot_area_sqm",), "Plot area (m2)"),
        _spec("rooms", FieldKind.FLOAT, ("rooms",), "Rooms"),
        _spec("floor", FieldKind.INTEGER, ("floor",), "Floor"),
        _spec("total_floors", FieldKind.INTEGER, ("total_floors",), "Total floors"),
        _spec("year_built", FieldKind.INTEGER, ("year_built",), "Year built"),
        _spec(
            "condition",
            FieldKind.ENUM,
            ("condition",),
            "Condition",
            enum_type=PropertyCondition,
        ),
        _spec("availability_date", FieldKind.DATE, ("availability_date",), "Available from"),
        # Location ----------------------------------------------------------
        _spec("street", FieldKind.STRING, ("location", "street"), "Street"),
        _spec("house_number", FieldKind.STRING, ("location", "house_number"), "House no."),
        _spec(
            "postal_code",
            FieldKind.STRING,
            ("location", "postal_code"),
            "Postal code",
            required=True,
        ),
        _spec("city", FieldKind.STRING, ("location", "city"), "City", required=True),
        _spec("district", FieldKind.STRING, ("location", "district"), "District"),
        # Features ----------------------------------------------------------
        _spec("balcony", FieldKind.BOOLEAN, ("features", "balcony"), "Balcony"),
        _spec("terrace", FieldKind.BOOLEAN, ("features", "terrace"), "Terrace"),
        _spec("garden", FieldKind.BOOLEAN, ("features", "garden"), "Garden"),
        _spec("parking", FieldKind.BOOLEAN, ("features", "parking"), "Parking"),
        _spec("cellar", FieldKind.BOOLEAN, ("features", "cellar"), "Cellar"),
        _spec("elevator", FieldKind.BOOLEAN, ("features", "elevator"), "Elevator"),
        _spec(
            "fitted_kitchen",
            FieldKind.BOOLEAN,
            ("features", "fitted_kitchen"),
            "Fitted kitchen",
        ),
        _spec("furnished", FieldKind.BOOLEAN, ("features", "furnished"), "Furnished"),
        _spec(
            "barrier_free",
            FieldKind.BOOLEAN,
            ("features", "barrier_free"),
            "Barrier-free",
        ),
        # Energy ------------------------------------------------------------
        _spec(
            "energy_class",
            FieldKind.ENUM,
            ("energy", "energy_class"),
            "Energy class",
            enum_type=EnergyClass,
        ),
        _spec(
            "heating_type",
            FieldKind.ENUM,
            ("energy", "heating_type"),
            "Heating type",
            enum_type=HeatingType,
        ),
        _spec(
            "energy_demand_kwh",
            FieldKind.FLOAT,
            ("energy", "energy_demand_kwh"),
            "Energy demand (kWh/m2a)",
        ),
        _spec(
            "energy_certificate_type",
            FieldKind.STRING,
            ("energy", "energy_certificate_type"),
            "Energy certificate",
        ),
    ]
)


def field_names() -> tuple[str, ...]:
    """Return all canonical field names in registry order.

    Returns
    -------
    tuple of str
        The ordered field names.
    """
    return tuple(PROPERTY_FIELDS.keys())


def required_field_names() -> tuple[str, ...]:
    """Return the names of the mandatory fields.

    Returns
    -------
    tuple of str
        Names of fields marked ``required`` in the registry.
    """
    return tuple(name for name, spec in PROPERTY_FIELDS.items() if spec.required)


def get_field(name: str) -> FieldSpec:
    """Look up a field specification by name.

    Parameters
    ----------
    name : str
        Canonical field name.

    Returns
    -------
    FieldSpec
        The matching specification.

    Raises
    ------
    KeyError
        If no field with that name is registered.
    """
    return PROPERTY_FIELDS[name]
