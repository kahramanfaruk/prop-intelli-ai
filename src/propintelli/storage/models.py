"""SQLAlchemy ORM models for the Silver layer.

The schema is deliberately normalised: a central ``properties`` row holds the
scalar attributes and quality summary, while sparse and repeating data live in
child tables (``property_features``, ``validation_findings``). Per-field
confidence and provenance are kept as JSON on the property row: they are always
read together with the record and never queried relationally. A ``processing_runs``
table records every pipeline execution, including failures, for auditability.

The repository targets SQLite locally; because everything goes through
SQLAlchemy, the same models map onto Azure SQL or PostgreSQL in production.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import JSON, DateTime, ForeignKey, Numeric, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all Silver-layer tables."""


def _utcnow() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class Property(Base):
    """A validated property record (one row per processed document)."""

    __tablename__ = "properties"

    property_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    source_document: Mapped[str] = mapped_column(String(512))
    schema_version: Mapped[str] = mapped_column(String(16))
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    listing_type: Mapped[str | None] = mapped_column(String(16))
    price_eur: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    price_kind: Mapped[str | None] = mapped_column(String(16))
    price_per_sqm: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    living_area_sqm: Mapped[float | None] = mapped_column()
    plot_area_sqm: Mapped[float | None] = mapped_column()
    rooms: Mapped[float | None] = mapped_column()
    floor: Mapped[int | None] = mapped_column()
    total_floors: Mapped[int | None] = mapped_column()
    year_built: Mapped[int | None] = mapped_column()
    condition: Mapped[str | None] = mapped_column(String(32))
    availability_date: Mapped[date | None] = mapped_column()
    title: Mapped[str | None] = mapped_column(String(512))

    street: Mapped[str | None] = mapped_column(String(256))
    house_number: Mapped[str | None] = mapped_column(String(32))
    postal_code: Mapped[str | None] = mapped_column(String(16))
    city: Mapped[str | None] = mapped_column(String(128))
    district: Mapped[str | None] = mapped_column(String(128))
    country: Mapped[str] = mapped_column(String(8), default="DE")

    energy_class: Mapped[str | None] = mapped_column(String(8))
    heating_type: Mapped[str | None] = mapped_column(String(32))
    energy_demand_kwh: Mapped[float | None] = mapped_column()
    energy_certificate_type: Mapped[str | None] = mapped_column(String(64))

    overall_confidence: Mapped[float] = mapped_column()
    completeness: Mapped[float] = mapped_column()
    validation_pass_rate: Mapped[float] = mapped_column()
    review_status: Mapped[str] = mapped_column(String(32))
    field_confidences: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    field_provenance: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    features: Mapped[list[PropertyFeature]] = relationship(
        back_populates="property", cascade="all, delete-orphan", lazy="selectin"
    )
    findings: Mapped[list[ValidationFindingRow]] = relationship(
        back_populates="property", cascade="all, delete-orphan", lazy="selectin"
    )


class PropertyFeature(Base):
    """A single boolean equipment feature for a property."""

    __tablename__ = "property_features"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    property_id: Mapped[str] = mapped_column(
        ForeignKey("properties.property_id", ondelete="CASCADE")
    )
    feature_name: Mapped[str] = mapped_column(String(32))
    value: Mapped[bool] = mapped_column()

    property: Mapped[Property] = relationship(back_populates="features")


class ValidationFindingRow(Base):
    """A persisted validation finding for a property."""

    __tablename__ = "validation_findings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    property_id: Mapped[str] = mapped_column(
        ForeignKey("properties.property_id", ondelete="CASCADE")
    )
    rule_id: Mapped[str] = mapped_column(String(64))
    field: Mapped[str | None] = mapped_column(String(64))
    severity: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(String(512))

    property: Mapped[Property] = relationship(back_populates="findings")


class ProcessingRun(Base):
    """An audit record of a single pipeline execution."""

    __tablename__ = "processing_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(64), index=True)
    source_document: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(16))
    review_status: Mapped[str | None] = mapped_column(String(32))
    property_id: Mapped[str | None] = mapped_column(String(32))
    error_code: Mapped[str | None] = mapped_column(String(16))
    error_message: Mapped[str | None] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
