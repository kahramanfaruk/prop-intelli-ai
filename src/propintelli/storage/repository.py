"""Silver layer — persistence of validated records via SQLAlchemy.

The :class:`SilverRepository` translates between the Pydantic
:class:`~propintelli.schemas.property_record.PropertyRecord` and the relational
ORM, exposing upsert/read operations and a processing-run audit log. SQLite is
used locally; the repository pattern keeps the rest of the codebase ignorant of
the backend, so swapping to Azure SQL or PostgreSQL is a connection-string change.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel
from sqlalchemy import NullPool, create_engine, select
from sqlalchemy.orm import sessionmaker

from propintelli.errors import ErrorSeverity, StorageError
from propintelli.logging_setup import get_logger
from propintelli.schemas.enums import (
    EnergyClass,
    HeatingType,
    ListingType,
    PriceKind,
    PropertyCondition,
    Provenance,
    ReviewStatus,
)
from propintelli.schemas.property_record import (
    EnergyProfile,
    Features,
    Location,
    PropertyRecord,
    QualityReport,
    ValidationFinding,
)
from propintelli.storage.models import (
    Base,
    ProcessingRun,
    Property,
    PropertyFeature,
    ValidationFindingRow,
)

logger = get_logger(__name__)

_FEATURE_NAMES = tuple(Features.model_fields)
_EnumT = TypeVar("_EnumT", bound=enum.StrEnum)


def _to_enum(enum_type: type[_EnumT], value: str | None) -> _EnumT | None:
    """Reconstruct an enum member from its stored value, or ``None``."""
    return enum_type(value) if value is not None else None


class ProcessingRunInfo(BaseModel):
    """A processing-run audit entry."""

    document_id: str
    source_document: str
    status: str
    review_status: ReviewStatus | None = None
    property_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class SilverRepository:
    """Persistence gateway for validated property records."""

    def __init__(self, db_path: Path) -> None:
        """Open (creating if needed) the Silver database at ``db_path``.

        Parameters
        ----------
        db_path : Path
            Path to the SQLite database file.
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # NullPool closes each connection on return, which suits the discrete
        # save/get/list access pattern over file-based SQLite and avoids leaking
        # pooled connections across many short-lived repositories.
        self._engine = create_engine(f"sqlite:///{db_path}", poolclass=NullPool)
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)

    def dispose(self) -> None:
        """Release the underlying engine and its connection pool.

        Call this when a repository is no longer needed in a long-running host;
        short-lived processes can rely on interpreter teardown.
        """
        self._engine.dispose()

    def save_record(self, record: PropertyRecord) -> None:
        """Insert or replace a property record.

        Parameters
        ----------
        record : PropertyRecord
            The validated record to persist.

        Raises
        ------
        StorageError
            If the record cannot be written.
        """
        try:
            with self._session_factory.begin() as session:
                existing = session.get(Property, record.property_id)
                if existing is not None:
                    session.delete(existing)
                    session.flush()
                session.add(_to_orm(record))
        except Exception as exc:  # surface any backend failure as a catalogued error
            raise StorageError(
                f"Failed to persist record {record.property_id}: {exc}",
                document_id=record.property_id,
            ) from exc

    def get_record(self, property_id: str) -> PropertyRecord | None:
        """Return a record by id, or ``None`` if absent.

        Parameters
        ----------
        property_id : str
            The record identifier.

        Returns
        -------
        PropertyRecord or None
            The reconstructed record.
        """
        with self._session_factory() as session:
            orm = session.get(Property, property_id)
            return _to_record(orm) if orm is not None else None

    def list_records(
        self,
        *,
        review_status: ReviewStatus | None = None,
        limit: int | None = None,
    ) -> list[PropertyRecord]:
        """List stored records, optionally filtered by review status.

        Parameters
        ----------
        review_status : ReviewStatus or None, optional
            If given, only records with this status are returned.
        limit : int or None, optional
            Maximum number of records to return.

        Returns
        -------
        list of PropertyRecord
            The matching records, newest first.
        """
        statement = select(Property).order_by(Property.created_at.desc())
        if review_status is not None:
            statement = statement.where(Property.review_status == review_status.value)
        if limit is not None:
            statement = statement.limit(limit)
        with self._session_factory() as session:
            return [_to_record(orm) for orm in session.scalars(statement)]

    def record_run(self, info: ProcessingRunInfo) -> None:
        """Append a processing-run audit entry.

        Parameters
        ----------
        info : ProcessingRunInfo
            The run outcome to record.
        """
        with self._session_factory.begin() as session:
            session.add(
                ProcessingRun(
                    document_id=info.document_id,
                    source_document=info.source_document,
                    status=info.status,
                    review_status=info.review_status.value if info.review_status else None,
                    property_id=info.property_id,
                    error_code=info.error_code,
                    error_message=info.error_message,
                )
            )

    def count(self) -> int:
        """Return the number of stored property records."""
        with self._session_factory() as session:
            return len(list(session.scalars(select(Property.property_id))))

    def processed_document_ids(self) -> set[str]:
        """Return the ids of documents that already have a processing-run audit.

        Includes both succeeded and failed runs, so a document is attempted at
        most once by the Bronze watcher and a permanently-unreadable file is not
        retried on every poll.

        Returns
        -------
        set of str
            Document ids recorded in the processing-run audit log.
        """
        with self._session_factory() as session:
            return set(session.scalars(select(ProcessingRun.document_id)))


def _to_orm(record: PropertyRecord) -> Property:
    """Map a :class:`PropertyRecord` onto the ORM graph."""
    quality = record.quality
    orm = Property(
        property_id=record.property_id,
        source_document=record.source_document,
        schema_version=record.schema_version,
        extracted_at=record.extracted_at,
        listing_type=record.listing_type.value if record.listing_type else None,
        price_eur=record.price_eur,
        price_kind=record.price_kind.value if record.price_kind else None,
        price_per_sqm=record.price_per_sqm,
        living_area_sqm=record.living_area_sqm,
        plot_area_sqm=record.plot_area_sqm,
        rooms=record.rooms,
        floor=record.floor,
        total_floors=record.total_floors,
        year_built=record.year_built,
        condition=record.condition.value if record.condition else None,
        availability_date=record.availability_date,
        title=record.title,
        street=record.location.street,
        house_number=record.location.house_number,
        postal_code=record.location.postal_code,
        city=record.location.city,
        district=record.location.district,
        country=record.location.country,
        energy_class=record.energy.energy_class.value if record.energy.energy_class else None,
        heating_type=record.energy.heating_type.value if record.energy.heating_type else None,
        energy_demand_kwh=record.energy.energy_demand_kwh,
        energy_certificate_type=record.energy.energy_certificate_type,
        overall_confidence=quality.overall_confidence,
        completeness=quality.completeness,
        validation_pass_rate=quality.validation_pass_rate,
        review_status=quality.review_status.value,
        field_confidences=dict(quality.field_confidences),
        field_provenance={name: prov.value for name, prov in quality.field_provenance.items()},
        warnings=list(quality.warnings),
    )
    orm.features = [
        PropertyFeature(feature_name=name, value=value)
        for name in _FEATURE_NAMES
        if (value := getattr(record.features, name)) is not None
    ]
    orm.findings = [
        ValidationFindingRow(
            rule_id=finding.rule_id,
            field=finding.field,
            severity=finding.severity.value,
            message=finding.message,
        )
        for finding in quality.findings
    ]
    return orm


def _to_record(orm: Property) -> PropertyRecord:
    """Reconstruct a :class:`PropertyRecord` from the ORM graph."""
    location = Location(
        street=orm.street,
        house_number=orm.house_number,
        postal_code=orm.postal_code,
        city=orm.city,
        district=orm.district,
        country=orm.country,
    )
    features = Features(**{feature.feature_name: feature.value for feature in orm.features})
    energy = EnergyProfile(
        energy_class=_to_enum(EnergyClass, orm.energy_class),
        heating_type=_to_enum(HeatingType, orm.heating_type),
        energy_demand_kwh=orm.energy_demand_kwh,
        energy_certificate_type=orm.energy_certificate_type,
    )
    quality = QualityReport(
        overall_confidence=orm.overall_confidence,
        field_confidences=dict(orm.field_confidences),
        field_provenance={name: Provenance(value) for name, value in orm.field_provenance.items()},
        completeness=orm.completeness,
        validation_pass_rate=orm.validation_pass_rate,
        review_status=ReviewStatus(orm.review_status),
        findings=[
            ValidationFinding(
                rule_id=row.rule_id,
                field=row.field,
                severity=ErrorSeverity(row.severity),
                message=row.message,
            )
            for row in orm.findings
        ],
        warnings=list(orm.warnings),
    )
    return PropertyRecord(
        schema_version=orm.schema_version,
        property_id=orm.property_id,
        source_document=orm.source_document,
        extracted_at=orm.extracted_at,
        title=orm.title,
        listing_type=_to_enum(ListingType, orm.listing_type),
        price_eur=orm.price_eur,
        price_kind=_to_enum(PriceKind, orm.price_kind),
        living_area_sqm=orm.living_area_sqm,
        plot_area_sqm=orm.plot_area_sqm,
        rooms=orm.rooms,
        floor=orm.floor,
        total_floors=orm.total_floors,
        year_built=orm.year_built,
        condition=_to_enum(PropertyCondition, orm.condition),
        availability_date=orm.availability_date,
        location=location,
        features=features,
        energy=energy,
        quality=quality,
    )
