"""Tests for the Bronze store and Silver/Gold medallion layers."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from propintelli.errors import ErrorSeverity, IngestionError
from propintelli.ingestion import DocumentStore
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
from propintelli.storage import ProcessingRunInfo, SilverRepository, build_gold


def _record(property_id: str = "abc123", city: str = "Nürnberg") -> PropertyRecord:
    return PropertyRecord(
        property_id=property_id,
        source_document="expose.pdf",
        listing_type=ListingType.SALE,
        price_eur=Decimal("449000.00"),
        price_kind=PriceKind.PURCHASE,
        living_area_sqm=92.0,
        rooms=3.0,
        year_built=1998,
        condition=PropertyCondition.MODERNISED,
        availability_date=date(2026, 7, 1),
        location=Location(
            street="Bucher Straße", house_number="42", postal_code="90408", city=city
        ),
        features=Features(balcony=True, cellar=True),
        energy=EnergyProfile(energy_class=EnergyClass.C, heating_type=HeatingType.GAS),
        quality=QualityReport(
            overall_confidence=0.91,
            field_confidences={"price_eur": 0.92, "city": 0.88},
            field_provenance={"price_eur": Provenance.RECONCILED, "city": Provenance.DETERMINISTIC},
            completeness=0.8,
            validation_pass_rate=1.0,
            review_status=ReviewStatus.AUTO_APPROVED,
            findings=[
                ValidationFinding(
                    rule_id="format.postal_code",
                    field="postal_code",
                    severity=ErrorSeverity.WARNING,
                    message="example",
                )
            ],
            warnings=["example warning"],
        ),
    )


# --- Bronze ----------------------------------------------------------------
def test_document_store_ingests_and_writes_manifest(tmp_path: Path, sample_pdf: Path) -> None:
    store = DocumentStore(tmp_path / "bronze")
    bronze = store.ingest_path(sample_pdf)
    assert bronze.stored_path.exists()
    assert bronze.size_bytes > 0
    assert len(bronze.sha256) == 64
    assert (bronze.stored_path.parent / "manifest.json").exists()


def test_document_store_rejects_empty(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "bronze")
    with pytest.raises(IngestionError):
        store.ingest_bytes(b"", "empty.pdf")


def test_document_store_iterates_resident_documents(tmp_path: Path, sample_pdf: Path) -> None:
    store = DocumentStore(tmp_path / "bronze")
    ingested = store.ingest_path(sample_pdf)
    # A stray non-document directory must be ignored by the enumerator.
    (tmp_path / "bronze" / "not-a-doc").mkdir()

    entries = list(store.iter_documents())
    assert [entry.document_id for entry in entries] == [ingested.document_id]
    assert entries[0].stored_path.exists()
    assert entries[0].source_document == sample_pdf.name


# --- Silver ----------------------------------------------------------------
def test_silver_round_trip_preserves_record(tmp_path: Path) -> None:
    repo = SilverRepository(tmp_path / "silver.sqlite")
    original = _record()
    repo.save_record(original)
    loaded = repo.get_record(original.property_id)

    assert loaded is not None
    assert loaded.price_eur == Decimal("449000.00")
    assert loaded.listing_type is ListingType.SALE
    assert loaded.location.city == "Nürnberg"
    assert loaded.features.balcony is True
    assert loaded.features.garden is None
    assert loaded.energy.energy_class is EnergyClass.C
    assert loaded.quality.review_status is ReviewStatus.AUTO_APPROVED
    assert loaded.quality.field_provenance["price_eur"] is Provenance.RECONCILED
    assert len(loaded.quality.findings) == 1
    assert loaded.availability_date == date(2026, 7, 1)


def test_silver_upsert_replaces_existing(tmp_path: Path) -> None:
    repo = SilverRepository(tmp_path / "silver.sqlite")
    repo.save_record(_record(city="Nürnberg"))
    repo.save_record(_record(city="Fürth"))  # same id, new city
    assert repo.count() == 1
    loaded = repo.get_record("abc123")
    assert loaded is not None
    assert loaded.location.city == "Fürth"


def test_silver_filters_by_review_status_and_records_runs(tmp_path: Path) -> None:
    repo = SilverRepository(tmp_path / "silver.sqlite")
    repo.save_record(_record(property_id="r1"))
    repo.record_run(
        ProcessingRunInfo(
            document_id="d1",
            source_document="expose.pdf",
            status="succeeded",
            review_status=ReviewStatus.AUTO_APPROVED,
            property_id="r1",
        )
    )
    approved = repo.list_records(review_status=ReviewStatus.AUTO_APPROVED)
    assert [record.property_id for record in approved] == ["r1"]
    assert repo.list_records(review_status=ReviewStatus.MANUAL_REQUIRED) == []


# --- Gold ------------------------------------------------------------------
def test_build_gold_exports_and_summarises(tmp_path: Path) -> None:
    records = [
        _record(property_id="g1", city="Nürnberg"),
        _record(property_id="g2", city="Nürnberg"),
    ]
    artifacts = build_gold(records, tmp_path / "gold")

    assert artifacts.properties_parquet.exists()
    assert artifacts.properties_csv.exists()
    assert artifacts.features_parquet.exists()
    assert artifacts.duckdb_path.exists()
    assert len(artifacts.summary) == 1
    row = artifacts.summary[0]
    assert row["city"] == "Nürnberg"
    assert row["listings"] == 2
    # 449000 / 92 ~= 4880 €/m²
    assert 4800 < float(row["avg_price_per_sqm"]) < 4900
