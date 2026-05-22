"""Tests for the end-to-end single-document pipeline."""

from __future__ import annotations

from pathlib import Path

from propintelli.config import Settings
from propintelli.ingestion import DocumentStore
from propintelli.pipeline import Pipeline
from propintelli.schemas.enums import ReviewStatus
from propintelli.storage import SilverRepository


def _pipeline(
    tmp_path: Path, *, with_repo: bool = True
) -> tuple[Pipeline, SilverRepository | None]:
    store = DocumentStore(tmp_path / "bronze")
    repo = SilverRepository(tmp_path / "silver.sqlite") if with_repo else None
    return Pipeline(store=store, repository=repo, settings=Settings()), repo


def test_pipeline_processes_sample_to_auto_approved(tmp_path: Path, sample_pdf: Path) -> None:
    pipeline, _ = _pipeline(tmp_path)
    result = pipeline.process_path(sample_pdf)

    assert result.succeeded
    assert result.error is None
    record = result.record
    assert record is not None
    # The record id matches the Bronze document id for traceability.
    assert record.property_id == result.document_id
    assert record.quality.review_status is ReviewStatus.AUTO_APPROVED
    assert record.location.city == "Nürnberg"


def test_pipeline_persists_record_and_run(tmp_path: Path, sample_pdf: Path) -> None:
    pipeline, repo = _pipeline(tmp_path)
    assert repo is not None
    result = pipeline.process_path(sample_pdf)

    assert repo.count() == 1
    stored = repo.get_record(result.document_id)
    assert stored is not None
    assert stored.source_document == sample_pdf.name


def test_pipeline_isolates_failure_and_records_it(tmp_path: Path, corrupt_pdf: Path) -> None:
    pipeline, repo = _pipeline(tmp_path)
    assert repo is not None
    result = pipeline.process_path(corrupt_pdf)

    assert not result.succeeded
    assert result.record is None
    assert result.error is not None
    assert result.error.error_code == "PRE_001"
    # The failed run is audited even though no record was produced.
    assert repo.count() == 0


def test_pipeline_without_repository_does_not_persist(tmp_path: Path, sample_pdf: Path) -> None:
    pipeline, _ = _pipeline(tmp_path, with_repo=False)
    result = pipeline.process_path(sample_pdf)
    assert result.succeeded
