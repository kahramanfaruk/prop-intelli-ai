"""Tests for the batch runner."""

from __future__ import annotations

import shutil
from pathlib import Path

from propintelli.batch import run_batch
from propintelli.config import Settings
from propintelli.ingestion import DocumentStore
from propintelli.pipeline import Pipeline
from propintelli.schemas.enums import ReviewStatus
from propintelli.storage import SilverRepository


def _pipeline(tmp_path: Path) -> Pipeline:
    return Pipeline(
        store=DocumentStore(tmp_path / "bronze"),
        repository=SilverRepository(tmp_path / "silver.sqlite"),
        settings=Settings(),
    )


def test_batch_processes_full_corpus(tmp_path: Path, sample_corpus: tuple[Path, Path]) -> None:
    raw_dir, _ = sample_corpus
    report = run_batch(raw_dir, _pipeline(tmp_path), show_progress=False)

    assert report.total == 10
    assert report.failed == 0
    assert report.succeeded == 10
    assert report.by_status[ReviewStatus.AUTO_APPROVED.value] >= 1


def test_batch_isolates_a_failing_document(
    tmp_path: Path, sample_corpus: tuple[Path, Path]
) -> None:
    raw_dir, _ = sample_corpus
    work_dir = tmp_path / "mixed"
    work_dir.mkdir()
    # One valid document plus one corrupt file in the same folder.
    shutil.copy(next(raw_dir.glob("*.pdf")), work_dir / "valid.pdf")
    (work_dir / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")

    report = run_batch(work_dir, _pipeline(tmp_path), show_progress=False)
    assert report.total == 2
    assert report.succeeded == 1
    assert report.failed == 1
    assert report.errors[0].error_code == "PRE_001"
