"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from propintelli.sampledata import SAMPLE_PROPERTIES, generate_samples


@pytest.fixture(scope="session")
def sample_corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Generate the sample corpus once per test session.

    Returns
    -------
    tuple of (Path, Path)
        ``(raw_dir, ground_truth_dir)`` containing the generated PDFs and labels.
    """
    root = tmp_path_factory.mktemp("corpus")
    raw_dir = root / "raw"
    truth_dir = root / "ground_truth"
    generate_samples(raw_dir, truth_dir)
    return raw_dir, truth_dir


@pytest.fixture
def sample_pdf(sample_corpus: tuple[Path, Path]) -> Path:
    """Return the path to the first generated sample exposé PDF."""
    raw_dir, _ = sample_corpus
    return raw_dir / f"{SAMPLE_PROPERTIES[0].document_stem}.pdf"


@pytest.fixture
def blank_pdf(tmp_path: Path) -> Path:
    """Create a single-page PDF with no text layer (simulates a scan)."""
    path = tmp_path / "blank.pdf"
    document = fitz.open()
    document.new_page()
    document.save(str(path))
    document.close()
    return path


@pytest.fixture
def corrupt_pdf(tmp_path: Path) -> Path:
    """Create a file with a ``.pdf`` suffix that is not a valid PDF."""
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.4 this is not a real pdf body")
    return path
