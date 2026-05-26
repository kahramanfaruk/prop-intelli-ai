"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import fitz
import pytest

from propintelli.config import get_settings
from propintelli.sampledata import SAMPLE_PROPERTIES, generate_samples


@pytest.fixture(autouse=True)
def _isolated_data_dir(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Route the configured data directory to a temp dir for every test.

    Keeps the repository clean: any Bronze/Silver/Gold artifacts written via the
    process settings land under a per-test temporary directory.
    """
    monkeypatch.setenv("PROPINTELLI_DATA_DIR", str(tmp_path / "propintelli-data"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    """Create an image-only PDF (rendered text, no text layer) simulating a scan.

    The page text is rasterised into an image and re-embedded, so direct text
    extraction yields nothing (the page classifies as ``SCANNED``) while a real
    OCR backend could still recover the rendered words.
    """
    source = fitz.open()
    page = source.new_page()
    page.insert_text((72, 96), "Kaufpreis 425000 EUR", fontsize=18)
    page.insert_text((72, 132), "Wohnflaeche 88 m2", fontsize=18)
    pixmap = page.get_pixmap(dpi=150)

    path = tmp_path / "scanned.pdf"
    image_only = fitz.open()
    image_page = image_only.new_page(width=pixmap.width, height=pixmap.height)
    image_page.insert_image(image_page.rect, pixmap=pixmap)
    image_only.save(str(path))
    image_only.close()
    source.close()
    return path
