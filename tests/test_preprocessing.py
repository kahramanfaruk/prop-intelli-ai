"""Tests for document classification and text extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from propintelli.config import Settings
from propintelli.errors import DocumentReadError, EmptyDocumentError, OcrUnavailableError
from propintelli.preprocessing import DocumentClass, TextSource, classify, extract_text


@pytest.mark.parametrize(
    ("lengths", "expected"),
    [
        ([200, 300, 250], DocumentClass.DIGITAL),
        ([0, 0, 0], DocumentClass.SCANNED),
        ([200, 5, 300], DocumentClass.HYBRID),
        ([], DocumentClass.SCANNED),
    ],
)
def test_classify(lengths: list[int], expected: DocumentClass) -> None:
    assert classify(lengths, scanned_threshold=50) == expected


def test_extract_text_digital_sample(sample_pdf: Path) -> None:
    result = extract_text(sample_pdf, document_id="doc-1")
    assert result.document_class is DocumentClass.DIGITAL
    assert result.text_source is TextSource.DIGITAL
    assert result.char_count > 0
    assert "Kaufpreis" in result.text
    assert result.page_count >= 1


def test_extract_text_corrupt_raises_read_error(corrupt_pdf: Path) -> None:
    with pytest.raises(DocumentReadError) as exc_info:
        extract_text(corrupt_pdf, document_id="doc-corrupt")
    error = exc_info.value.as_processing_error()
    assert error.error_code == "PRE_001"
    assert error.document_id == "doc-corrupt"


def test_extract_text_blank_without_ocr_raises_empty(blank_pdf: Path) -> None:
    settings = Settings(ocr_enabled=False)
    with pytest.raises(EmptyDocumentError):
        extract_text(blank_pdf, document_id="doc-blank", settings=settings)


def test_extract_text_blank_with_unavailable_ocr_raises_ocr_unavailable(
    blank_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the "OCR requested but Tesseract missing" branch deterministically.
    monkeypatch.setattr(
        "propintelli.preprocessing.text_extractor.tesseract_available",
        lambda: False,
    )
    settings = Settings(ocr_enabled=True)
    with pytest.raises(OcrUnavailableError) as exc_info:
        extract_text(blank_pdf, document_id="doc-scan", settings=settings)
    assert exc_info.value.as_processing_error().error_code == "OCR_001"
