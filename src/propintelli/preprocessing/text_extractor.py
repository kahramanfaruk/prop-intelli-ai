"""Extract text from a PDF, with a guarded optional OCR fallback.

The digital-text path (PyMuPDF) is the primary, lossless route and is what the
synthetic corpus and most real exposés exercise. OCR (Tesseract) is engaged only
when a page lacks a usable text layer *and* OCR is both enabled and available.
The OCR backend is imported lazily and probed at runtime, so the package imports
and the digital path run on machines without Tesseract installed.
"""

from __future__ import annotations

import enum
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from pydantic import BaseModel, Field

from propintelli.config import Settings, get_settings
from propintelli.errors import DocumentReadError, EmptyDocumentError, OcrUnavailableError
from propintelli.logging_setup import get_logger
from propintelli.preprocessing.pdf_classifier import DocumentClass, classify

logger = get_logger(__name__)


class TextSource(enum.StrEnum):
    """Where a document's extracted text ultimately came from."""

    DIGITAL = "digital"
    OCR = "ocr"
    HYBRID = "hybrid"


class PageStat(BaseModel):
    """Per-page extraction bookkeeping."""

    index: int
    char_count: int
    used_ocr: bool


class PreprocessedDocument(BaseModel):
    """The text and provenance metadata produced by preprocessing.

    Attributes
    ----------
    document_id : str
        Identifier assigned at ingestion.
    source_document : str
        Original filename.
    text : str
        Concatenated, whitespace-trimmed text across all pages.
    document_class : DocumentClass
        How the document's text was encoded.
    text_source : TextSource
        Whether the text came from the digital layer, OCR, or both.
    page_count : int
        Number of pages.
    char_count : int
        Total number of characters in :attr:`text`.
    pages : list of PageStat
        Per-page bookkeeping.
    warnings : list of str
        Non-fatal notes (e.g. an OCR failure on a single page).
    """

    document_id: str
    source_document: str
    text: str
    document_class: DocumentClass
    text_source: TextSource
    page_count: int
    char_count: int
    pages: list[PageStat] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def tesseract_available() -> bool:
    """Report whether the Tesseract OCR backend can be used.

    Tesseract is a free, open-source OCR engine that extracts text from images
    (or image-only PDF pages); it is reached here through the ``pytesseract``
    package, which wraps the ``tesseract`` binary. The result is cached
    (``lru_cache``) so the binary lookup and the import check run at most once
    per process.

    Returns
    -------
    bool
        ``True`` if both the ``pytesseract`` package imports and the
        ``tesseract`` binary is on ``PATH``.
    """
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def _ocr_page(page: Any, language: str) -> str:
    """Run OCR on a single rendered PDF page.

    Parameters
    ----------
    page : fitz.Page
        The page to render and OCR.
    language : str
        Tesseract language code (e.g. ``"deu"``).

    Returns
    -------
    str
        The recognised text.
    """
    import pytesseract

    pixmap = page.get_pixmap(dpi=300)
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        pixmap.save(handle.name)
        text: str = pytesseract.image_to_string(handle.name, lang=language)
    return text


def extract_text(
    pdf_path: Path,
    *,
    document_id: str,
    settings: Settings | None = None,
) -> PreprocessedDocument:
    """Extract text from a PDF document.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.
    document_id : str
        Identifier assigned at ingestion, propagated into errors and the result.
    settings : Settings or None, optional
        Settings controlling the OCR behaviour and the scanned-page threshold.

    Returns
    -------
    PreprocessedDocument
        The extracted text and its provenance metadata.

    Raises
    ------
    DocumentReadError
        If the PDF cannot be opened or is password-protected.
    OcrUnavailableError
        If the document is image-only, OCR is requested, but Tesseract is not
        available.
    EmptyDocumentError
        If no usable text could be recovered from the document.
    """
    settings = settings or get_settings()
    source_document = pdf_path.name

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:  # PyMuPDF raises several unrelated types on bad input
        raise DocumentReadError(
            f"Failed to open PDF {pdf_path}: {exc}", document_id=document_id
        ) from exc

    try:
        if document.needs_pass:
            raise DocumentReadError(
                f"PDF {pdf_path} is password-protected", document_id=document_id
            )

        threshold = settings.scanned_text_threshold
        page_digital_text = [str(page.get_text("text")) for page in document]
        document_class = classify([len(text.strip()) for text in page_digital_text], threshold)
        ocr_ready = settings.ocr_enabled and tesseract_available()
        ocr_requested_unavailable = settings.ocr_enabled and not tesseract_available()

        parts, page_stats, warnings = _collect_pages(
            document, page_digital_text, threshold, settings, ocr_ready
        )
    finally:
        document.close()

    text = "\n".join(part for part in parts if part).strip()
    if not text:
        if document_class is DocumentClass.SCANNED and ocr_requested_unavailable:
            raise OcrUnavailableError(
                f"Document {source_document} is image-only and OCR is unavailable",
                document_id=document_id,
            )
        raise EmptyDocumentError(
            f"No usable text recovered from {source_document}", document_id=document_id
        )

    used_ocr = any(stat.used_ocr for stat in page_stats)
    used_digital = any(not stat.used_ocr and stat.char_count > 0 for stat in page_stats)
    text_source = _resolve_source(used_ocr=used_ocr, used_digital=used_digital)

    return PreprocessedDocument(
        document_id=document_id,
        source_document=source_document,
        text=text,
        document_class=document_class,
        text_source=text_source,
        page_count=len(page_stats),
        char_count=len(text),
        pages=page_stats,
        warnings=warnings,
    )


def _collect_pages(
    document: Any,
    page_digital_text: list[str],
    threshold: int,
    settings: Settings,
    ocr_ready: bool,
) -> tuple[list[str], list[PageStat], list[str]]:
    """Gather text for every page, applying OCR where needed and available."""
    parts: list[str] = []
    page_stats: list[PageStat] = []
    warnings: list[str] = []

    for index, page in enumerate(document):
        digital_text = page_digital_text[index]
        if len(digital_text.strip()) >= threshold:
            parts.append(digital_text)
            page_stats.append(
                PageStat(index=index, char_count=len(digital_text.strip()), used_ocr=False)
            )
            continue

        page_text = digital_text
        used_ocr = False
        if ocr_ready:
            try:
                page_text = _ocr_page(page, settings.ocr_language)
                used_ocr = True
            except Exception as exc:  # OCR failures degrade one page, not the run
                warnings.append(f"OCR failed on page {index + 1}: {exc}")
                logger.warning("ocr_page_failed", extra={"page": index + 1, "error": str(exc)})

        parts.append(page_text)
        page_stats.append(
            PageStat(index=index, char_count=len(page_text.strip()), used_ocr=used_ocr)
        )

    return parts, page_stats, warnings


def _resolve_source(*, used_ocr: bool, used_digital: bool) -> TextSource:
    """Map the per-page usage flags onto an overall text source."""
    if used_ocr and used_digital:
        return TextSource.HYBRID
    if used_ocr:
        return TextSource.OCR
    return TextSource.DIGITAL
