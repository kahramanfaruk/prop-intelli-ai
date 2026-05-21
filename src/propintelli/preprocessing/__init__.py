"""Preprocessing: document classification and text extraction."""

from __future__ import annotations

from propintelli.preprocessing.pdf_classifier import DocumentClass, classify
from propintelli.preprocessing.text_extractor import (
    PageStat,
    PreprocessedDocument,
    TextSource,
    extract_text,
    tesseract_available,
)

__all__ = [
    "DocumentClass",
    "PageStat",
    "PreprocessedDocument",
    "TextSource",
    "classify",
    "extract_text",
    "tesseract_available",
]
