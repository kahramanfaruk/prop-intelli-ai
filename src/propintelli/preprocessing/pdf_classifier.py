"""Classify a PDF by how its text is encoded.

The pipeline must cope with three broad document categories that demand
different handling:

* ``DIGITAL`` — every page carries an embedded text layer; direct extraction is
  fast and lossless.
* ``SCANNED`` — pages are images with no text layer; OCR is required.
* ``HYBRID`` — a mix (e.g. a digital brochure with a scanned floor-plan page).

Classification is a pure function of per-page text density, which keeps it
trivially testable and decoupled from the PDF library.
"""

from __future__ import annotations

import enum


class DocumentClass(enum.StrEnum):
    """How a document's text is encoded across its pages."""

    DIGITAL = "digital"
    HYBRID = "hybrid"
    SCANNED = "scanned"


def classify(page_text_lengths: list[int], scanned_threshold: int) -> DocumentClass:
    """Classify a document from the text length of each page.

    Parameters
    ----------
    page_text_lengths : list of int
        Number of embedded-text characters on each page, in page order.
    scanned_threshold : int
        Minimum character count for a page to count as carrying a usable text
        layer. Pages below this are treated as scanned/image pages.

    Returns
    -------
    DocumentClass
        ``DIGITAL`` if every page has a text layer, ``SCANNED`` if none do, and
        ``HYBRID`` otherwise. An empty document is treated as ``SCANNED`` so it
        is routed to the OCR/recovery path rather than silently accepted.
    """
    if not page_text_lengths:
        return DocumentClass.SCANNED
    digital_pages = sum(1 for length in page_text_lengths if length >= scanned_threshold)
    if digital_pages == len(page_text_lengths):
        return DocumentClass.DIGITAL
    if digital_pages == 0:
        return DocumentClass.SCANNED
    return DocumentClass.HYBRID
