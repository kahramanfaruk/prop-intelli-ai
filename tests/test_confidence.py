"""Tests for confidence scoring and human-in-the-loop routing."""

from __future__ import annotations

from propintelli.confidence import compute_quality, source_quality_score
from propintelli.config import Settings
from propintelli.errors import ErrorSeverity
from propintelli.preprocessing import TextSource
from propintelli.schemas.enums import Provenance, ReviewStatus
from propintelli.schemas.fields import field_names
from propintelli.schemas.property_record import ValidationFinding
from propintelli.transformation.normalize import NormalizedFields


def _full_normalized() -> NormalizedFields:
    names = field_names()
    return NormalizedFields(
        values=dict.fromkeys(names, "x"),
        confidences=dict.fromkeys(names, 0.95),
        provenance=dict.fromkeys(names, Provenance.RECONCILED),
    )


def test_source_quality_ranking() -> None:
    assert source_quality_score(TextSource.DIGITAL) == 1.0
    assert source_quality_score(TextSource.HYBRID) == 0.8
    assert source_quality_score(TextSource.OCR) == 0.6


def test_high_quality_record_is_auto_approved() -> None:
    quality = compute_quality(
        normalized=_full_normalized(),
        findings=[],
        validation_pass_rate=1.0,
        source_quality=1.0,
        warnings=[],
        settings=Settings(),
    )
    assert quality.review_status is ReviewStatus.AUTO_APPROVED
    assert quality.overall_confidence > 0.85
    assert quality.completeness == 1.0


def test_validation_error_blocks_auto_approval() -> None:
    error = ValidationFinding(
        rule_id="mandatory.price_eur",
        field="price_eur",
        severity=ErrorSeverity.ERROR,
        message="missing",
    )
    quality = compute_quality(
        normalized=_full_normalized(),
        findings=[error],
        validation_pass_rate=0.9,
        source_quality=1.0,
        warnings=[],
        settings=Settings(),
    )
    # Score would auto-approve, but a hard error caps it at needs_review.
    assert quality.review_status is ReviewStatus.NEEDS_REVIEW


def test_sparse_low_confidence_record_requires_manual() -> None:
    sparse = NormalizedFields(
        values={"city": "Berlin"},
        confidences={"city": 0.3},
        provenance={"city": Provenance.DETERMINISTIC},
    )
    quality = compute_quality(
        normalized=sparse,
        findings=[],
        validation_pass_rate=0.3,
        source_quality=0.6,
        warnings=[],
        settings=Settings(),
    )
    assert quality.review_status is ReviewStatus.MANUAL_REQUIRED
    assert quality.overall_confidence < 0.6
