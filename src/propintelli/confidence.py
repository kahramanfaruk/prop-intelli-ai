"""Confidence scoring and human-in-the-loop routing.

The overall confidence is a weighted blend of four signals:

``overall = 0.40·extraction + 0.30·completeness + 0.20·validation_pass_rate
           + 0.10·source_quality``

The result drives a three-way routing decision (auto-approve / review / manual)
against configurable thresholds. A hard validation error (e.g. a missing
mandatory field) prevents auto-approval regardless of the numeric score, so a
high-confidence-but-incomplete record is still seen by a human.
"""

from __future__ import annotations

from propintelli.config import Settings, get_settings
from propintelli.errors import ErrorSeverity
from propintelli.preprocessing.text_extractor import TextSource
from propintelli.schemas.enums import ReviewStatus
from propintelli.schemas.fields import PROPERTY_FIELDS, FieldKind
from propintelli.schemas.property_record import QualityReport, ValidationFinding
from propintelli.transformation.normalize import NormalizedFields

_WEIGHT_EXTRACTION = 0.40
_WEIGHT_COMPLETENESS = 0.30
_WEIGHT_VALIDATION = 0.20
_WEIGHT_SOURCE = 0.10

# Completeness is measured over the non-boolean "core" fields, since boolean
# equipment features are legitimately sparse and would otherwise depress it.
_CORE_FIELDS: tuple[str, ...] = tuple(
    name for name, spec in PROPERTY_FIELDS.items() if spec.kind is not FieldKind.BOOLEAN
)

_SOURCE_QUALITY: dict[TextSource, float] = {
    TextSource.DIGITAL: 1.0,
    TextSource.HYBRID: 0.8,
    TextSource.OCR: 0.6,
}


def source_quality_score(text_source: TextSource) -> float:
    """Map a text source to a quality score in ``[0, 1]``.

    Parameters
    ----------
    text_source : TextSource
        How the document text was obtained.

    Returns
    -------
    float
        Higher for lossless digital text, lower for OCR.
    """
    return _SOURCE_QUALITY.get(text_source, 0.6)


def _mean(values: list[float]) -> float:
    """Arithmetic mean, or 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def _route(overall: float, findings: list[ValidationFinding], settings: Settings) -> ReviewStatus:
    """Decide the routing status from the score and validation errors."""
    if overall >= settings.confidence_auto_approve:
        status = ReviewStatus.AUTO_APPROVED
    elif overall >= settings.confidence_review_floor:
        status = ReviewStatus.NEEDS_REVIEW
    else:
        status = ReviewStatus.MANUAL_REQUIRED

    has_error = any(finding.severity is ErrorSeverity.ERROR for finding in findings)
    if has_error and status is ReviewStatus.AUTO_APPROVED:
        return ReviewStatus.NEEDS_REVIEW
    return status


def compute_quality(
    *,
    normalized: NormalizedFields,
    findings: list[ValidationFinding],
    validation_pass_rate: float,
    source_quality: float,
    warnings: list[str],
    settings: Settings | None = None,
) -> QualityReport:
    """Compute the aggregate quality report and routing decision.

    Parameters
    ----------
    normalized : NormalizedFields
        Typed values with per-field confidence and provenance.
    findings : list of ValidationFinding
        Findings produced by the validation rules.
    validation_pass_rate : float
        Fraction of applicable validation rules that passed.
    source_quality : float
        Quality of the text source (see :func:`source_quality_score`).
    warnings : list of str
        Accumulated processing warnings to attach to the report.
    settings : Settings or None, optional
        Provides the routing thresholds.

    Returns
    -------
    QualityReport
        The overall confidence, component scores, and routing decision.
    """
    settings = settings or get_settings()
    extraction_confidence = _mean(list(normalized.confidences.values()))
    present_core = sum(1 for name in _CORE_FIELDS if name in normalized.values)
    completeness = present_core / len(_CORE_FIELDS) if _CORE_FIELDS else 0.0

    overall = (
        _WEIGHT_EXTRACTION * extraction_confidence
        + _WEIGHT_COMPLETENESS * completeness
        + _WEIGHT_VALIDATION * validation_pass_rate
        + _WEIGHT_SOURCE * source_quality
    )
    overall = max(0.0, min(1.0, overall))

    return QualityReport(
        overall_confidence=overall,
        field_confidences=dict(normalized.confidences),
        field_provenance=dict(normalized.provenance),
        completeness=completeness,
        validation_pass_rate=validation_pass_rate,
        review_status=_route(overall, findings, settings),
        findings=findings,
        warnings=warnings,
    )
