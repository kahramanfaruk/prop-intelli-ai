"""Systematic, field-level evaluation against ground truth.

This module compares the pipeline's output to machine-readable ground-truth
labels and reports, per field, accuracy and precision/recall/F1, plus
corpus-level macro-F1 and an exact-match ratio. Because real corpora are small,
proportion metrics are reported with **Wilson score confidence intervals** so a
"75 %" on four observations is not mistaken for a precise estimate.

It also measures whether the pipeline's per-field **confidence scores are
calibrated** — i.e. whether a field predicted with confidence 0.8 is correct
about 80 % of the time — via the Brier score and a reliability table. The
heuristic confidences are priors, not learned probabilities, so this turns "how
sure is it?" into a measured, falsifiable claim rather than an assertion.

Independence of the metric from the pipeline: ground-truth labels are produced
directly from canonical records (synthetic corpus) or written by hand (holdout
corpus) and never pass through extraction or normalisation, while predictions go
through the full pipeline. The only shared code is the value comparator in
:mod:`propintelli.comparison`, which is unit-tested in isolation — so a
normalisation regression changes predictions but not labels and is caught.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from propintelli.comparison import values_match
from propintelli.config import LlmProvider, PromptVariant, Settings
from propintelli.ingestion.document_store import DocumentStore
from propintelli.pipeline import Pipeline
from propintelli.schemas.fields import PROPERTY_FIELDS, field_names, get_field
from propintelli.schemas.property_record import PropertyRecord

# Standard normal quantile for a two-sided 95% interval.
_Z_95 = 1.96
# Reliability-diagram bin edges over the confidence range [0, 1].
_CALIBRATION_BIN_EDGES: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)


def wilson_interval(successes: int, trials: int, *, z: float = _Z_95) -> tuple[float, float]:
    """Return the Wilson score confidence interval for a proportion.

    The Wilson interval is well-behaved for the small samples and extreme
    proportions (0 % / 100 %) typical of per-field evaluation, where the normal
    approximation collapses to a zero-width interval.

    Parameters
    ----------
    successes : int
        Number of successes observed.
    trials : int
        Number of trials. A value of zero yields the maximally uncertain
        ``(0.0, 1.0)``.
    z : float, optional
        Standard-normal quantile (default 1.96 for 95 % coverage).

    Returns
    -------
    tuple of (float, float)
        Lower and upper bounds in ``[0, 1]``.
    """
    if trials <= 0:
        return (0.0, 1.0)
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    margin = (z / denominator) * math.sqrt(
        phat * (1 - phat) / trials + z * z / (4 * trials * trials)
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True, slots=True)
class FieldMetrics:
    """Per-field evaluation metrics.

    Attributes
    ----------
    field : str
        Canonical field name.
    support : int
        Number of documents in which the field was expected (present in GT).
    true_positive, false_positive, false_negative : int
        Confusion-matrix counts (a false positive is a hallucinated field).
    accuracy, precision, recall, f1 : float or None
        Standard metrics, or ``None`` when undefined for this field.
    accuracy_ci : tuple of (float, float) or None
        95 % Wilson interval for accuracy (correct ÷ support).
    """

    field: str
    support: int
    true_positive: int
    false_positive: int
    false_negative: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    accuracy_ci: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One bin of the reliability table.

    Attributes
    ----------
    lower, upper : float
        Confidence-bin bounds.
    count : int
        Number of predicted fields whose confidence falls in the bin.
    mean_confidence : float
        Mean predicted confidence within the bin.
    empirical_accuracy : float
        Fraction of those predictions that were correct.
    """

    lower: float
    upper: float
    count: int
    mean_confidence: float
    empirical_accuracy: float


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Calibration of the pipeline's per-field confidence scores.

    Attributes
    ----------
    sample_size : int
        Number of predicted (field, confidence, correctness) observations.
    brier_score : float
        Mean squared error between confidence and correctness in ``[0, 1]``;
        lower is better (0 is perfect, 0.25 is an uninformative 0.5 guess).
    bins : list of CalibrationBin
        Non-empty reliability bins, low to high confidence.
    """

    sample_size: int
    brier_score: float
    bins: list[CalibrationBin]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Corpus-level evaluation report.

    Attributes
    ----------
    document_count : int
        Number of documents evaluated.
    macro_f1 : float
        Mean F1 over fields seen in ground truth or predictions.
    micro_field_accuracy : float
        Correct expected occurrences ÷ total expected occurrences.
    micro_accuracy_ci : tuple of (float, float)
        95 % Wilson interval for the micro field accuracy.
    exact_match_ratio : float
        Fraction of documents whose every expected field was correct.
    per_field : list of FieldMetrics
        Metrics per field, in registry order.
    calibration : CalibrationReport or None
        Confidence calibration, when per-field confidences were supplied.
    """

    document_count: int
    macro_f1: float
    micro_field_accuracy: float
    micro_accuracy_ci: tuple[float, float]
    exact_match_ratio: float
    per_field: list[FieldMetrics]
    calibration: CalibrationReport | None = None


def record_to_canonical(record: PropertyRecord) -> dict[str, Any]:
    """Flatten a record into canonical, comparable field values.

    Parameters
    ----------
    record : PropertyRecord
        The record to flatten.

    Returns
    -------
    dict
        Canonical field name to a JSON-comparable value (enums as their value,
        Decimals/dates as primitives). Absent fields are omitted.
    """
    flat: dict[str, Any] = {}
    for name, spec in PROPERTY_FIELDS.items():
        target: Any = record
        for part in spec.record_path:
            target = getattr(target, part, None)
            if target is None:
                break
        if target is None:
            continue
        flat[name] = _to_comparable(target)
    return flat


def _to_comparable(value: Any) -> Any:
    """Convert a typed value into a JSON-comparable primitive."""
    if hasattr(value, "value"):  # StrEnum
        return value.value
    if hasattr(value, "isoformat"):  # date
        return value.isoformat()
    if isinstance(value, bool | int | str):
        return value
    return float(value)  # Decimal / float


def _match(name: str, expected: Any, predicted: Any) -> bool:
    """Whether a predicted value matches the expected value for a field."""
    return values_match(get_field(name), expected, predicted)


class _FieldCounter:
    """Mutable per-field tally used while scoring."""

    __slots__ = ("fn", "fp", "support", "tp")

    def __init__(self) -> None:
        self.support = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0


def evaluate_records(
    predicted: dict[str, dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
) -> EvaluationReport:
    """Compute evaluation metrics from predicted and ground-truth field maps.

    Parameters
    ----------
    predicted : dict of str to dict
        Predicted canonical field values keyed by document filename.
    ground_truth : dict of str to dict
        Expected canonical field values keyed by document filename.

    Returns
    -------
    EvaluationReport
        The per-field and corpus-level metrics (without calibration).
    """
    counters = {name: _FieldCounter() for name in field_names()}
    documents = sorted(ground_truth)
    exact_matches = 0

    for document in documents:
        expected_fields = ground_truth[document]
        predicted_fields = predicted.get(document, {})
        document_correct = True
        for name in field_names():
            expected = expected_fields.get(name)
            prediction = predicted_fields.get(name)
            counter = counters[name]
            expected_present = expected is not None
            predicted_present = prediction is not None
            matched = expected_present and predicted_present and _match(name, expected, prediction)
            if expected_present:
                counter.support += 1
            if matched:
                counter.tp += 1
            if predicted_present and not matched:
                counter.fp += 1
            if expected_present and not matched:
                counter.fn += 1
                document_correct = False
        if document_correct:
            exact_matches += 1

    per_field = [_field_metrics(name, counters[name]) for name in field_names()]
    return _aggregate(per_field, len(documents), exact_matches)


def _field_metrics(name: str, counter: _FieldCounter) -> FieldMetrics:
    """Build the metrics for one field from its tally."""
    tp, fp, fn, support = counter.tp, counter.fp, counter.fn, counter.support
    accuracy = tp / support if support else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None and recall is None:
        f1: float | None = None
    elif precision and recall and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return FieldMetrics(
        field=name,
        support=support,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy_ci=wilson_interval(tp, support) if support else None,
    )


def _aggregate(
    per_field: list[FieldMetrics], document_count: int, exact_matches: int
) -> EvaluationReport:
    """Combine per-field metrics into the corpus-level report."""
    scored = [metrics.f1 for metrics in per_field if metrics.f1 is not None]
    macro_f1 = sum(scored) / len(scored) if scored else 0.0
    total_support = sum(metrics.support for metrics in per_field)
    total_correct = sum(metrics.true_positive for metrics in per_field)
    micro_accuracy = total_correct / total_support if total_support else 0.0
    exact_ratio = exact_matches / document_count if document_count else 0.0
    return EvaluationReport(
        document_count=document_count,
        macro_f1=macro_f1,
        micro_field_accuracy=micro_accuracy,
        micro_accuracy_ci=wilson_interval(total_correct, total_support),
        exact_match_ratio=exact_ratio,
        per_field=per_field,
    )


def compute_calibration(
    confidences: dict[str, dict[str, float]],
    predicted: dict[str, dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
) -> CalibrationReport:
    """Measure calibration of per-field confidence scores against correctness.

    Every *predicted* field contributes one ``(confidence, correct)`` pair:
    correct iff the field is in ground truth and matches it, so an overconfident
    hallucination is penalised. Missing predictions have no confidence and are
    excluded — calibration concerns the scores the pipeline actually emits.

    Parameters
    ----------
    confidences : dict of str to (dict of str to float)
        Per-document, per-field confidence in ``[0, 1]``.
    predicted : dict of str to dict
        Per-document predicted canonical field values.
    ground_truth : dict of str to dict
        Per-document expected canonical field values.

    Returns
    -------
    CalibrationReport
        The Brier score and reliability bins.
    """
    observations: list[tuple[float, bool]] = []
    for document, field_confidences in confidences.items():
        expected_fields = ground_truth.get(document, {})
        predicted_fields = predicted.get(document, {})
        for name, confidence in field_confidences.items():
            expected = expected_fields.get(name)
            prediction = predicted_fields.get(name)
            correct = (
                expected is not None
                and prediction is not None
                and _match(name, expected, prediction)
            )
            observations.append((confidence, correct))

    if not observations:
        return CalibrationReport(sample_size=0, brier_score=0.0, bins=[])

    brier = sum((conf - (1.0 if correct else 0.0)) ** 2 for conf, correct in observations) / len(
        observations
    )
    return CalibrationReport(
        sample_size=len(observations),
        brier_score=brier,
        bins=_reliability_bins(observations),
    )


def _reliability_bins(observations: list[tuple[float, bool]]) -> list[CalibrationBin]:
    """Group ``(confidence, correct)`` observations into reliability bins."""
    bins: list[CalibrationBin] = []
    for lower, upper in pairwise(_CALIBRATION_BIN_EDGES):
        members = [obs for obs in observations if lower <= obs[0] < upper]
        if not members:
            continue
        mean_conf = sum(conf for conf, _ in members) / len(members)
        accuracy = sum(1 for _, correct in members if correct) / len(members)
        bins.append(
            CalibrationBin(
                lower=lower,
                upper=min(upper, 1.0),
                count=len(members),
                mean_confidence=mean_conf,
                empirical_accuracy=accuracy,
            )
        )
    return bins


def load_ground_truth(ground_truth_dir: Path) -> dict[str, dict[str, Any]]:
    """Load ground-truth labels keyed by document filename.

    Parameters
    ----------
    ground_truth_dir : Path
        Directory of ``*.json`` label files.

    Returns
    -------
    dict of str to dict
        Mapping of document filename to its expected field values.
    """
    labels: dict[str, dict[str, Any]] = {}
    for path in sorted(ground_truth_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        labels[data["document"]] = data["fields"]
    return labels


def evaluate_corpus(
    raw_dir: Path,
    ground_truth_dir: Path,
    *,
    settings: Settings | None = None,
) -> EvaluationReport:
    """Run the pipeline over a corpus and evaluate against ground truth.

    Parameters
    ----------
    raw_dir : Path
        Directory of source PDFs.
    ground_truth_dir : Path
        Directory of ground-truth label JSON files.
    settings : Settings or None, optional
        Settings for the run; defaults to a deterministic, offline configuration.

    Returns
    -------
    EvaluationReport
        The evaluation metrics, including confidence calibration.
    """
    settings = settings or Settings(llm_provider=LlmProvider.NONE)
    ground_truth = load_ground_truth(ground_truth_dir)
    # Evaluation does not persist; it uses an in-memory Bronze store under data_dir.
    pipeline = Pipeline(
        store=DocumentStore(settings.bronze_dir), repository=None, settings=settings
    )

    predicted: dict[str, dict[str, Any]] = {}
    confidences: dict[str, dict[str, float]] = {}
    for pdf in sorted(raw_dir.glob("*.pdf")):
        result = pipeline.process_path(pdf)
        if result.record is None:
            predicted[pdf.name] = {}
            continue
        predicted[pdf.name] = record_to_canonical(result.record)
        confidences[pdf.name] = dict(result.record.quality.field_confidences)

    report = evaluate_records(predicted, ground_truth)
    calibration = compute_calibration(confidences, predicted, ground_truth)
    return EvaluationReport(
        document_count=report.document_count,
        macro_f1=report.macro_f1,
        micro_field_accuracy=report.micro_field_accuracy,
        micro_accuracy_ci=report.micro_accuracy_ci,
        exact_match_ratio=report.exact_match_ratio,
        per_field=report.per_field,
        calibration=calibration,
    )


def compare_prompt_variants(
    raw_dir: Path,
    ground_truth_dir: Path,
    *,
    settings: Settings,
) -> list[tuple[PromptVariant, EvaluationReport]]:
    """Evaluate every documented prompt variant with the configured backend.

    Re-runs :func:`evaluate_corpus` once per :class:`PromptVariant`, holding the
    LLM provider fixed and changing only the prompt, so the variants are
    compared apples-to-apples on the same corpus and metric.

    Parameters
    ----------
    raw_dir : Path
        Directory of source PDFs.
    ground_truth_dir : Path
        Directory of ground-truth label JSON files.
    settings : Settings
        Base settings; the provider is taken from here and the prompt variant is
        overridden per run.

    Returns
    -------
    list of (PromptVariant, EvaluationReport)
        One report per variant, in declaration order.
    """
    results: list[tuple[PromptVariant, EvaluationReport]] = []
    for variant in PromptVariant:
        variant_settings = settings.model_copy(update={"llm_prompt_variant": variant})
        report = evaluate_corpus(raw_dir, ground_truth_dir, settings=variant_settings)
        results.append((variant, report))
    return results
