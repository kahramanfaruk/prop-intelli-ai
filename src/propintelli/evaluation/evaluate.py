"""Systematic, field-level evaluation against ground truth.

Because the synthetic corpus is generated from canonical records, every PDF has a
machine-readable ground-truth label. This module compares the pipeline's output
to those labels and reports, per field, accuracy and precision/recall/F1, plus
corpus-level macro-F1 and an exact-match ratio.

The comparison rewards correct *absence*: predicting a field the document never
stated counts against precision, so the metrics reflect hallucination as well as
extraction quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from propintelli.config import LlmProvider, Settings
from propintelli.ingestion.document_store import DocumentStore
from propintelli.pipeline import Pipeline
from propintelli.schemas.fields import PROPERTY_FIELDS, FieldKind, FieldSpec, field_names, get_field
from propintelli.schemas.property_record import PropertyRecord

_NUMERIC_REL_TOLERANCE = 0.01


@dataclass(frozen=True, slots=True)
class FieldMetrics:
    """Per-field evaluation metrics.

    Attributes
    ----------
    field : str
        Canonical field name.
    support : int
        Number of documents in which the field was expected (present in GT).
    accuracy : float or None
        Fraction of expected occurrences extracted correctly, or ``None`` when
        the field is never expected.
    precision, recall, f1 : float or None
        Standard metrics treating "extracted and correct" as a true positive.
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


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Corpus-level evaluation report.

    Attributes
    ----------
    document_count : int
        Number of documents evaluated.
    macro_f1 : float
        Mean F1 over fields that appear in ground truth or predictions.
    micro_field_accuracy : float
        Correct expected occurrences divided by total expected occurrences.
    exact_match_ratio : float
        Fraction of documents whose every expected field was extracted correctly.
    per_field : list of FieldMetrics
        Metrics per field, in registry order.
    """

    document_count: int
    macro_f1: float
    micro_field_accuracy: float
    exact_match_ratio: float
    per_field: list[FieldMetrics]


def record_to_canonical(record: PropertyRecord) -> dict[str, Any]:
    """Flatten a record back into canonical, comparable field values.

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


def _match(spec: FieldSpec, expected: Any, predicted: Any) -> bool:
    """Decide whether a predicted value matches the expected value."""
    if spec.is_numeric:
        try:
            expected_value, predicted_value = float(expected), float(predicted)
        except (TypeError, ValueError):
            return False
        if spec.kind is FieldKind.INTEGER:
            return round(expected_value) == round(predicted_value)
        tolerance = _NUMERIC_REL_TOLERANCE * max(abs(expected_value), abs(predicted_value), 1.0)
        return abs(expected_value - predicted_value) <= tolerance
    if spec.kind is FieldKind.BOOLEAN:
        return bool(expected) == bool(predicted)
    return str(expected).strip().casefold() == str(predicted).strip().casefold()


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
        The per-field and corpus-level metrics.
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
            matched = (
                expected_present
                and predicted_present
                and _match(get_field(name), expected, prediction)
            )
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
        exact_match_ratio=exact_ratio,
        per_field=per_field,
    )


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
        The evaluation metrics.
    """
    settings = settings or Settings(llm_provider=LlmProvider.NONE)
    ground_truth = load_ground_truth(ground_truth_dir)
    # Evaluation does not persist; it uses an in-memory Bronze store under data_dir.
    pipeline = Pipeline(
        store=DocumentStore(settings.bronze_dir), repository=None, settings=settings
    )

    predicted: dict[str, dict[str, Any]] = {}
    for pdf in sorted(raw_dir.glob("*.pdf")):
        result = pipeline.process_path(pdf)
        predicted[pdf.name] = record_to_canonical(result.record) if result.record else {}
    return evaluate_records(predicted, ground_truth)
