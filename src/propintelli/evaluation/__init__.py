"""Evaluation: field-level accuracy against ground-truth labels."""

from __future__ import annotations

from propintelli.evaluation.evaluate import (
    EvaluationReport,
    FieldMetrics,
    evaluate_corpus,
    evaluate_records,
    record_to_canonical,
)

__all__ = [
    "EvaluationReport",
    "FieldMetrics",
    "evaluate_corpus",
    "evaluate_records",
    "record_to_canonical",
]
