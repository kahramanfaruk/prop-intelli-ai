"""Evaluation: field-level accuracy against ground-truth labels."""

from __future__ import annotations

from propintelli.evaluation.evaluate import (
    CalibrationReport,
    EvaluationReport,
    FieldMetrics,
    compute_calibration,
    evaluate_corpus,
    evaluate_records,
    record_to_canonical,
    wilson_interval,
)

__all__ = [
    "CalibrationReport",
    "EvaluationReport",
    "FieldMetrics",
    "compute_calibration",
    "evaluate_corpus",
    "evaluate_records",
    "record_to_canonical",
    "wilson_interval",
]
