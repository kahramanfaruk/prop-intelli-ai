"""Synthetic sample-data generation for development, demos, and evaluation."""

from __future__ import annotations

from propintelli.sampledata.generator import (
    SAMPLE_PROPERTIES,
    SyntheticProperty,
    generate_samples,
    ground_truth,
)
from propintelli.sampledata.holdout import (
    HOLDOUT_DOCUMENTS,
    HoldoutDocument,
    generate_holdout,
    holdout_ground_truth,
)

__all__ = [
    "HOLDOUT_DOCUMENTS",
    "SAMPLE_PROPERTIES",
    "HoldoutDocument",
    "SyntheticProperty",
    "generate_holdout",
    "generate_samples",
    "ground_truth",
    "holdout_ground_truth",
]
