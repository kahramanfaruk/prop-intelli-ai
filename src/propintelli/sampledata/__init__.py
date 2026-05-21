"""Synthetic sample-data generation for development, demos, and evaluation."""

from __future__ import annotations

from propintelli.sampledata.generator import (
    SAMPLE_PROPERTIES,
    SyntheticProperty,
    generate_samples,
    ground_truth,
)

__all__ = [
    "SAMPLE_PROPERTIES",
    "SyntheticProperty",
    "generate_samples",
    "ground_truth",
]
