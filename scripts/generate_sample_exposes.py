#!/usr/bin/env python3
"""Generate the synthetic sample exposes and their ground-truth labels.

Thin wrapper around :func:`propintelli.sampledata.generate_samples` so the
sample corpus can be regenerated with a plain ``python`` invocation as well as
via the ``propintelli generate-samples`` CLI command.

Usage
-----
    python scripts/generate_sample_exposes.py
"""

from __future__ import annotations

from pathlib import Path

from propintelli.sampledata import generate_samples

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "sample_data" / "raw"
GROUND_TRUTH_DIR = ROOT / "sample_data" / "ground_truth"


def main() -> None:
    """Generate the sample corpus into ``sample_data/``."""
    generated = generate_samples(RAW_DIR, GROUND_TRUTH_DIR)
    print(f"Generated {len(generated)} exposes into {RAW_DIR}")
    for path in generated:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
