"""Tests for the synthetic sample-data generator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from propintelli.sampledata import SAMPLE_PROPERTIES, generate_samples, ground_truth
from propintelli.sampledata.generator import render_pdf
from propintelli.schemas.fields import required_field_names


def test_sample_set_covers_layouts_and_listing_types() -> None:
    layouts = {prop.layout for prop in SAMPLE_PROPERTIES}
    assert layouts == {"tabular", "prose", "sectioned"}
    listing_types = {prop.listing_type.value for prop in SAMPLE_PROPERTIES}
    assert "sale" in listing_types
    assert "rent" in listing_types


def test_document_stems_are_unique() -> None:
    stems = [prop.document_stem for prop in SAMPLE_PROPERTIES]
    assert len(stems) == len(set(stems))


def test_ground_truth_always_contains_required_fields() -> None:
    for prop in SAMPLE_PROPERTIES:
        truth = ground_truth(prop)["fields"]
        for required in required_field_names():
            assert required in truth, f"{prop.document_stem} missing {required}"


def test_sparse_listing_omits_unstated_fields() -> None:
    sparse = next(p for p in SAMPLE_PROPERTIES if "sparse" in p.document_stem)
    truth = ground_truth(sparse)["fields"]
    # The sparse listing intentionally states no construction year or energy data.
    assert "year_built" not in truth
    assert "energy_class" not in truth


def test_generate_samples_writes_pdf_and_label_pairs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    truth_dir = tmp_path / "ground_truth"
    generated = generate_samples(raw_dir, truth_dir)

    assert len(generated) == len(SAMPLE_PROPERTIES)
    for pdf_path in generated:
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0
        label = truth_dir / f"{pdf_path.stem}.json"
        assert label.exists()


def test_render_pdf_rejects_unknown_layout(tmp_path: Path) -> None:
    broken = replace(SAMPLE_PROPERTIES[0], layout="mosaic")
    try:
        render_pdf(broken, tmp_path / "x.pdf")
    except ValueError as exc:
        assert "Unknown layout" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for unknown layout")
