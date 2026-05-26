"""Tests for the synthetic sample-data generator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from propintelli.sampledata import SAMPLE_PROPERTIES, generate_samples, ground_truth
from propintelli.sampledata.generator import render_pdf
from propintelli.schemas.enums import HeatingType, PriceKind
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


def test_explicitly_absent_features_are_ground_truth_false() -> None:
    # A feature marked absent must appear as False in the label (a stated fact),
    # while genuinely unstated features are omitted entirely.
    absent_props = [p for p in SAMPLE_PROPERTIES if any(not v for v in p.features.values())]
    assert absent_props, "expected at least one listing with an explicitly-absent feature"
    for prop in absent_props:
        truth = ground_truth(prop)["fields"]
        for name, present in prop.features.items():
            assert truth.get(name) is present


def test_ancillary_costs_are_not_ground_truth() -> None:
    # Service charges, broker fees, and deposits are rendered but must never be
    # mistaken for the headline price.
    with_extras = [p for p in SAMPLE_PROPERTIES if p.ancillary_costs]
    assert with_extras, "expected at least one listing with ancillary costs"
    for prop in with_extras:
        truth = ground_truth(prop)["fields"]
        assert float(truth["price_eur"]) == float(prop.price_eur)


def test_sample_set_exercises_full_value_space() -> None:
    # Guards against the corpus silently leaving parts of the schema untested.
    assert PriceKind.WARM_RENT.value in {
        ground_truth(p)["fields"]["price_kind"] for p in SAMPLE_PROPERTIES
    }
    heating = {p.heating_type for p in SAMPLE_PROPERTIES if p.heating_type is not None}
    assert {HeatingType.ELECTRIC, HeatingType.PELLET, HeatingType.SOLAR} <= heating
    energy = {p.energy_class.value for p in SAMPLE_PROPERTIES if p.energy_class is not None}
    assert {"A", "G", "H"} <= energy
    assert any(p.features.get("furnished") for p in SAMPLE_PROPERTIES)


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
