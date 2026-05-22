"""Tests for the evaluation harness."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from propintelli.config import Settings
from propintelli.evaluation import evaluate_corpus, evaluate_records, record_to_canonical
from propintelli.schemas.enums import EnergyClass, ListingType
from propintelli.schemas.property_record import (
    EnergyProfile,
    Location,
    PropertyRecord,
    QualityReport,
)


def test_record_to_canonical_flattens_and_serialises() -> None:
    record = PropertyRecord(
        source_document="x.pdf",
        listing_type=ListingType.SALE,
        price_eur=Decimal("450000.00"),
        location=Location(city="Berlin"),
        energy=EnergyProfile(energy_class=EnergyClass.B),
        quality=QualityReport(overall_confidence=0.9),
    )
    flat = record_to_canonical(record)
    assert flat["listing_type"] == "sale"
    assert flat["price_eur"] == 450000.0
    assert flat["city"] == "Berlin"
    assert flat["energy_class"] == "B"
    assert "rooms" not in flat  # absent fields are omitted


def test_evaluate_records_scores_correct_and_incorrect_fields() -> None:
    ground_truth = {"d1": {"price_eur": 100000.0, "city": "Berlin", "balcony": True}}
    predicted = {
        "d1": {"price_eur": 100000.0, "city": "Munich", "garden": True},  # city wrong, garden FP
    }
    report = evaluate_records(predicted, ground_truth)
    by_field = {metrics.field: metrics for metrics in report.per_field}

    assert by_field["price_eur"].f1 == 1.0
    assert by_field["city"].accuracy == 0.0  # expected present but wrong
    assert by_field["balcony"].recall == 0.0  # expected but not predicted
    assert by_field["garden"].false_positive == 1  # hallucinated
    assert report.exact_match_ratio == 0.0  # city + balcony wrong


def test_evaluate_records_perfect_corpus() -> None:
    truth = {"d1": {"price_eur": 100000.0, "city": "Berlin"}}
    report = evaluate_records(dict(truth), truth)
    assert report.exact_match_ratio == 1.0
    assert report.micro_field_accuracy == 1.0


def test_evaluate_corpus_on_samples_is_accurate(
    tmp_path: Path, sample_corpus: tuple[Path, Path]
) -> None:
    raw_dir, truth_dir = sample_corpus
    report = evaluate_corpus(raw_dir, truth_dir, settings=Settings(data_dir=tmp_path))
    assert report.document_count == 10
    # The deterministic baseline alone extracts the corpus near-perfectly.
    assert report.micro_field_accuracy >= 0.95
    assert report.macro_f1 >= 0.95
