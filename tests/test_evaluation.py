"""Tests for the evaluation harness."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from propintelli.config import LlmProvider, PromptVariant, Settings
from propintelli.evaluation import (
    compute_calibration,
    evaluate_corpus,
    evaluate_records,
    record_to_canonical,
    wilson_interval,
)
from propintelli.evaluation.evaluate import compare_prompt_variants
from propintelli.extraction.llm.base import LlmExtraction
from propintelli.sampledata import SAMPLE_PROPERTIES, generate_holdout
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
    assert report.document_count == len(SAMPLE_PROPERTIES)
    # The deterministic baseline alone extracts the corpus near-perfectly.
    assert report.micro_field_accuracy >= 0.95
    assert report.macro_f1 >= 0.95
    # Confidence intervals are attached for every field with support.
    supported = [m for m in report.per_field if m.support > 0]
    assert all(m.accuracy_ci is not None for m in supported)


@pytest.mark.parametrize(
    ("successes", "trials"),
    [(0, 0), (3, 4), (10, 10), (0, 5), (1, 100)],
)
def test_wilson_interval_is_bounded_and_contains_estimate(successes: int, trials: int) -> None:
    low, high = wilson_interval(successes, trials)
    assert 0.0 <= low <= high <= 1.0
    if trials > 0:
        assert low <= successes / trials <= high
    else:
        assert (low, high) == (0.0, 1.0)


def test_wilson_interval_for_small_sample_is_wide() -> None:
    # The critique's example: "75% on 3/4" must not be read as precise.
    low, high = wilson_interval(3, 4)
    assert high - low > 0.4


def test_calibration_rewards_confident_correctness_and_penalises_overconfidence() -> None:
    ground_truth = {"d1": {"price_eur": 100000.0, "city": "Berlin"}}
    predicted = {"d1": {"price_eur": 100000.0, "city": "Munich"}}  # city wrong
    confidences = {"d1": {"price_eur": 0.95, "city": 0.95}}  # both overconfident-or-correct
    report = compute_calibration(confidences, predicted, ground_truth)
    assert report.sample_size == 2
    # One correct at 0.95 -> (0.95-1)^2; one wrong at 0.95 -> (0.95-0)^2.
    expected = ((0.95 - 1.0) ** 2 + (0.95 - 0.0) ** 2) / 2
    assert report.brier_score == pytest.approx(expected)


def test_evaluate_corpus_attaches_calibration(
    tmp_path: Path, sample_corpus: tuple[Path, Path]
) -> None:
    raw_dir, truth_dir = sample_corpus
    report = evaluate_corpus(raw_dir, truth_dir, settings=Settings(data_dir=tmp_path))
    assert report.calibration is not None
    assert report.calibration.sample_size > 0
    assert 0.0 <= report.calibration.brier_score <= 1.0


def test_holdout_corpus_is_harder_than_synthetic(tmp_path: Path) -> None:
    # The authored holdout exercises wording the generator never produces, so the
    # deterministic baseline scores below the synthetic ceiling, the honest
    # generalization signal, yet still recovers most fields.
    raw_dir = tmp_path / "raw"
    truth_dir = tmp_path / "ground_truth"
    generate_holdout(raw_dir, truth_dir)
    report = evaluate_corpus(raw_dir, truth_dir, settings=Settings(data_dir=tmp_path / "data"))
    assert report.document_count == 3
    assert 0.6 <= report.macro_f1 < 1.0
    assert report.calibration is not None


class _StubProvider:
    """Deterministic offline LLM stand-in, to verify the comparison harness."""

    name = "stub"

    def extract(self, text: str) -> LlmExtraction:
        # Returns one plausible field so the variants run end to end without a model.
        return LlmExtraction(fields={"city": "Nürnberg"}, field_confidences={"city": 0.7})


def test_compare_prompt_variants_runs_for_every_variant(
    tmp_path: Path, sample_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir, truth_dir = sample_corpus
    monkeypatch.setattr(
        "propintelli.extraction.engine.build_provider", lambda _settings: _StubProvider()
    )
    settings = Settings(llm_provider=LlmProvider.OLLAMA, data_dir=tmp_path)
    results = compare_prompt_variants(raw_dir, truth_dir, settings=settings)
    assert [variant for variant, _ in results] == list(PromptVariant)
    assert all(report.document_count == len(SAMPLE_PROPERTIES) for _, report in results)
