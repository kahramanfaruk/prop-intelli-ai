"""Tests for the Typer command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from propintelli.cli import app
from propintelli.extraction.llm.base import LlmExtraction
from propintelli.sampledata import SAMPLE_PROPERTIES

runner = CliRunner()


class _StubProvider:
    """Offline LLM stand-in so the CLI comparison path runs without a model."""

    name = "stub"

    def extract(self, text: str) -> LlmExtraction:
        return LlmExtraction(fields={"city": "Nürnberg"}, field_confidences={"city": 0.7})


def test_info_command_runs() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "configuration" in result.stdout.lower()


def test_generate_samples_command(tmp_path: Path) -> None:
    result = runner.invoke(app, ["generate-samples", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert len(list((tmp_path / "raw").glob("*.pdf"))) == len(SAMPLE_PROPERTIES)


def test_run_command_on_sample(sample_pdf: Path) -> None:
    result = runner.invoke(app, ["run", str(sample_pdf), "--no-persist", "--json"])
    assert result.exit_code == 0
    assert "property_id" in result.stdout


def test_evaluate_command(sample_corpus: tuple[Path, Path]) -> None:
    raw_dir, truth_dir = sample_corpus
    result = runner.invoke(
        app, ["evaluate", "--raw-dir", str(raw_dir), "--truth-dir", str(truth_dir)]
    )
    assert result.exit_code == 0
    assert "Macro F1" in result.stdout


def test_run_command_reports_error_on_corrupt(corrupt_pdf: Path) -> None:
    result = runner.invoke(app, ["run", str(corrupt_pdf), "--no-persist"])
    assert result.exit_code == 1
    assert "PRE_001" in result.stdout


def test_generate_holdout_command(tmp_path: Path) -> None:
    result = runner.invoke(app, ["generate-holdout", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert list((tmp_path / "raw").glob("*.pdf"))


def test_process_bronze_command_on_empty_store() -> None:
    # With nothing ingested, the one-shot worker reports zero pending documents.
    result = runner.invoke(app, ["process-bronze"])
    assert result.exit_code == 0
    assert "Processed 0 pending" in result.stdout


def test_compare_prompts_refuses_without_llm_backend(sample_corpus: tuple[Path, Path]) -> None:
    # With the default 'none' backend the prompt variants are not exercised, so
    # the command must refuse rather than print a meaningless identical table.
    raw_dir, truth_dir = sample_corpus
    result = runner.invoke(
        app, ["compare-prompts", "--raw-dir", str(raw_dir), "--truth-dir", str(truth_dir)]
    )
    assert result.exit_code == 1
    assert "No LLM backend" in result.stdout


def test_compare_prompts_renders_table_with_backend(
    sample_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # With a backend configured and the provider stubbed, the command runs every
    # variant and prints the comparison table (verifying the LLM-comparison path).
    raw_dir, truth_dir = sample_corpus
    monkeypatch.setenv("PROPINTELLI_LLM_PROVIDER", "ollama")
    monkeypatch.setattr(
        "propintelli.extraction.engine.build_provider", lambda _settings: _StubProvider()
    )
    result = runner.invoke(
        app, ["compare-prompts", "--raw-dir", str(raw_dir), "--truth-dir", str(truth_dir)]
    )
    assert result.exit_code == 0
    assert "Macro F1" in result.stdout
    assert "v2_schema" in result.stdout
