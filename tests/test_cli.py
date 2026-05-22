"""Tests for the Typer command-line interface."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from propintelli.cli import app

runner = CliRunner()


def test_info_command_runs() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "configuration" in result.stdout.lower()


def test_generate_samples_command(tmp_path: Path) -> None:
    result = runner.invoke(app, ["generate-samples", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert len(list((tmp_path / "raw").glob("*.pdf"))) == 10


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
