"""Command-line interface for PropIntelli AI.

Exposes the pipeline as a small set of commands: generate the sample corpus,
process a single document, batch-process a folder, evaluate against ground
truth, and publish the Gold layer. Built on Typer + Rich for a clear terminal UX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from propintelli.batch.runner import run_batch
from propintelli.config import get_settings
from propintelli.evaluation.evaluate import EvaluationReport, evaluate_corpus
from propintelli.ingestion.document_store import DocumentStore
from propintelli.logging_setup import configure_logging
from propintelli.pipeline import Pipeline, build_default_pipeline
from propintelli.sampledata import generate_samples
from propintelli.storage.gold import build_gold
from propintelli.storage.repository import SilverRepository

app = typer.Typer(
    help="PropIntelli AI — extract structured data from real-estate exposés.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback()
def _main() -> None:
    """Configure logging before any command runs."""
    configure_logging()


@app.command("generate-samples")
def generate_samples_command(
    output_dir: Annotated[Path, typer.Option(help="Root for raw/ and ground_truth/.")] = Path(
        "sample_data"
    ),
) -> None:
    """Generate the synthetic sample exposés and ground-truth labels."""
    paths = generate_samples(output_dir / "raw", output_dir / "ground_truth")
    console.print(f"[green]Generated {len(paths)} exposés[/green] into {output_dir / 'raw'}")


@app.command("run")
def run_command(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="PDF to process.")],
    persist: Annotated[bool, typer.Option(help="Persist to the Silver store.")] = True,
    json_output: Annotated[bool, typer.Option("--json", help="Print raw JSON only.")] = False,
) -> None:
    """Process a single document and print the structured record."""
    settings = get_settings()
    if persist:
        pipeline = build_default_pipeline(settings)
    else:
        pipeline = Pipeline(store=DocumentStore(settings.bronze_dir), settings=settings)

    result = pipeline.process_path(path)
    if result.error is not None:
        console.print(f"[red]{result.error.error_code}[/red]: {result.error.user_message}")
        raise typer.Exit(1)

    record = result.record
    assert record is not None  # success implies a record
    if json_output:
        console.print_json(record.model_dump_json())
        return

    quality = record.quality
    console.print(
        f"[bold]{record.source_document}[/bold] → "
        f"[cyan]{quality.review_status.value}[/cyan] "
        f"(confidence {quality.overall_confidence:.2f}, completeness {quality.completeness:.2f})"
    )
    console.print_json(record.model_dump_json())


@app.command("batch")
def batch_command(
    input_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Folder of PDFs.")
    ],
) -> None:
    """Batch-process every PDF in a folder and print a summary."""
    report = run_batch(input_dir, build_default_pipeline())
    table = Table(title=f"Batch summary — {report.total} documents")
    table.add_column("Outcome")
    table.add_column("Count", justify="right")
    table.add_row("Succeeded", str(report.succeeded))
    for status, count in sorted(report.by_status.items()):
        table.add_row(f"  · {status}", str(count))
    table.add_row("Failed", str(report.failed))
    console.print(table)

    for error in report.errors:
        console.print(f"[red]{error.error_code}[/red] {error.document_id}: {error.user_message}")


@app.command("evaluate")
def evaluate_command(
    raw_dir: Annotated[Path, typer.Option(help="Folder of source PDFs.")] = Path("sample_data/raw"),
    truth_dir: Annotated[Path, typer.Option(help="Folder of ground-truth JSON.")] = Path(
        "sample_data/ground_truth"
    ),
) -> None:
    """Evaluate extraction accuracy against ground-truth labels."""
    report = evaluate_corpus(raw_dir, truth_dir)
    _render_evaluation(report)


@app.command("export")
def export_command() -> None:
    """Publish the Gold analytics layer from the Silver store."""
    settings = get_settings()
    repository = SilverRepository(settings.silver_db_path)
    records = repository.list_records()
    if not records:
        console.print("[yellow]No records in the Silver store; run 'batch' first.[/yellow]")
        raise typer.Exit(1)

    artifacts = build_gold(records, settings.gold_dir)
    console.print(f"[green]Exported {len(records)} records[/green] to {settings.gold_dir}")
    columns = ("city", "listings", "avg_price_per_sqm", "avg_living_area_sqm")
    table = Table(title="Market summary (sale listings)")
    for column in columns:
        table.add_column(column)
    for row in artifacts.summary:
        table.add_row(*(str(row.get(column, "")) for column in columns))
    console.print(table)


@app.command("info")
def info_command() -> None:
    """Show the active configuration."""
    settings = get_settings()
    table = Table(title="PropIntelli AI configuration")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("LLM provider", settings.llm_provider.value)
    table.add_row("Prompt variant", settings.llm_prompt_variant.value)
    table.add_row("OCR enabled", str(settings.ocr_enabled))
    table.add_row("Auto-approve ≥", str(settings.confidence_auto_approve))
    table.add_row("Review floor ≥", str(settings.confidence_review_floor))
    table.add_row("Data dir", str(settings.data_dir))
    console.print(table)


def _render_evaluation(report: EvaluationReport) -> None:
    """Print an evaluation report as a Rich table with headline metrics."""
    table = Table(title=f"Field-level evaluation — {report.document_count} documents")
    table.add_column("Field")
    table.add_column("Support", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    for metrics in report.per_field:
        if metrics.support == 0 and metrics.false_positive == 0:
            continue
        table.add_row(
            metrics.field,
            str(metrics.support),
            _pct(metrics.accuracy),
            _pct(metrics.precision),
            _pct(metrics.recall),
            _pct(metrics.f1),
        )
    console.print(table)
    console.print(
        f"[bold]Macro F1[/bold]: {report.macro_f1:.3f}  ·  "
        f"[bold]Field accuracy[/bold]: {report.micro_field_accuracy:.3f}  ·  "
        f"[bold]Exact-match ratio[/bold]: {report.exact_match_ratio:.3f}"
    )


def _pct(value: float | None) -> str:
    """Format a metric in ``[0, 1]`` as a percentage, or ``—`` when undefined."""
    return "—" if value is None else f"{value * 100:.1f}%"


if __name__ == "__main__":  # pragma: no cover
    app()
