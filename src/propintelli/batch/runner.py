"""Batch processing with progress reporting and per-document isolation.

Processes every PDF in a folder through the pipeline, isolating each document so
one failure cannot abort the run, and aggregates outcomes by review status. A
``rich`` progress bar is shown for interactive use and can be disabled for tests
and non-TTY environments.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rich.progress import Progress

from propintelli.errors import ProcessingError
from propintelli.logging_setup import get_logger
from propintelli.pipeline import Pipeline, PipelineResult
from propintelli.schemas.enums import ReviewStatus

logger = get_logger(__name__)


@dataclass
class BatchReport:
    """Aggregated outcome of a batch run.

    Attributes
    ----------
    total : int
        Number of documents processed.
    by_status : Counter of str
        Count of successful records per review status.
    failed : int
        Number of documents that failed with an error.
    results : list of PipelineResult
        The per-document results, in processing order.
    errors : list of ProcessingError
        The structured errors for failed documents.
    """

    total: int = 0
    by_status: Counter[str] = field(default_factory=Counter)
    failed: int = 0
    results: list[PipelineResult] = field(default_factory=list)
    errors: list[ProcessingError] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        """Number of documents that produced a record."""
        return self.total - self.failed

    def record(self, result: PipelineResult) -> None:
        """Fold a single pipeline result into the report.

        Parameters
        ----------
        result : PipelineResult
            The outcome to incorporate.
        """
        self.total += 1
        self.results.append(result)
        if result.record is not None:
            self.by_status[result.record.quality.review_status.value] += 1
        elif result.error is not None:
            self.failed += 1
            self.errors.append(result.error)


def discover_pdfs(input_dir: Path) -> list[Path]:
    """Return the sorted PDF paths in a directory.

    Parameters
    ----------
    input_dir : Path
        Directory to scan (non-recursively) for ``*.pdf`` files.

    Returns
    -------
    list of Path
        Sorted PDF paths.
    """
    return sorted(input_dir.glob("*.pdf"))


def run_batch(
    input_dir: Path,
    pipeline: Pipeline,
    *,
    show_progress: bool = True,
) -> BatchReport:
    """Process every PDF in a directory through the pipeline.

    Parameters
    ----------
    input_dir : Path
        Directory containing the PDFs to process.
    pipeline : Pipeline
        The configured pipeline.
    show_progress : bool, optional
        Whether to render a progress bar (disable for tests/non-TTY).

    Returns
    -------
    BatchReport
        The aggregated outcome.
    """
    pdfs = discover_pdfs(input_dir)
    report = BatchReport()

    with Progress(disable=not show_progress) as progress:
        task = progress.add_task("Processing exposés", total=len(pdfs))
        for pdf in pdfs:
            result = pipeline.process_path(pdf)
            report.record(result)
            if result.error is not None:
                logger.warning(
                    "batch_document_failed",
                    extra={"document": pdf.name, "error_code": result.error.error_code},
                )
            progress.advance(task)

    logger.info(
        "batch_complete",
        extra={
            "total": report.total,
            "succeeded": report.succeeded,
            "failed": report.failed,
            "auto_approved": report.by_status.get(ReviewStatus.AUTO_APPROVED.value, 0),
        },
    )
    return report
