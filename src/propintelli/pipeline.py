"""End-to-end single-document orchestration.

Wires the stages together — Bronze ingestion, preprocessing, hybrid extraction,
normalisation, validation, confidence scoring, record assembly, and Silver
persistence — into one call. Stage failures are caught and converted into a
:class:`~propintelli.errors.ProcessingError`, so a single bad document never
aborts a batch; the failure is recorded in the run audit and returned in the
result for routing to human review.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from propintelli.confidence import compute_quality, source_quality_score
from propintelli.config import Settings, get_settings
from propintelli.errors import ProcessingError, PropIntelliError
from propintelli.extraction.engine import run_extraction
from propintelli.ingestion.document_store import DocumentStore
from propintelli.logging_setup import get_logger
from propintelli.preprocessing.text_extractor import extract_text
from propintelli.schemas.property_record import PropertyRecord
from propintelli.storage.repository import ProcessingRunInfo, SilverRepository
from propintelli.transformation.assembler import assemble_record
from propintelli.transformation.normalize import normalize
from propintelli.validation.rules import validate

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Outcome of processing a single document.

    Attributes
    ----------
    document_id : str
        Bronze document identifier.
    source_document : str
        Original filename.
    record : PropertyRecord or None
        The assembled record on success, otherwise ``None``.
    error : ProcessingError or None
        The structured error on failure, otherwise ``None``.
    """

    document_id: str
    source_document: str
    record: PropertyRecord | None
    error: ProcessingError | None

    @property
    def succeeded(self) -> bool:
        """Whether a record was produced."""
        return self.record is not None


class Pipeline:
    """Orchestrates the full extraction pipeline for one document at a time."""

    def __init__(
        self,
        *,
        store: DocumentStore,
        repository: SilverRepository | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Construct a pipeline.

        Parameters
        ----------
        store : DocumentStore
            The Bronze store every document is ingested into.
        repository : SilverRepository or None, optional
            If given, successful records and run audits are persisted.
        settings : Settings or None, optional
            Runtime settings; defaults to the process settings.
        """
        self._store = store
        self._repository = repository
        self._settings = settings or get_settings()

    def process_path(self, path: Path) -> PipelineResult:
        """Ingest and process a document from a local path.

        Parameters
        ----------
        path : Path
            Path to the source PDF.

        Returns
        -------
        PipelineResult
            The outcome (record or structured error).
        """
        bronze = self._store.ingest_path(path)
        return self._process(bronze.document_id, bronze.stored_path, bronze.source_document)

    def process_bytes(self, data: bytes, filename: str) -> PipelineResult:
        """Ingest and process raw document bytes.

        Parameters
        ----------
        data : bytes
            Raw PDF contents.
        filename : str
            Original filename.

        Returns
        -------
        PipelineResult
            The outcome (record or structured error).
        """
        bronze = self._store.ingest_bytes(data, filename)
        return self._process(bronze.document_id, bronze.stored_path, bronze.source_document)

    def _process(self, document_id: str, pdf_path: Path, source_document: str) -> PipelineResult:
        """Run all stages for an already-ingested document."""
        try:
            record = self._extract_record(document_id, pdf_path, source_document)
        except PropIntelliError as exc:
            return self._handle_failure(document_id, source_document, exc.as_processing_error())
        except Exception as exc:  # convert any unforeseen failure into a structured error
            error = PropIntelliError(
                f"Unexpected pipeline failure: {exc}", document_id=document_id
            ).as_processing_error()
            logger.exception("pipeline_unexpected_failure", extra={"document_id": document_id})
            return self._handle_failure(document_id, source_document, error)

        self._persist(record)
        logger.info(
            "document_processed",
            extra={
                "document_id": document_id,
                "review_status": record.quality.review_status.value,
                "confidence": round(record.quality.overall_confidence, 3),
            },
        )
        return PipelineResult(document_id, source_document, record, None)

    def _extract_record(
        self, document_id: str, pdf_path: Path, source_document: str
    ) -> PropertyRecord:
        """Execute the transformation stages and assemble the record."""
        preprocessed = extract_text(pdf_path, document_id=document_id, settings=self._settings)
        extraction = run_extraction(preprocessed, self._settings)
        normalized = normalize(extraction)
        findings, pass_rate = validate(normalized.values)
        quality = compute_quality(
            normalized=normalized,
            findings=findings,
            validation_pass_rate=pass_rate,
            source_quality=source_quality_score(preprocessed.text_source),
            warnings=[*extraction.warnings, *normalized.warnings],
            settings=self._settings,
        )
        return assemble_record(normalized, quality, source_document, property_id=document_id)

    def _persist(self, record: PropertyRecord) -> None:
        """Persist a successful record and its run audit, if a repository exists."""
        if self._repository is None:
            return
        self._repository.save_record(record)
        self._repository.record_run(
            ProcessingRunInfo(
                document_id=record.property_id,
                source_document=record.source_document,
                status="succeeded",
                review_status=record.quality.review_status,
                property_id=record.property_id,
            )
        )

    def _handle_failure(
        self, document_id: str, source_document: str, error: ProcessingError
    ) -> PipelineResult:
        """Record a failed run and return a failure result."""
        if self._repository is not None:
            self._repository.record_run(
                ProcessingRunInfo(
                    document_id=document_id,
                    source_document=source_document,
                    status="failed",
                    error_code=error.error_code,
                    error_message=error.message,
                )
            )
        return PipelineResult(document_id, source_document, None, error)


def build_default_pipeline(settings: Settings | None = None) -> Pipeline:
    """Build a pipeline wired to the configured Bronze and Silver locations.

    Parameters
    ----------
    settings : Settings or None, optional
        Runtime settings; defaults to the process settings.

    Returns
    -------
    Pipeline
        A pipeline persisting to the configured medallion layers.
    """
    settings = settings or get_settings()
    return Pipeline(
        store=DocumentStore(settings.bronze_dir),
        repository=SilverRepository(settings.silver_db_path),
        settings=settings,
    )
