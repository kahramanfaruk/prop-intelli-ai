"""Structured, actionable error handling.

This module implements the platform's error philosophy: failures are *classification
signals* that route a document to automatic processing, a fallback path, or a human
review queue — never silent exceptions or bare strings.

Two layers cooperate:

* :class:`ProcessingError` — a serialisable record carrying both a *developer*
  message (for logs) and a *user* message (for the API/UI). It is safe to return
  across a service boundary.
* :class:`PropIntelliError` and its subclasses — the raised exceptions. Each
  subclass declares stable defaults (error code, stage, severity, recoverability,
  user message) and can be converted to a :class:`ProcessingError` for transport.

The mapping from code to user message is documented in ``docs/error_catalog.md``.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class PipelineStage(enum.StrEnum):
    """The pipeline stage in which an error originated."""

    INGESTION = "ingestion"
    PREPROCESSING = "preprocessing"
    EXTRACTION = "extraction"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    STORAGE = "storage"


class ErrorSeverity(enum.StrEnum):
    """Severity of a processing error, ordered from least to most disruptive."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class ProcessingError(BaseModel):
    """A transport-safe description of a processing failure.

    Attributes
    ----------
    error_code : str
        Stable, catalogued identifier (e.g. ``"OCR_001"``).
    message : str
        Detailed technical message intended for developers and logs.
    user_message : str
        Plain, actionable message intended for an end user or API client.
    stage : PipelineStage
        Stage in which the error occurred.
    severity : ErrorSeverity
        How disruptive the error is.
    recoverable : bool
        Whether the document can still proceed (e.g. via a fallback path or a
        human-in-the-loop queue) rather than being dropped.
    document_id : str or None
        Identifier of the affected document, when known.
    details : dict of str to str
        Optional structured context (filenames, provider names, counts).
    """

    error_code: str
    message: str
    user_message: str
    stage: PipelineStage
    severity: ErrorSeverity
    recoverable: bool
    document_id: str | None = None
    details: dict[str, str] = Field(default_factory=dict)


class PropIntelliError(Exception):
    """Base class for all platform exceptions.

    Subclasses declare class-level defaults that describe the failure category.
    Instances may override the user message, recoverability, and details.

    Parameters
    ----------
    message : str
        Technical message for developers and logs.
    document_id : str or None, optional
        Identifier of the affected document.
    user_message : str or None, optional
        Overrides the class default user-facing message.
    recoverable : bool or None, optional
        Overrides the class default recoverability.
    details : dict of str to str or None, optional
        Structured context attached to the error.
    """

    error_code: str = "GEN_000"
    stage: PipelineStage = PipelineStage.INGESTION
    severity: ErrorSeverity = ErrorSeverity.ERROR
    recoverable: bool = False
    default_user_message: str = (
        "The document could not be processed. Please retry or contact support."
    )

    def __init__(
        self,
        message: str,
        *,
        document_id: str | None = None,
        user_message: str | None = None,
        recoverable: bool | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.document_id = document_id
        self.user_message = user_message or self.default_user_message
        self.recoverable = self.recoverable if recoverable is None else recoverable
        self.details = details or {}

    def as_processing_error(self) -> ProcessingError:
        """Convert this exception into a transport-safe error record.

        Returns
        -------
        ProcessingError
            A serialisable representation suitable for logging, API responses,
            and the human-in-the-loop UI.
        """
        return ProcessingError(
            error_code=self.error_code,
            message=self.message,
            user_message=self.user_message,
            stage=self.stage,
            severity=self.severity,
            recoverable=self.recoverable,
            document_id=self.document_id,
            details=self.details,
        )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
class IngestionError(PropIntelliError):
    """Raised when a document cannot be accepted into the Bronze store."""

    error_code = "ING_001"
    stage = PipelineStage.INGESTION
    severity = ErrorSeverity.ERROR
    recoverable = False
    default_user_message = (
        "The document could not be received. Please verify the file and upload it again."
    )


# ---------------------------------------------------------------------------
# Preprocessing / OCR
# ---------------------------------------------------------------------------
class DocumentReadError(PropIntelliError):
    """Raised when a PDF is corrupt, encrypted, or otherwise unreadable."""

    error_code = "PRE_001"
    stage = PipelineStage.PREPROCESSING
    severity = ErrorSeverity.ERROR
    recoverable = False
    default_user_message = (
        "The document could not be opened. It may be corrupted or password-protected. "
        "Please upload a readable PDF."
    )


class EmptyDocumentError(PropIntelliError):
    """Raised when no usable text could be recovered from a document."""

    error_code = "PRE_002"
    stage = PipelineStage.PREPROCESSING
    severity = ErrorSeverity.WARNING
    recoverable = True
    default_user_message = (
        "No readable text was found in the document. If it is a scanned image, enable OCR "
        "or upload a higher-resolution version. The document has been flagged for review."
    )


class OcrUnavailableError(PropIntelliError):
    """Raised when OCR is requested but the Tesseract backend is unavailable."""

    error_code = "OCR_001"
    stage = PipelineStage.PREPROCESSING
    severity = ErrorSeverity.WARNING
    recoverable = True
    default_user_message = (
        "This document appears to be scanned, but OCR is not available in the current "
        "environment. Install Tesseract or upload a text-based PDF."
    )


class OcrFailureError(PropIntelliError):
    """Raised when the OCR engine fails to process a scanned page."""

    error_code = "OCR_002"
    stage = PipelineStage.PREPROCESSING
    severity = ErrorSeverity.WARNING
    recoverable = True
    default_user_message = (
        "The document could not be read by OCR. It may be a low-quality scan. "
        "Please upload a clearer or higher-resolution version."
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
class ExtractionError(PropIntelliError):
    """Raised when the deterministic extraction layer fails unexpectedly."""

    error_code = "EXT_001"
    stage = PipelineStage.EXTRACTION
    severity = ErrorSeverity.ERROR
    recoverable = False
    default_user_message = (
        "Structured property data could not be extracted from this document. "
        "The document has been flagged for review."
    )


class LlmError(PropIntelliError):
    """Raised when the LLM layer fails; the pipeline downgrades to Layer A.

    Notes
    -----
    This error is recoverable by design: the deterministic baseline still
    produces a result, so an LLM failure degrades quality rather than blocking
    the document.
    """

    error_code = "LLM_002"
    stage = PipelineStage.EXTRACTION
    severity = ErrorSeverity.WARNING
    recoverable = True
    default_user_message = (
        "The AI assistance step was unavailable, so a deterministic extraction was used "
        "instead. Results may be less complete and have been flagged for review."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class DataValidationError(PropIntelliError):
    """Raised when mandatory fields are missing or fail plausibility checks."""

    error_code = "VAL_003"
    stage = PipelineStage.VALIDATION
    severity = ErrorSeverity.WARNING
    recoverable = True
    default_user_message = (
        "Some required property fields are missing or implausible (e.g. price, location, "
        "or area). Please verify the document or complete the missing values."
    )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
class StorageError(PropIntelliError):
    """Raised when a record cannot be persisted to a medallion layer."""

    error_code = "STO_001"
    stage = PipelineStage.STORAGE
    severity = ErrorSeverity.ERROR
    recoverable = False
    default_user_message = (
        "The extracted data could not be saved. Please retry; if the problem persists, "
        "contact support."
    )
