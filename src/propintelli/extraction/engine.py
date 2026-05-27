"""The extraction engine: orchestrates Layers A, B, and C.

The engine always runs the deterministic Layer A, optionally runs the LLM
Layer B, and reconciles them in Layer C. A failure in the LLM layer is caught and
recorded as a warning so the document still completes on the deterministic
result, the LLM degrades quality, never availability.
"""

from __future__ import annotations

from propintelli.config import LlmProvider, PromptVariant, Settings, get_settings
from propintelli.errors import ExtractionError, LlmError
from propintelli.extraction.deterministic import extract_deterministic
from propintelli.extraction.llm.base import LlmExtraction, build_provider
from propintelli.extraction.reconciliation import reconcile
from propintelli.logging_setup import get_logger
from propintelli.preprocessing.text_extractor import PreprocessedDocument
from propintelli.schemas.enums import Provenance
from propintelli.schemas.extraction import ExtractionResult, FieldValue

logger = get_logger(__name__)

# Default per-field confidence assigned to LLM values when the variant does not
# return explicit confidences. The schema-anchored variants are trusted more.
_LLM_DEFAULT_CONFIDENCE: dict[PromptVariant, float] = {
    PromptVariant.V1_DIRECT: 0.55,
    PromptVariant.V2_SCHEMA: 0.70,
    PromptVariant.V3_REASONING: 0.70,
}


def run_extraction(
    document: PreprocessedDocument,
    settings: Settings | None = None,
) -> ExtractionResult:
    """Extract structured fields from a preprocessed document.

    Parameters
    ----------
    document : PreprocessedDocument
        The text and metadata produced by preprocessing.
    settings : Settings or None, optional
        Settings selecting the LLM backend and prompt variant.

    Returns
    -------
    ExtractionResult
        The reconciled field map with accumulated warnings.

    Raises
    ------
    ExtractionError
        If the deterministic layer fails unexpectedly. (An LLM failure is
        recoverable and recorded as a warning instead.)
    """
    settings = settings or get_settings()
    warnings = list(document.warnings)

    try:
        layer_a = extract_deterministic(document.text)
    except Exception as exc:  # the deterministic baseline must not crash silently
        raise ExtractionError(
            f"Deterministic extraction failed: {exc}", document_id=document.document_id
        ) from exc

    layer_b: dict[str, FieldValue] = {}
    if settings.llm_provider is not LlmProvider.NONE:
        layer_b, llm_warnings = _run_llm_layer(document, settings)
        warnings.extend(llm_warnings)

    fields, reconciliation_warnings = reconcile(layer_a, layer_b)
    warnings.extend(reconciliation_warnings)

    return ExtractionResult(
        document_id=document.document_id,
        source_document=document.source_document,
        fields=fields,
        warnings=warnings,
    )


def _run_llm_layer(
    document: PreprocessedDocument,
    settings: Settings,
) -> tuple[dict[str, FieldValue], list[str]]:
    """Run the optional LLM layer, downgrading gracefully on any failure."""
    try:
        provider = build_provider(settings)
        extraction = provider.extract(document.text)
        return _to_field_values(extraction, settings), []
    except LlmError as exc:
        logger.warning("llm_layer_downgraded", extra={"document_id": document.document_id})
        return {}, [exc.user_message]
    except Exception as exc:  # any unexpected provider failure still degrades safely
        error = LlmError(f"Unexpected LLM layer failure: {exc}", document_id=document.document_id)
        logger.warning("llm_layer_failed", extra={"document_id": document.document_id})
        return {}, [error.user_message]


def _to_field_values(extraction: LlmExtraction, settings: Settings) -> dict[str, FieldValue]:
    """Convert a raw LLM extraction into provenance-tagged field values."""
    default_confidence = _LLM_DEFAULT_CONFIDENCE[settings.llm_prompt_variant]
    fields: dict[str, FieldValue] = {}
    for name, value in extraction.fields.items():
        raw = "true" if value is True else "false" if value is False else str(value).strip()
        if not raw:
            continue
        fields[name] = FieldValue(
            raw_value=raw,
            confidence=extraction.field_confidences.get(name, default_confidence),
            provenance=Provenance.LLM,
        )
    return fields
