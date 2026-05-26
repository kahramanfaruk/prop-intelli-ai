# Error Catalog

**Philosophy:** errors are *classification signals*, not silent failures. Every
failure is captured as a structured `ProcessingError` carrying both a **developer
message** (for logs) and an **actionable user message** (for the API/UI), the
stage, severity, and whether it is recoverable. The pipeline routes each document
to automatic processing, a fallback path, or the human-in-the-loop queue — it
never crashes a batch.

The model and exception hierarchy live in `src/propintelli/errors.py`.

## Structured error model

```python
ProcessingError(
    error_code:   str,   # stable, catalogued (e.g. "OCR_001")
    message:      str,   # developer/log detail
    user_message: str,   # plain, actionable
    stage:        PipelineStage,
    severity:     ErrorSeverity,   # info | warning | error | fatal
    recoverable:  bool,
    document_id:  str | None,
    details:      dict[str, str],
)
```

Every raised `PropIntelliError` subclass can be converted to a `ProcessingError`
via `as_processing_error()` for transport across the API/UI boundary.

## Catalog

| Code | Exception | Stage | Severity | Recoverable | User message (abridged) |
| --- | --- | --- | --- | --- | --- |
| `ING_001` | `IngestionError` | ingestion | error | no | The document could not be received. Please verify the file and upload it again. |
| `PRE_001` | `DocumentReadError` | preprocessing | error | no | The document could not be opened. It may be corrupted or password-protected. |
| `PRE_002` | `EmptyDocumentError` | preprocessing | warning | yes | No readable text was found. If it is a scan, enable OCR or upload a clearer file. Flagged for review. |
| `OCR_001` | `OcrUnavailableError` | preprocessing | warning | yes | This document appears scanned, but OCR is unavailable here. Install Tesseract or upload a text PDF. |
| `OCR_002` | `OcrFailureError` | preprocessing | warning | yes | The document could not be read by OCR. Please upload a higher-resolution version. |
| `EXT_001` | `ExtractionError` | extraction | error | no | Structured data could not be extracted. The document has been flagged for review. |
| `LLM_002` | `LlmError` | extraction | warning | **yes** | The AI step was unavailable; a deterministic extraction was used instead. Flagged for review. |
| `VAL_003` | `DataValidationError` | validation | warning | yes | Some required fields are missing or implausible (price/location/area). Please verify. |
| `STO_001` | `StorageError` | storage | error | no | The extracted data could not be saved. Please retry; contact support if it persists. |

## Retry / fallback strategy per stage

| Stage | Failure | Strategy |
| --- | --- | --- |
| Ingestion | empty / unreadable input | Reject early (`ING_001`) with an actionable message. |
| Preprocessing | scanned page, no text layer | Try OCR if enabled+available; else `OCR_001`/`PRE_002` → review queue. |
| Preprocessing | OCR fails on a page | Degrade that **page** only (warning); keep the rest of the document. |
| Extraction | LLM backend down/timeout/bad JSON | **Downgrade to the deterministic baseline** (`LLM_002` warning); never block. |
| Extraction | deterministic crash | `EXT_001` → review (should not happen; defensive). |
| Validation | missing/implausible fields | Keep data, attach findings, lower confidence → HITL. |
| Storage | transient DB error | `STO_001`; safe to retry (ingestion is idempotent via UUID). |

## Worked examples

OCR on a low-quality scan:

```json
{
  "error_code": "OCR_002",
  "user_message": "The document could not be read by OCR. It may be a low-quality scan. Please upload a clearer or higher-resolution version.",
  "stage": "preprocessing",
  "severity": "warning",
  "recoverable": true
}
```

LLM failure (recoverable downgrade):

```json
{
  "error_code": "LLM_002",
  "user_message": "The AI assistance step was unavailable, so a deterministic extraction was used instead. Results may be less complete and have been flagged for review.",
  "stage": "extraction",
  "severity": "warning",
  "recoverable": true
}
```

## API boundary (C# ingestion service)

Before a document ever reaches the pipeline, the C# upload endpoint rejects bad
input early and explicitly, so the catalogued pipeline errors above only ever see
plausible documents:

| Condition | HTTP status | Response |
| --- | --- | --- |
| Empty file | `400 Bad Request` | "A non-empty file is required." |
| Not a PDF (missing `%PDF-` signature) | `400 Bad Request` | "Only PDF documents are accepted…" |
| Larger than `Upload:MaxBytes` (default 25 MB) | `413 Payload Too Large` | size-limit message |

## Senior-level summary

> We do not treat exceptions as failures, but as classification signals that
> determine whether a document enters automatic processing, a fallback pipeline,
> or a human review queue.
