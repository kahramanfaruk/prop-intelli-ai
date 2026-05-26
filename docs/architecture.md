# Architecture

PropIntelli AI is a **document-intelligence platform** organised as a
**medallion** data architecture with a **hybrid (deterministic + LLM)** extraction
core and a **confidence-driven human-in-the-loop (HITL)** loop. This document
covers the pipeline, the key technology decisions and trade-offs, how variance is
handled, and the local → Azure production mapping.

## 1. High-level pipeline

```mermaid
flowchart LR
    subgraph Bronze["Bronze — raw ingestion"]
      U[PDF / image] --> API[C# .NET API\n/api/documents/upload]
      U --> CLI[Python CLI / worker]
      API --> RAW[(Raw store + manifest\nUUID, SHA-256)]
      CLI --> RAW
    end

    subgraph Pre["Preprocessing"]
      RAW --> CLS[Classifier\ndigital / hybrid / scanned]
      CLS --> TXT[Text extraction\nPyMuPDF + pdfplumber\nOCR fallback: Tesseract]
    end

    subgraph Extract["Hybrid AI extraction"]
      TXT --> LA[Layer A: deterministic\nGerman regex / heuristics]
      TXT --> LB[Layer B: LLM (optional)\nnone / ollama / openai / azure]
      LA --> REC[Layer C: reconciliation\n+ per-field confidence]
      LB --> REC
    end

    subgraph Silver["Silver — validated records"]
      REC --> NORM[Normalisation\ntypes, enums, dates, amounts]
      NORM --> VAL[Validation rules\nmandatory / range / plausibility]
      VAL --> CONF[Confidence model\n+ HITL routing]
      CONF --> DB[(SQLite via SQLAlchemy\nproperties + features + findings + runs)]
    end

    CONF -->|>= 0.85| GOLD
    CONF -->|0.60–0.85| HITL[Streamlit HITL UI\ncorrect flagged fields]
    CONF -->|< 0.60| HITL
    HITL --> DB

    subgraph Gold["Gold — analytics"]
      DB --> G[(DuckDB + Parquet/CSV\nwide + features + market summary)]
    end
```

**Medallion layers**

- **Bronze** — immutable raw documents + manifest (filename, SHA-256, size, time).
- **Silver** — cleaned, typed, validated `PropertyRecord`s with quality metadata.
- **Gold** — analytics-ready wide table, long features table, and a market summary.

## 2. Stage responsibilities

| Stage | Module | Responsibility |
| --- | --- | --- |
| Ingestion | `ingestion/document_store.py` | Assign UUID, persist raw bytes + manifest. |
| Classification | `preprocessing/pdf_classifier.py` | Decide digital / hybrid / scanned. |
| Text extraction | `preprocessing/text_extractor.py` | PyMuPDF/pdfplumber; optional Tesseract OCR. |
| Layer A | `extraction/deterministic.py` | Regex/heuristic German extraction (always on). |
| Layer B | `extraction/llm/*` | Optional LLM extraction via a provider protocol. |
| Layer C | `extraction/reconciliation.py` | Merge A+B; agreement boosts, disagreement flags. |
| Normalisation | `transformation/normalize.py` | Parse to typed values (Decimal, date, enum…). |
| Validation | `validation/rules.py` | Mandatory/range/plausibility → findings + pass rate. |
| Confidence | `confidence.py` | Weighted score → auto / review / manual routing. |
| Persistence | `storage/repository.py` | Upsert Silver record + run audit. |
| Gold | `storage/gold.py` | Publish analytics exports via DuckDB. |
| Orchestration | `pipeline.py`, `batch/runner.py` | Single-doc, batch, and Bronze-watch execution. |

### Ingestion → extraction integration

The C# ingestion API and the Python extractor are decoupled through the shared
Bronze store, not a direct call. The API writes an uploaded document (UUID +
manifest) into Bronze; the Python worker runs `pipeline.process_pending()` — via
`propintelli watch` (a poll loop) or `propintelli process-bronze` (one-shot) —
which enumerates Bronze, processes any document without a prior run **in place**
(preserving its id, not re-ingesting), and persists to Silver. This makes the
`API → Bronze → preprocess → …` arrow fire end to end, and is idempotent (a
document is attempted once) and producer-agnostic (it reads both the Python
snake_case and the C# camelCase manifest). In `docker compose`, the worker runs
`watch`, so a `POST /api/documents/upload` is extracted automatically. In
production this poll is replaced by an event trigger (Blob → Event Grid → queue).

## 3. Key technology decisions & trade-offs

| Decision | Choice | Why | Alternatives / trade-offs |
| --- | --- | --- | --- |
| Extraction strategy | **Hybrid: deterministic baseline + optional LLM** | Deterministic is free, fast, offline, and auditable; the LLM adds recall on free-text. The baseline is also the LLM's safety net. | Pure-LLM: higher recall but cost, latency, hallucination, non-determinism. Pure-regex: brittle on novel layouts. |
| LLM access | **Provider protocol (none/ollama/openai/azure)** | Swap backends by config; no vendor lock-in; offline by default. | Hard-coding one SDK couples the core to a vendor and breaks offline/CI. |
| Schema | **Pydantic v2 + central field registry** | One source of truth drives extraction, validation, scoring, storage, eval, and UI; eliminates drift. | Hand-maintained field lists per module drift and rot. |
| PDF text | **PyMuPDF (+ pdfplumber)** | Fast, robust layout/text extraction; pure-Python wheels. | pdfminer is slower; Camelot/tabula target tables only. |
| OCR | **Tesseract, optional & probed at runtime** | Free, local; engaged only when needed and available. | Cloud OCR (Azure DI) is better but paid; made the prod mapping. |
| Storage | **SQLite (Silver) + DuckDB/Parquet (Gold)** | Zero-config, runs in CI, columnar analytics for free; SQLAlchemy keeps it swappable. | Postgres needs a running server (not offline-friendly for a take-home). |
| Confidence/HITL | **Weighted score + thresholds** | Transparent, tunable, explainable to stakeholders. | A learned router needs labelled data we don't yet have. |
| Tooling | **uv, ruff, mypy --strict, pytest** | Fast, reproducible, strict quality gates. | — |

## 4. Handling document variance (a core requirement)

Different agencies render the same concept very differently. The pipeline is
robust to this on three independent levels:

1. **Layout-agnostic deterministic extraction.** Layer A anchors on German
   **labels and units** (`Kaufpreis`, `m²`, `Baujahr`, a 5-digit postal code, the
   `…straße N` address pattern), not on positions. It tolerates the label and
   value being inline *or* on separate lines — exactly the difference between the
   tabular, prose, and sectioned sample layouts, all of which extract correctly.
2. **Shared synonym vocabulary** (`extraction/vocabulary.py`) maps many German
   surface forms onto canonical values (e.g. `Fernwärme`/`Wärmepumpe`/`Gas-…` →
   heating types), so new wording is a one-line addition.
3. **Negation and multi-amount handling.** Feature extraction scopes German
   negation cues (`kein`, `ohne`, …) per clause, so "kein Balkon" yields an
   explicit `False` rather than a false positive, and a single positive mention
   still wins ("kein Stellplatz, aber Tiefgarage" → parking present). Price
   selection rejects amounts governed by ancillary-cost labels (Hausgeld,
   Nebenkosten, Provision, Kaution), so a listing with several monetary lines
   still yields the right headline price.
4. **LLM fallback for the long tail.** When a layout defeats the regexes, the
   optional LLM layer — which does not depend on fixed positions — recovers the
   field, and reconciliation decides which value to trust.

The **synthetic** corpus deliberately spans three layouts, sale/cold-rent/warm-rent
listings, a sparse listing, negated and explicitly-absent features, and multi-amount
price lines. An **independently-authored holdout** corpus then measures how well
this transfers to unseen wording (see [`evaluation.md`](evaluation.md)); the gap
between the two is the honest robustness signal, and the residual misses it
surfaces are exactly the long-tail cases the LLM layer targets.

## 5. Error handling philosophy

Failures are **classification signals**, not silent exceptions. Each stage is
isolated; a failure is converted to a structured `ProcessingError` (developer +
user message, stage, severity, recoverability) and routed:

- **fatal** → reject early with an actionable message (e.g. corrupt PDF);
- **recoverable** → fall back (LLM failure → deterministic; OCR page failure →
  skip that page) and continue;
- **uncertain** → send to the HITL review queue.

A single bad document never aborts a batch. The full code → message → retry
mapping is in [`error_catalog.md`](error_catalog.md).

## 6. Confidence & human-in-the-loop

```
overall = 0.40·extraction + 0.30·completeness + 0.20·validation_pass_rate + 0.10·source_quality
```

`completeness` is the fraction of the **required** fields (price, living area,
postal code, city) that were extracted — the fields every valid listing must
carry. It deliberately does *not* count optional fields: a studio with no plot
area or a sparse listing without an energy certificate is legitimately sparse, so
penalising it would conflate "absent in the document" with "missed by the
extractor" and route correct records to review for no reason.

| Overall confidence | Routing |
| --- | --- |
| ≥ 0.85 | **auto-approved** |
| 0.60 – 0.85 | **needs review** (HITL) |
| < 0.60 | **manual required** (HITL) |

A hard validation error (e.g. a missing mandatory field) blocks auto-approval
regardless of the numeric score. Human corrections are written back with `manual`
provenance — turning the platform into a continuously improving system rather
than a static parser.

## 7. Azure production mapping

| Local (this repo) | Azure production |
| --- | --- |
| Local FS Bronze store | **Azure Blob Storage** (+ Event Grid trigger) |
| Bronze poll (`watch`) | **Event Grid → Service Bus** push (no polling) |
| API auth: none (take-home) | **Azure API Management + Microsoft Entra ID** (or managed identity); private networking |
| C# ingestion/validation API | **Azure Container Apps** / App Service |
| Python extraction worker | **Azure Functions / Container Apps** + Service Bus queue |
| Deterministic + Ollama LLM | **Azure OpenAI** (GPT-4o-mini) in AI Foundry |
| Tesseract OCR fallback | **Azure Document Intelligence** (prebuilt + custom) |
| SQLite (Silver) | **Azure SQL** |
| DuckDB / Parquet (Gold) | **Microsoft Fabric** Lakehouse / Warehouse (OneLake) |
| Streamlit HITL | **Power Apps / Power Platform** review queue |
| Structured JSON logs | **Application Insights / Azure Monitor** |
| GitHub Actions | **Azure DevOps Pipelines** |

The medallion layout maps 1:1 onto a Fabric **Lakehouse** (Bronze/Silver/Gold),
and the repository pattern means swapping SQLite → Azure SQL is a connection
string, not a rewrite.

## 8. Scalability

- **Stateless workers.** The pipeline is per-document and side-effect-isolated, so
  scaling out is "add more workers behind a queue" (Service Bus / Azure Functions).
- **Idempotent ingestion.** UUID + content hash enable dedup and safe retries.
- **Batch isolation.** Failures are per-document; throughput degrades gracefully.
- **Columnar Gold.** DuckDB/Parquet (→ Fabric) serves analytics without loading the
  transactional store.
