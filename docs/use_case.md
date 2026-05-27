# Use Case: Immobilien-Exposé Intelligence

## Problem

Real-estate listings (Immobilien-Exposés) arrive as PDFs from dozens of agencies
and portals (e.g. ImmoScout exports). They are **visually unstructured** (every
agency has its own layout), **semantically semi-structured** (the same concepts,
price, area, year built, appear under different German labels), often
**image-heavy** (floor plans, badges), and **inconsistent** (missing fields,
mixed number formats). Today this data is re-keyed by hand into CRMs and
spreadsheets.

That manual step is slow, expensive, and error-prone, and it is the bottleneck in
lead processing: every minute a listing sits un-digitised is a minute competitors
are ahead.

## Target users (personas)

| Persona | Need |
| --- | --- |
| **Real-estate analyst** | Screen many listings quickly; consistent, comparable fields. |
| **PropTech / listing aggregator** | Ingest third-party exposés into a normalised catalogue at scale. |
| **Investment / valuation team** | Reliable price, area, year, and energy data for portfolio screening. |
| **CRM integration** | Automatic, structured ingestion instead of copy-paste. |

## Input → Output

- **Input**: a German real-estate exposé as a PDF (digital text, scanned, or hybrid).
- **Output**: a validated, normalised `PropertyRecord` (Pydantic v2), see
  [`data_model.md`](data_model.md), covering **Fläche, Preis, Lage, Ausstattung,
  Baujahr** and more, with per-field confidence, provenance, validation findings,
  and a human-in-the-loop routing decision. Persisted to a medallion store
  (SQLite → DuckDB/Parquet) and exportable as JSON/CSV.

## Business value

- **Eliminates manual re-keying.** A reviewer confirms instead of transcribes.
- **Faster lead processing.** Auto-approved listings flow straight through; only
  uncertain ones reach a human.
- **Consistent data quality.** Normalisation + plausibility rules catch bad values
  before they pollute the CRM.
- **Auditable & continuously improving.** Every run is recorded; human corrections
  are captured as high-trust labels that feed future prompt/rule tuning.

### Quantified illustration

Assume a team digitises 200 exposés/day, ~6 minutes of manual entry each
(20 hours/day). With the pipeline, ~85% are auto-approved and ~15% need a
~1-minute confirmation:

```
Manual today:        200 × 6 min            = 1200 min/day (~20 h)
With PropIntelli:     30 × 1 min (review)   =   30 min/day
Reduction:           ~97.5% of manual effort
```

Even at conservative auto-approval rates, the operational saving is the headline
business case; the data-quality and audit improvements are the durable ones.

## Success metrics

Two corpora are reported because they answer different questions: a **synthetic**
corpus measures round-trip *consistency*, and an independently-authored
**holdout** measures *generalization* to wording the extractor was not built
around. All proportions carry 95% Wilson confidence intervals (small samples).

| Metric | Definition | Target | Synthetic (consistency) | Holdout (generalization) |
| --- | --- | --- | --- | --- |
| **Field accuracy** | Correct values among expected fields | ≥ 90% | **100%** (CI 98.7–100) | **90.6%** (CI 81.0–95.6) |
| **Macro F1** | Mean per-field F1 (rewards correct absence) | ≥ 0.90 | **0.996** | **0.896** |
| **Exact-match ratio** | Documents with every expected field correct | ≥ 0.70 | **1.00** | **0.33** |
| **Confidence calibration** | Brier score (confidence vs. correctness) | well-calibrated | 0.030 | **0.038** |
| **Auto-approval rate** | Share routed without human review | maximise safely | tunable via thresholds | n/a |

The synthetic number reflects consistency, not real-world accuracy; the holdout
number is the honest generalization estimate. Methodology, the full per-field
tables with CIs, the calibration reliability table, and the catalogue of honest
residual misses are in [`evaluation.md`](evaluation.md).

## Scope & assumptions

- German-language sale and rent listings; the field registry is the single source
  of truth and is straightforward to extend.
- The deterministic baseline runs fully offline; an optional LLM layer (Ollama /
  OpenAI / Azure OpenAI) improves recall on free-text and unusual layouts.
- OCR is an optional path for scanned PDFs (Tesseract); the digital-text path is
  primary.
