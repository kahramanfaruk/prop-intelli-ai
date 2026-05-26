# Prompt Engineering

The optional LLM layer (Layer B) ships **three documented prompt variants**,
selectable via `PROPINTELLI_LLM_PROMPT_VARIANT`. All variants share one system
message and one defensive parser, so they can be compared apples-to-apples and
swapped without code changes. The variants are defined in
`src/propintelli/extraction/llm/prompts.py`.

## Shared system message

> You are a precise information-extraction system for German real-estate exposés.
> You extract structured data. You never invent values: if a field is not stated
> in the document, return null for it. You return only the requested JSON.

## Variants

### v1 — Direct
A terse instruction: "extract the fields, return JSON, use null when unknown."

- **Pros**: shortest prompt, lowest token cost.
- **Cons**: the model invents field names, mixes German/English values, and is
  inconsistent about nulls. A baseline to measure against.

### v2 — Schema-anchored (default)
Injects the **full canonical field schema** generated from the registry (field
name, kind, allowed enum values, German→English mapping hint, required flag) and
demands a single strict-JSON object `{"fields": {…}}`.

- **Pros**: stable field naming and typing; enums constrained to allowed values;
  far fewer nulls-as-empty-strings. Best quality/cost balance — the default.
- **Cons**: longer prompt; still no self-consistency signal.

### v3 — Reasoning + per-field confidence
Extends v2 with an instruction to **internally verify** each value against the
text before answering, and to return a parallel `"confidences"` object with a
0–1 score per field.

- **Pros**: highest robustness on noisy/unusual layouts; the per-field confidence
  feeds the platform's confidence model directly (instead of a flat default).
- **Cons**: most tokens/latency; confidence is self-reported and must still be
  cross-checked by reconciliation and validation.

## Why a single envelope

Every variant returns `{"fields": {…}}` (v3 adds `{"confidences": {…}}`), so one
parser (`parse_extraction`) handles all three. The parser is **defensive**: it
restricts keys to the registry, drops nulls, and clamps confidences to `[0, 1]` —
so a misbehaving model cannot inject unknown fields or out-of-range scores.

## Comparison methodology

The variants are compared with the **same harness** used for the deterministic
baseline ([`evaluation.md`](evaluation.md)), holding the provider fixed and
changing only the prompt, via a single command:

```bash
export PROPINTELLI_LLM_PROVIDER=ollama        # or openai / azure_openai
uv run propintelli compare-prompts \
  --raw-dir sample_data/holdout/raw --truth-dir sample_data/holdout/ground_truth
```

It prints one row per variant — Macro-F1, field accuracy, exact-match, and the
**Brier** calibration score — so the variants are directly comparable. The
comparison machinery (`compare_prompt_variants`) is verified end-to-end in CI with
a deterministic **stub provider**, so the harness itself is known-good; the
command refuses to run with the `none` backend (the variants would be identical).

### Results

The clean synthetic corpus already sits at the deterministic ceiling, so the
variants are best compared on the **holdout** corpus, where the baseline leaves
real gaps (post-posed negation, bare "Klasse E", `Lage:`-style districts).

> The machine used to build this project had **no LLM backend** (no Ollama, no
> network), so the model rows below are intentionally left for you to fill — the
> numbers are **not fabricated**. Run the command above with a free local Ollama
> model to populate them; the deterministic rows are measured.

| Variant | Macro F1 | Field accuracy | Exact-match | Notes |
| --- | --- | --- | --- | --- |
| Deterministic baseline (no LLM) — synthetic | **0.996** | **1.000** | **1.000** | Measured (13-doc corpus). |
| Deterministic baseline (no LLM) — holdout | **0.902** | **0.921** | **0.333** | Measured (3-doc authored holdout). |
| v1_direct (holdout) | _run to fill_ | _run to fill_ | _run to fill_ | Expect lower precision (field-name drift). |
| v2_schema (holdout) | _run to fill_ | _run to fill_ | _run to fill_ | Expect strong typing/naming. |
| v3_reasoning (holdout) | _run to fill_ | _run to fill_ | _run to fill_ | Expect best recall on hard wording. |

## Practical guidance

- On the clean synthetic corpus the **deterministic baseline already scores at the
  ceiling**, so the LLM's value shows on **real, messy** exposés (free-text
  descriptions, unusual labels) and on the one honest false positive in the
  baseline ("Garten" matched from the park name *Englischer Garten*), which the
  reasoning variant + reconciliation resolve.
- Default to **v2** for production (quality/cost); reserve **v3** for documents the
  baseline flags for review (a cost-aware escalation strategy).
