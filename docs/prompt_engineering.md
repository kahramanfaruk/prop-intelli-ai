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

- **Intended benefit**: self-verification on hard layouts, plus a per-field
  confidence that feeds the platform's confidence model directly.
- **Cons**: most tokens/latency; the confidence is **self-reported**. Measured on
  an 8B local model this backfired — its high self-confidence let wrong values win
  in reconciliation, making it the worst variant on the holdout (see Results). The
  premise may hold on a larger/hosted model; on a small one it does not.

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

### Results (measured)

The clean synthetic corpus sits at the deterministic ceiling, so the variants are
compared on the **holdout** corpus, where the baseline leaves real gaps. Measured
with **Ollama, llama3.1 (8B, Q4_K_M), `temperature=0`** (each LLM row is the hybrid
deterministic + variant result via `compare-prompts`):

| Variant | Macro F1 | Field accuracy | Exact-match | Brier | Finding |
| --- | ---: | ---: | ---: | ---: | --- |
| Deterministic (no LLM) — synthetic | 0.996 | 100.0 % | 100.0 % | 0.030 | Consistency ceiling (13 docs). |
| Deterministic (no LLM) — holdout | 0.896 | 90.6 % | 33.3 % | 0.038 | Generalization baseline (3 docs). |
| + v1_direct — holdout | 0.896 | 90.6 % | 33.3 % | 0.034 | No net effect — low-confidence outputs lose in reconciliation. |
| + v2_schema — holdout | 0.888 | **95.3 %** | 33.3 % | 0.093 | **Recovers** missed fields (accuracy↑) but **hallucinates** (macro-F1/Brier↓). |
| + v3_reasoning — holdout | 0.856 | 87.5 % | 0.0 % | 0.089 | **Worst** — high self-confidence overrides correct regex values. |

The measured ranking **contradicts the naive expectation** that more reasoning is
better: on an 8B local model, v2 (schema-anchored) is the only net-useful variant,
and v3 (reasoning + self-reported confidence) *degrades* below the deterministic
baseline because reconciliation lets its over-confident wrong values win. With
**n = 3** the point differences are within noise; the durable conclusions are
qualitative (see [`evaluation.md`](evaluation.md#llm-prompt-variant-comparison-measured)).

## Practical guidance

- On the clean synthetic corpus the **deterministic baseline already scores at the
  ceiling**, so the LLM's value shows only on **real, messy** wording — exactly
  what the holdout measures.
- **Default to v2** for production: it is the only variant measured to *recover*
  fields without overriding correct deterministic values, because its flat 0.70
  confidence stays below the regex confidences in reconciliation.
- **Do not trust a small model's self-reported confidence (v3).** On the 8B model
  it overrode correct values and was the worst configuration. If v3's per-field
  confidence is to be used, gate or down-weight it in reconciliation, or reserve
  v3 for a larger/hosted model where its self-verification is more reliable.
- Accept the trade-off explicitly: the LLM raises recall but can hallucinate, so
  it belongs **behind** reconciliation and human-in-the-loop review, not in front
  of them.
