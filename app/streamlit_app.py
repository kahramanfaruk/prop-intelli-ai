"""Streamlit human-in-the-loop demo for PropIntelli AI.

Upload an exposé PDF, run the extraction pipeline, and review the structured
result with uncertain fields highlighted. Reviewers can correct flagged values
and persist the approved record to the Silver store, then download it as JSON,
demonstrating the confidence-driven HITL loop end to end. A sidebar panel
publishes the Gold analytics layer (DuckDB + Parquet/CSV and a city-level market
summary) from the Silver store on demand, so the full Bronze -> Silver -> Gold
medallion is visible in the demo; the market summary is shown both as a table
and as an average-price-per-m2-by-city bar chart.

The active extraction backend is shown so it is clear whether the optional LLM
layer is engaged; the per-field source breakdown and the reconciliation notes
make the hybrid (deterministic + LLM) decisions visible, including where the two
layers disagreed. A corpus-analytics section charts the stored Silver records,
the review-status distribution and, for sale listings, price against living area,
so the medallion's downstream value is visible, not just per-document output.

Run deterministic (offline)::

    uv run streamlit run app/streamlit_app.py

Run with the local Ollama LLM second opinion enabled (slower)::

    make ui-llm
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from propintelli.config import LlmProvider, Settings, get_settings
from propintelli.ingestion.document_store import DocumentStore
from propintelli.logging_setup import configure_logging
from propintelli.pipeline import Pipeline
from propintelli.schemas.enums import ListingType, Provenance, ReviewStatus
from propintelli.schemas.extraction import FieldValue
from propintelli.schemas.fields import PROPERTY_FIELDS
from propintelli.schemas.property_record import PropertyRecord
from propintelli.storage import GoldArtifacts, SilverRepository, build_gold
from propintelli.transformation import normalize_value

_STATUS_STYLE: dict[ReviewStatus, tuple[str, str]] = {
    ReviewStatus.AUTO_APPROVED: ("✅", "green"),
    ReviewStatus.NEEDS_REVIEW: ("⚠️", "orange"),
    ReviewStatus.MANUAL_REQUIRED: ("⛔", "red"),
}
_REVIEW_CONFIDENCE_CEILING = 0.85


@st.cache_resource
def _services() -> tuple[Pipeline, SilverRepository]:
    """Construct the pipeline and repository once per session."""
    configure_logging()
    settings = get_settings()
    repository = SilverRepository(settings.silver_db_path)
    pipeline = Pipeline(
        store=DocumentStore(settings.bronze_dir), repository=None, settings=settings
    )
    return pipeline, repository


def _flatten(record: PropertyRecord) -> dict[str, Any]:
    """Flatten a record into ``{field: value}`` for the canonical registry fields."""
    flat: dict[str, Any] = {}
    for name, spec in PROPERTY_FIELDS.items():
        target: Any = record
        for part in spec.record_path:
            target = getattr(target, part, None)
            if target is None:
                break
        flat[name] = target.value if hasattr(target, "value") else target
    return flat


def _backend_caption(settings: Settings) -> str:
    """Describe the active extraction backend for display.

    Parameters
    ----------
    settings : Settings
        The resolved runtime settings.

    Returns
    -------
    str
        A one-line description: the deterministic baseline alone when no LLM
        backend is configured, otherwise the deterministic baseline plus the
        configured LLM provider, model, and prompt variant.
    """
    if settings.llm_provider is LlmProvider.NONE:
        return "Extraction backend: deterministic baseline only (offline, no LLM)."
    model = {
        LlmProvider.OLLAMA: settings.ollama_model,
        LlmProvider.OPENAI: settings.openai_model,
        LlmProvider.AZURE_OPENAI: settings.azure_openai_deployment or "deployment",
    }[settings.llm_provider]
    return (
        "Extraction backend: deterministic baseline + LLM second opinion "
        f"({settings.llm_provider.value} · {model} · prompt {settings.llm_prompt_variant.value})."
    )


def _provenance_breakdown(record: PropertyRecord) -> str | None:
    """Summarise how many extracted fields came from each source layer.

    Parameters
    ----------
    record : PropertyRecord
        The processed record whose field provenance is summarised.

    Returns
    -------
    str or None
        A compact ``"deterministic: 18 · llm: 2 · reconciled: 4"`` summary in
        canonical provenance order, or ``None`` when no field provenance is
        recorded (e.g. an empty extraction).
    """
    counts = Counter(prov.value for prov in record.quality.field_provenance.values())
    if not counts:
        return None
    parts = [
        f"{origin.value}: {counts[origin.value]}" for origin in Provenance if counts[origin.value]
    ]
    return " · ".join(parts)


def _review_status_counts(records: list[PropertyRecord]) -> dict[str, int]:
    """Count stored records by their human-in-the-loop review status.

    Parameters
    ----------
    records : list of PropertyRecord
        The records to tally.

    Returns
    -------
    dict of str to int
        Count per :class:`~propintelli.schemas.enums.ReviewStatus` value, in
        canonical status order and including statuses with a zero count, so the
        chart axis is stable as records accumulate.
    """
    counts = Counter(record.quality.review_status.value for record in records)
    return {status.value: counts[status.value] for status in ReviewStatus}


def _sale_price_area_points(records: list[PropertyRecord]) -> list[dict[str, float]]:
    """Extract (living area, price) points for sale listings.

    Only sale listings with both a price and a living area are returned: rent and
    sale prices are not mixed (their scales differ by orders of magnitude), and
    each point is a real listing rather than an aggregate, so the chart stays
    honest at the small sample sizes typical of a demo store.

    Parameters
    ----------
    records : list of PropertyRecord
        The records to extract from.

    Returns
    -------
    list of dict
        One ``{"living_area_sqm": float, "price_eur": float}`` mapping per
        qualifying sale listing.
    """
    return [
        {"living_area_sqm": record.living_area_sqm, "price_eur": float(record.price_eur)}
        for record in records
        if record.listing_type is ListingType.SALE
        and record.price_eur is not None
        and record.living_area_sqm is not None
    ]


def _render_header(record: PropertyRecord) -> None:
    """Render the status banner and headline quality metrics."""
    quality = record.quality
    icon, colour = _STATUS_STYLE[quality.review_status]
    st.markdown(f"### {icon} :{colour}[{quality.review_status.value.replace('_', ' ').title()}]")
    columns = st.columns(4)
    columns[0].metric("Overall confidence", f"{quality.overall_confidence:.0%}")
    columns[1].metric("Completeness", f"{quality.completeness:.0%}")
    columns[2].metric("Validation pass rate", f"{quality.validation_pass_rate:.0%}")
    columns[3].metric("Findings", str(len(quality.findings)))
    breakdown = _provenance_breakdown(record)
    if breakdown is not None:
        st.caption(f"Field sources: {breakdown}")


def _render_processing_notes(record: PropertyRecord) -> None:
    """Render reconciliation and processing notes, if any were recorded.

    These warnings surface where the two extraction layers disagreed (the
    deterministic and LLM values differed and one was chosen, with its
    confidence penalised) and any recoverable downgrade (e.g. the LLM layer was
    unavailable), so a reviewer can see how the hybrid result was reached.

    Parameters
    ----------
    record : PropertyRecord
        The processed record whose quality warnings are displayed.
    """
    warnings = record.quality.warnings
    if not warnings:
        return
    with st.expander(f"Reconciliation & processing notes ({len(warnings)})"):
        for note in warnings:
            st.write(f"- {note}")


def _render_corpus_analytics(repository: SilverRepository) -> None:
    """Render corpus-level charts over the stored Silver records.

    Shows the review-status distribution (the routing outcome of the confidence
    model, an exact count that is honest at any sample size) and, for sale
    listings, price against living area (raw per-listing points, not an
    aggregate). Both read from the Silver store, so they reflect every persisted
    record rather than just the current upload. Nothing is rendered when the
    store is empty.

    Parameters
    ----------
    repository : SilverRepository
        The Silver store to read records from.
    """
    records = repository.list_records()
    if not records:
        return
    st.subheader(f"Corpus analytics · {len(records)} stored record(s)")
    status_column, scatter_column = st.columns(2)
    with status_column:
        st.caption("Review-status distribution")
        st.bar_chart(pd.DataFrame({"records": _review_status_counts(records)}))
    with scatter_column:
        points = _sale_price_area_points(records)
        if points:
            st.caption(f"Sale listings: price vs living area (n={len(points)})")
            st.scatter_chart(pd.DataFrame(points), x="living_area_sqm", y="price_eur")
        else:
            st.caption("No sale listings with both price and living area yet.")


def _render_review_form(record: PropertyRecord) -> dict[str, str]:
    """Render an editable form, flagging low-confidence fields, and return edits."""
    flat = _flatten(record)
    confidences = record.quality.field_confidences
    provenance = record.quality.field_provenance
    edits: dict[str, str] = {}

    with st.form("review"):
        st.caption("Fields below the auto-approve threshold are flagged for your review.")
        for name, spec in PROPERTY_FIELDS.items():
            value = flat.get(name)
            confidence = confidences.get(name)
            label = spec.label
            if confidence is not None and confidence < _REVIEW_CONFIDENCE_CEILING:
                label = f"⚠️ {label} ({confidence:.0%})"
            origin = provenance.get(name, Provenance.DETERMINISTIC).value
            edits[name] = st.text_input(
                label,
                value="" if value is None else str(value),
                help=f"Source: {origin}" if name in confidences else "Not extracted",
                key=f"field_{name}",
            )
        submitted = st.form_submit_button("Approve & save to Silver")
    return edits if submitted else {}


def _apply_edits(record: PropertyRecord, edits: dict[str, str]) -> PropertyRecord:
    """Apply human corrections to a record, marking changed fields as manual."""
    flat = _flatten(record)
    updates: dict[str, dict[str, Any]] = {"location": {}, "features": {}, "energy": {}, "_top": {}}
    provenance = dict(record.quality.field_provenance)

    for name, raw in edits.items():
        spec = PROPERTY_FIELDS[name]
        # Reuse the pipeline's typed coercion so an edited date becomes a
        # ``date``, an enum becomes its member, a price becomes ``Decimal`` and
        # so on. ``model_copy(update=...)`` does not validate, so the value must
        # already be the field's declared type before it reaches storage.
        new_value = normalize_value(spec, FieldValue(raw_value=raw, provenance=Provenance.MANUAL))
        if new_value == flat.get(name):
            continue
        provenance[name] = Provenance.MANUAL
        bucket = spec.record_path[0] if len(spec.record_path) == 2 else "_top"
        key = spec.record_path[-1]
        updates[bucket][key] = new_value

    quality = record.quality.model_copy(
        update={"field_provenance": provenance, "review_status": ReviewStatus.AUTO_APPROVED}
    )
    return record.model_copy(
        update={
            **updates["_top"],
            "location": record.location.model_copy(update=updates["location"]),
            "features": record.features.model_copy(update=updates["features"]),
            "energy": record.energy.model_copy(update=updates["energy"]),
            "quality": quality,
        }
    )


def _publish_gold(repository: SilverRepository, gold_dir: Path) -> GoldArtifacts | None:
    """Publish the Gold analytics layer from the current Silver store.

    Parameters
    ----------
    repository : SilverRepository
        The Silver store to read validated records from.
    gold_dir : Path
        Output directory for the Gold artifacts.

    Returns
    -------
    GoldArtifacts or None
        The build artifacts, or ``None`` when the Silver store is empty.
    """
    records = repository.list_records()
    if not records:
        return None
    return build_gold(records, gold_dir)


def _market_summary_chart(summary: list[dict[str, Any]]) -> pd.DataFrame:
    """Shape the Gold market summary into a frame for a price-per-m2 bar chart.

    The aggregation itself is performed once in the Gold layer (DuckDB); this only
    reshapes the resulting rows for display. Each city label embeds the number of
    listings behind it, so the (often small) per-city support is visible and a
    single-listing average is not read as a robust market figure.

    Parameters
    ----------
    summary : list of dict
        The Gold market-summary rows (city-level aggregates for sale listings),
        as returned by :func:`propintelli.storage.gold.build_gold`.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``"<city> (n=<listings>)"`` with a single
        ``"avg_price_per_sqm"`` column, ready for :func:`streamlit.bar_chart`.
        Cities without a name or without an average price are omitted.
    """
    prices = {
        f"{row['city']} (n={row['listings']})": row["avg_price_per_sqm"]
        for row in summary
        if row.get("city") and row.get("avg_price_per_sqm") is not None
    }
    return pd.DataFrame({"avg_price_per_sqm": prices})


def _render_gold_panel(repository: SilverRepository, gold_dir: Path) -> None:
    """Render the sidebar Gold panel: publish analytics from Silver on demand.

    Gold is an aggregate, recomputable view rebuilt from the whole Silver store,
    so it is published explicitly rather than on every save. The panel surfaces
    that step in the demo and shows the resulting market summary, as a table and
    an average-price-per-m2 bar chart, plus the file exports.
    """
    with st.sidebar:
        st.header("Gold analytics layer")
        count = repository.count()
        st.caption(f"Silver store: {count} record(s). Gold is rebuilt from Silver on demand.")
        if not st.button("Publish Gold from Silver", disabled=count == 0):
            return
        artifacts = _publish_gold(repository, gold_dir)
        if artifacts is None:
            st.info("No records in the Silver store yet.")
            return
        st.success(f"Published Gold to `{gold_dir}`.")
        if artifacts.summary:
            st.caption("Market summary (sale listings)")
            st.dataframe(artifacts.summary, use_container_width=True)
            price_chart = _market_summary_chart(artifacts.summary)
            if not price_chart.empty:
                st.caption("Average sale price per m² by city")
                st.bar_chart(price_chart)
        else:
            st.caption("No sale listings to summarise yet.")
        st.download_button(
            "Download properties.csv",
            data=artifacts.properties_csv.read_bytes(),
            file_name="properties.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download market_summary.csv",
            data=artifacts.summary_csv.read_bytes(),
            file_name="market_summary.csv",
            mime="text/csv",
        )


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="PropIntelli AI", page_icon="🏠", layout="wide")
    st.title("🏠 PropIntelli AI: Exposé Extraction")
    st.write(
        "Upload a German real-estate exposé (PDF). The pipeline extracts structured "
        "data, scores its confidence, and routes uncertain results to you for review."
    )

    settings = get_settings()
    st.caption(_backend_caption(settings))

    pipeline, repository = _services()
    _render_gold_panel(repository, settings.gold_dir)
    upload = st.file_uploader("Exposé PDF", type=["pdf"])
    if upload is None:
        st.info("Upload a PDF to begin. Sample exposés live in `sample_data/raw/`.")
        _render_corpus_analytics(repository)
        return

    spinner_message = (
        "Extracting structured data…"
        if settings.llm_provider is LlmProvider.NONE
        else "Extracting structured data (querying the local LLM may take a minute)…"
    )
    with st.spinner(spinner_message):
        result = pipeline.process_bytes(upload.getvalue(), upload.name)
    if result.error is not None:
        st.error(f"**{result.error.error_code}**: {result.error.user_message}")
        return

    record = result.record
    assert record is not None
    _render_header(record)

    if record.quality.findings:
        with st.expander(f"Validation findings ({len(record.quality.findings)})"):
            for finding in record.quality.findings:
                st.write(f"- **{finding.severity.value}** `{finding.rule_id}`: {finding.message}")

    _render_processing_notes(record)

    edits = _render_review_form(record)
    if edits:
        corrected = _apply_edits(record, edits)
        repository.save_record(corrected)
        st.success(f"Saved record `{corrected.property_id}` to the Silver store.")
        record = corrected

    st.download_button(
        "Download JSON",
        data=json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2),
        file_name=f"{record.property_id}.json",
        mime="application/json",
    )

    _render_corpus_analytics(repository)


if __name__ == "__main__":
    main()
