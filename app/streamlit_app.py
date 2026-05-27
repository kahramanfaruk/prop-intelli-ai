"""Streamlit human-in-the-loop demo for PropIntelli AI.

Upload an exposé PDF, run the extraction pipeline, and review the structured
result with uncertain fields highlighted. Reviewers can correct flagged values
and persist the approved record to the Silver store, then download it as JSON,
demonstrating the confidence-driven HITL loop end to end.

Run with::

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from propintelli.config import get_settings
from propintelli.ingestion.document_store import DocumentStore
from propintelli.logging_setup import configure_logging
from propintelli.pipeline import Pipeline
from propintelli.schemas.enums import Provenance, ReviewStatus
from propintelli.schemas.extraction import FieldValue
from propintelli.schemas.fields import PROPERTY_FIELDS
from propintelli.schemas.property_record import PropertyRecord
from propintelli.storage.repository import SilverRepository
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


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="PropIntelli AI", page_icon="🏠", layout="wide")
    st.title("🏠 PropIntelli AI: Exposé Extraction")
    st.write(
        "Upload a German real-estate exposé (PDF). The pipeline extracts structured "
        "data, scores its confidence, and routes uncertain results to you for review."
    )

    pipeline, repository = _services()
    upload = st.file_uploader("Exposé PDF", type=["pdf"])
    if upload is None:
        st.info("Upload a PDF to begin. Sample exposés live in `sample_data/raw/`.")
        return

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


if __name__ == "__main__":
    main()
