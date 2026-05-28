"""Tests for the Streamlit HITL helper logic.

The app module is loaded by path because ``app/`` is not an installed package;
its ``_flatten``/``_apply_edits`` helpers do not touch the Streamlit runtime, so
they can be exercised directly. This guards the "Approve & save" path, where an
edited record must keep each field's declared Python type (a ``date`` stays a
``date``, not the string the form rendered) so it persists to SQLite.
"""

from __future__ import annotations

import importlib.util
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("streamlit")

from propintelli.config import LlmProvider, PromptVariant, Settings
from propintelli.schemas.enums import ListingType, PriceKind, Provenance, ReviewStatus
from propintelli.schemas.property_record import Location, PropertyRecord, QualityReport
from propintelli.storage import SilverRepository

_APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def _load_app() -> ModuleType:
    spec = importlib.util.spec_from_file_location("propintelli_streamlit_app", _APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record() -> PropertyRecord:
    return PropertyRecord(
        property_id="hitl1",
        source_document="x.pdf",
        listing_type=ListingType.SALE,
        price_eur=Decimal("449000.00"),
        price_kind=PriceKind.PURCHASE,
        living_area_sqm=92.0,
        year_built=1998,
        availability_date=date(2026, 7, 1),
        location=Location(city="Nürnberg", postal_code="90408"),
        quality=QualityReport(overall_confidence=0.9, review_status=ReviewStatus.NEEDS_REVIEW),
    )


def test_apply_edits_keeps_typed_values_and_persists(tmp_path: Path) -> None:
    app = _load_app()
    record = _record()
    # Mirror the form: every field rendered as a string, left unchanged.
    flat = app._flatten(record)
    edits = {name: ("" if value is None else str(value)) for name, value in flat.items()}

    corrected = app._apply_edits(record, edits)

    # Edited strings must coerce back to the declared types, not stay strings.
    assert isinstance(corrected.availability_date, date)
    assert corrected.availability_date == date(2026, 7, 1)
    assert isinstance(corrected.price_eur, Decimal)
    assert corrected.listing_type is ListingType.SALE

    # The corrected record must persist without a SQLite type error.
    repo = SilverRepository(tmp_path / "silver" / "db.sqlite")
    repo.save_record(corrected)
    assert repo.get_record("hitl1") is not None


def test_apply_edits_applies_a_changed_date(tmp_path: Path) -> None:
    app = _load_app()
    record = _record()
    flat = app._flatten(record)
    edits = {name: ("" if value is None else str(value)) for name, value in flat.items()}
    edits["availability_date"] = "2026-09-15"  # reviewer corrects the date

    corrected = app._apply_edits(record, edits)

    assert corrected.availability_date == date(2026, 9, 15)
    repo = SilverRepository(tmp_path / "silver" / "db.sqlite")
    repo.save_record(corrected)
    assert repo.get_record("hitl1") is not None


def test_publish_gold_returns_none_for_empty_store(tmp_path: Path) -> None:
    app = _load_app()
    repo = SilverRepository(tmp_path / "silver" / "db.sqlite")
    assert app._publish_gold(repo, tmp_path / "gold") is None


def test_publish_gold_builds_artifacts_from_silver(tmp_path: Path) -> None:
    app = _load_app()
    repo = SilverRepository(tmp_path / "silver" / "db.sqlite")
    repo.save_record(_record())

    artifacts = app._publish_gold(repo, tmp_path / "gold")

    assert artifacts is not None
    assert artifacts.properties_csv.exists()
    assert artifacts.summary_csv.exists()
    # The sale listing appears in the city-level market summary.
    assert any(row["city"] == "Nürnberg" for row in artifacts.summary)


def test_backend_caption_reports_deterministic_only_without_llm() -> None:
    app = _load_app()
    caption = app._backend_caption(Settings(llm_provider=LlmProvider.NONE))
    assert "deterministic baseline only" in caption


def test_backend_caption_names_provider_model_and_variant() -> None:
    app = _load_app()
    settings = Settings(
        llm_provider=LlmProvider.OLLAMA,
        ollama_model="llama3.1",
        llm_prompt_variant=PromptVariant.V2_SCHEMA,
    )
    caption = app._backend_caption(settings)
    assert "ollama" in caption
    assert "llama3.1" in caption
    assert "v2_schema" in caption


def test_provenance_breakdown_counts_sources_in_canonical_order() -> None:
    app = _load_app()
    record = _record()
    record.quality.field_provenance = {
        "price_eur": Provenance.DETERMINISTIC,
        "postal_code": Provenance.DETERMINISTIC,
        "district": Provenance.LLM,
        "city": Provenance.RECONCILED,
    }
    assert app._provenance_breakdown(record) == "deterministic: 2 · llm: 1 · reconciled: 1"


def test_provenance_breakdown_is_none_when_no_fields_were_extracted() -> None:
    app = _load_app()
    assert app._provenance_breakdown(_record()) is None


def test_review_status_counts_includes_every_status_in_canonical_order() -> None:
    app = _load_app()
    auto = _record()
    auto.quality.review_status = ReviewStatus.AUTO_APPROVED
    review = _record()  # _record() defaults to NEEDS_REVIEW
    counts = app._review_status_counts([auto, review])
    assert counts == {"auto_approved": 1, "needs_review": 1, "manual_required": 0}
    assert list(counts) == ["auto_approved", "needs_review", "manual_required"]


def test_sale_price_area_points_keeps_only_sale_listings_with_price_and_area() -> None:
    app = _load_app()
    sale = _record()  # SALE, price 449000.00, area 92.0
    rent = PropertyRecord(
        property_id="rent1",
        source_document="r.pdf",
        listing_type=ListingType.RENT,
        price_eur=Decimal("980.00"),
        living_area_sqm=58.5,
        quality=QualityReport(overall_confidence=0.9, review_status=ReviewStatus.AUTO_APPROVED),
    )
    sale_without_area = PropertyRecord(
        property_id="sale2",
        source_document="s2.pdf",
        listing_type=ListingType.SALE,
        price_eur=Decimal("500000.00"),
        quality=QualityReport(overall_confidence=0.9, review_status=ReviewStatus.AUTO_APPROVED),
    )

    points = app._sale_price_area_points([sale, rent, sale_without_area])

    assert points == [{"living_area_sqm": 92.0, "price_eur": 449000.0}]


def test_app_renders_headlessly_without_error() -> None:
    # Smoke-test the whole script via Streamlit's headless AppTest: it must run
    # (no upload -> early return) and expose the Gold panel in the sidebar.
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file(str(_APP_PATH), default_timeout=30).run()

    assert not app_test.exception
    assert any("PropIntelli AI" in title.value for title in app_test.title)
    assert any(header.value == "Gold analytics layer" for header in app_test.sidebar.header)
