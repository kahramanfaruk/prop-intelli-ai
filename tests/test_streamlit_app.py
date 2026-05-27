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

from propintelli.schemas.enums import ListingType, PriceKind, ReviewStatus
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


def test_app_renders_headlessly_without_error() -> None:
    # Smoke-test the whole script via Streamlit's headless AppTest: it must run
    # (no upload -> early return) and expose the Gold panel in the sidebar.
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file(str(_APP_PATH), default_timeout=30).run()

    assert not app_test.exception
    assert any("PropIntelli AI" in title.value for title in app_test.title)
    assert any(header.value == "Gold analytics layer" for header in app_test.sidebar.header)
