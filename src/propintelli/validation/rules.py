"""Data-quality validation rules.

The rules engine runs plausibility, mandatory-field, range, and cross-field
checks over the normalised values. Each *applicable* rule contributes to a
validation pass rate that feeds the confidence model; failures become
:class:`~propintelli.schemas.property_record.ValidationFinding` records. Rules do
not mutate or reject data — implausible values are kept and flagged so the
document can be routed to human review rather than dropped.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date
from typing import Any

from propintelli.errors import ErrorSeverity
from propintelli.schemas.enums import ListingType
from propintelli.schemas.fields import required_field_names
from propintelli.schemas.property_record import ValidationFinding

_POSTAL_RE = re.compile(r"^\d{5}$")
_PRICE_MAX = 100_000_000.0
_AREA_MIN, _AREA_MAX = 5.0, 1000.0
_PLOT_MAX = 100_000.0
_YEAR_MIN = 1800
_ROOMS_MAX = 50.0
_DEMAND_MAX = 1000.0
# Plausible price per square metre, separated by listing type.
_SALE_PPSQM = (300.0, 40_000.0)
_RENT_PPSQM = (2.0, 80.0)


class _RuleRunner:
    """Accumulates rule outcomes and computes the pass rate."""

    def __init__(self) -> None:
        self.findings: list[ValidationFinding] = []
        self.executed = 0
        self.passed = 0

    def check(
        self,
        *,
        applicable: bool,
        ok: bool,
        finding: Callable[[], ValidationFinding],
    ) -> None:
        """Record one rule outcome.

        Parameters
        ----------
        applicable : bool
            Whether the rule applies (e.g. its field is present). Inapplicable
            rules are not counted toward the pass rate.
        ok : bool
            Whether the applicable rule passed.
        finding : callable
            Factory invoked to build the finding when an applicable rule fails.
        """
        if not applicable:
            return
        self.executed += 1
        if ok:
            self.passed += 1
        else:
            self.findings.append(finding())

    @property
    def pass_rate(self) -> float:
        """Fraction of executed rules that passed (1.0 when none executed)."""
        return self.passed / self.executed if self.executed else 1.0


def _finding(
    rule_id: str, field: str | None, severity: ErrorSeverity, message: str
) -> ValidationFinding:
    """Construct a validation finding."""
    return ValidationFinding(rule_id=rule_id, field=field, severity=severity, message=message)


def validate(values: dict[str, Any]) -> tuple[list[ValidationFinding], float]:
    """Validate normalised field values.

    Parameters
    ----------
    values : dict of str to object
        Normalised, typed field values keyed by canonical field name.

    Returns
    -------
    tuple of (list of ValidationFinding, float)
        The findings and the validation pass rate in ``[0, 1]``.
    """
    runner = _RuleRunner()
    _check_mandatory(runner, values)
    _check_ranges(runner, values)
    _check_cross_field(runner, values)
    return runner.findings, runner.pass_rate


def _check_mandatory(runner: _RuleRunner, values: dict[str, Any]) -> None:
    """Ensure every required field is present and non-empty."""
    for name in required_field_names():
        present = values.get(name) not in (None, "")
        runner.check(
            applicable=True,
            ok=present,
            finding=lambda name=name: _finding(  # type: ignore[misc]
                f"mandatory.{name}",
                name,
                ErrorSeverity.ERROR,
                f"Required field '{name}' is missing.",
            ),
        )


def _check_ranges(runner: _RuleRunner, values: dict[str, Any]) -> None:
    """Apply per-field range and format checks."""
    price = values.get("price_eur")
    runner.check(
        applicable=price is not None,
        ok=price is not None and 0 < float(price) < _PRICE_MAX,
        finding=lambda: _finding(
            "range.price_eur",
            "price_eur",
            ErrorSeverity.ERROR,
            f"Price {price} is outside the plausible range (0, {_PRICE_MAX:.0f}).",
        ),
    )

    area = values.get("living_area_sqm")
    runner.check(
        applicable=area is not None,
        ok=area is not None and _AREA_MIN <= float(area) <= _AREA_MAX,
        finding=lambda: _finding(
            "range.living_area_sqm",
            "living_area_sqm",
            ErrorSeverity.WARNING,
            f"Living area {area} m² is outside the plausible range "
            f"[{_AREA_MIN:.0f}, {_AREA_MAX:.0f}].",
        ),
    )

    plot = values.get("plot_area_sqm")
    runner.check(
        applicable=plot is not None,
        ok=plot is not None and 0 < float(plot) < _PLOT_MAX,
        finding=lambda: _finding(
            "range.plot_area_sqm",
            "plot_area_sqm",
            ErrorSeverity.WARNING,
            f"Plot area {plot} m² is implausible.",
        ),
    )

    rooms = values.get("rooms")
    runner.check(
        applicable=rooms is not None,
        ok=rooms is not None and 0 < float(rooms) <= _ROOMS_MAX,
        finding=lambda: _finding(
            "range.rooms",
            "rooms",
            ErrorSeverity.WARNING,
            f"Room count {rooms} is implausible.",
        ),
    )

    year = values.get("year_built")
    runner.check(
        applicable=year is not None,
        ok=year is not None and _YEAR_MIN <= int(year) <= date.today().year,
        finding=lambda: _finding(
            "range.year_built",
            "year_built",
            ErrorSeverity.WARNING,
            f"Construction year {year} is outside [{_YEAR_MIN}, {date.today().year}].",
        ),
    )

    demand = values.get("energy_demand_kwh")
    runner.check(
        applicable=demand is not None,
        ok=demand is not None and 0 <= float(demand) <= _DEMAND_MAX,
        finding=lambda: _finding(
            "range.energy_demand_kwh",
            "energy_demand_kwh",
            ErrorSeverity.WARNING,
            f"Energy demand {demand} kWh/(m²·a) is implausible.",
        ),
    )

    postal = values.get("postal_code")
    runner.check(
        applicable=postal is not None,
        ok=postal is not None and bool(_POSTAL_RE.match(str(postal))),
        finding=lambda: _finding(
            "format.postal_code",
            "postal_code",
            ErrorSeverity.WARNING,
            f"Postal code {postal!r} is not a valid 5-digit German code.",
        ),
    )


def _check_cross_field(runner: _RuleRunner, values: dict[str, Any]) -> None:
    """Apply checks that depend on more than one field."""
    floor = values.get("floor")
    total = values.get("total_floors")
    runner.check(
        applicable=floor is not None and total is not None,
        ok=floor is None or total is None or int(floor) <= int(total),
        finding=lambda: _finding(
            "cross.floor_le_total",
            "floor",
            ErrorSeverity.WARNING,
            f"Floor {floor} exceeds the building's total floors {total}.",
        ),
    )

    price = values.get("price_eur")
    area = values.get("living_area_sqm")
    listing = values.get("listing_type")
    if price is not None and area is not None and listing is not None and float(area) > 0:
        per_sqm = float(price) / float(area)
        low, high = _SALE_PPSQM if listing is ListingType.SALE else _RENT_PPSQM
        listing_label = listing.value if isinstance(listing, ListingType) else str(listing)
        runner.check(
            applicable=True,
            ok=low <= per_sqm <= high,
            finding=lambda: _finding(
                "plausibility.price_per_sqm",
                "price_eur",
                ErrorSeverity.WARNING,
                f"Price per m² ({per_sqm:.0f} €) is implausible for a {listing_label} listing.",
            ),
        )
