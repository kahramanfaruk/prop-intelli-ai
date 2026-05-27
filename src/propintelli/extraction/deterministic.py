"""Layer A: deterministic, layout-agnostic extraction.

A regex/heuristic extractor that is always run, regardless of whether an LLM
backend is configured. It is fast, free, fully offline, and deterministic, which
makes it both a reliable baseline and the safety net the pipeline downgrades to
when the LLM layer is unavailable.

The extractor is intentionally *layout-agnostic*: it anchors on German labels and
units (``Kaufpreis``, ``m²``, ``Baujahr``, a 5-digit postal code, …) and tolerates
the label and value appearing inline or on separate lines, which is exactly how
the same field is rendered across the tabular, prose, and sectioned layouts.
"""

from __future__ import annotations

import re

from propintelli.extraction.vocabulary import (
    CONDITION_PATTERNS,
    ENERGY_CERTIFICATE_PATTERNS,
    FEATURE_PATTERNS,
    HEATING_PATTERNS,
    NEGATION_PATTERN,
    NON_PRICE_LABEL_PATTERN,
    PRICE_LABEL_PATTERNS,
)
from propintelli.schemas.enums import ListingType, PriceKind, Provenance
from propintelli.schemas.extraction import FieldValue

_FLAGS = re.IGNORECASE
# A German monetary/decimal amount: thousands separated by dots, decimals by comma.
_AMOUNT = r"(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)"
_AREA_UNIT = r"(?:m²|m2|qm)"
# Maximum characters allowed between a price label and its amount. Kept tight so
# the extractor cannot skip across an "auf Anfrage" gap onto an unrelated number.
_PRICE_GAP = 30
_NON_PRICE = re.compile(NON_PRICE_LABEL_PATTERN, _FLAGS)
_NEGATION = re.compile(NEGATION_PATTERN, _FLAGS)
# Clause separators that bound the window scanned for a feature's negation cue.
_CLAUSE_SEPARATORS = (".", ";", ":", "\n", ",", "•", "·", "|")
_STREET_SUFFIX_SUBSTRINGS = (
    "straße",
    "strasse",
    "str.",
    "weg",
    "allee",
    "platz",
    "gasse",
    "chaussee",
    "ring",
    "damm",
    "ufer",
    "steig",
    "berg",
)


def _snippet(text: str, start: int, end: int, pad: int = 25) -> str:
    """Return a whitespace-normalised context window around a match."""
    window = text[max(0, start - pad) : min(len(text), end + pad)]
    return re.sub(r"\s+", " ", window).strip()


def _value_span(text: str, start: int, end: int, raw: str, confidence: float) -> FieldValue:
    """Build a deterministic :class:`FieldValue` from an absolute text span."""
    return FieldValue(
        raw_value=raw,
        confidence=confidence,
        provenance=Provenance.DETERMINISTIC,
        source_snippet=_snippet(text, start, end),
    )


def _value(text: str, match: re.Match[str], raw: str, confidence: float) -> FieldValue:
    """Build a deterministic :class:`FieldValue` from a regex match."""
    return _value_span(text, match.start(), match.end(), raw, confidence)


def _extract_price(text: str, fields: dict[str, FieldValue]) -> None:
    """Extract price, price kind, and the implied listing type.

    Price labels are tried in priority order. For each, the nearest following
    amount is read within a tight gap and rejected if an ancillary-cost label
    (Hausgeld, Nebenkosten, Provision, …) governs it, so an exposé listing
    several monetary amounts does not yield the wrong one as the headline price.
    """
    amount_re = re.compile(rf"[^0-9]{{0,{_PRICE_GAP}}}({_AMOUNT})\s*(€|eur|euro)?", _FLAGS)
    for label_pattern, kind in PRICE_LABEL_PATTERNS:
        for label in re.finditer(label_pattern, text, _FLAGS):
            tail = text[label.end() :]
            amount = amount_re.match(tail)
            if amount is None:
                continue
            if _NON_PRICE.search(tail[: amount.start(1)]):
                continue  # the amount belongs to an ancillary cost, not the price
            confidence = 0.92 if amount.group(2) else 0.82
            listing = ListingType.SALE if kind is PriceKind.PURCHASE else ListingType.RENT
            end = label.end() + amount.end(1)
            fields["price_eur"] = _value_span(text, label.start(), end, amount.group(1), confidence)
            fields["price_kind"] = _value_span(text, label.start(), end, kind.value, confidence)
            fields["listing_type"] = _value_span(
                text, label.start(), end, listing.value, confidence - 0.05
            )
            return


def _extract_area(text: str, fields: dict[str, FieldValue]) -> None:
    """Extract living area and (when present) plot area."""
    living = re.search(
        rf"(?:wohnfl[äa]che|wohnfl\.?|wfl\.?|wohnraum)[^0-9]{{0,30}}({_AMOUNT})\s*{_AREA_UNIT}",
        text,
        _FLAGS,
    )
    if living:
        fields["living_area_sqm"] = _value(text, living, living.group(1), 0.9)
    plot = re.search(
        rf"(?:grundst[üu]cksfl[äa]che|grundst[üu]ck)[^0-9]{{0,30}}({_AMOUNT})\s*{_AREA_UNIT}",
        text,
        _FLAGS,
    )
    if plot:
        fields["plot_area_sqm"] = _value(text, plot, plot.group(1), 0.85)


def _extract_rooms(text: str, fields: dict[str, FieldValue]) -> None:
    """Extract the number of rooms (``Zimmer``)."""
    match = re.search(rf"({_AMOUNT})\s*[-\s]?zimmer", text, _FLAGS) or re.search(
        rf"zimmer(?:zahl)?[^0-9]{{0,15}}({_AMOUNT})", text, _FLAGS
    )
    if match:
        fields["rooms"] = _value(text, match, match.group(1), 0.85)


def _extract_year(text: str, fields: dict[str, FieldValue]) -> None:
    """Extract the construction year (``Baujahr``)."""
    match = re.search(
        r"(?:baujahr|bj\.?|errichtet(?:\s*im(?:\s*jahr)?)?)[^0-9]{0,15}(\d{4})",
        text,
        _FLAGS,
    )
    if match:
        fields["year_built"] = _value(text, match, match.group(1), 0.9)


def _extract_floor(text: str, fields: dict[str, FieldValue]) -> None:
    """Extract floor and total floors (``Etage: 2 von 4``)."""
    match = re.search(r"(?:etage|geschoss)[^0-9]{0,15}(\d{1,2})\s*von\s*(\d{1,2})", text, _FLAGS)
    if match:
        fields["floor"] = _value(text, match, match.group(1), 0.85)
        fields["total_floors"] = _value(text, match, match.group(2), 0.85)
        return
    single = re.search(r"(\d{1,2})\.\s*(?:og|obergeschoss)", text, _FLAGS)
    if single:
        fields["floor"] = _value(text, single, single.group(1), 0.75)


def _extract_availability(text: str, fields: dict[str, FieldValue]) -> None:
    """Extract the availability date (``Bezugsfrei ab dd.mm.yyyy``)."""
    match = re.search(
        r"(?:bezugsfrei|bezugsfertig|verf[üu]gbar|frei|bezug)\s*(?:ab)?[^0-9]{0,15}"
        r"(\d{1,2}\.\d{1,2}\.\d{4})",
        text,
        _FLAGS,
    )
    if match:
        fields["availability_date"] = _value(text, match, match.group(1), 0.85)


def _extract_location(text: str, fields: dict[str, FieldValue]) -> None:
    """Extract postal code, city, street, house number, and district."""
    # Postal code + city: the city is a run of capitalised tokens after the code.
    # Periods are excluded from the token so a sentence boundary ("90408 Nürnberg.
    # Bezugsfrei …") does not absorb the next sentence's leading word.
    place = re.search(
        r"\b(\d{5})[ ]+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+(?:[ ][A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+)*)",
        text,
    )
    if place:
        fields["postal_code"] = _value(text, place, place.group(1), 0.9)
        fields["city"] = _value(text, place, place.group(2).strip(), 0.88)

    # Street + house number: iterate over every "Title-case run + number" candidate
    # and accept the first one that actually contains a street suffix, so unrelated
    # matches such as "Baujahr 1910" are skipped.
    address_pattern = re.compile(
        r"([A-ZÄÖÜ][\wäöüÄÖÜß.\-]+(?:[ ][A-ZÄÖÜ][\wäöüÄÖÜß.\-]+){0,3})[ ]+(\d{1,4}[a-zA-Z]?)"
        r"(?=[\s,.]|$)"
    )
    for address in address_pattern.finditer(text):
        if any(token in address.group(1).lower() for token in _STREET_SUFFIX_SUBSTRINGS):
            # Drop any preceding sentence fragment captured before a ". " boundary
            # ("Lage: Wiehre. Sundgauallee" -> "Sundgauallee").
            street = address.group(1).strip().rsplit(". ", 1)[-1].strip()
            fields["street"] = _value(text, address, street, 0.8)
            fields["house_number"] = _value(text, address, address.group(2).strip(), 0.8)
            break

    # A parenthetical district is only trusted when it follows a postal code (an
    # address context); otherwise "Verhandlungsbasis (Kaufpreis)" would be read as
    # a district. The "Stadtteil <name>" phrasing is an independent fallback.
    district = re.search(
        r"\b\d{5}\b[^\n(]{0,40}?\(([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]{2,})\)", text
    ) or re.search(r"stadtteil\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+)", text, _FLAGS)
    if district:
        fields["district"] = _value(text, district, district.group(1).strip(), 0.6)


def _extract_energy(text: str, fields: dict[str, FieldValue]) -> None:
    """Extract energy class, demand, certificate type, and heating system."""
    energy_class = re.search(
        r"energie(?:effizienz|ausweis)?klasse[^A-Za-z0-9]{0,10}(A\+|[A-H])(?![A-Za-z])",
        text,
        _FLAGS,
    )
    if energy_class:
        fields["energy_class"] = _value(text, energy_class, energy_class.group(1).upper(), 0.9)

    demand = re.search(
        rf"(?:end)?energie(?:bedarf|verbrauch|kennwert)[^0-9]{{0,20}}({_AMOUNT})\s*kwh",
        text,
        _FLAGS,
    )
    if demand:
        fields["energy_demand_kwh"] = _value(text, demand, demand.group(1), 0.85)

    for pattern, label in ENERGY_CERTIFICATE_PATTERNS:
        cert = re.search(pattern, text, _FLAGS)
        if cert:
            fields["energy_certificate_type"] = _value(text, cert, label, 0.85)
            break

    for pattern, heating in HEATING_PATTERNS:
        match = re.search(pattern, text, _FLAGS)
        if match:
            fields["heating_type"] = _value(text, match, heating.value, 0.85)
            break


def _extract_condition(text: str, fields: dict[str, FieldValue]) -> None:
    """Extract the building condition (``Objektzustand``)."""
    for pattern, condition in CONDITION_PATTERNS:
        match = re.search(pattern, text, _FLAGS)
        if match:
            fields["condition"] = _value(text, match, condition.value, 0.8)
            return


def _is_negated(text: str, match_start: int) -> bool:
    """Whether a negation cue precedes a feature mention within its clause.

    The window scanned runs from the nearest preceding clause separator up to
    the mention, so "kein Balkon" negates *balcony* while a later, separate
    "Keller vorhanden" in the same sentence is unaffected.

    Parameters
    ----------
    text : str
        The full document text.
    match_start : int
        Start offset of the feature keyword match.

    Returns
    -------
    bool
        ``True`` if a negation token governs the mention.
    """
    clause_start = max(
        (text.rfind(separator, 0, match_start) + 1 for separator in _CLAUSE_SEPARATORS),
        default=0,
    )
    return _NEGATION.search(text[clause_start:match_start]) is not None


def _extract_features(text: str, fields: dict[str, FieldValue]) -> None:
    """Extract tri-state equipment features, honouring negated mentions.

    Each feature keyword may appear several times; a single non-negated mention
    is decisive evidence of presence (``true``), so "kein Stellplatz, aber
    Tiefgarage" still yields parking. Only when every mention is negated is the
    feature recorded as an explicit absence (``false``); with no mention at all
    the feature is left unset (not stated).
    """
    for name, pattern in FEATURE_PATTERNS.items():
        positive: re.Match[str] | None = None
        negated: re.Match[str] | None = None
        for match in re.finditer(pattern, text, _FLAGS):
            if _is_negated(text, match.start()):
                negated = negated or match
            else:
                positive = match
                break
        if positive is not None:
            fields[name] = _value(text, positive, "true", 0.9)
        elif negated is not None:
            fields[name] = _value(text, negated, "false", 0.85)


_STOP_LINE = re.compile(
    r"^(angebotsart|objektdaten|lage|preis|adresse|ausstattung|energie|wir bieten|"
    r"der |die |das |zum |zur )",
    re.IGNORECASE,
)


def _extract_title(text: str, fields: dict[str, FieldValue]) -> None:
    """Extract the listing title from the leading lines of the document."""
    title_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if title_lines:
                break
            continue
        if _STOP_LINE.match(line):
            break
        title_lines.append(line)
        if len(title_lines) >= 2:
            break
    if title_lines:
        title = " ".join(title_lines)
        fields["title"] = FieldValue(
            raw_value=title, confidence=0.7, provenance=Provenance.DETERMINISTIC
        )


# Field extractors run in a fixed order; each writes into the shared mapping.
_EXTRACTORS = (
    _extract_title,
    _extract_price,
    _extract_area,
    _extract_rooms,
    _extract_year,
    _extract_floor,
    _extract_availability,
    _extract_location,
    _extract_energy,
    _extract_condition,
    _extract_features,
)


def extract_deterministic(text: str) -> dict[str, FieldValue]:
    """Extract canonical fields from document text using regex/heuristics.

    Parameters
    ----------
    text : str
        The document text produced by preprocessing.

    Returns
    -------
    dict of str to FieldValue
        Extracted values keyed by canonical field name. Only fields for which
        positive evidence was found are present; the values carry
        :class:`~propintelli.schemas.enums.Provenance.DETERMINISTIC` provenance.
    """
    fields: dict[str, FieldValue] = {}
    for extractor in _EXTRACTORS:
        extractor(text, fields)
    return fields
