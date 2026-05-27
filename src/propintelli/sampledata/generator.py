"""Synthetic Immobilien-Exposé generator with ground-truth labels.

The generator starts from canonical, normalised property records and *renders*
them into realistic German exposé PDFs across three distinct layouts (tabular,
prose, sectioned). Because generation is the inverse of extraction, each PDF is
emitted together with a ground-truth JSON label holding the canonical values,
which directly powers the evaluation harness without any manual annotation.

The data is hard-coded (not randomised) so the committed ``sample_data`` and the
evaluation results are reproducible in CI. Layout and wording vary deliberately
to exercise the pipeline's tolerance to document variance, including listings
with missing fields and a rental (vs. sale) listing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from propintelli.schemas.enums import (
    EnergyClass,
    HeatingType,
    ListingType,
    PriceKind,
    PropertyCondition,
)

# Reproducible output: reportlab otherwise embeds a creation timestamp and a
# random document ID, which would make the committed PDFs differ on every run.
# Invariant mode fixes both so the corpus is byte-stable across regenerations.
rl_config.invariant = 1

# --- German rendering vocabularies -----------------------------------------
_HEATING_DE: dict[HeatingType, str] = {
    HeatingType.GAS: "Gas-Zentralheizung",
    HeatingType.OIL: "Ölheizung",
    HeatingType.DISTRICT_HEATING: "Fernwärme",
    HeatingType.HEAT_PUMP: "Wärmepumpe",
    HeatingType.ELECTRIC: "Elektroheizung",
    HeatingType.PELLET: "Pelletheizung",
    HeatingType.SOLAR: "Solarthermie",
    HeatingType.UNDERFLOOR: "Fußbodenheizung",
    HeatingType.OTHER: "Sonstige",
}
_CONDITION_DE: dict[PropertyCondition, str] = {
    PropertyCondition.NEW_BUILD: "Neubau",
    PropertyCondition.FIRST_OCCUPANCY: "Erstbezug",
    PropertyCondition.MODERNISED: "modernisiert",
    PropertyCondition.RENOVATED: "saniert",
    PropertyCondition.WELL_KEPT: "gepflegt",
    PropertyCondition.NEEDS_RENOVATION: "renovierungsbedürftig",
}
_FEATURE_DE: dict[str, str] = {
    "balcony": "Balkon",
    "terrace": "Terrasse",
    "garden": "Garten",
    "parking": "Stellplatz",
    "cellar": "Keller",
    "elevator": "Aufzug",
    "fitted_kitchen": "Einbauküche",
    "furnished": "möbliert",
    "barrier_free": "barrierefrei",
}
_PRICE_KIND_DE: dict[PriceKind, str] = {
    PriceKind.PURCHASE: "Kaufpreis",
    PriceKind.COLD_RENT: "Kaltmiete",
    PriceKind.WARM_RENT: "Warmmiete",
}


@dataclass(frozen=True, slots=True)
class SyntheticProperty:
    """A canonical property used to render one exposé and its ground truth.

    The attribute names mirror the canonical field registry so the ground-truth
    label can be produced mechanically.
    """

    document_stem: str
    layout: str
    title: str
    listing_type: ListingType
    price_kind: PriceKind
    price_eur: Decimal
    living_area_sqm: float
    street: str
    house_number: str
    postal_code: str
    city: str
    rooms: float | None = None
    plot_area_sqm: float | None = None
    floor: int | None = None
    total_floors: int | None = None
    year_built: int | None = None
    condition: PropertyCondition | None = None
    availability_date: date | None = None
    district: str | None = None
    energy_class: EnergyClass | None = None
    heating_type: HeatingType | None = None
    energy_demand_kwh: float | None = None
    energy_certificate_type: str | None = None
    features: dict[str, bool] = field(default_factory=dict)
    # Ancillary monetary lines (service charge, broker fee, deposit, …) rendered
    # alongside the price. They are *not* part of the ground truth: the extractor
    # must keep them out of the headline price. Each entry is a (label, value).
    ancillary_costs: tuple[tuple[str, str], ...] = ()


# --- Number / date formatting helpers ---------------------------------------
def _fmt_eur(value: Decimal) -> str:
    """Format an amount with German thousands separators (``450.000``)."""
    return f"{int(value):,}".replace(",", ".")


def _fmt_de_number(value: float) -> str:
    """Format a number with a German decimal comma; drop a trailing ``,0``."""
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}".replace(".", ",")


def _fmt_de_date(value: date) -> str:
    """Format a date as ``dd.mm.yyyy``."""
    return value.strftime("%d.%m.%Y")


def _present_features_de(prop: SyntheticProperty) -> list[str]:
    """Return German labels for the features that are present (``True``)."""
    return [_FEATURE_DE[name] for name, present in prop.features.items() if present]


def _absent_sentence(prop: SyntheticProperty) -> str | None:
    """Render explicitly-absent features as a negated German sentence.

    Returns ``None`` when no feature is marked absent. Each item carries its own
    "ohne" so a comma-separated list negates every feature individually, matching
    how the deterministic extractor scopes negation to a clause.
    """
    absent = [f"ohne {_FEATURE_DE[name]}" for name, present in prop.features.items() if not present]
    if not absent:
        return None
    sentence = ", ".join(absent)
    return sentence[0].upper() + sentence[1:] + "."


# --- Ground truth ------------------------------------------------------------
def ground_truth(prop: SyntheticProperty) -> dict[str, Any]:
    """Build the canonical ground-truth label for a property.

    Only fields the document actually states are included; everything else is
    expected to be absent (``None``) in the extracted record. This rewards the
    pipeline for *not* hallucinating unstated fields.

    Parameters
    ----------
    prop : SyntheticProperty
        The source property.

    Returns
    -------
    dict
        Mapping of canonical field name to normalised value (JSON-serialisable).
    """
    fields: dict[str, Any] = {
        "title": prop.title,
        "listing_type": prop.listing_type.value,
        "price_kind": prop.price_kind.value,
        "price_eur": float(prop.price_eur),
        "living_area_sqm": prop.living_area_sqm,
        "postal_code": prop.postal_code,
        "city": prop.city,
        "street": prop.street,
        "house_number": prop.house_number,
    }
    optional: dict[str, Any] = {
        "rooms": prop.rooms,
        "plot_area_sqm": prop.plot_area_sqm,
        "floor": prop.floor,
        "total_floors": prop.total_floors,
        "year_built": prop.year_built,
        "condition": prop.condition.value if prop.condition else None,
        "availability_date": prop.availability_date.isoformat() if prop.availability_date else None,
        "district": prop.district,
        "energy_class": prop.energy_class.value if prop.energy_class else None,
        "heating_type": prop.heating_type.value if prop.heating_type else None,
        "energy_demand_kwh": prop.energy_demand_kwh,
        "energy_certificate_type": prop.energy_certificate_type,
    }
    fields.update({key: value for key, value in optional.items() if value is not None})
    # Every *stated* feature is ground truth, whether present (True) or
    # explicitly absent (False); only unstated features (omitted from the dict)
    # are expected to be None in the extracted record.
    fields.update(dict(prop.features))
    return {"document": f"{prop.document_stem}.pdf", "fields": fields}


# --- PDF rendering -----------------------------------------------------------
def _styles() -> dict[str, ParagraphStyle]:
    """Return the paragraph styles used by all layouts."""
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {
        "title": ParagraphStyle("ExposeTitle", parent=base["Title"], fontSize=18, spaceAfter=10),
        "h2": ParagraphStyle(
            "ExposeH2", parent=base["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4
        ),
        "body": ParagraphStyle("ExposeBody", parent=base["BodyText"], fontSize=10, leading=14),
    }
    return styles


def _data_rows(prop: SyntheticProperty) -> list[tuple[str, str]]:
    """Build labelled key/value rows shared by the tabular and sectioned layouts."""
    rows: list[tuple[str, str]] = [
        (_PRICE_KIND_DE[prop.price_kind], f"{_fmt_eur(prop.price_eur)} €"),
        ("Wohnfläche", f"ca. {_fmt_de_number(prop.living_area_sqm)} m²"),
    ]
    if prop.rooms is not None:
        rows.append(("Zimmer", _fmt_de_number(prop.rooms)))
    if prop.plot_area_sqm is not None:
        rows.append(("Grundstück", f"{_fmt_de_number(prop.plot_area_sqm)} m²"))
    if prop.floor is not None and prop.total_floors is not None:
        rows.append(("Etage", f"{prop.floor} von {prop.total_floors}"))
    if prop.year_built is not None:
        rows.append(("Baujahr", str(prop.year_built)))
    if prop.condition is not None:
        rows.append(("Zustand", _CONDITION_DE[prop.condition]))
    if prop.availability_date is not None:
        rows.append(("Bezugsfrei ab", _fmt_de_date(prop.availability_date)))
    address = f"{prop.street} {prop.house_number}, {prop.postal_code} {prop.city}"
    if prop.district:
        address += f" ({prop.district})"
    rows.append(("Adresse", address))
    if prop.energy_class is not None:
        rows.append(("Energieeffizienzklasse", prop.energy_class.value))
    if prop.heating_type is not None:
        rows.append(("Heizung", _HEATING_DE[prop.heating_type]))
    if prop.energy_demand_kwh is not None:
        rows.append(("Energiebedarf", f"{_fmt_de_number(prop.energy_demand_kwh)} kWh/(m²·a)"))
    if prop.energy_certificate_type is not None:
        rows.append(("Energieausweis", prop.energy_certificate_type))
    return rows


def _flowables_tabular(prop: SyntheticProperty, styles: dict[str, ParagraphStyle]) -> list[Any]:
    """Render a tabular "Objektdaten" layout."""
    flow: list[Any] = [Paragraph(prop.title, styles["title"])]
    flow.append(
        Paragraph(
            f"Angebotsart: {'Kauf' if prop.listing_type is ListingType.SALE else 'Miete'}",
            styles["body"],
        )
    )
    flow.append(Spacer(1, 6))
    flow.append(Paragraph("Objektdaten", styles["h2"]))
    rows = [*_data_rows(prop), *prop.ancillary_costs]
    table = Table([[k, v] for k, v in rows], colWidths=[55 * mm, 110 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#444444")),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
            ]
        )
    )
    flow.append(table)
    features = _present_features_de(prop)
    if features:
        flow.append(Paragraph("Ausstattung", styles["h2"]))
        flow.append(
            ListFlowable(
                [ListItem(Paragraph(label, styles["body"])) for label in features],
                bulletType="bullet",
            )
        )
    absent = _absent_sentence(prop)
    if absent:
        flow.append(Paragraph(absent, styles["body"]))
    return flow


def _flowables_prose(prop: SyntheticProperty, styles: dict[str, ParagraphStyle]) -> list[Any]:
    """Render a free-text prose layout that embeds values in sentences."""
    flow: list[Any] = [Paragraph(prop.title, styles["title"])]
    intro = (
        f"Wir bieten Ihnen diese Immobilie in {prop.postal_code} {prop.city}"
        f"{f' ({prop.district})' if prop.district else ''} "
        f"{'zum Kauf' if prop.listing_type is ListingType.SALE else 'zur Miete'} an. "
        f"Die Wohnfläche beträgt ca. {_fmt_de_number(prop.living_area_sqm)} m²"
    )
    if prop.rooms is not None:
        intro += f" und verteilt sich auf {_fmt_de_number(prop.rooms)} Zimmer"
    intro += "."
    flow.append(Paragraph(intro, styles["body"]))
    flow.append(Spacer(1, 6))

    price_sentence = (
        f"Der {_PRICE_KIND_DE[prop.price_kind]} liegt bei {_fmt_eur(prop.price_eur)} EUR."
    )
    flow.append(Paragraph(price_sentence, styles["body"]))
    if prop.ancillary_costs:
        extras = ", ".join(f"{label} {value}" for label, value in prop.ancillary_costs)
        flow.append(Paragraph(f"Hinzu kommen: {extras}.", styles["body"]))

    detail_bits: list[str] = []
    if prop.year_built is not None:
        detail_bits.append(f"Das Objekt wurde im Baujahr {prop.year_built} errichtet")
    if prop.condition is not None:
        detail_bits.append(f"und präsentiert sich {_CONDITION_DE[prop.condition]}")
    if detail_bits:
        flow.append(Paragraph(" ".join(detail_bits) + ".", styles["body"]))

    if prop.floor is not None and prop.total_floors is not None:
        flow.append(
            Paragraph(
                f"Die Einheit liegt auf Etage {prop.floor} von {prop.total_floors}.", styles["body"]
            )
        )
    if prop.plot_area_sqm is not None:
        flow.append(
            Paragraph(
                f"Die Grundstücksfläche beträgt {_fmt_de_number(prop.plot_area_sqm)} m².",
                styles["body"],
            )
        )

    features = _present_features_de(prop)
    if features:
        flow.append(
            Paragraph("Zur Ausstattung zählen: " + ", ".join(features) + ".", styles["body"])
        )
    absent = _absent_sentence(prop)
    if absent:
        flow.append(Paragraph(absent, styles["body"]))
    if prop.availability_date is not None:
        flow.append(
            Paragraph(f"Bezugsfrei ab {_fmt_de_date(prop.availability_date)}.", styles["body"])
        )

    energy_bits: list[str] = []
    if prop.energy_class is not None:
        energy_bits.append(f"Energieeffizienzklasse {prop.energy_class.value}")
    if prop.heating_type is not None:
        energy_bits.append(f"Heizungsart: {_HEATING_DE[prop.heating_type]}")
    if prop.energy_demand_kwh is not None:
        energy_bits.append(f"Energiebedarf {_fmt_de_number(prop.energy_demand_kwh)} kWh/(m²·a)")
    if prop.energy_certificate_type is not None:
        energy_bits.append(f"Energieausweis: {prop.energy_certificate_type}")
    if energy_bits:
        flow.append(Spacer(1, 6))
        flow.append(Paragraph(" · ".join(energy_bits) + ".", styles["body"]))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(f"Adresse: {prop.street} {prop.house_number}.", styles["body"]))
    return flow


def _flowables_sectioned(prop: SyntheticProperty, styles: dict[str, ParagraphStyle]) -> list[Any]:
    """Render a sectioned layout with headed groups of lines."""
    flow: list[Any] = [Paragraph(prop.title, styles["title"])]
    flow.append(Paragraph("Lage", styles["h2"]))
    address = f"{prop.street} {prop.house_number}, {prop.postal_code} {prop.city}"
    if prop.district:
        address += f", Stadtteil {prop.district}"
    flow.append(Paragraph(address, styles["body"]))

    flow.append(Paragraph("Preis & Flächen", styles["h2"]))
    for label, value in _data_rows(prop):
        if label in {
            "Adresse",
            "Energieeffizienzklasse",
            "Heizung",
            "Energiebedarf",
            "Energieausweis",
        }:
            continue
        flow.append(Paragraph(f"{label}: {value}", styles["body"]))
    for label, value in prop.ancillary_costs:
        flow.append(Paragraph(f"{label}: {value}", styles["body"]))

    features = _present_features_de(prop)
    if features:
        flow.append(Paragraph("Ausstattung", styles["h2"]))
        flow.append(Paragraph(", ".join(features), styles["body"]))
    absent = _absent_sentence(prop)
    if absent:
        flow.append(Paragraph(absent, styles["body"]))

    flow.append(Paragraph("Energie", styles["h2"]))
    if prop.energy_class is not None:
        flow.append(Paragraph(f"Energieklasse: {prop.energy_class.value}", styles["body"]))
    if prop.heating_type is not None:
        flow.append(Paragraph(f"Heizung: {_HEATING_DE[prop.heating_type]}", styles["body"]))
    if prop.energy_demand_kwh is not None:
        flow.append(
            Paragraph(
                f"Endenergiebedarf: {_fmt_de_number(prop.energy_demand_kwh)} kWh/(m²·a)",
                styles["body"],
            )
        )
    if prop.energy_certificate_type is not None:
        flow.append(Paragraph(f"Energieausweistyp: {prop.energy_certificate_type}", styles["body"]))
    return flow


_LAYOUTS = {
    "tabular": _flowables_tabular,
    "prose": _flowables_prose,
    "sectioned": _flowables_sectioned,
}


def render_pdf(prop: SyntheticProperty, output_path: Path) -> None:
    """Render a single property to a PDF file.

    Parameters
    ----------
    prop : SyntheticProperty
        The property to render.
    output_path : Path
        Destination PDF path; parent directories are created if needed.

    Raises
    ------
    ValueError
        If the property declares an unknown layout.
    """
    builder = _LAYOUTS.get(prop.layout)
    if builder is None:
        msg = f"Unknown layout {prop.layout!r}; expected one of {sorted(_LAYOUTS)}"
        raise ValueError(msg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        title=prop.title,
        author="PropIntelli AI: synthetic sample",
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )
    doc.build(builder(prop, _styles()))


def generate_samples(raw_dir: Path, ground_truth_dir: Path) -> list[Path]:
    """Generate every sample exposé PDF and its ground-truth label.

    Parameters
    ----------
    raw_dir : Path
        Directory to write the PDF files into.
    ground_truth_dir : Path
        Directory to write the ground-truth JSON labels into.

    Returns
    -------
    list of Path
        Paths of the generated PDF files, in registry order.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for prop in SAMPLE_PROPERTIES:
        pdf_path = raw_dir / f"{prop.document_stem}.pdf"
        render_pdf(prop, pdf_path)
        label_path = ground_truth_dir / f"{prop.document_stem}.json"
        label_path.write_text(
            json.dumps(ground_truth(prop), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        generated.append(pdf_path)
    return generated


# --- The fixed sample set ----------------------------------------------------
SAMPLE_PROPERTIES: list[SyntheticProperty] = [
    SyntheticProperty(
        document_stem="expose_01_nuernberg_eigentumswohnung",
        layout="tabular",
        title="Moderne 3-Zimmer-Eigentumswohnung mit Balkon in Nürnberg-Gärten",
        listing_type=ListingType.SALE,
        price_kind=PriceKind.PURCHASE,
        price_eur=Decimal("449000"),
        living_area_sqm=92.0,
        rooms=3.0,
        floor=2,
        total_floors=4,
        year_built=1998,
        condition=PropertyCondition.MODERNISED,
        availability_date=date(2026, 7, 1),
        street="Bucher Straße",
        house_number="42",
        postal_code="90408",
        city="Nürnberg",
        district="Nordstadt",
        energy_class=EnergyClass.C,
        heating_type=HeatingType.GAS,
        energy_demand_kwh=98.0,
        energy_certificate_type="Bedarfsausweis",
        features={"balcony": True, "cellar": True, "elevator": True, "fitted_kitchen": True},
    ),
    SyntheticProperty(
        document_stem="expose_02_muenchen_altbau",
        layout="prose",
        title="Charmante Altbauwohnung nahe Englischer Garten, München",
        listing_type=ListingType.SALE,
        price_kind=PriceKind.PURCHASE,
        price_eur=Decimal("1190000"),
        living_area_sqm=124.5,
        rooms=4.5,
        floor=3,
        total_floors=5,
        year_built=1910,
        condition=PropertyCondition.WELL_KEPT,
        street="Königinstraße",
        house_number="7",
        postal_code="80539",
        city="München",
        district="Schwabing",
        energy_class=EnergyClass.E,
        heating_type=HeatingType.DISTRICT_HEATING,
        energy_demand_kwh=156.0,
        features={"balcony": True, "cellar": True, "fitted_kitchen": True},
    ),
    SyntheticProperty(
        document_stem="expose_03_fuerth_reihenhaus_sparse",
        layout="sectioned",
        title="Reihenmittelhaus mit Garten in Fürth",
        listing_type=ListingType.SALE,
        price_kind=PriceKind.PURCHASE,
        price_eur=Decimal("520000"),
        living_area_sqm=138.0,
        rooms=5.0,
        plot_area_sqm=210.0,
        street="Schwabacher Straße",
        house_number="115",
        postal_code="90763",
        city="Fürth",
        features={"garden": True, "cellar": True, "parking": True},
    ),
    SyntheticProperty(
        document_stem="expose_04_berlin_neubau",
        layout="tabular",
        title="Erstbezug: 2-Zimmer-Neubauwohnung in Berlin-Mitte",
        listing_type=ListingType.SALE,
        price_kind=PriceKind.PURCHASE,
        price_eur=Decimal("675000"),
        living_area_sqm=68.0,
        rooms=2.0,
        floor=5,
        total_floors=7,
        year_built=2024,
        condition=PropertyCondition.FIRST_OCCUPANCY,
        availability_date=date(2026, 9, 1),
        street="Invalidenstraße",
        house_number="20",
        postal_code="10115",
        city="Berlin",
        district="Mitte",
        energy_class=EnergyClass.A_PLUS,
        heating_type=HeatingType.HEAT_PUMP,
        energy_demand_kwh=28.5,
        energy_certificate_type="Bedarfsausweis",
        features={
            "balcony": True,
            "elevator": True,
            "fitted_kitchen": True,
            "barrier_free": True,
            "parking": True,
        },
    ),
    SyntheticProperty(
        document_stem="expose_05_hamburg_villa",
        layout="prose",
        title="Freistehende Stadtvilla mit Garten in Hamburg-Blankenese",
        listing_type=ListingType.SALE,
        price_kind=PriceKind.PURCHASE,
        price_eur=Decimal("2450000"),
        living_area_sqm=265.0,
        rooms=7.0,
        plot_area_sqm=820.0,
        year_built=1985,
        condition=PropertyCondition.RENOVATED,
        street="Elbchaussee",
        house_number="350",
        postal_code="22609",
        city="Hamburg",
        district="Blankenese",
        energy_class=EnergyClass.D,
        heating_type=HeatingType.OIL,
        energy_demand_kwh=132.0,
        features={"garden": True, "terrace": True, "cellar": True, "parking": True},
    ),
    SyntheticProperty(
        document_stem="expose_06_leipzig_dachgeschoss",
        layout="sectioned",
        title="Sanierte Dachgeschosswohnung in Leipzig-Süd",
        listing_type=ListingType.SALE,
        price_kind=PriceKind.PURCHASE,
        price_eur=Decimal("389500"),
        living_area_sqm=104.0,
        rooms=3.5,
        floor=4,
        total_floors=4,
        year_built=1920,
        condition=PropertyCondition.RENOVATED,
        availability_date=date(2026, 6, 15),
        street="Karl-Liebknecht-Straße",
        house_number="88",
        postal_code="04275",
        city="Leipzig",
        district="Südvorstadt",
        energy_class=EnergyClass.B,
        heating_type=HeatingType.GAS,
        energy_demand_kwh=74.0,
        features={"balcony": True, "fitted_kitchen": True, "cellar": True},
    ),
    SyntheticProperty(
        document_stem="expose_07_nuernberg_mietwohnung",
        layout="tabular",
        title="Helle 2-Zimmer-Mietwohnung in Nürnberg-Gostenhof",
        listing_type=ListingType.RENT,
        price_kind=PriceKind.COLD_RENT,
        price_eur=Decimal("980"),
        living_area_sqm=58.5,
        rooms=2.0,
        floor=1,
        total_floors=3,
        year_built=1965,
        condition=PropertyCondition.WELL_KEPT,
        availability_date=date(2026, 8, 1),
        street="Fürther Straße",
        house_number="210",
        postal_code="90429",
        city="Nürnberg",
        district="Gostenhof",
        energy_class=EnergyClass.D,
        heating_type=HeatingType.GAS,
        energy_demand_kwh=118.0,
        features={"balcony": True, "cellar": True},
    ),
    SyntheticProperty(
        document_stem="expose_08_stuttgart_penthouse",
        layout="prose",
        title="Exklusives Penthouse mit Dachterrasse in Stuttgart-West",
        listing_type=ListingType.SALE,
        price_kind=PriceKind.PURCHASE,
        price_eur=Decimal("1340000"),
        living_area_sqm=176.0,
        rooms=4.0,
        floor=6,
        total_floors=6,
        year_built=2016,
        condition=PropertyCondition.MODERNISED,
        street="Rotebühlstraße",
        house_number="55",
        postal_code="70178",
        city="Stuttgart",
        district="Stuttgart-West",
        energy_class=EnergyClass.B,
        heating_type=HeatingType.UNDERFLOOR,
        energy_demand_kwh=62.0,
        energy_certificate_type="Verbrauchsausweis",
        features={
            "terrace": True,
            "elevator": True,
            "fitted_kitchen": True,
            "parking": True,
            "barrier_free": True,
        },
    ),
    SyntheticProperty(
        document_stem="expose_09_dresden_bungalow",
        layout="sectioned",
        title="Bungalow mit großem Grundstück in Dresden-Loschwitz",
        listing_type=ListingType.SALE,
        price_kind=PriceKind.PURCHASE,
        price_eur=Decimal("615000"),
        living_area_sqm=119.0,
        rooms=4.0,
        plot_area_sqm=640.0,
        year_built=1972,
        condition=PropertyCondition.NEEDS_RENOVATION,
        street="Pillnitzer Landstraße",
        house_number="9",
        postal_code="01326",
        city="Dresden",
        district="Loschwitz",
        energy_class=EnergyClass.F,
        heating_type=HeatingType.OIL,
        energy_demand_kwh=189.0,
        features={"garden": True, "cellar": True, "parking": True},
    ),
    SyntheticProperty(
        document_stem="expose_10_koeln_loft",
        layout="prose",
        title="Loft im umgebauten Speicher, Köln-Ehrenfeld",
        listing_type=ListingType.SALE,
        price_kind=PriceKind.PURCHASE,
        price_eur=Decimal("729000"),
        living_area_sqm=142.0,
        rooms=3.0,
        floor=2,
        total_floors=3,
        year_built=2009,
        condition=PropertyCondition.MODERNISED,
        availability_date=date(2026, 10, 1),
        street="Venloer Straße",
        house_number="389",
        postal_code="50825",
        city="Köln",
        district="Ehrenfeld",
        energy_class=EnergyClass.C,
        heating_type=HeatingType.HEAT_PUMP,
        energy_demand_kwh=88.0,
        features={"terrace": True, "elevator": True, "fitted_kitchen": True},
    ),
    # --- Variance & edge-case coverage --------------------------------------
    # Warm rent (not just sale/cold-rent), furnished, electric heating, energy
    # class G, an ancillary deposit (must not be read as the price), and an
    # explicitly-absent feature ("ohne Keller").
    SyntheticProperty(
        document_stem="expose_11_dresden_warmmiete",
        layout="prose",
        title="Möblierte 2,5-Zimmer-Wohnung zur Miete in Dresden-Neustadt",
        listing_type=ListingType.RENT,
        price_kind=PriceKind.WARM_RENT,
        price_eur=Decimal("1450"),
        living_area_sqm=70.0,
        rooms=2.5,
        floor=2,
        total_floors=5,
        year_built=1955,
        condition=PropertyCondition.WELL_KEPT,
        availability_date=date(2026, 9, 1),
        street="Alaunstraße",
        house_number="12",
        postal_code="01099",
        city="Dresden",
        district="Neustadt",
        energy_class=EnergyClass.G,
        heating_type=HeatingType.ELECTRIC,
        energy_demand_kwh=212.0,
        features={"furnished": True, "balcony": True, "cellar": False},
        ancillary_costs=(("Kaution", "4.350 €"),),
    ),
    # New build, pellet heating, energy class A, barrier-free, with service
    # charge and broker commission listed beside the price, plus "ohne Garten".
    SyntheticProperty(
        document_stem="expose_12_essen_neubau",
        layout="tabular",
        title="Neubau: barrierefreie 3-Zimmer-Wohnung in Essen-Rüttenscheid",
        listing_type=ListingType.SALE,
        price_kind=PriceKind.PURCHASE,
        price_eur=Decimal("580000"),
        living_area_sqm=95.0,
        rooms=3.0,
        floor=1,
        total_floors=4,
        year_built=2025,
        condition=PropertyCondition.NEW_BUILD,
        availability_date=date(2026, 12, 1),
        street="Rüttenscheider Straße",
        house_number="88",
        postal_code="45130",
        city="Essen",
        district="Rüttenscheid",
        energy_class=EnergyClass.A,
        heating_type=HeatingType.PELLET,
        energy_demand_kwh=35.0,
        energy_certificate_type="Bedarfsausweis",
        features={
            "barrier_free": True,
            "elevator": True,
            "fitted_kitchen": True,
            "parking": True,
            "garden": False,
        },
        ancillary_costs=(("Hausgeld", "320 €"), ("Provision", "3,57%")),
    ),
    # Old building needing renovation, solar heating, worst energy class H, a
    # service charge, and "ohne Balkon".
    SyntheticProperty(
        document_stem="expose_13_hannover_altbau",
        layout="sectioned",
        title="Sanierungsbedürftige Altbauwohnung in Hannover-List",
        listing_type=ListingType.SALE,
        price_kind=PriceKind.PURCHASE,
        price_eur=Decimal("410000"),
        living_area_sqm=88.0,
        rooms=3.0,
        floor=3,
        total_floors=4,
        year_built=1925,
        condition=PropertyCondition.NEEDS_RENOVATION,
        street="Podbielskistraße",
        house_number="25",
        postal_code="30161",
        city="Hannover",
        district="List",
        energy_class=EnergyClass.H,
        heating_type=HeatingType.SOLAR,
        energy_demand_kwh=265.0,
        features={"cellar": True, "parking": True, "balcony": False},
        ancillary_costs=(("Hausgeld", "280 €"),),
    ),
]
