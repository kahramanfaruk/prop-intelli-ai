"""Independently-authored holdout corpus for measuring generalization.

The synthetic generator (:mod:`propintelli.sampledata.generator`) renders
canonical records with the *same* German vocabulary the deterministic extractor
anchors on, so evaluating against it measures round-trip **consistency**, not
real-world accuracy. This module is the complement: a small set of exposés
written by hand to mimic the messiness of genuine listings —

* free-text prose rather than labelled key/value rows,
* abbreviations (``Bj.``, ``Wfl.``, ``EBK``, ``OG``),
* several monetary amounts (price plus Hausgeld / Nebenkosten / Provision / Kaution),
* negations, including post-posed ones ("einen Aufzug gibt es nicht"),
* alternative phrasings the generator never emits ("Verhandlungsbasis", bare
  "Klasse E", "Lage: <Stadtteil>", compound "Gasetagenheizung").

The labels are the values a human reader would record, *independent* of what the
extractor happens to produce. Evaluating against this corpus therefore measures
**generalization** to unseen wording. The text is authored rather than scraped:
genuine portal PDFs cannot be redistributed for licensing reasons and the build
assumes no network — but the documents are deliberately independent of the
extractor's templates, which is what a generalization metric requires.

These documents intentionally include fields the deterministic baseline is
expected to miss (compound heating words, bare energy-class labels, post-posed
negations); the resulting sub-100% score is the honest signal that the synthetic
ceiling does not transfer unchanged to real input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from propintelli.sampledata.generator import _styles  # shared paragraph styles


@dataclass(frozen=True, slots=True)
class HoldoutDocument:
    """A hand-authored exposé and its hand-labelled ground truth.

    Attributes
    ----------
    stem : str
        File stem for the rendered PDF and its label.
    body : str
        The exposé text; the first non-empty line is rendered as the title and
        the remaining lines as body paragraphs.
    fields : dict
        Ground-truth canonical field values a human reader would record. Only
        stated fields are present; unstated fields are omitted (expected
        ``None``), matching the synthetic-label convention.
    """

    stem: str
    body: str
    fields: dict[str, Any]


HOLDOUT_DOCUMENTS: list[HoldoutDocument] = [
    HoldoutDocument(
        stem="holdout_01_karlsruhe_reihenhaus",
        body=(
            "Charmantes Reihenmittelhaus in Karlsruhe-Durlach\n"
            "Wir verkaufen ein gepflegtes Reihenhaus (Bj. 1981) in ruhiger Lage.\n"
            "Wohnfläche ca. 138 m², Grundstücksfläche rund 220 m², 5 Zimmer auf drei Ebenen.\n"
            "Verhandlungsbasis (Kaufpreis): 489.000 EUR. Hausgeld entfällt. "
            "Provision: 3,57 % inkl. MwSt.\n"
            "Eine Einbauküche ist vorhanden, ebenso ein Keller. Einen Aufzug gibt es nicht.\n"
            "Energieausweis: Verbrauchsausweis, Energieverbrauch 142 kWh, Klasse E. "
            "Heizung: Gasetagenheizung.\n"
            "Adresse: Pfinztalstraße 30, 76227 Karlsruhe. Bezugsfrei ab sofort."
        ),
        fields={
            "title": "Charmantes Reihenmittelhaus in Karlsruhe-Durlach",
            "listing_type": "sale",
            "price_kind": "purchase",
            "price_eur": 489000.0,
            "living_area_sqm": 138.0,
            "plot_area_sqm": 220.0,
            "rooms": 5.0,
            "year_built": 1981,
            "condition": "well_kept",
            "street": "Pfinztalstraße",
            "house_number": "30",
            "postal_code": "76227",
            "city": "Karlsruhe",
            "energy_class": "E",
            "heating_type": "gas",
            "energy_demand_kwh": 142.0,
            "energy_certificate_type": "Verbrauchsausweis",
            "fitted_kitchen": True,
            "cellar": True,
            "elevator": False,
        },
    ),
    HoldoutDocument(
        stem="holdout_02_freiburg_etagenwohnung",
        body=(
            "Helle 3-Zimmer-Wohnung zur Miete - Freiburg im Breisgau\n"
            "Objekt: Etagenwohnung, 2. OG (von 4).\n"
            "Kaltmiete: 1.180 € · Nebenkosten: 230 € · Kaution: 3 Monatsmieten.\n"
            "Wohnfläche: 82,5 m² · Zimmer: 3 · Baujahr: 2004.\n"
            "Ausstattung: Balkon, Einbauküche, Tiefgarage. Kein Keller.\n"
            "Energieeffizienzklasse B, Bedarfsausweis, 68 kWh/(m²a). Fernwärme.\n"
            "Lage: Wiehre. Sundgauallee 14, 79110 Freiburg. Frei ab 01.10.2026."
        ),
        fields={
            "title": "Helle 3-Zimmer-Wohnung zur Miete - Freiburg im Breisgau",
            "listing_type": "rent",
            "price_kind": "cold_rent",
            "price_eur": 1180.0,
            "living_area_sqm": 82.5,
            "rooms": 3.0,
            "floor": 2,
            "total_floors": 4,
            "year_built": 2004,
            "street": "Sundgauallee",
            "house_number": "14",
            "postal_code": "79110",
            "city": "Freiburg",
            "district": "Wiehre",
            "availability_date": "2026-10-01",
            "energy_class": "B",
            "heating_type": "district_heating",
            "energy_demand_kwh": 68.0,
            "energy_certificate_type": "Bedarfsausweis",
            "balcony": True,
            "fitted_kitchen": True,
            "parking": True,
            "cellar": False,
        },
    ),
    HoldoutDocument(
        stem="holdout_03_potsdam_villa",
        body=(
            "Exklusive Villa in Potsdam - Erstbezug nach Sanierung\n"
            "Diese vollständig sanierte Stadtvilla (Baujahr 1908, kernsaniert 2023) bietet auf\n"
            "zwei Etagen großzügigen Wohnraum. Wohnfläche: 240 m². Grundstück: 900 m². 8 Zimmer.\n"
            "Kaufpreis: 1.950.000 €. Käuferprovision: 5,95 %.\n"
            "Zur Ausstattung gehören Terrasse, Garten, Weinkeller und Fußbodenheizung.\n"
            "Wärmepumpe, Energiebedarf 45 kWh/(m²·a), Energieeffizienzklasse A+.\n"
            "Standort: Berliner Straße 100, 14467 Potsdam (Innenstadt). Frei nach Vereinbarung."
        ),
        fields={
            "title": "Exklusive Villa in Potsdam - Erstbezug nach Sanierung",
            "listing_type": "sale",
            "price_kind": "purchase",
            "price_eur": 1950000.0,
            "living_area_sqm": 240.0,
            "plot_area_sqm": 900.0,
            "rooms": 8.0,
            "year_built": 1908,
            "condition": "first_occupancy",
            "street": "Berliner Straße",
            "house_number": "100",
            "postal_code": "14467",
            "city": "Potsdam",
            "district": "Innenstadt",
            "energy_class": "A+",
            "heating_type": "heat_pump",
            "energy_demand_kwh": 45.0,
            "terrace": True,
            "garden": True,
            "cellar": True,
        },
    ),
]


def holdout_ground_truth(document: HoldoutDocument) -> dict[str, Any]:
    """Return the corpus-format ground-truth label for a holdout document.

    Parameters
    ----------
    document : HoldoutDocument
        The source document.

    Returns
    -------
    dict
        ``{"document": "<stem>.pdf", "fields": {...}}``, matching the schema
        consumed by :func:`propintelli.evaluation.evaluate.load_ground_truth`.
    """
    return {"document": f"{document.stem}.pdf", "fields": dict(document.fields)}


def render_holdout_pdf(document: HoldoutDocument, output_path: Path) -> None:
    """Render a holdout document to a PDF file.

    Parameters
    ----------
    document : HoldoutDocument
        The document to render.
    output_path : Path
        Destination PDF path; parent directories are created if needed.
    """
    styles = _styles()
    lines = [line.strip() for line in document.body.splitlines() if line.strip()]
    flow: list[Any] = []
    for index, line in enumerate(lines):
        flow.append(Paragraph(line, styles["title"] if index == 0 else styles["body"]))
        flow.append(Spacer(1, 4))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        title=lines[0] if lines else document.stem,
        author="PropIntelli AI — authored holdout",
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    ).build(flow)


def generate_holdout(raw_dir: Path, ground_truth_dir: Path) -> list[Path]:
    """Render every holdout exposé PDF and write its hand-labelled ground truth.

    Parameters
    ----------
    raw_dir : Path
        Directory to write the PDF files into.
    ground_truth_dir : Path
        Directory to write the ground-truth JSON labels into.

    Returns
    -------
    list of Path
        Paths of the generated PDF files.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for document in HOLDOUT_DOCUMENTS:
        pdf_path = raw_dir / f"{document.stem}.pdf"
        render_holdout_pdf(document, pdf_path)
        label_path = ground_truth_dir / f"{document.stem}.json"
        label_path.write_text(
            json.dumps(holdout_ground_truth(document), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        generated.append(pdf_path)
    return generated
