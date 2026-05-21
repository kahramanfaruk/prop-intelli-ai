"""Shared German→canonical vocabularies for the property domain.

Both the deterministic extractor (which must *recognise* German terms in the
text) and the normalisation step (which must *map* free-form terms onto the
canonical enums) rely on the same vocabulary. Defining it once keeps recognition
and mapping in lock-step and makes adding a new synonym a single-line change.

Patterns are ordered most-specific first so that, e.g., ``Fernwärme`` is matched
before the bare token ``Gas`` and ``renovierungsbedürftig`` before ``saniert``.
"""

from __future__ import annotations

from propintelli.schemas.enums import HeatingType, PriceKind, PropertyCondition

# Heating system synonyms, most specific first.
HEATING_PATTERNS: tuple[tuple[str, HeatingType], ...] = (
    (r"fernw[äa]rme", HeatingType.DISTRICT_HEATING),
    (r"w[äa]rmepumpe", HeatingType.HEAT_PUMP),
    (r"fu[ßs]bodenheizung", HeatingType.UNDERFLOOR),
    (r"pellet", HeatingType.PELLET),
    (r"solarthermie|solaranlage|solar", HeatingType.SOLAR),
    (r"gasheizung|gas-?zentralheizung|erdgas|\bgas\b", HeatingType.GAS),
    (r"[öo]lheizung|\b[öo]l\b", HeatingType.OIL),
    (r"elektroheizung|nachtspeicher|\belektro\b", HeatingType.ELECTRIC),
)

# Building-condition synonyms, most specific first.
CONDITION_PATTERNS: tuple[tuple[str, PropertyCondition], ...] = (
    (r"erstbezug", PropertyCondition.FIRST_OCCUPANCY),
    (r"neubau|neuwertig", PropertyCondition.NEW_BUILD),
    (
        r"renovierungsbed[üu]rftig|sanierungsbed[üu]rftig|unsaniert",
        PropertyCondition.NEEDS_RENOVATION,
    ),
    (r"modernisiert|teilsaniert", PropertyCondition.MODERNISED),
    (r"\bsaniert|kernsaniert|vollsaniert", PropertyCondition.RENOVATED),
    (r"gepflegt|gut erhalten|guter zustand", PropertyCondition.WELL_KEPT),
)

# Boolean equipment-feature keyword patterns.
FEATURE_PATTERNS: dict[str, str] = {
    "balcony": r"balkon",
    "terrace": r"terrasse|dachterrasse",
    "garden": r"garten",
    "parking": r"stellplatz|tiefgarage|garage|parkplatz|carport|duplexparker",
    "cellar": r"keller|kellerabteil|souterrain",
    "elevator": r"aufzug|fahrstuhl|\blift\b|personenaufzug",
    "fitted_kitchen": r"einbauk[üu]che|\bebk\b",
    "furnished": r"m[öo]bliert|vollm[öo]bliert",
    "barrier_free": r"barrierefrei|altersgerecht|rollstuhlgerecht|stufenlos",
}

# Energy-certificate type keyword patterns mapped to a normalised label.
ENERGY_CERTIFICATE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"bedarfsausweis", "Bedarfsausweis"),
    (r"verbrauchsausweis", "Verbrauchsausweis"),
)

# Price-label synonyms mapped to a price kind, most specific first.
PRICE_LABEL_PATTERNS: tuple[tuple[str, PriceKind], ...] = (
    (r"warmmiete|gesamtmiete|bruttomiete", PriceKind.WARM_RENT),
    (r"kaltmiete|nettokaltmiete|nettomiete", PriceKind.COLD_RENT),
    (r"kaufpreis|kauf-?preis", PriceKind.PURCHASE),
    (r"monatsmiete|\bmiete\b", PriceKind.COLD_RENT),
    (r"\bpreis\b", PriceKind.PURCHASE),
)

# Recognised German street-name suffixes used to locate an address line.
STREET_SUFFIXES: tuple[str, ...] = (
    "stra[ßs]e",
    "str\\.",
    "weg",
    "allee",
    "platz",
    "gasse",
    "chaussee",
    "ring",
    "landstra[ßs]e",
    "damm",
    "ufer",
    "steig",
    "berg",
)
