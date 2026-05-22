"""Assemble a :class:`PropertyRecord` from normalised fields.

Uses each field's ``record_path`` from the registry to route flat canonical
values into the nested record structure (top-level, ``location``, ``features``,
or ``energy``), so adding a field to the registry automatically places it
correctly without touching this module.
"""

from __future__ import annotations

from typing import Any

from propintelli.schemas.fields import PROPERTY_FIELDS
from propintelli.schemas.property_record import (
    EnergyProfile,
    Features,
    Location,
    PropertyRecord,
    QualityReport,
)
from propintelli.transformation.normalize import NormalizedFields

_NESTED_MODELS = ("location", "features", "energy")


def assemble_record(
    normalized: NormalizedFields,
    quality: QualityReport,
    source_document: str,
    *,
    property_id: str | None = None,
) -> PropertyRecord:
    """Build a structured property record from normalised values.

    Parameters
    ----------
    normalized : NormalizedFields
        The typed field values.
    quality : QualityReport
        The computed quality report to attach.
    source_document : str
        Original document filename.
    property_id : str or None, optional
        Identifier for the record. When given (typically the Bronze document id),
        it links the record to its source document; otherwise one is generated.

    Returns
    -------
    PropertyRecord
        The assembled, validated record.
    """
    top_level: dict[str, Any] = {}
    nested: dict[str, dict[str, Any]] = {group: {} for group in _NESTED_MODELS}

    for name, value in normalized.values.items():
        path = PROPERTY_FIELDS[name].record_path
        if len(path) == 1:
            top_level[path[0]] = value
        else:
            nested[path[0]][path[1]] = value

    if property_id is not None:
        top_level["property_id"] = property_id

    return PropertyRecord(
        source_document=source_document,
        location=Location(**nested["location"]),
        features=Features(**nested["features"]),
        energy=EnergyProfile(**nested["energy"]),
        quality=quality,
        **top_level,
    )
