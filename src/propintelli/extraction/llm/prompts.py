"""Prompt engineering for the LLM extraction layer.

Three documented variants trade off cost against robustness:

* ``v1_direct`` — a terse instruction. Cheapest, least robust; a baseline.
* ``v2_schema`` — anchors the model on the explicit field schema and demands a
  single strict-JSON object. Far more reliable field naming and typing.
* ``v3_reasoning`` — ``v2`` plus an internal self-check and a parallel
  per-field confidence object, which feeds the platform's confidence model.

All variants instruct the model to return ``{"fields": {...}}`` (with an extra
``"confidences"`` object for ``v3``), so a single parser handles every variant.
A comparison of the variants against ground truth is documented in
``docs/prompt_engineering.md``.
"""

from __future__ import annotations

from propintelli.config import PromptVariant
from propintelli.schemas.fields import PROPERTY_FIELDS, FieldKind

_SYSTEM = (
    "You are a precise information-extraction system for German real-estate "
    "exposés (Immobilien-Exposés). You extract structured data. You never invent "
    "values: if a field is not stated in the document, return null for it. You "
    "return only the requested JSON and no prose."
)


def _schema_block() -> str:
    """Render the canonical field schema as instruction text for the model."""
    lines: list[str] = []
    for name, spec in PROPERTY_FIELDS.items():
        descriptor = spec.kind.value
        if spec.kind is FieldKind.ENUM and spec.enum_type is not None:
            allowed = ", ".join(member.value for member in spec.enum_type)
            descriptor = f"one of [{allowed}]"
        elif spec.kind is FieldKind.DATE:
            descriptor = "date as YYYY-MM-DD"
        elif spec.kind is FieldKind.BOOLEAN:
            descriptor = "true/false"
        required = " (required)" if spec.required else ""
        lines.append(f"- {name}: {descriptor}{required} — {spec.label}")
    return "\n".join(lines)


def build_messages(text: str, variant: PromptVariant) -> tuple[str, str]:
    """Build the ``(system, user)`` messages for a prompt variant.

    Parameters
    ----------
    text : str
        The document text to extract from.
    variant : PromptVariant
        Which documented prompt variant to construct.

    Returns
    -------
    tuple of (str, str)
        The system and user message contents.
    """
    if variant is PromptVariant.V1_DIRECT:
        user = (
            "Extract the real-estate fields from the document below and return a "
            'JSON object {"fields": {<field>: <value>}}. Use null when unknown.\n\n'
            f"DOCUMENT:\n{text}"
        )
        return _SYSTEM, user

    schema = _schema_block()
    if variant is PromptVariant.V2_SCHEMA:
        user = (
            "Extract the following fields from the German exposé. Map German terms "
            "to the canonical English values shown. Amounts are in EUR; use a dot as "
            "the decimal separator and no thousands separators. Return ONLY a JSON "
            'object of the form {"fields": {<field>: <value-or-null>}}.\n\n'
            f"FIELDS:\n{schema}\n\nDOCUMENT:\n{text}"
        )
        return _SYSTEM, user

    user = (
        "Extract the following fields from the German exposé. Map German terms to "
        "the canonical English values shown. Before answering, internally verify "
        "that every value is supported by the text and that required fields are "
        "present; set unsupported values to null. Return ONLY a JSON object of the "
        'form {"fields": {<field>: <value-or-null>}, "confidences": {<field>: '
        "<number between 0 and 1>}}. The confidence reflects how certain you are "
        "that the value is correct.\n\n"
        f"FIELDS:\n{schema}\n\nDOCUMENT:\n{text}"
    )
    return _SYSTEM, user
