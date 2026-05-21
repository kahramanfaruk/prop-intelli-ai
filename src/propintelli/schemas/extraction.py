"""Extraction-time data structures.

These models carry the *raw* output of the extraction layers — one
:class:`FieldValue` per canonical field — before normalisation and validation
turn them into a typed :class:`~propintelli.schemas.property_record.PropertyRecord`.
Keeping confidence and provenance attached at the field level is what enables
per-field confidence reporting and targeted human review.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from propintelli.schemas.enums import Provenance


class FieldValue(BaseModel):
    """A single extracted field value with its quality metadata.

    Attributes
    ----------
    raw_value : str or None
        The value exactly as extracted from the document text, before
        normalisation. ``None`` denotes "not found".
    confidence : float
        Confidence in this value, in ``[0, 1]``.
    provenance : Provenance
        Which layer produced the value.
    source_snippet : str or None
        A short surrounding text fragment, used to explain and highlight the
        value in the human-in-the-loop UI.
    """

    model_config = ConfigDict(frozen=True)

    raw_value: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance: Provenance = Provenance.DETERMINISTIC
    source_snippet: str | None = None

    @property
    def is_present(self) -> bool:
        """Whether a non-empty value was extracted."""
        return self.raw_value is not None and self.raw_value.strip() != ""


class ExtractionResult(BaseModel):
    """The reconciled output of the extraction engine for one document.

    Attributes
    ----------
    document_id : str
        Stable identifier assigned at ingestion.
    source_document : str
        Original document filename.
    fields : dict of str to FieldValue
        Extracted values keyed by canonical field name.
    warnings : list of str
        Non-fatal notes accumulated during extraction (e.g. an LLM downgrade).
    """

    document_id: str
    source_document: str
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def get(self, name: str) -> FieldValue | None:
        """Return the value for a field, or ``None`` if absent.

        Parameters
        ----------
        name : str
            Canonical field name.

        Returns
        -------
        FieldValue or None
            The extracted value, or ``None`` when the field was not produced.
        """
        return self.fields.get(name)

    def present_field_names(self) -> tuple[str, ...]:
        """Return the names of fields that hold a non-empty value.

        Returns
        -------
        tuple of str
            Names of present fields.
        """
        return tuple(name for name, value in self.fields.items() if value.is_present)
