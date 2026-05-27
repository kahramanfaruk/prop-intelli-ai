"""Layer C: reconciliation of the deterministic and LLM extractions.

When both layers produce a value for a field, agreement is strong evidence the
value is correct, so the merged confidence is boosted. Disagreement is a signal
of uncertainty: the higher-confidence value is kept, its confidence is penalised,
and a warning is recorded so the document is more likely to be routed to human
review. Fields produced by only one layer pass through unchanged.
"""

from __future__ import annotations

from propintelli.comparison import numeric_match, text_match
from propintelli.schemas.enums import Provenance
from propintelli.schemas.extraction import FieldValue
from propintelli.schemas.fields import PROPERTY_FIELDS, FieldKind
from propintelli.transformation.parsing import parse_bool, parse_number

_AGREEMENT_BOOST = 0.1
_DISAGREEMENT_PENALTY = 0.6


def _is_present(value: FieldValue | None) -> bool:
    """Whether a slot holds a usable value."""
    return value is not None and value.is_present


def _numeric(value: FieldValue) -> float | None:
    """Parse a value to float using the convention implied by its provenance."""
    return parse_number(value.raw_value, german=value.provenance is Provenance.DETERMINISTIC)


def _values_agree(field_name: str, left: FieldValue, right: FieldValue) -> bool:
    """Decide whether two values for the same field are equivalent.

    Comparison is type-aware: numeric fields compare within a relative
    tolerance, booleans compare by parsed truth value, and everything else
    compares case-insensitively.
    """
    spec = PROPERTY_FIELDS.get(field_name)
    if spec is None:
        return text_match(left.raw_value, right.raw_value)

    if spec.is_numeric:
        left_value, right_value = _numeric(left), _numeric(right)
        if left_value is None or right_value is None:
            return False
        return numeric_match(left_value, right_value, integer=spec.kind is FieldKind.INTEGER)

    if spec.kind is FieldKind.BOOLEAN:
        return parse_bool(left.raw_value) == parse_bool(right.raw_value)

    return text_match(left.raw_value, right.raw_value)


def reconcile(
    layer_a: dict[str, FieldValue],
    layer_b: dict[str, FieldValue],
) -> tuple[dict[str, FieldValue], list[str]]:
    """Merge the deterministic and LLM field maps into a single result.

    Parameters
    ----------
    layer_a : dict of str to FieldValue
        Deterministic (Layer A) field values.
    layer_b : dict of str to FieldValue
        LLM (Layer B) field values; empty when no LLM backend is configured.

    Returns
    -------
    tuple of (dict of str to FieldValue, list of str)
        The reconciled field map and any disagreement warnings.
    """
    merged: dict[str, FieldValue] = {}
    warnings: list[str] = []

    for name in sorted(set(layer_a) | set(layer_b)):
        left, right = layer_a.get(name), layer_b.get(name)
        left_present, right_present = _is_present(left), _is_present(right)

        if left_present and not right_present:
            merged[name] = left  # type: ignore[assignment]
        elif right_present and not left_present:
            merged[name] = right  # type: ignore[assignment]
        elif left_present and right_present:
            merged[name], warning = _merge_pair(name, left, right)  # type: ignore[arg-type]
            if warning:
                warnings.append(warning)

    return merged, warnings


def _merge_pair(
    name: str,
    left: FieldValue,
    right: FieldValue,
) -> tuple[FieldValue, str | None]:
    """Combine two present values for the same field."""
    if _values_agree(name, left, right):
        confidence = min(1.0, max(left.confidence, right.confidence) + _AGREEMENT_BOOST)
        return (
            FieldValue(
                raw_value=left.raw_value,
                confidence=confidence,
                provenance=Provenance.RECONCILED,
                source_snippet=left.source_snippet or right.source_snippet,
            ),
            None,
        )

    winner = left if left.confidence >= right.confidence else right
    warning = (
        f"Disagreement on '{name}': deterministic={left.raw_value!r} vs "
        f"llm={right.raw_value!r}; kept {winner.raw_value!r}"
    )
    return (
        FieldValue(
            raw_value=winner.raw_value,
            confidence=winner.confidence * _DISAGREEMENT_PENALTY,
            provenance=winner.provenance,
            source_snippet=winner.source_snippet,
        ),
        warning,
    )
