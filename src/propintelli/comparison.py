"""Shared, type-aware value comparison.

Two independent parts of the pipeline need to decide whether two values for the
same field are equivalent: Layer-C reconciliation (does the deterministic value
agree with the LLM value?) and the evaluation harness (does a prediction match
ground truth?). Defining the comparison once, here, guarantees the two can never
drift and that the numeric tolerance is stated in a single place.

The comparison is *type-aware*: numeric fields compare within a relative
tolerance (integers must match exactly), booleans compare by truth value, and
everything else compares case-insensitively on its canonical form.
"""

from __future__ import annotations

from typing import Any

from propintelli.schemas.fields import FieldKind, FieldSpec

# Relative tolerance for non-integer numeric comparison. Absorbs lossless
# formatting differences (German "449.000" vs "449000") without masking real
# disagreements; integers (years, counts) are required to match exactly.
NUMERIC_REL_TOLERANCE = 0.01


def numeric_match(left: float, right: float, *, integer: bool) -> bool:
    """Compare two numbers, exactly for integers and within tolerance otherwise.

    Parameters
    ----------
    left, right : float
        The values to compare.
    integer : bool
        If ``True``, both values are rounded and compared exactly (a year off by
        one is wrong). Otherwise a relative tolerance is applied.

    Returns
    -------
    bool
        Whether the values are considered equal.
    """
    if integer:
        return round(left) == round(right)
    tolerance = NUMERIC_REL_TOLERANCE * max(abs(left), abs(right), 1.0)
    return abs(left - right) <= tolerance


def text_match(left: str | None, right: str | None) -> bool:
    """Compare two strings case-insensitively after trimming whitespace.

    Parameters
    ----------
    left, right : str or None
        The values to compare; ``None`` is treated as the empty string.

    Returns
    -------
    bool
        Whether the trimmed, case-folded strings are equal.
    """
    return (left or "").strip().casefold() == (right or "").strip().casefold()


def values_match(spec: FieldSpec, expected: Any, predicted: Any) -> bool:
    """Compare two already-typed/comparable values for one field.

    This operates on canonical, comparable primitives (numbers, booleans, and
    enum/date values rendered as strings) — the form produced both by the
    evaluation harness when it flattens a record and by the ground-truth labels.
    Reconciliation, which works on raw extracted strings, instead composes the
    lower-level :func:`numeric_match` and :func:`text_match` helpers.

    Parameters
    ----------
    spec : FieldSpec
        Registry metadata for the field, which selects the comparison strategy.
    expected, predicted : object
        The values to compare.

    Returns
    -------
    bool
        Whether the two values are equivalent under the field's data kind.
    """
    if spec.is_numeric:
        try:
            left, right = float(expected), float(predicted)
        except (TypeError, ValueError):
            return False
        return numeric_match(left, right, integer=spec.kind is FieldKind.INTEGER)
    if spec.kind is FieldKind.BOOLEAN:
        return bool(expected) == bool(predicted)
    return text_match(str(expected), str(predicted))
