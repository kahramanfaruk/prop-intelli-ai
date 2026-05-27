"""Locale-aware primitive parsing.

Deterministic extraction yields German-formatted numbers (``1.234,56``: dot
thousands, comma decimal), while the LLM layer is instructed to emit
dot-decimal numbers. Both conventions are handled here behind a single
``german`` flag, so the rest of the codebase never re-implements number parsing.
"""

from __future__ import annotations

import re

_NUMERIC_CHARS = re.compile(r"[^0-9.,\-]")
_TRUE_TOKENS = frozenset({"true", "ja", "yes", "1", "vorhanden", "x"})
_FALSE_TOKENS = frozenset({"false", "nein", "no", "0", "nicht vorhanden"})


def parse_number(raw: str | None, *, german: bool) -> float | None:
    """Parse a numeric string under either the German or dot-decimal convention.

    Parameters
    ----------
    raw : str or None
        The raw value (may include currency symbols, units, or separators).
    german : bool
        If ``True``, interpret ``.`` as a thousands separator and ``,`` as the
        decimal separator. If ``False``, interpret ``.`` as the decimal point and
        ``,`` as a thousands separator.

    Returns
    -------
    float or None
        The parsed value, or ``None`` if the string holds no parseable number.
    """
    if raw is None:
        return None
    cleaned = _NUMERIC_CHARS.sub("", raw.strip())
    if not cleaned or cleaned in {"-", ".", ","}:
        return None

    if german:
        cleaned = (
            cleaned.replace(".", "").replace(",", ".")
            if "," in cleaned
            else cleaned.replace(".", "")
        )
    elif cleaned.count(",") > 1 or ("," in cleaned and "." in cleaned):
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")

    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_bool(raw: str | None) -> bool | None:
    """Parse a boolean from common German/English tokens.

    Parameters
    ----------
    raw : str or None
        The raw value.

    Returns
    -------
    bool or None
        The parsed boolean, or ``None`` if the token is unrecognised.
    """
    if raw is None:
        return None
    token = raw.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return None
