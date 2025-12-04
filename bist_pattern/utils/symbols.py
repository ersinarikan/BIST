from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def sanitize_symbol(symbol: str) -> str:
    """Normalize raw ticker text to a clean alphanumeric BIST code.

    - Strips zero-width and whitespace
    - Uppercases
    - Removes all non-alphanumeric characters (e.g., '$')
    - Validates reasonable length (2..10)
    """
    if not symbol:
        return ""

    try:
        raw = str(symbol).strip().replace("\ufeff", "").replace("\u200b", "").upper()
    except Exception:
        raw = str(symbol).strip().upper()

    cleaned = "".join(ch for ch in raw if ch.isalnum())

    if not (2 <= len(cleaned) <= 10):
        return ""

    if cleaned != raw:
        try:
            logger.debug("sanitize_symbol: '%s' -> '%s'", symbol, cleaned)
        except Exception as e:
            logger.debug(f"Failed to log sanitize_symbol debug: {e}")

    return cleaned


def to_yf_symbol(symbol: str) -> str:
    """Convert input to Yahoo Finance BIST symbol (append .IS).

    Input is sanitized first. Returns empty string if invalid after sanitize.
    """
    s = sanitize_symbol(symbol)
    if not s:
        return ""
    return s if s.endswith('.IS') else f"{s}.IS"

