"""
Lightweight news provider for FinGPT integration.

Reads RSS feeds from environment (NEWS_SOURCES) and returns recent
headlines that likely relate to a given BIST symbol.

Environment variables (override via systemd):
  - NEWS_SOURCES: comma-separated RSS URLs (e.g., Milliyet Ekonomi RSS, BloombergHT RSS)
  - NEWS_LOOKBACK_HOURS: time window to accept items (default 24)
  - NEWS_MAX_ITEMS: maximum items to return (default 10)
  - NEWS_CACHE_TTL: seconds to cache results in-process (default 600)

Return value of get_recent_news(symbol): list[str]
Suitable to be passed directly to FinGPT analyzer.
"""

from __future__ import annotations

import os
import time
from typing import List


_cache: dict[str, dict] = {}


def _now_ts() -> float:
    try:
        return time.time()
    except Exception:
        return 0.0


def _get_env_list(name: str) -> List[str]:
    try:
        raw = os.getenv(name, "")
        return [u.strip() for u in raw.split(",") if u.strip()]
    except Exception:
        return []

 
def _within_lookback(published_ts: float, lookback_hours: int) -> bool:
    try:
        if not published_ts:
            return True  # keep if feed has no date
        return (_now_ts() - float(published_ts)) <= (lookback_hours * 3600.0)
    except Exception:
        return True


def _entry_text(entry) -> str:
    parts: List[str] = []
    try:
        title = getattr(entry, "title", None) or ""
        if title:
            parts.append(str(title))
    except Exception:
        pass
    try:
        summary = getattr(entry, "summary", None) or getattr(entry, "description", None) or ""
        if summary:
            parts.append(str(summary))
    except Exception:
        pass
    try:
        link = getattr(entry, "link", None) or ""
        if link:
            parts.append(str(link))
    except Exception:
        pass
    txt = " - ".join(parts)
    # Normalize whitespace
    try:
        txt = " ".join(txt.split())
    except Exception:
        pass
    return txt[:2000]


def _normalize(text: str) -> str:
    try:
        t = (text or "").upper()
        # Basic Turkish normalization
        trans = str.maketrans({
            "İ": "I", "I": "I", "Ş": "S", "Ğ": "G", "Ü": "U", "Ö": "O", "Ç": "C",
            "ı": "I", "ş": "S", "ğ": "G", "ü": "U", "ö": "O", "ç": "C",
        })
        return t.translate(trans)
    except Exception:
        return (text or "").upper()


def _strip_company_suffixes(name: str) -> str:
    t = _normalize(name)
    suffixes = [
        # TR common
        " A.S.", " A.S", " AŞ", " AS", " A.Ş.", " A.Ş", " T.A.S.", " T.A.S",
        " TIC.", " TIC", " VE TIC.", " VE TIC", " SAN.", " SAN",
        " YATIRIM", " HOLDING", " HOLDING A.S.", " HOLDING AS",
        # EN common
        " CORPORATION", " CORP.", " CORP", " INC.", " INC", " LTD.", " LTD",
        " LIMITED", " CO.", " COMPANY",
    ]
    for suf in suffixes:
        if t.endswith(suf):
            t = t[: -len(suf)]
    return t.strip()


def _get_company_names(symbol: str) -> List[str]:
    names: List[str] = []
    try:
        # Try to read from DB via app context if available
        from app import app
        with app.app_context():
            try:
                from models import Stock  # type: ignore
                s = Stock.query.filter_by(symbol=symbol.upper()).first()
                if s and getattr(s, "name", None):
                    names.append(str(s.name))
            except Exception:
                pass
    except Exception:
        pass
    # Normalize and expand variants
    out: List[str] = []
    for n in names:
        base = _strip_company_suffixes(n)
        if base:
            out.append(base)
        if n and n not in out:
            out.append(_normalize(n))
    return out


def _match_symbol_or_name(text: str, symbol: str, company_names: List[str]) -> bool:
    try:
        t = _normalize(text)
        s = _normalize(symbol)
        if not s or not t:
            return False
        # Symbol patterns
        sym_patterns = [
            f" {s} ", f"({s})", f"[{s}]", f"{s}:", f"{s}-", f"{s}.", f"{s},",
            f" {s} HISSE", f" {s} HISSESI", f" {s} IS", f" {s}.IS",
        ]
        if any(p in t for p in sym_patterns):
            return True
        # Company name presence (word-boundary heuristic)
        for name in company_names or []:
            n = _strip_company_suffixes(name)
            if not n:
                continue
            if f" {n} " in t or t.startswith(n + " ") or t.endswith(" " + n) or n + ":" in t:
                return True
        return False
    except Exception:
        return False


def get_recent_news(symbol: str, *, max_items: int | None = None, lookback_hours: int | None = None) -> List[str]:
    try:
        sources = _get_env_list("NEWS_SOURCES")
        if not sources:
            return []

        try:
            import feedparser  # type: ignore
        except Exception:
            return []

        if max_items is None:
            try:
                max_items = int(os.getenv("NEWS_MAX_ITEMS", "10"))
            except Exception:
                max_items = 10
        if lookback_hours is None:
            try:
                lookback_hours = int(os.getenv("NEWS_LOOKBACK_HOURS", "24"))
            except Exception:
                lookback_hours = 24
        try:
            cache_ttl = int(os.getenv("NEWS_CACHE_TTL", "600"))
        except Exception:
            cache_ttl = 600

        # Cache key by symbol + sources signature
        cache_key = f"{symbol}|{','.join(sources)}|{max_items}|{lookback_hours}"
        c = _cache.get(cache_key)
        if c and (_now_ts() - float(c.get("ts", 0.0))) < cache_ttl:
            data = c.get("data")
            if isinstance(data, list):
                return data

        hits: List[str] = []
        generals: List[str] = []
        sym = (symbol or "").upper().strip()
        comp_names = _get_company_names(sym)

        for url in sources:
            try:
                feed = feedparser.parse(url)
                for e in getattr(feed, "entries", [])[: max_items * 3]:  # soft cap per feed
                    # Compute published timestamp
                    published_ts = 0.0
                    try:
                        if hasattr(e, "published_parsed") and e.published_parsed:
                            published_ts = time.mktime(e.published_parsed)
                    except Exception:
                        published_ts = 0.0
                    if not _within_lookback(published_ts, lookback_hours):
                        continue
                    text = _entry_text(e)
                    if not text:
                        continue
                    if _match_symbol_or_name(text, sym, comp_names):
                        hits.append(text)
                    elif len(generals) < max_items:
                        # Keep a pool of general market headlines
                        if any(k in text.upper() for k in ["BIST", "BORSA İSTANBUL", "BORSA", "HİSSE"]):
                            generals.append(text)
            except Exception:
                continue

        # Prefer symbol-specific headlines; fallback to general market news
        result: List[str] = hits[: max_items]
        if len(result) < (max_items or 0):
            for t in generals:
                if len(result) >= max_items:
                    break
                result.append(t)

        _cache[cache_key] = {"ts": _now_ts(), "data": result}
        return result
    except Exception:
        return []
