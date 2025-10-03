"""
Shadow evaluation for pattern+sentiment meta-confidence impact.

Writes /opt/bist-pattern/logs/shadow_eval.json with summary and details.
Run:
  source venv && export DATABASE_URL=... && python scripts/shadow_eval.py
"""
from __future__ import annotations

import json
from datetime import datetime, date, timedelta

from app import app


def get_top_symbols(n: int = 40) -> list[str]:
    from models import db, Stock, StockPrice
    from sqlalchemy import func

    cutoff = date.today() - timedelta(days=60)
    rows = (
        db.session.query(Stock.symbol, func.avg(StockPrice.volume).label("avg_vol"))
        .join(StockPrice, Stock.id == StockPrice.stock_id)
        .filter(Stock.is_active.is_(True), StockPrice.date >= cutoff)
        .group_by(Stock.id, Stock.symbol)
        .order_by(func.avg(StockPrice.volume).desc())
        .limit(n)
        .all()
    )
    return [r[0] for r in rows] if rows else []


def analyze_symbol_from_cache(symbol: str) -> dict:
    """Read cached analysis for a symbol from pattern_cache.

    Returns a dict per horizon with fields: conf, delta, label, evidence.
    """
    import pathlib

    cache_path = pathlib.Path(f"/opt/bist-pattern/logs/pattern_cache/{symbol}.json")
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    uni = raw.get("ml_unified", {}) or {}

    thr_map = {"1d": 0.008, "3d": 0.021, "7d": 0.03, "14d": 0.03, "30d": 0.025}

    def label(delta_pct: float | None, h: str) -> str:
        thr = thr_map.get(h, 0.03)
        if not isinstance(delta_pct, (int, float)):
            return "HOLD"
        if delta_pct >= thr:
            return "BUY"
        if delta_pct <= -thr:
            return "SELL"
        return "HOLD"

    out: dict[str, dict] = {}
    for h in ["1d", "3d", "7d", "14d", "30d"]:
        x = uni.get(h, {}) or {}
        # Prefer best if present, else enhanced
        best_key = x.get("best") if isinstance(x.get("best"), str) else None
        node = x.get(best_key) if best_key in ("enhanced", "basic") else x.get("enhanced")
        if not isinstance(node, dict):
            node = {}
        out[h] = {
            "conf": node.get("confidence"),
            "delta": node.get("delta_pct"),
            "label": label(node.get("delta_pct"), h),
            "evidence": node.get("evidence"),
        }
    return out


def main() -> int:
    with app.app_context():
        symbols = get_top_symbols(40)
        if not symbols:
            # Fallback: any active 40
            from models import Stock

            symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).limit(40).all()]

        results: dict[str, dict] = {}
        changed = {h: 0 for h in ["1d", "3d", "7d", "14d", "30d"]}
        conf_delta_sum = {h: 0.0 for h in changed}
        strong_delta_sum = {h: 0.0 for h in changed}
        count = {h: 0 for h in changed}
        strong_threshold = 0.6

        skipped: list[str] = []
        for sym in symbols:
            meta = analyze_symbol_from_cache(sym)
            if not meta:
                skipped.append(sym)
                continue
            # Reconstruct base confidence by undoing evidence adjustment when possible
            per = {}
            for h in changed.keys():
                m = meta.get(h, {}) or {}
                mc = m.get("conf")
                ev = m.get("evidence") or {}
                adj = ev.get("confidence_adjustment")
                bc = None
                if isinstance(mc, (int, float)):
                    if isinstance(adj, (int, float)):
                        # If adj looks like multiplicative factor (typ >0.5), divide; else treat additive
                        if adj > 0.5:
                            bc = max(0.0, min(1.0, mc / adj))
                        else:
                            bc = max(0.0, min(1.0, mc - adj))
                    else:
                        bc = mc
                # labels identical because label uses delta only; keep for completeness
                b = {"conf": bc, "delta": m.get("delta"), "label": m.get("label")}
                if isinstance(bc, (int, float)) and isinstance(mc, (int, float)):
                    conf_delta_sum[h] += (mc - bc)
                    # strong signal count delta
                    strong_delta_sum[h] += (1 if mc >= strong_threshold else 0) - (1 if bc >= strong_threshold else 0)
                    count[h] += 1
                per[h] = {"base": b, "meta": m}
            results[sym] = per

        summary = {
            "generated_at": datetime.now().isoformat(),
            "symbols": symbols,
            "skipped": skipped,
            "changed_counts": changed,
            "avg_conf_delta": {h: (conf_delta_sum[h] / count[h] if count[h] > 0 else None) for h in changed},
            "strong_signal_delta": {h: (strong_delta_sum[h] / count[h] if count[h] > 0 else None) for h in changed},
        }
        payload = {"summary": summary, "details": results}

        out_path = "/opt/bist-pattern/logs/shadow_eval.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(json.dumps(summary, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
