#!/usr/bin/env python3
"""
Optimize evidence weights (w_pat, w_sent) per horizon from recent outcomes.

Heuristic approach:
- Join PredictionsLog with OutcomesLog for the last N days (default 30)
- Compute correlation between pattern/sentiment scores and direction hit
- Scale default horizon weights by positive correlation strength
- Persist into logs/param_store.json under key:
  {
    "weights_generated_at": ts,
    "weights": { "1d": {"w_pat": x, "w_sent": y}, ... }
  }

Notes:
- Falls back to previous weights or static defaults if data is insufficient
- Designed to be lightweight and safe for nightly runs
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from app import app  # Flask app context
from models import db, PredictionsLog, OutcomesLog


DEFAULTS = {
    '1d': (0.12, 0.10),
    '3d': (0.10, 0.08),
    '7d': (0.06, 0.05),
    '14d': (0.04, 0.03),
    '30d': (0.03, 0.02),
}


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _load_param_store(out_dir: str) -> Dict:
    try:
        path = os.path.join(out_dir, 'param_store.json')
        if os.path.exists(path):
            with open(path, 'r') as rf:
                return json.load(rf) or {}
    except Exception:
        pass
    return {}


def _save_param_store(out_dir: str, store: Dict) -> str:
    """Atomic write with checksum for safety."""
    path = os.path.join(out_dir, 'param_store.json')
    tmp = path + '.tmp'
    # embed checksum of weights section for quick validation
    try:
        import hashlib
        checksum = hashlib.sha256(json.dumps(store.get('weights', {}), sort_keys=True).encode('utf-8')).hexdigest()
        store['weights_checksum'] = checksum  # type: ignore[assignment]
    except Exception:
        pass
    content = json.dumps(store, ensure_ascii=False, indent=2)
    with open(tmp, 'w') as wf:
        wf.write(content)
        try:
            os.fsync(wf.fileno())
        except Exception:
            pass
    try:
        os.replace(tmp, path)
    except Exception:
        # last resort
        with open(path, 'w') as wf:
            wf.write(content)
    return path


def _corr(xs: List[float], ys: List[float]) -> float:
    try:
        import math
        n = len(xs)
        if n == 0 or n != len(ys):
            return 0.0
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        deny = math.sqrt(sum((y - my) ** 2 for y in ys))
        if denx == 0 or deny == 0:
            return 0.0
        return max(-1.0, min(1.0, num / (denx * deny)))
    except Exception:
        return 0.0


def learn_weights(window_days: int = 30, min_samples: int = 150) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with app.app_context():
        cutoff = datetime.utcnow() - timedelta(days=window_days)
        horizons = ['1d', '3d', '7d', '14d', '30d']
        for h in horizons:
            q = (
                db.session.query(PredictionsLog, OutcomesLog)
                .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)
                .filter(PredictionsLog.horizon == h)
                .filter(PredictionsLog.ts_pred >= cutoff)
            )
            p_list: List[Tuple[float, float, int]] = []  # (pat_score, sent_score, hit)
            for p, o in q.all():
                try:
                    ps = _safe_float(getattr(p, 'pat_score', None))
                    ss = _safe_float(getattr(p, 'sent_score', None))
                    if ps is None and ss is None:
                        continue
                    hit = 1 if bool(getattr(o, 'dir_hit', False)) else 0
                    p_list.append((ps or 0.0, ss or 0.0, hit))
                except Exception:
                    continue
            if len(p_list) < min_samples:
                # insufficient; keep defaults
                base = DEFAULTS.get(h, (0.03, 0.02))
                out[h] = {'w_pat': base[0], 'w_sent': base[1], 'samples': len(p_list)}
                continue
            pats = [it[0] for it in p_list]
            sents = [it[1] for it in p_list]
            hits = [float(it[2]) for it in p_list]
            # Positive correlations only; negative evidence should not boost
            c_pat = max(0.0, _corr(pats, hits))
            c_sent = max(0.0, _corr(sents, hits))
            # Scale defaults by correlation strength (soft cap 1.0)
            base_pat, base_sent = DEFAULTS.get(h, (0.03, 0.02))
            w_pat = round(base_pat * (0.5 + 0.5 * c_pat), 4)
            w_sent = round(base_sent * (0.5 + 0.5 * c_sent), 4)
            out[h] = {'w_pat': w_pat, 'w_sent': w_sent, 'samples': len(p_list)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-days', type=int, default=30)
    parser.add_argument('--min-samples', type=int, default=150)
    args = parser.parse_args()

    out_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
    os.makedirs(out_dir, exist_ok=True)
    store = _load_param_store(out_dir)

    weights = learn_weights(args.window_days, args.min_samples)
    store = store if isinstance(store, dict) else {}
    store['weights_generated_at'] = datetime.utcnow().isoformat()
    store['weights'] = weights
    # Initialize simple bandit section if absent (keep traffic small by default)
    store.setdefault('bandit', {}).setdefault('horizons', {}).setdefault('1d', {'traffic': 0.10})
    # File-locked atomic save
    try:
        from bist_pattern.utils.param_store_lock import file_lock  # type: ignore
    except Exception:
        file_lock = None  # type: ignore
    if file_lock is not None:
        with file_lock(os.path.join(out_dir, 'param_store.json')):
            path = _save_param_store(out_dir, store)
    else:
        path = _save_param_store(out_dir, store)
    print(json.dumps({'status': 'ok', 'output': path, 'weights': weights}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
