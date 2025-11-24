#!/usr/bin/env python3
"""
Calibrate confidence and optimize per-horizon delta/conf thresholds.

Outputs param_store.json with:
  - isotonic bins (horizon -> [(c_in, c_out), ...])
  - thresholds (horizon -> {delta_thr, conf_thr})
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Tuple

from app import app
from models import db, PredictionsLog, OutcomesLog
import numpy as np  # type: ignore


def isotonic_bins(pairs: List[Tuple[float, float]], bins: int = 10) -> List[Tuple[float, float]]:
    """Build reliability bins (c_in -> c_out) using sklearn IsotonicRegression if available.

    Fallback: decile grouping with monotonic smoothing.
    """
    if not pairs:
        return []
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore
        x = np.array([float(c) for c, _ in pairs], dtype=float)
        y = np.array([float(h) for _, h in pairs], dtype=float)
        # Bound inputs to [0,1]
        x = np.clip(x, 0.0, 1.0)
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
        y_fit = ir.fit_transform(x, y)
        # Build bins based on equal-frequency quantiles
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y_fit[order]
        n = len(x_sorted)
        bins_out: List[Tuple[float, float]] = []
        for i in range(bins):
            lo = int(i * n / bins)
            hi = int((i + 1) * n / bins)
            hi = max(hi, lo + 1)
            xs = x_sorted[lo:hi]
            ys = y_sorted[lo:hi]
            if xs.size == 0:
                continue
            bins_out.append((float(xs.mean()), float(ys.mean())))
        return bins_out
    except Exception:
        # Fallback: simple deciles with monotonic smoothing (PAV-lite)
        pairs = sorted(pairs, key=lambda x: x[0])
        fallback_bins: List[Tuple[float, float]] = []
        n = len(pairs)
        for i in range(bins):
            lo = int(i * n / bins)
            hi = int((i + 1) * n / bins) or n
            chunk = pairs[lo:hi]
            if not chunk:
                continue
            c_in = sum(x for x, _ in chunk) / len(chunk)
            c_out = sum(y for _, y in chunk) / len(chunk)
            fallback_bins.append((c_in, c_out))
        for i in range(1, len(fallback_bins)):
            if fallback_bins[i][1] < fallback_bins[i - 1][1]:
                fallback_bins[i] = (fallback_bins[i][0], fallback_bins[i - 1][1])
        return fallback_bins


def optimize_thresholds(pairs: List[Tuple[float, float]], deltas: List[float]) -> Dict[str, float]:
    # pairs: (conf, hit) ; deltas: |delta_pred|
    if not pairs or not deltas:
        return {'delta_thr': 0.03, 'conf_thr': 0.65}
    best = {'delta_thr': 0.03, 'conf_thr': 0.65}
    best_f1 = -1.0
    # Small grid
    for conf_thr in [0.55, 0.6, 0.65, 0.7, 0.75]:
        for delta_thr in [0.015, 0.02, 0.03, 0.04, 0.05]:
            tp = fp = fn = 0
            for (c, hit), d in zip(pairs, deltas):
                pred = (c >= conf_thr) and (abs(d) >= delta_thr)
                if pred and hit >= 0.5:
                    tp += 1
                elif pred and hit < 0.5:
                    fp += 1
                elif (not pred) and hit >= 0.5:
                    fn += 1
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best = {'delta_thr': delta_thr, 'conf_thr': conf_thr}
    return best


def optimize_thresholds_challenger(pairs: List[Tuple[float, float]], deltas: List[float], prod_th: Dict[str, float]) -> Dict[str, float]:
    """Optimize challenger thresholds using different ranges than production.
    
    Challenger uses more aggressive ranges (slightly lower thresholds) to test
    if more signals can be generated with acceptable precision.
    """
    if not pairs or not deltas:
        # Default challenger values (slightly more aggressive than production default)
        return {'delta_thr': 0.02, 'conf_thr': 0.6}
    
    prod_delta = prod_th.get('delta_thr', 0.03)
    prod_conf = prod_th.get('conf_thr', 0.65)
    
    best = {'delta_thr': max(0.01, prod_delta * 0.75), 'conf_thr': max(0.50, prod_conf * 0.92)}
    best_f1 = -1.0
    
    # Challenger grid: More aggressive (lower thresholds) than production
    # Delta range: slightly lower than production (0.75x to 1.0x)
    delta_min = max(0.01, prod_delta * 0.7)
    delta_max = min(0.05, prod_delta * 1.1)
    delta_candidates = [
        delta_min,
        prod_delta * 0.8,
        prod_delta * 0.9,
        prod_delta,
        min(delta_max, prod_delta * 1.05),
        delta_max
    ]
    # Remove duplicates and sort
    delta_candidates = sorted(set([round(d, 4) for d in delta_candidates if 0.01 <= d <= 0.05]))
    
    # Conf range: slightly lower than production (0.85x to 1.0x)
    conf_min = max(0.50, prod_conf * 0.85)
    conf_max = min(0.80, prod_conf * 1.05)
    conf_candidates = [
        conf_min,
        prod_conf * 0.90,
        prod_conf * 0.95,
        prod_conf,
        min(conf_max, prod_conf * 1.02),
        conf_max
    ]
    # Remove duplicates and sort
    conf_candidates = sorted(set([round(c, 3) for c in conf_candidates if 0.50 <= c <= 0.80]))
    
    for conf_thr in conf_candidates:
        for delta_thr in delta_candidates:
            tp = fp = fn = 0
            for (c, hit), d in zip(pairs, deltas):
                pred = (c >= conf_thr) and (abs(d) >= delta_thr)
                if pred and hit >= 0.5:
                    tp += 1
                elif pred and hit < 0.5:
                    fp += 1
                elif (not pred) and hit >= 0.5:
                    fn += 1
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best = {'delta_thr': round(delta_thr, 4), 'conf_thr': round(conf_thr, 3)}
    
    return best


def analyze_ab_test_results(cutoff: datetime, horizon: str, min_samples_chall: int = 50) -> Dict[str, any]:
    """Analyze A/B test results for a specific horizon.
    
    Note: This function must be called within app.app_context().
    
    Returns:
        {
            'prod': {'acc': float, 'n': int},
            'chall': {'acc': float, 'n': int},
            'chall_better': bool,
            'significant': bool
        }
    """
    rows = (
        db.session.query(PredictionsLog, OutcomesLog)
        .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)
        .filter(PredictionsLog.horizon == horizon)
        .filter(PredictionsLog.ts_pred >= cutoff)
        .all()
    )
    
    prod_hits = []
    chall_hits = []
    
    for p, o in rows:
        try:
            pv = str(getattr(p, 'param_version', '') or '')
            grp = 'chall' if 'ab:chall' in pv else 'prod'
            hit = 1.0 if bool(getattr(o, 'dir_hit')) else 0.0
            
            if grp == 'prod':
                prod_hits.append(hit)
            else:
                chall_hits.append(hit)
        except Exception:
            continue
    
    prod_n = len(prod_hits)
    chall_n = len(chall_hits)
    prod_acc = (sum(prod_hits) / prod_n) if prod_n > 0 else None
    chall_acc = (sum(chall_hits) / chall_n) if chall_n > 0 else None
    
    # Check if challenger is significantly better (at least 2% improvement with enough samples)
    chall_better = False
    significant = False
    if prod_acc is not None and chall_acc is not None and chall_n >= min_samples_chall:
        chall_better = chall_acc > prod_acc
        # Simple significance: challenger must be at least 2% better
        improvement = chall_acc - prod_acc
        significant = improvement >= 0.02 and chall_n >= min_samples_chall
    
    return {
        'prod': {'acc': prod_acc, 'n': prod_n},
        'chall': {'acc': chall_acc, 'n': chall_n},
        'chall_better': chall_better,
        'significant': significant,
        'improvement': (chall_acc - prod_acc) if (prod_acc is not None and chall_acc is not None) else None
    }


def _load_existing_store(out_dir: str) -> Dict:
    try:
        path = os.path.join(out_dir, 'param_store.json')
        if os.path.exists(path):
            with open(path, 'r') as rf:
                return json.load(rf) or {}
    except Exception:
        pass
    return {}


def run(window_days: int = 30, go_live: str | None = None, min_samples: int = 150,
        enable_challenger_opt: bool = False, enable_production_promote: bool = False,
        min_samples_chall: int = 50, improvement_threshold: float = 0.02) -> int:
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    # Optional hard cutoff to exclude pre-live/backfilled data
    if go_live:
        try:
            go_live_dt = datetime.fromisoformat(go_live)
            # Avoid overly narrow window: ensure at least half window after go_live
            if go_live_dt > cutoff:
                cutoff = go_live_dt
        except Exception:
            pass
    out_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
    os.makedirs(out_dir, exist_ok=True)
    prev_store = _load_existing_store(out_dir)
    store = {
        'generated_at': datetime.utcnow().isoformat(),
        'window_days': window_days,
        'go_live': go_live,
        'min_samples': min_samples,
        'horizons': {},
        'skipped_horizons': [],
    }
    state: Dict[str, Dict[str, int | bool | str]] = {'meta': {}}  # horizon -> info
    with app.app_context():
        horizons = ['1d', '3d', '7d', '14d', '30d']
        for h in horizons:
            q = (
                db.session.query(PredictionsLog, OutcomesLog)
                .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)
                .filter(PredictionsLog.horizon == h)
                .filter(PredictionsLog.ts_pred >= cutoff)
            )
            pairs: List[Tuple[float, float]] = []
            deltas: List[float] = []
            for p, o in q.all():
                try:
                    c = float(p.confidence) if p.confidence is not None else None
                    if c is None:
                        continue
                    hit = 1.0 if bool(o.dir_hit) else 0.0
                    pairs.append((c, hit))
                    d = float(p.delta_pred) if p.delta_pred is not None else 0.0
                    deltas.append(abs(d))
                except Exception:
                    continue
            # Min-sample gating: if insufficient data, reuse prior or defaults
            if len(pairs) < min_samples:
                store['skipped_horizons'].append({'horizon': h, 'reason': f'not_enough_samples:{len(pairs)}'})
                prev = (prev_store.get('horizons', {}) if isinstance(prev_store, dict) else {}).get(h)
                used_prev = False
                if prev:
                    store['horizons'][h] = prev
                    used_prev = True
                else:
                    store['horizons'][h] = {
                        'isotonic': [],
                        'thresholds': {'delta_thr': 0.03, 'conf_thr': 0.65},
                    }
                state[h] = {
                    'n_pairs': len(pairs),
                    'used_prev': used_prev,  # type: ignore[typeddict-item]
                    'used_default': (not used_prev),  # type: ignore[typeddict-item]
                }
                continue
            bins = isotonic_bins(pairs, bins=10)
            th = optimize_thresholds(pairs, deltas)
            store['horizons'][h] = {'isotonic': bins, 'thresholds': th}
            state[h] = {'n_pairs': len(pairs), 'used_prev': False, 'used_default': False}  # type: ignore[typeddict-item]
            
            # ⚡ NEW: Challenger optimization (optional)
            # Note: We optimize challenger FIRST, then promote if better
            # This ensures we promote the newly optimized challenger, not old values
            optimized_chall_th = None  # Will hold newly optimized challenger thresholds
            
            if enable_challenger_opt:
                # Analyze A/B test results for this horizon
                ab_results = analyze_ab_test_results(cutoff, h, min_samples_chall=min_samples_chall)
                
                # Option 2: Optimize challenger thresholds using A/B test data (DO THIS FIRST)
                # Collect only challenger predictions for optimization
                chall_pairs = []
                chall_deltas = []
                
                chall_rows = (
                    db.session.query(PredictionsLog, OutcomesLog)
                    .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)
                    .filter(PredictionsLog.horizon == h)
                    .filter(PredictionsLog.ts_pred >= cutoff)
                    .all()
                )
                
                for p, o in chall_rows:
                    try:
                        pv = str(getattr(p, 'param_version', '') or '')
                        if 'ab:chall' not in pv:
                            continue  # Skip non-challenger predictions
                        c = float(p.confidence) if p.confidence is not None else None
                        if c is None:
                            continue
                        hit = 1.0 if bool(o.dir_hit) else 0.0
                        chall_pairs.append((c, hit))
                        d = float(p.delta_pred) if p.delta_pred is not None else 0.0
                        chall_deltas.append(abs(d))
                    except Exception:
                        continue
                
                # Optimize challenger thresholds if enough samples
                if len(chall_pairs) >= min_samples_chall:
                    chall_th = optimize_thresholds_challenger(chall_pairs, chall_deltas, th)
                    
                    # Initialize bandit structure if not exists
                    if 'bandit' not in store:
                        store['bandit'] = {'horizons': {}}
                    if 'horizons' not in store['bandit']:
                        store['bandit']['horizons'] = {}
                    if h not in store['bandit']['horizons']:
                        # Preserve existing traffic setting
                        prev_bandit = prev_store.get('bandit', {})
                        prev_bandit_h = (prev_bandit.get('horizons', {}) or {}).get(h, {})
                        store['bandit']['horizons'][h] = {
                            'traffic': prev_bandit_h.get('traffic', 0.1),
                            'challenger': {}
                        }
                    
                    store['bandit']['horizons'][h]['challenger'] = chall_th
                    optimized_chall_th = chall_th  # Save optimized challenger thresholds
                    print(f"[challenger-opt] {h}: Optimized challenger thresholds "
                          f"(delta_thr: {chall_th['delta_thr']}, conf_thr: {chall_th['conf_thr']}, "
                          f"samples: {len(chall_pairs)})", flush=True)
                else:
                    # Not enough challenger samples, preserve existing challenger config
                    prev_bandit = prev_store.get('bandit', {})
                    prev_bandit_h = (prev_bandit.get('horizons', {}) or {}).get(h, {})
                    if prev_bandit_h and prev_bandit_h.get('challenger'):
                        if 'bandit' not in store:
                            store['bandit'] = {'horizons': {}}
                        if 'horizons' not in store['bandit']:
                            store['bandit']['horizons'] = {}
                        store['bandit']['horizons'][h] = prev_bandit_h.copy()
                        optimized_chall_th = prev_bandit_h.get('challenger')  # Use existing if not optimized
                        print(f"[challenger-opt] {h}: Not enough samples ({len(chall_pairs)} < {min_samples_chall}), "
                              f"preserving existing challenger config", flush=True)
                
                # Option 1: Promote challenger to production if significantly better (DO THIS AFTER OPTIMIZATION)
                # ✅ FIX: Use newly optimized challenger thresholds, not old ones from prev_store
                if enable_production_promote and ab_results['significant'] and ab_results['chall_better']:
                    # Use optimized challenger thresholds (if available) or existing ones
                    chall_to_promote = optimized_chall_th or prev_store.get('bandit', {}).get('horizons', {}).get(h, {}).get('challenger', {})
                    
                    if chall_to_promote:
                        # Promote challenger to production
                        print(f"[challenger-promote] {h}: Promoting challenger to production "
                              f"(improvement: {ab_results['improvement']*100:.2f}%, "
                              f"prod_acc: {ab_results['prod']['acc']*100:.1f}%, "
                              f"chall_acc: {ab_results['chall']['acc']*100:.1f}%)", flush=True)
                        th = chall_to_promote.copy()  # Use optimized challenger thresholds as production
                        store['horizons'][h]['thresholds'] = th

    # Preserve existing bandit config if not optimizing challenger
    if not enable_challenger_opt:
        prev_bandit = prev_store.get('bandit', {})
        if prev_bandit:
            store['bandit'] = prev_bandit.copy()
    
    # Add checksum of horizons
    try:
        import hashlib  # lazy import
        _h = hashlib.sha256(json.dumps(store.get('horizons', {}), sort_keys=True).encode('utf-8')).hexdigest()
        store['checksum_horizons'] = _h  # type: ignore[assignment]
    except Exception:
        pass

    # Atomic write for param_store.json (with file lock)
    out_path = os.path.join(out_dir, 'param_store.json')
    tmp_path = out_path + '.tmp'
    content = json.dumps(store, ensure_ascii=False, indent=2)
    try:
        from bist_pattern.utils.param_store_lock import file_lock  # type: ignore
    except Exception:
        file_lock = None  # type: ignore
    try:
        if file_lock is not None:
            with file_lock(out_path):
                with open(tmp_path, 'w') as wf:
                    wf.write(content)
                    wf.flush()
                    try:
                        import os as _os
                        _os.fsync(wf.fileno())
                    except Exception:
                        pass
                import os as _os
                _os.replace(tmp_path, out_path)
        else:
            with open(tmp_path, 'w') as wf:
                wf.write(content)
                wf.flush()
                try:
                    import os as _os
                    _os.fsync(wf.fileno())
                except Exception:
                    pass
            import os as _os
            _os.replace(tmp_path, out_path)
    except Exception:
        # Fallback non-atomic write
        with open(out_path, 'w') as wf:
            wf.write(content)
    print(json.dumps({'status': 'ok', 'output': out_path}))

    # Write calibration_state.json with horizon sample counts
    try:
        state_out = {
            'generated_at': store['generated_at'],
            'window_days': store['window_days'],
            'go_live': store['go_live'],
            'min_samples': store['min_samples'],
            'horizons': state,
        }
        cstate_path = os.path.join(out_dir, 'calibration_state.json')
        tmp_state = cstate_path + '.tmp'
        scontent = json.dumps(state_out, ensure_ascii=False, indent=2)
        with open(tmp_state, 'w') as wf:
            wf.write(scontent)
            wf.flush()
            try:
                import os as _os
                _os.fsync(wf.fileno())
            except Exception:
                pass
        import os as _os
        _os.replace(tmp_state, cstate_path)
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-days', type=int, default=30)
    parser.add_argument('--go-live', type=str, default=os.getenv('CALIB_GO_LIVE', ''))
    parser.add_argument('--min-samples', type=int, default=int(os.getenv('CALIB_MIN_SAMPLES', '150')))
    parser.add_argument('--enable-challenger-opt', action='store_true',
                        default=os.getenv('ENABLE_CHALLENGER_OPTIMIZATION', '0').lower() in ('1', 'true', 'yes', 'on'))
    parser.add_argument('--enable-production-promote', action='store_true',
                        default=os.getenv('ENABLE_PRODUCTION_PROMOTE', '0').lower() in ('1', 'true', 'yes', 'on'))
    parser.add_argument('--min-samples-chall', type=int, default=int(os.getenv('CHALL_MIN_SAMPLES', '50')))
    parser.add_argument('--improvement-threshold', type=float, default=float(os.getenv('CHALL_IMPROVEMENT_THRESHOLD', '0.02')))
    args = parser.parse_args()
    raise SystemExit(
        run(
            window_days=args.window_days,
            go_live=(args.go_live or None),
            min_samples=args.min_samples,
            enable_challenger_opt=args.enable_challenger_opt,
            enable_production_promote=args.enable_production_promote,
            min_samples_chall=args.min_samples_chall,
            improvement_threshold=args.improvement_threshold,
        )
    )
