import logging
from typing import Dict, Any
 
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    try:
        d = df.copy()
        mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        }
        for k, v in mapping.items():
            if k in d.columns and v not in d.columns:
                d[v] = d[k]
        return d
    except Exception:
        return df


def _rolling_last_true(series: pd.Series, window: int) -> pd.Series:
    # Returns 1.0 if any True in last window, else 0.0
    try:
        return series.rolling(window, min_periods=1).max().astype(float)
    except Exception:
        return pd.Series(index=series.index, data=0.0)


def _head_shoulders_heuristic(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Minimal H&S / Inverse H&S heuristic on last N bars.
    Produces per-row signals using rolling detection to avoid lookahead.
    Outputs:
      - hs_signal: -1 bear (H&S), +1 bull (Inverse H&S), 0 otherwise
      - hs_conf:   0..1 confidence
      - hs_neckline: approximate neckline level (float)
      - hs_measured_move_pct: signed measured move as ratio of close (float)
    """
    try:
        close = df['close'].astype(float).values
        n = len(close)
        if n < 50:
            zeros = np.zeros(n, dtype=float)
            return {
                'hs_signal': zeros,
                'hs_conf': zeros,
                'hs_neckline': np.full(n, np.nan),
                'hs_measured_move_pct': zeros,
            }

        # Simple peak find
        def _peaks(arr: np.ndarray, dist: int = 3) -> list[int]:
            idx = []
            for i in range(dist, len(arr) - dist):
                w = arr[i - dist:i + dist + 1]
                if arr[i] == w.max() and w.argmax() == dist:
                    idx.append(i)
            return idx

        window = 50
        hs_signal = np.zeros(n, dtype=float)
        hs_conf = np.zeros(n, dtype=float)
        hs_neck = np.full(n, np.nan)
        hs_mm = np.zeros(n, dtype=float)

        for end in range(20, n):
            start = max(0, end - window)
            seg = close[start:end + 1]
            peaks = _peaks(seg, 3)
            if len(peaks) < 3:
                continue
            # take consecutive triples
            found = False
            for i in range(len(peaks) - 2):
                a, b, c = peaks[i], peaks[i + 1], peaks[i + 2]
                L, H, R = seg[a], seg[b], seg[c]
                # Bearish H&S
                if H > L * 1.01 and H > R * 1.01 and abs(L - R) / max(H, 1e-8) < 0.03:
                    neckline = (min(seg[a:b]) + min(seg[b:c])) / 2.0
                    head_height = max(H - neckline, 0.0)
                    conf = float(min(0.95, 0.55 + (head_height / max(H, 1e-8)) * 3.0))
                    hs_signal[end] = -1.0
                    hs_conf[end] = conf
                    hs_neck[end] = neckline
                    # measured move (expected drop) as fraction of price
                    hs_mm[end] = - head_height / max(H, 1e-8)
                    found = True
                    break
                # Bullish Inverse H&S
                if H < L * 0.99 and H < R * 0.99 and abs(L - R) / max(L, 1e-8) < 0.03:
                    neckline = (max(seg[a:b]) + max(seg[b:c])) / 2.0
                    head_depth = max(neckline - H, 0.0)
                    conf = float(min(0.95, 0.55 + (head_depth / max(neckline, 1e-8)) * 3.0))
                    hs_signal[end] = 1.0
                    hs_conf[end] = conf
                    hs_neck[end] = neckline
                    hs_mm[end] = + head_depth / max(neckline, 1e-8)
                    found = True
                    break
            if not found:
                continue

        return {
            'hs_signal': hs_signal,
            'hs_conf': hs_conf,
            'hs_neckline': hs_neck,
            'hs_measured_move_pct': hs_mm,
        }
    except Exception as e:
        logger.debug(f"H&S heuristic error: {e}")
        n = len(df)
        zeros = np.zeros(n, dtype=float)
        return {
            'hs_signal': zeros,
            'hs_conf': zeros,
            'hs_neckline': np.full(n, np.nan),
            'hs_measured_move_pct': zeros,
        }


def build_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Event-anchored pattern feature builder.
    Adds pattern-aware columns:
      - hs_signal, hs_conf, hs_neckline, hs_measured_move_pct
      - hs_breakout (neckline cross), hs_event_window_7/14 (recent event flag)
    """
    try:
        d = _ensure_ohlc(df)
        if not all(c in d.columns for c in ('open', 'high', 'low', 'close')):
            return d

        hs = _head_shoulders_heuristic(d)
        for k, v in hs.items():
            d[k] = v

        # Double Top / Bottom heuristic with measured move
        try:
            dtb = _double_top_bottom_heuristic(d)
            for k, v in dtb.items():
                d[k] = v
        except Exception as _e:
            logger.debug(f"DTB heuristic skipped: {_e}")

        # Triangle heuristic (ascending/descending/symmetrical)
        try:
            tri = _triangle_heuristic(d)
            for k, v in tri.items():
                d[k] = v
        except Exception as _e:
            logger.debug(f"Triangle heuristic skipped: {_e}")

        # Flag heuristic (bull/bear flag)
        try:
            flg = _flag_heuristic(d)
            for k, v in flg.items():
                d[k] = v
        except Exception as _e:
            logger.debug(f"Flag heuristic skipped: {_e}")

        # Wedge heuristic (rising/falling wedge)
        try:
            wdg = _wedge_heuristic(d)
            for k, v in wdg.items():
                d[k] = v
        except Exception as _e:
            logger.debug(f"Wedge heuristic skipped: {_e}")

        # Breakout detection (leakage-safe using only current/previous)
        try:
            neck = pd.Series(d['hs_neckline']).astype(float)
            close = pd.Series(d['close']).astype(float)
            prev_close = close.shift(1)
            prev_neck = neck.shift(1)
            # Cross from below (bull) or above (bear)
            cross_up = (prev_close < prev_neck) & (close >= neck)
            cross_dn = (prev_close > prev_neck) & (close <= neck)
            d['hs_breakout'] = np.where(cross_up | cross_dn, 1.0, 0.0)
        except Exception:
            d['hs_breakout'] = 0.0

        # DTB breakout
        try:
            neck = pd.Series(d['dtb_neckline']).astype(float)
            close = pd.Series(d['close']).astype(float)
            prev_close = close.shift(1)
            prev_neck = neck.shift(1)
            cross_up = (prev_close < prev_neck) & (close >= neck)
            cross_dn = (prev_close > prev_neck) & (close <= neck)
            d['dtb_breakout'] = np.where(cross_up | cross_dn, 1.0, 0.0)
        except Exception:
            d['dtb_breakout'] = 0.0

        # Triangle breakout
        try:
            neck = pd.Series(d['tri_neckline']).astype(float)
            close = pd.Series(d['close']).astype(float)
            prev_close = close.shift(1)
            prev_neck = neck.shift(1)
            cross_up = (prev_close < prev_neck) & (close >= neck)
            cross_dn = (prev_close > prev_neck) & (close <= neck)
            d['tri_breakout'] = np.where(cross_up | cross_dn, 1.0, 0.0)
        except Exception:
            d['tri_breakout'] = 0.0

        # Flag breakout
        try:
            neck = pd.Series(d['flag_neckline']).astype(float)
            close = pd.Series(d['close']).astype(float)
            prev_close = close.shift(1)
            prev_neck = neck.shift(1)
            cross_up = (prev_close < prev_neck) & (close >= neck)
            cross_dn = (prev_close > prev_neck) & (close <= neck)
            d['flag_breakout'] = np.where(cross_up | cross_dn, 1.0, 0.0)
        except Exception:
            d['flag_breakout'] = 0.0

        # Wedge breakout
        try:
            neck = pd.Series(d['wedge_neckline']).astype(float)
            close = pd.Series(d['close']).astype(float)
            prev_close = close.shift(1)
            prev_neck = neck.shift(1)
            cross_up = (prev_close < prev_neck) & (close >= neck)
            cross_dn = (prev_close > prev_neck) & (close <= neck)
            d['wedge_breakout'] = np.where(cross_up | cross_dn, 1.0, 0.0)
        except Exception:
            d['wedge_breakout'] = 0.0

        # Breakout strength (signed, clipped) per pattern
        try:
            close = pd.to_numeric(d['close'], errors='coerce')
        except Exception:
            close = pd.Series(index=d.index, data=np.nan)

        def _breakout_strength(sig_col: str, neck_col: str, brk_col: str, out_col: str) -> None:
            try:
                sig = pd.to_numeric(d.get(sig_col, 0.0), errors='coerce').fillna(0.0)
                neck = pd.to_numeric(d.get(neck_col, np.nan), errors='coerce')
                brk = pd.to_numeric(d.get(brk_col, 0.0), errors='coerce').fillna(0.0)
                denom = neck.abs().replace(0, np.nan)
                raw = (close - neck) / denom
                signed = raw * np.sign(sig)
                # Only count when breakout flagged; else 0
                strength = np.where(brk > 0.0, np.clip(signed, -0.2, 0.2), 0.0)
                d[out_col] = pd.Series(strength, index=d.index).fillna(0.0).astype(float)
            except Exception:
                d[out_col] = 0.0

        _breakout_strength('hs_signal', 'hs_neckline', 'hs_breakout', 'hs_breakout_strength')
        _breakout_strength('dtb_signal', 'dtb_neckline', 'dtb_breakout', 'dtb_breakout_strength')
        _breakout_strength('tri_signal', 'tri_neckline', 'tri_breakout', 'tri_breakout_strength')
        _breakout_strength('flag_signal', 'flag_neckline', 'flag_breakout', 'flag_breakout_strength')
        _breakout_strength('wedge_signal', 'wedge_neckline', 'wedge_breakout', 'wedge_breakout_strength')

        # Pattern consensus (direction and measured-move) and any-breakout/any-event windows
        try:
            comp = []  # (signal, conf, mm, breakout)
            for base in ('hs', 'dtb', 'tri', 'flag', 'wedge'):
                sig = pd.to_numeric(d.get(f'{base}_signal', 0.0), errors='coerce').fillna(0.0)
                conf = pd.to_numeric(d.get(f'{base}_conf', 0.0), errors='coerce').fillna(0.0).clip(0.0, 1.0)
                mm = pd.to_numeric(d.get(f'{base}_measured_move_pct', 0.0), errors='coerce').fillna(0.0).clip(-0.5, 0.5)
                brk = pd.to_numeric(d.get(f'{base}_breakout', 0.0), errors='coerce').fillna(0.0)
                comp.append((sig, conf, mm, brk))

            if comp:
                sig_stack = [c[0] for c in comp]
                conf_stack = [c[1] for c in comp]
                mm_stack = [c[2] for c in comp]
                brk_stack = [c[3] for c in comp]

                wsum = None
                for w in conf_stack:
                    wsum = w if wsum is None else (wsum + w)
                wsum = (wsum if wsum is not None else pd.Series(index=d.index, data=0.0)) + 1e-8

                # Direction consensus: weighted average of signals by confidence
                num_dir = None
                for s, w in zip(sig_stack, conf_stack):
                    term = (s.apply(np.sign) * w)
                    num_dir = term if num_dir is None else (num_dir + term)
                dir_cons = (num_dir / wsum).clip(-1.0, 1.0)
                d['pattern_dir_consensus'] = dir_cons.astype(float)

                # Measured-move consensus: weighted average of mm by confidence
                num_mm = None
                for mm, w in zip(mm_stack, conf_stack):
                    term = (mm * w)
                    num_mm = term if num_mm is None else (num_mm + term)
                mm_cons = (num_mm / wsum).clip(-0.5, 0.5)
                d['pattern_mm_consensus'] = mm_cons.astype(float)

                # Any breakout flag
                any_brk = None
                for b in brk_stack:
                    any_brk = b if any_brk is None else np.maximum(any_brk, b)
                d['pattern_breakout_any'] = pd.Series(any_brk, index=d.index).fillna(0.0).astype(float)
            else:
                d['pattern_dir_consensus'] = 0.0
                d['pattern_mm_consensus'] = 0.0
                d['pattern_breakout_any'] = 0.0
        except Exception:
            d['pattern_dir_consensus'] = 0.0
            d['pattern_mm_consensus'] = 0.0
            d['pattern_breakout_any'] = 0.0

        # Unified event recency windows across patterns
        try:
            hs_ev = ((pd.Series(d.get('hs_signal', 0.0)).abs() > 0) & (pd.Series(d.get('hs_conf', 0.0)) > 0.5))
            dtb_ev = ((pd.Series(d.get('dtb_signal', 0.0)).abs() > 0) & (pd.Series(d.get('dtb_conf', 0.0)) > 0.5))
            tri_ev = ((pd.Series(d.get('tri_signal', 0.0)).abs() > 0) & (pd.Series(d.get('tri_conf', 0.0)) > 0.5))
            flag_ev = ((pd.Series(d.get('flag_signal', 0.0)).abs() > 0) & (pd.Series(d.get('flag_conf', 0.0)) > 0.5))
            wedge_ev = ((pd.Series(d.get('wedge_signal', 0.0)).abs() > 0) & (pd.Series(d.get('wedge_conf', 0.0)) > 0.5))
            any_ev = (hs_ev | dtb_ev | tri_ev | flag_ev | wedge_ev).astype(float)
            d['pattern_event_window_7'] = _rolling_last_true(any_ev, 7)
            d['pattern_event_window_14'] = _rolling_last_true(any_ev, 14)
        except Exception:
            d['pattern_event_window_7'] = 0.0
            d['pattern_event_window_14'] = 0.0

        # Event recency windows
        try:
            event_flag = ((pd.Series(d['hs_signal']).abs() > 0) & (pd.Series(d['hs_conf']) > 0.5)).astype(float)
            d['hs_event_window_7'] = _rolling_last_true(event_flag, 7)
            d['hs_event_window_14'] = _rolling_last_true(event_flag, 14)
        except Exception:
            d['hs_event_window_7'] = 0.0
            d['hs_event_window_14'] = 0.0

        try:
            dtb_flag = ((pd.Series(d.get('dtb_signal', 0)).abs() > 0) & (pd.Series(d.get('dtb_conf', 0)) > 0.5)).astype(float)
            d['dtb_event_window_7'] = _rolling_last_true(dtb_flag, 7)
            d['dtb_event_window_14'] = _rolling_last_true(dtb_flag, 14)
        except Exception:
            d['dtb_event_window_7'] = 0.0
            d['dtb_event_window_14'] = 0.0

        # Triangle/Flag/Wedge event recency
        try:
            tri_flag = ((pd.Series(d.get('tri_signal', 0)).abs() > 0) & (pd.Series(d.get('tri_conf', 0)) > 0.5)).astype(float)
            d['tri_event_window_7'] = _rolling_last_true(tri_flag, 7)
            d['tri_event_window_14'] = _rolling_last_true(tri_flag, 14)
        except Exception:
            d['tri_event_window_7'] = 0.0
            d['tri_event_window_14'] = 0.0

        try:
            flag_flag = ((pd.Series(d.get('flag_signal', 0)).abs() > 0) & (pd.Series(d.get('flag_conf', 0)) > 0.5)).astype(float)
            d['flag_event_window_7'] = _rolling_last_true(flag_flag, 7)
            d['flag_event_window_14'] = _rolling_last_true(flag_flag, 14)
        except Exception:
            d['flag_event_window_7'] = 0.0
            d['flag_event_window_14'] = 0.0

        try:
            wedge_flag = ((pd.Series(d.get('wedge_signal', 0)).abs() > 0) & (pd.Series(d.get('wedge_conf', 0)) > 0.5)).astype(float)
            d['wedge_event_window_7'] = _rolling_last_true(wedge_flag, 7)
            d['wedge_event_window_14'] = _rolling_last_true(wedge_flag, 14)
        except Exception:
            d['wedge_event_window_7'] = 0.0
            d['wedge_event_window_14'] = 0.0

        # Clip and clean
        for col in ('hs_conf', 'hs_measured_move_pct'):
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors='coerce')
                if col == 'hs_conf':
                    d[col] = d[col].clip(0.0, 1.0)
                if col == 'hs_measured_move_pct':
                    d[col] = d[col].clip(-0.5, 0.5)  # avoid extreme ratios

        for col in ('dtb_conf', 'dtb_measured_move_pct', 'tri_conf', 'tri_measured_move_pct', 'flag_conf', 'flag_measured_move_pct', 'wedge_conf', 'wedge_measured_move_pct'):
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors='coerce')
                if col.endswith('conf'):
                    d[col] = d[col].clip(0.0, 1.0)
                if col.endswith('measured_move_pct'):
                    d[col] = d[col].clip(-0.5, 0.5)

        return d
    except Exception as e:
        logger.error(f"pattern_features error: {e}")
        return df


def _double_top_bottom_heuristic(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple Double Top / Double Bottom heuristic with measured move and neckline.
    Returns per-row arrays:
      - dtb_signal: -1 (double top), +1 (double bottom), 0 otherwise
      - dtb_conf: confidence 0..1
      - dtb_neckline: approximate neckline level
      - dtb_measured_move_pct: signed measured move ratio
    """
    try:
        close = df['close'].astype(float).values
        n = len(close)
        if n < 40:
            zeros = np.zeros(n, dtype=float)
            return {
                'dtb_signal': zeros,
                'dtb_conf': zeros,
                'dtb_neckline': np.full(n, np.nan),
                'dtb_measured_move_pct': zeros,
            }

        def _peaks(arr: np.ndarray, dist: int = 3):
            idx_max, idx_min = [], []
            for i in range(dist, len(arr) - dist):
                w = arr[i - dist:i + dist + 1]
                if arr[i] == w.max() and w.argmax() == dist:
                    idx_max.append(i)
                if arr[i] == w.min() and w.argmin() == dist:
                    idx_min.append(i)
            return idx_max, idx_min

        window = 50
        sig = np.zeros(n, dtype=float)
        conf = np.zeros(n, dtype=float)
        neck = np.full(n, np.nan)
        mm = np.zeros(n, dtype=float)

        for end in range(20, n):
            start = max(0, end - window)
            seg = close[start:end + 1]
            peaks, troughs = _peaks(seg, 3)
            # Double Top (two highs within ~1% with dip between)
            found = False
            for i in range(len(peaks) - 1):
                a, b = peaks[i], peaks[i + 1]
                H1, H2 = seg[a], seg[b]
                tol = max(H1, H2) * 0.01
                if abs(H1 - H2) <= tol and b - a >= 3:
                    left, right = min(a, b), max(a, b)
                    valley = np.min(seg[left:right]) if right - left > 2 else max(H1, H2)
                    if valley < max(H1, H2) * 0.985:
                        height = max(H1, H2) - valley
                        if height > 0:
                            conf_val = min(0.95, 0.55 + (height / max(H1, H2)) * 3.0)
                            sig[end] = -1.0
                            conf[end] = conf_val
                            neck[end] = valley
                            mm[end] = - height / max(H1, H2)
                            found = True
                            break
            if found:
                continue
            # Double Bottom (two lows within ~1% with peak between)
            for i in range(len(troughs) - 1):
                a, b = troughs[i], troughs[i + 1]
                L1, L2 = seg[a], seg[b]
                tol = max(abs(L1), abs(L2)) * 0.01 if max(abs(L1), abs(L2)) != 0 else 0.01
                if abs(L1 - L2) <= tol and b - a >= 3:
                    left, right = min(a, b), max(a, b)
                    peak = np.max(seg[left:right]) if right - left > 2 else min(L1, L2)
                    if peak > min(L1, L2) * 1.015:
                        depth = peak - min(L1, L2)
                        if depth > 0:
                            conf_val = min(0.95, 0.55 + (depth / max(peak, 1e-8)) * 3.0)
                            sig[end] = 1.0
                            conf[end] = conf_val
                            neck[end] = peak
                            mm[end] = + depth / max(peak, 1e-8)
                            break
        return {
            'dtb_signal': sig,
            'dtb_conf': conf,
            'dtb_neckline': neck,
            'dtb_measured_move_pct': mm,
        }
    except Exception as e:
        logger.debug(f"DTB heuristic error: {e}")
        n = len(df)
        zeros = np.zeros(n, dtype=float)
        return {
            'dtb_signal': zeros,
            'dtb_conf': zeros,
            'dtb_neckline': np.full(n, np.nan),
            'dtb_measured_move_pct': zeros,
        }


def _triangle_heuristic(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple triangle detection using last N highs/lows convergence and flat line.
    Outputs tri_signal (-1 bear/+1 bull/0), tri_conf (0..1), tri_neckline, tri_measured_move_pct.
    """
    try:
        close = df['close'].astype(float).values
        high = df['high'].astype(float).values
        low = df['low'].astype(float).values
        n = len(close)
        zeros = np.zeros(n, dtype=float)
        if n < 30:
            return {
                'tri_signal': zeros,
                'tri_conf': zeros,
                'tri_neckline': np.full(n, np.nan),
                'tri_measured_move_pct': zeros,
            }
        win = 40
        tri_sig = zeros.copy()
        tri_conf = zeros.copy()
        tri_neck = np.full(n, np.nan)
        tri_mm = zeros.copy()
        for end in range(20, n):
            start = max(0, end - win)
            hseg = high[start:end+1]
            lseg = low[start:end+1]
            if len(hseg) < 10:
                continue
            top = np.max(hseg[-10:])
            bot = np.min(lseg[-10:])
            height = max(top - bot, 0.0)
            if height <= 0:
                continue
            # Flat top (ascending triangle) or flat bottom (descending triangle)
            flat_top = np.std(hseg[-6:]) / (top + 1e-8) < 0.002
            flat_bot = np.std(lseg[-6:]) / (abs(bot) + 1e-8) < 0.002
            # Convergence: range shrink
            rng_now = np.max(hseg[-6:]) - np.min(lseg[-6:])
            rng_prev = np.max(hseg[:6]) - np.min(lseg[:6]) if len(hseg) >= 12 else rng_now
            converging = rng_now < rng_prev * 0.6
            if converging and (flat_top or flat_bot):
                if flat_top:
                    # expect bullish breakout; neckline near top
                    tri_sig[end] = 1.0
                    tri_neck[end] = top
                    tri_mm[end] = + height / max(top, 1e-8)
                elif flat_bot:
                    tri_sig[end] = -1.0
                    tri_neck[end] = bot
                    tri_mm[end] = - height / max(top, 1e-8)
                tri_conf[end] = min(0.95, 0.55 + (height / max(top, 1e-8)) * 1.5)
        return {
            'tri_signal': tri_sig,
            'tri_conf': tri_conf,
            'tri_neckline': tri_neck,
            'tri_measured_move_pct': tri_mm,
        }
    except Exception:
        n = len(df)
        zeros = np.zeros(n, dtype=float)
        return {
            'tri_signal': zeros,
            'tri_conf': zeros,
            'tri_neckline': np.full(n, np.nan),
            'tri_measured_move_pct': zeros,
        }


def _flag_heuristic(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple flag detection: sharp move (flagpole) followed by tight channel consolidation.
    Outputs flag_signal, flag_conf, flag_neckline, flag_measured_move_pct.
    """
    try:
        close = df['close'].astype(float).values
        n = len(close)
        zeros = np.zeros(n, dtype=float)
        if n < 30:
            return {
                'flag_signal': zeros,
                'flag_conf': zeros,
                'flag_neckline': np.full(n, np.nan),
                'flag_measured_move_pct': zeros,
            }
        flg_sig = zeros.copy()
        flg_conf = zeros.copy()
        flg_neck = np.full(n, np.nan)
        flg_mm = zeros.copy()
        for end in range(15, n):
            # flagpole: last 10 bars change
            if end < 12:
                continue
            pole = close[end-10] if end-10 >= 0 else close[end]
            pct = (close[end-1] - pole) / max(pole, 1e-8)
            # consolidation: last 5 bars small range
            rng = (np.max(close[end-5:end]) - np.min(close[end-5:end])) / max(close[end-1], 1e-8) if end >= 5 else 0
            if pct > 0.08 and rng < 0.01:  # bullish flag
                flg_sig[end] = 1.0
                flg_neck[end] = np.max(close[end-5:end])
                flg_mm[end] = + pct  # measured move ~ pole percent
                flg_conf[end] = min(0.95, 0.55 + pct * 2.0)
            elif pct < -0.08 and rng < 0.01:  # bearish flag
                flg_sig[end] = -1.0
                flg_neck[end] = np.min(close[end-5:end])
                flg_mm[end] = - abs(pct)
                flg_conf[end] = min(0.95, 0.55 + abs(pct) * 2.0)
        return {
            'flag_signal': flg_sig,
            'flag_conf': flg_conf,
            'flag_neckline': flg_neck,
            'flag_measured_move_pct': flg_mm,
        }
    except Exception:
        n = len(df)
        zeros = np.zeros(n, dtype=float)
        return {
            'flag_signal': zeros,
            'flag_conf': zeros,
            'flag_neckline': np.full(n, np.nan),
            'flag_measured_move_pct': zeros,
        }


def _wedge_heuristic(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple wedge detection: converging higher highs/higher lows (rising) or lower highs/lower lows (falling).
    Outputs wedge_signal, wedge_conf, wedge_neckline, wedge_measured_move_pct.
    """
    try:
        high = df['high'].astype(float).values
        low = df['low'].astype(float).values
        close = df['close'].astype(float).values
        n = len(close)
        zeros = np.zeros(n, dtype=float)
        if n < 30:
            return {
                'wedge_signal': zeros,
                'wedge_conf': zeros,
                'wedge_neckline': np.full(n, np.nan),
                'wedge_measured_move_pct': zeros,
            }
        w_sig = zeros.copy()
        w_conf = zeros.copy()
        w_neck = np.full(n, np.nan)
        w_mm = zeros.copy()
        win = 40
        for end in range(20, n):
            start = max(0, end - win)
            hseg = high[start:end+1]
            lseg = low[start:end+1]
            if len(hseg) < 10:
                continue
            # simple slopes via linear fit
            try:
                x = np.arange(len(hseg))
                a_h, b_h = np.polyfit(x, hseg, 1)
                a_l, b_l = np.polyfit(x, lseg, 1)
            except Exception:
                continue
            converging = (np.max(hseg[-6:]) - np.min(lseg[-6:])) < (np.max(hseg[:6]) - np.min(lseg[:6])) * 0.7 if len(hseg) >= 12 else False
            if converging:
                top = np.max(hseg[-6:])
                bot = np.min(lseg[-6:])
                height = max(top - bot, 0.0)
                if height <= 0:
                    continue
                if a_h > 0 and a_l > 0:  # rising wedge (bearish)
                    w_sig[end] = -1.0
                    w_neck[end] = bot
                    w_mm[end] = - height / max(top, 1e-8)
                elif a_h < 0 and a_l < 0:  # falling wedge (bullish)
                    w_sig[end] = 1.0
                    w_neck[end] = top
                    w_mm[end] = + height / max(top, 1e-8)
                w_conf[end] = min(0.95, 0.55 + (height / max(top, 1e-8)) * 1.5)
        return {
            'wedge_signal': w_sig,
            'wedge_conf': w_conf,
            'wedge_neckline': w_neck,
            'wedge_measured_move_pct': w_mm,
        }
    except Exception:
        n = len(df)
        zeros = np.zeros(n, dtype=float)
        return {
            'wedge_signal': zeros,
            'wedge_conf': zeros,
            'wedge_neckline': np.full(n, np.nan),
            'wedge_measured_move_pct': zeros,
        }
