#!/usr/bin/env python3
"""
Summarize short-horizon (1/3/7d) metrics from the latest apply CSV and compare
against the latest baseline multi-horizon CSV.

Env:
  - APPLY_CSV: optional explicit path to apply CSV
  - LOG_DIR: optional directory (default /opt/bist-pattern/logs)
"""

import os
import glob
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def find_latest(pattern: str) -> Optional[str]:
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def agg_metrics(df: pd.DataFrame, horizons: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for h in horizons:
        s = df[df['horizon'] == h]
        out[h] = {
            'n': int(len(s)),
            'r2': float(np.nanmean(s['r2'])) if len(s) else float('nan'),
            'dirHit': float(np.nanmean(s['dir_hit_pct'])) if len(s) else float('nan'),
            'MAPE': float(np.nanmean(s['mape'])) if len(s) else float('nan'),
            'nRMSE': float(np.nanmean(s['nrmse'])) if len(s) else float('nan'),
        }
    return out


def main() -> None:
    log_dir = os.getenv('LOG_DIR', '/opt/bist-pattern/logs')
    apply_csv = os.getenv('APPLY_CSV')
    if not apply_csv:
        apply_csv = find_latest(os.path.join(log_dir, 'bist30_short_apply_*.csv'))
    if not apply_csv or not os.path.exists(apply_csv):
        print('❌ apply CSV not found')
        return

    base_csv = find_latest(os.path.join(log_dir, 'bist30_multihorizon_metrics_*.csv'))

    df_apply = pd.read_csv(apply_csv)
    horizons = ['1d', '3d', '7d']
    agg_apply = agg_metrics(df_apply, horizons)

    agg_base: Optional[Dict[str, Dict[str, float]]] = None
    if base_csv and os.path.exists(base_csv):
        df_base = pd.read_csv(base_csv)
        df_base = df_base[df_base['horizon'].isin(horizons)]
        agg_base = agg_metrics(df_base, horizons)

    # Print concise summary
    print(f"APPLY: {apply_csv}")
    if base_csv:
        print(f"BASE : {base_csv}")
    print()
    header = "horizon,n,r2,dirHit,MAPE,nRMSE"
    if agg_base:
        header += ",Δr2,ΔdirHit,ΔMAPE,ΔnRMSE"
    print(header)
    for h in horizons:
        a = agg_apply.get(h, {})
        if agg_base:
            b = agg_base.get(h, {})
            dr2 = (a.get('r2') or 0.0) - (b.get('r2') or 0.0)
            dh = (a.get('dirHit') or 0.0) - (b.get('dirHit') or 0.0)
            dm = (a.get('MAPE') or 0.0) - (b.get('MAPE') or 0.0)
            dn = (a.get('nRMSE') or 0.0) - (b.get('nRMSE') or 0.0)
            line = (
                f"{h},{a.get('n', 0)},{a.get('r2', float('nan')):.4f},"
                f"{a.get('dirHit', float('nan')):.2f},{a.get('MAPE', float('nan')):.3f},"
                f"{a.get('nRMSE', float('nan')):.3f},{dr2:.4f},{dh:.2f},{dm:.3f},{dn:.3f}"
            )
            print(line)
        else:
            line = (
                f"{h},{a.get('n', 0)},{a.get('r2', float('nan')):.4f},"
                f"{a.get('dirHit', float('nan')):.2f},{a.get('MAPE', float('nan')):.3f},"
                f"{a.get('nRMSE', float('nan')):.3f}"
            )
            print(line)


if __name__ == '__main__':
    main()


