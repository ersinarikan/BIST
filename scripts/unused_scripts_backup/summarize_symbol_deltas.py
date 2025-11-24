#!/usr/bin/env python3
"""
Summarize per-symbol deltas between latest short-apply CSV and baseline
multi-horizon CSV for horizons 1d/3d/7d.

Outputs:
 - Prints top/bottom by DirHit delta per horizon
 - Writes a CSV with per-symbol deltas to logs

Env (optional):
 - APPLY_CSV: explicit path to apply CSV
 - BASE_CSV: explicit path to baseline CSV
 - LOG_DIR: default /opt/bist-pattern/logs
"""

import os
import glob
from typing import Optional

import pandas as pd


def find_latest(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def main() -> None:
    log_dir = os.getenv('LOG_DIR', '/opt/bist-pattern/logs')
    apply_csv = os.getenv('APPLY_CSV') or find_latest(os.path.join(log_dir, 'bist30_short_apply_*.csv'))
    base_csv = os.getenv('BASE_CSV') or find_latest(os.path.join(log_dir, 'bist30_multihorizon_metrics_*.csv'))

    if not apply_csv or not os.path.exists(apply_csv):
        print('❌ apply CSV not found')
        return
    if not base_csv or not os.path.exists(base_csv):
        print('❌ baseline CSV not found')
        return

    df_a = pd.read_csv(apply_csv)
    df_b = pd.read_csv(base_csv)

    horizons = ['1d', '3d', '7d']
    df_a = df_a[df_a['horizon'].isin(horizons)].copy()
    df_b = df_b[df_b['horizon'].isin(horizons)].copy()

    m = pd.merge(
        df_a,
        df_b,
        on=['symbol', 'horizon'],
        suffixes=('_a', '_b'),
        how='inner'
    )
    if m.empty:
        print('⚠️ No overlap between apply and baseline rows.')
        return

    m['d_r2'] = m['r2_a'] - m['r2_b']
    m['d_dirHit'] = m['dir_hit_pct_a'] - m['dir_hit_pct_b']
    m['d_mape'] = m['mape_a'] - m['mape_b']
    m['d_nrmse'] = m['nrmse_a'] - m['nrmse_b']

    # Save full deltas CSV
    os.makedirs(log_dir, exist_ok=True)
    out_csv = os.path.join(log_dir, 'bist30_short_symbol_deltas.csv')
    m[['symbol', 'horizon', 'd_dirHit', 'd_r2', 'd_mape', 'd_nrmse', 'dir_hit_pct_a', 'r2_a', 'mape_a', 'nrmse_a', 'dir_hit_pct_b', 'r2_b', 'mape_b', 'nrmse_b']].to_csv(out_csv, index=False)

    print(f"APPLY: {apply_csv}")
    print(f"BASE : {base_csv}")
    print(f"WROTE: {out_csv}")
    print()
    for hz in horizons:
        mh = m[m['horizon'] == hz].copy()
        if mh.empty:
            continue
        print('=' * 100)
        print(f"HORIZON {hz}  (n={len(mh)})")
        print('=' * 100)
        top5 = mh.sort_values('d_dirHit', ascending=False).head(5)
        bot5 = mh.sort_values('d_dirHit', ascending=True).head(5)
        print('Top +ΔDirHit: symbol, ΔDirHit(pp), ΔR2, ΔMAPE, ΔnRMSE')
        for _, r in top5.iterrows():
            print(f"  {r['symbol']}, {r['d_dirHit']:.2f}, {r['d_r2']:.4f}, {r['d_mape']:.3f}, {r['d_nrmse']:.3f}")
        print('Bottom ΔDirHit: symbol, ΔDirHit(pp), ΔR2, ΔMAPE, ΔnRMSE')
        for _, r in bot5.iterrows():
            print(f"  {r['symbol']}, {r['d_dirHit']:.2f}, {r['d_r2']:.4f}, {r['d_mape']:.3f}, {r['d_nrmse']:.3f}")


if __name__ == '__main__':
    main()


