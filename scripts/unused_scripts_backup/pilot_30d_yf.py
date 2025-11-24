#!/usr/bin/env python3
"""
Pilot: DB bağımsız 30g eğitim (yfinance ile veri çekerek)

Kullanım:
  ML_HORIZONS=30 ML_USE_DIRECTIONAL_LOSS=0 python3 scripts/pilot_30d_yf.py
"""

import os
import sys
import json
from typing import List

import pandas as pd

# Proje kökünü path'e ekle
sys.path.insert(0, '/opt/bist-pattern')

from enhanced_ml_system import EnhancedMLSystem  # noqa: E402


SYMBOLS: List[str] = ['HEKTS', 'KRDMD', 'AKBNK']


def _yf_symbol(sym: str) -> str:
    # Yahoo Finance BIST sembolleri .IS uzantısı ile kullanılır
    sym = sym.strip().upper()
    if not sym.endswith('.IS'):
        return f"{sym}.IS"
    return sym


def _fetch_yf(sym: str, limit_days: int = 1000) -> pd.DataFrame:
    import yfinance as yf  # noqa: WPS433 (runtime import)
    t = _yf_symbol(sym)
    df = yf.download(t, period='max', auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Son 'limit_days' adet barı al
    df = df.tail(limit_days)
    # Sütunları beklenen forma çevir
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    df = df.reset_index(drop=True)
    # Eksik kolonları oluştur ve tip dönüşümleri
    cols = ['open', 'high', 'low', 'close', 'volume']
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    df[cols] = df[cols].apply(lambda s: pd.to_numeric(s, errors='coerce'))
    df[cols] = df[cols].fillna(0.0)
    return df[cols]


def run_pilot() -> None:
    # Varsayılanları ata
    os.environ.setdefault('ML_HORIZONS', '30')
    os.environ.setdefault('ML_USE_DIRECTIONAL_LOSS', '0')

    print('=' * 100)
    print('🎯 30G PILOT (yfinance, DB bağımsız)')
    print('=' * 100)
    print()
    print(f"Config: ML_HORIZONS={os.getenv('ML_HORIZONS')} ML_USE_DIRECTIONAL_LOSS={os.getenv('ML_USE_DIRECTIONAL_LOSS')}")
    print()

    ml = EnhancedMLSystem()
    results: dict[str, dict] = {}

    for sym in SYMBOLS:
        print(f"\n{'='*100}")
        print(f"Training: {sym}")
        print(f"{'='*100}\n")

        data = _fetch_yf(sym, limit_days=1000)
        if data is None or data.empty or len(data) < 200:
            print(f"❌ {sym} için yeterli veri yok ({0 if data is None else len(data)})")
            continue

        ok = ml.train_enhanced_models(sym, data)
        if not ok:
            print(f"❌ Eğitim başarısız: {sym}")
            continue
        perf = ml.model_performance.get(f"{sym}_30d", {})
        results[sym] = perf

    print('\nRESULTS:')
    for sym, perf in results.items():
        print(sym, json.dumps(perf))


if __name__ == '__main__':
    run_pilot()
