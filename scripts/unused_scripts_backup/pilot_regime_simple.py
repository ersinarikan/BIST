#!/usr/bin/env python3
"""
Simple Regime Detection Pilot: Rule-Based

Detects 3 regimes:
1. Low Volatility (vol < 0.015)
2. Medium Volatility (0.015 <= vol < 0.03)
3. High Volatility (vol >= 0.03)

Tests regime-specific strategies
"""

import os
import sys
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, '/opt/bist-pattern')


SYMS = ['GARAN', 'AKBNK', 'EREGL']


def fetch_prices(engine, symbol: str) -> pd.DataFrame:
    """Fetch all available historical prices"""
    q = text(
        """
        SELECT p.date, p.open_price, p.high_price, p.low_price, p.close_price, p.volume
        FROM stock_prices p
        JOIN stocks s ON s.id = p.stock_id
        WHERE s.symbol = :sym
        ORDER BY p.date DESC
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(q, {"sym": symbol}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([
        {
            'date': r[0],
            'open': float(r[1]),
            'high': float(r[2]),
            'low': float(r[3]),
            'close': float(r[4]),
            'volume': float(r[5]) if r[5] is not None else 0.0,
        }
        for r in rows
    ])
    df = df.sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df[['open', 'high', 'low', 'close', 'volume']]


def detect_regime_simple(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Simple rule-based regime detection
    
    Returns:
        Series with regime labels: 'low_vol', 'medium_vol', 'high_vol'
    """
    # Calculate rolling volatility
    returns = df['close'].pct_change()
    volatility = returns.rolling(window).std()
    
    # Classify regimes
    regimes = pd.Series(index=df.index, dtype=str)
    regimes[volatility < 0.015] = 'low_vol'
    regimes[(volatility >= 0.015) & (volatility < 0.03)] = 'medium_vol'
    regimes[volatility >= 0.03] = 'high_vol'
    
    return regimes


def analyze_regimes(df: pd.DataFrame, regimes: pd.Series) -> dict:
    """Analyze regime characteristics"""
    results = {}
    
    for regime in ['low_vol', 'medium_vol', 'high_vol']:
        mask = regimes == regime
        if mask.sum() == 0:
            continue
        
        regime_df = df[mask]
        returns = regime_df['close'].pct_change()
        
        results[regime] = {
            'days': int(mask.sum()),
            'pct': float(mask.sum() / len(df) * 100),
            'mean_return': float(returns.mean()),
            'std_return': float(returns.std()),
            'sharpe': float(returns.mean() / returns.std()) if returns.std() > 0 else 0.0
        }
    
    return results


def run() -> None:
    os.environ.setdefault('DATABASE_URL', 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db')
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('❌ DATABASE_URL not set')
        sys.exit(1)
    engine = create_engine(db_url)
    
    print("🚀 Simple Regime Detection Pilot")
    print("="*80)
    
    all_results = []
    
    for sym in SYMS:
        print(f"\n📊 {sym}")
        df = fetch_prices(engine, sym)
        if df is None or df.empty or len(df) < 100:
            print("❌ Insufficient data")
            continue
        
        print(f"  Total days: {len(df)}")
        
        # Detect regimes
        regimes = detect_regime_simple(df, window=20)
        
        # Analyze
        analysis = analyze_regimes(df, regimes)
        
        print("\n  Regime Analysis:")
        print("  " + "-"*70)
        print(f"  {'Regime':<15} {'Days':<8} {'%':<8} {'Mean Ret':<12} {'Std':<10} {'Sharpe':<8}")
        print("  " + "-"*70)
        
        for regime, stats in analysis.items():
            print(f"  {regime:<15} {stats['days']:<8} {stats['pct']:<8.1f} "
                  f"{stats['mean_return']:<12.4f} {stats['std_return']:<10.4f} {stats['sharpe']:<8.2f}")
            
            all_results.append({
                'symbol': sym,
                'regime': regime,
                **stats
            })
        
        # Regime transitions
        transitions = (regimes != regimes.shift()).sum()
        print(f"\n  Regime transitions: {transitions}")
        print(f"  Avg regime duration: {len(df) / transitions:.1f} days")
    
    # Save results
    if all_results:
        out_dir = '/opt/bist-pattern/logs'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        pd.DataFrame(all_results).to_csv(out_path, index=False)
        print(f"\n✅ Results saved: {out_path}")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("1. Implement regime-specific model selection")
    print("2. Test regime-based hyperparameter tuning")
    print("3. Integrate with EnhancedMLSystem")
    print("="*80)


if __name__ == '__main__':
    run()
