#!/usr/bin/env python3
"""
Symbol Data Characteristics Analyzer

BESLR gibi semboller için neden çok az anlamlı tahmin olduğunu analiz eder.
Veri özelliklerini (volatilite, trend, vs.) inceler.

Kullanım:
    python3 scripts/analyze_symbol_data_characteristics.py BESLR 1
"""

import sys
import sqlite3
import json
from pathlib import Path
from typing import Dict, Optional, List, Any
import numpy as np
import pandas as pd
import optuna

sys.path.insert(0, '/opt/bist-pattern')

HPO_STUDIES_DIR = Path('/opt/bist-pattern/hpo_studies')
RESULTS_DIR = Path('/opt/bist-pattern/results')


def load_state() -> Dict:
    """Load pipeline state"""
    state_file = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
    if not state_file.exists():
        return {}
    try:
        with open(state_file, 'r') as f:
            content = f.read().strip()
            if content.count('{') > 1:
                last_brace = content.rfind('}')
                if last_brace > 0:
                    brace_count = 0
                    start_pos = last_brace
                    for i in range(last_brace, -1, -1):
                        if content[i] == '}':
                            brace_count += 1
                        elif content[i] == '{':
                            brace_count -= 1
                            if brace_count == 0:
                                start_pos = i
                                break
                    content = content[start_pos:last_brace+1]
            return json.loads(content)
    except Exception:
        return {}


def analyze_trials_mask_counts(db_file: Path, symbol: str, horizon: int) -> Dict[str, Any]:
    """Analyze mask counts across all trials"""
    result = {
        'total_trials': 0,
        'complete_trials': 0,
        'trials_with_metrics': 0,
        'mask_count_stats': {
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'percentiles': {}
        },
        'low_support_trials': [],
        'high_support_trials': []
    }
    
    try:
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_file}")
        
        result['total_trials'] = len(study.trials)
        result['complete_trials'] = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        
        mask_counts = []
        
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            
            symbol_metrics = trial.user_attrs.get('symbol_metrics')
            if not symbol_metrics:
                continue
            
            symbol_key = f"{symbol}_{horizon}d"
            if symbol_key not in symbol_metrics:
                continue
            
            result['trials_with_metrics'] += 1
            metrics = symbol_metrics[symbol_key]
            split_metrics = metrics.get('split_metrics', [])
            
            # Calculate total mask_count across all splits
            total_mask_count = sum(s.get('mask_count', 0) for s in split_metrics)
            avg_mask_pct = np.mean([s.get('mask_pct', 0.0) for s in split_metrics if s.get('mask_pct') is not None])
            
            mask_counts.append({
                'trial_number': trial.number,
                'mask_count': total_mask_count,
                'mask_pct': avg_mask_pct,
                'dirhit': metrics.get('avg_dirhit'),
                'split_count': len(split_metrics)
            })
            
            if total_mask_count < 10 or avg_mask_pct < 5.0:
                result['low_support_trials'].append({
                    'trial': trial.number,
                    'mask_count': total_mask_count,
                    'mask_pct': avg_mask_pct,
                    'dirhit': metrics.get('avg_dirhit')
                })
            else:
                result['high_support_trials'].append({
                    'trial': trial.number,
                    'mask_count': total_mask_count,
                    'mask_pct': avg_mask_pct,
                    'dirhit': metrics.get('avg_dirhit')
                })
        
        if mask_counts:
            mask_count_values = [m['mask_count'] for m in mask_counts]
            result['mask_count_stats'] = {
                'min': int(np.min(mask_count_values)),
                'max': int(np.max(mask_count_values)),
                'mean': float(np.mean(mask_count_values)),
                'median': float(np.median(mask_count_values)),
                'percentiles': {
                    'p10': float(np.percentile(mask_count_values, 10)),
                    'p25': float(np.percentile(mask_count_values, 25)),
                    'p50': float(np.percentile(mask_count_values, 50)),
                    'p75': float(np.percentile(mask_count_values, 75)),
                    'p90': float(np.percentile(mask_count_values, 90)),
                    'p95': float(np.percentile(mask_count_values, 95)),
                    'p99': float(np.percentile(mask_count_values, 99))
                }
            }
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def analyze_price_data(symbol: str, horizon: int) -> Dict[str, Any]:
    """Analyze price data characteristics"""
    result = {
        'symbol': symbol,
        'horizon': horizon,
        'data_available': False,
        'total_days': 0,
        'volatility': None,
        'mean_return': None,
        'std_return': None,
        'significant_returns_pct': None,
        'trend': None
    }
    
    try:
        from sqlalchemy import create_engine
        from bist_pattern.core.config_manager import ConfigManager
        
        config = ConfigManager()
        db_url = config.get('DATABASE_URL')
        if not db_url:
            result['error'] = 'DATABASE_URL not found'
            return result
        
        engine = create_engine(db_url)
        
        # Fetch prices
        query = f"""
            SELECT date, close
            FROM prices
            WHERE symbol = '{symbol}'
            ORDER BY date ASC
        """
        df = pd.read_sql(query, engine)
        
        if len(df) == 0:
            result['error'] = 'No price data found'
            return result
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        result['data_available'] = True
        result['total_days'] = len(df)
        
        # Calculate returns
        returns = df['close'].pct_change().dropna()
        result['mean_return'] = float(returns.mean() * 100)
        result['std_return'] = float(returns.std() * 100)
        result['volatility'] = float(returns.std() * np.sqrt(252) * 100)  # Annualized volatility
        
        # Calculate forward returns (for horizon)
        forward_returns = (df['close'].shift(-horizon) / df['close'] - 1.0).dropna()
        significant_returns = forward_returns[abs(forward_returns) > 0.005]
        result['significant_returns_pct'] = float(len(significant_returns) / len(forward_returns) * 100) if len(forward_returns) > 0 else 0.0
        
        # Trend analysis
        if len(df) > 0:
            first_price = df['close'].iloc[0]
            last_price = df['close'].iloc[-1]
            total_return = (last_price / first_price - 1.0) * 100
            result['trend'] = {
                'total_return_pct': float(total_return),
                'first_price': float(first_price),
                'last_price': float(last_price),
                'first_date': str(df.index[0]),
                'last_date': str(df.index[-1])
            }
    
    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()
    
    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/analyze_symbol_data_characteristics.py SYMBOL HORIZON")
        print("Example: python3 scripts/analyze_symbol_data_characteristics.py BESLR 1")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    horizon = int(sys.argv[2])
    
    import json
    
    print("=" * 100)
    print(f"🔍 SYMBOL DATA CHARACTERISTICS ANALİZİ: {symbol}_{horizon}d")
    print("=" * 100)
    print()
    
    # Get cycle
    state = load_state()
    cycle = state.get('cycle', 1)
    print(f"🔄 Cycle: {cycle}")
    print()
    
    # 1. Analyze price data
    print("📊 1. PRICE DATA ANALİZİ")
    print("-" * 100)
    price_analysis = analyze_price_data(symbol, horizon)
    
    if price_analysis.get('error'):
        print(f"   ❌ Error: {price_analysis['error']}")
        print(f"   ⚠️ Skipping price data analysis")
        price_analysis['significant_returns_pct'] = None
    else:
        print(f"   Total Days: {price_analysis['total_days']}")
        print(f"   Mean Daily Return: {price_analysis['mean_return']:.4f}%")
        print(f"   Std Daily Return: {price_analysis['std_return']:.4f}%")
        print(f"   Annualized Volatility: {price_analysis['volatility']:.2f}%")
        print(f"   Significant Returns (>0.5%): {price_analysis['significant_returns_pct']:.1f}%")
        
        if price_analysis.get('trend'):
            trend = price_analysis['trend']
            print(f"   Trend:")
            print(f"      Period: {trend['first_date']} to {trend['last_date']}")
            print(f"      Total Return: {trend['total_return_pct']:.2f}%")
            print(f"      First Price: {trend['first_price']:.2f}")
            print(f"      Last Price: {trend['last_price']:.2f}")
        
        # Interpretation
        if price_analysis['significant_returns_pct'] < 10:
            print(f"   ⚠️ WARNING: Only {price_analysis['significant_returns_pct']:.1f}% of returns are significant (>0.5%)")
            print(f"      This explains why mask_count is low in HPO trials!")
    print()
    
    # 2. Analyze trials mask counts
    print("🔬 2. HPO TRIALS MASK COUNT ANALİZİ")
    print("-" * 100)
    db_file = HPO_STUDIES_DIR / f"hpo_with_features_{symbol}_h{horizon}_c{cycle}.db"
    if not db_file.exists():
        # Try legacy format
        if cycle == 1:
            db_file = HPO_STUDIES_DIR / f"hpo_with_features_{symbol}_h{horizon}.db"
        # Try cycle 2 if cycle 1 doesn't work
        if not db_file.exists():
            db_file = HPO_STUDIES_DIR / f"hpo_with_features_{symbol}_h{horizon}_c2.db"
    
    if db_file.exists():
        trials_analysis = analyze_trials_mask_counts(db_file, symbol, horizon)
        
        print(f"   Total Trials: {trials_analysis['total_trials']}")
        print(f"   Complete Trials: {trials_analysis['complete_trials']}")
        print(f"   Trials with Metrics: {trials_analysis['trials_with_metrics']}")
        
        if trials_analysis.get('mask_count_stats', {}).get('min') is not None:
            stats = trials_analysis['mask_count_stats']
            print(f"   Mask Count Statistics:")
            print(f"      Min: {stats['min']}")
            print(f"      Max: {stats['max']}")
            print(f"      Mean: {stats['mean']:.1f}")
            print(f"      Median: {stats['median']:.1f}")
            print(f"      Percentiles:")
            print(f"         P10: {stats['percentiles']['p10']:.1f}")
            print(f"         P25: {stats['percentiles']['p25']:.1f}")
            print(f"         P50: {stats['percentiles']['p50']:.1f}")
            print(f"         P75: {stats['percentiles']['p75']:.1f}")
            print(f"         P90: {stats['percentiles']['p90']:.1f}")
            print(f"         P95: {stats['percentiles']['p95']:.1f}")
            print(f"         P99: {stats['percentiles']['p99']:.1f}")
        
        print(f"   Low Support Trials (<10 mask or <5%): {len(trials_analysis['low_support_trials'])}")
        if trials_analysis['low_support_trials']:
            print(f"      Examples (first 5):")
            for trial_info in trials_analysis['low_support_trials'][:5]:
                print(f"         Trial {trial_info['trial']}: mask_count={trial_info['mask_count']}, mask_pct={trial_info['mask_pct']:.1f}%, dirhit={trial_info['dirhit']:.1f}%")
        
        print(f"   High Support Trials (>=10 mask and >=5%): {len(trials_analysis['high_support_trials'])}")
        if trials_analysis['high_support_trials']:
            print(f"      Examples (first 5):")
            for trial_info in trials_analysis['high_support_trials'][:5]:
                print(f"         Trial {trial_info['trial']}: mask_count={trial_info['mask_count']}, mask_pct={trial_info['mask_pct']:.1f}%, dirhit={trial_info['dirhit']:.1f}%")
        
        # Interpretation
        if len(trials_analysis['low_support_trials']) > len(trials_analysis['high_support_trials']):
            print(f"   ⚠️ WARNING: Most trials have low support!")
            print(f"      This is a data characteristic issue, not an HPO problem.")
            print(f"      The symbol has very few significant price movements.")
    else:
        print(f"   ⚠️ Study database not found")
    print()
    
    # 3. Summary
    print("📋 3. ÖZET VE YORUM")
    print("-" * 100)
    
    significant_returns_pct = price_analysis.get('significant_returns_pct')
    if significant_returns_pct is not None and significant_returns_pct < 10:
        print(f"   ✅ SORUN TESPİT EDİLDİ:")
        print(f"      BESLR sembolü çok az anlamlı fiyat hareketi gösteriyor.")
        print(f"      Sadece {price_analysis['significant_returns_pct']:.1f}% günlük getiri >0.5% threshold'unu geçiyor.")
        print(f"      Bu yüzden HPO'da mask_count çok düşük.")
        print()
        print(f"   💡 ÇÖZÜM ÖNERİLERİ:")
        print(f"      1. Bu sembol için DirHit threshold'u düşürülebilir (0.005 → 0.003)")
        print(f"      2. Veya bu sembol için farklı bir metrik kullanılabilir (RMSE, MAPE)")
        print(f"      3. Veya bu sembol HPO'dan çıkarılabilir (çok az sinyal)")
    
    if trials_analysis.get('trials_with_metrics', 0) > 0:
        low_support_ratio = len(trials_analysis.get('low_support_trials', [])) / trials_analysis['trials_with_metrics'] * 100
        if low_support_ratio > 50:
            print(f"   ⚠️ {low_support_ratio:.1f}% of trials have low support")
            print(f"      This confirms it's a data issue, not an HPO issue")
    
    print()
    print("=" * 100)


if __name__ == '__main__':
    main()

