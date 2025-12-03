#!/usr/bin/env python3
"""
HPO study dosyalarından best trial'ın hangi filtre değerleriyle bulunduğunu
ve hangi split'lerin low support olduğunu analiz eder.

Kullanım:
    /opt/bist-pattern/venv/bin/python3 scripts/analyze_best_trial_filters.py --symbol ADEL --horizon 1
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import optuna

# Add project root to path
sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

HPO_STUDIES_DIR = Path('/opt/bist-pattern/hpo_studies')


def load_state() -> Dict:
    """Load pipeline state to get cycle"""
    state_file = Path('/opt/bist-pattern/results/hpo_state.json')
    if not state_file.exists():
        return {}
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def find_study_db(symbol: str, horizon: int, cycle: Optional[int] = None) -> Optional[Path]:
    """Find study database file"""
    if cycle is None:
        state = load_state()
        cycle = state.get('cycle', 1)
    
    if not HPO_STUDIES_DIR.exists():
        return None
    
    # Priority 1: Cycle format
    cycle_file = HPO_STUDIES_DIR / f"hpo_with_features_{symbol}_h{horizon}_c{cycle}.db"
    if cycle_file.exists():
        return cycle_file
    
    # Priority 2: Legacy format
    if cycle == 1:
        legacy_file = HPO_STUDIES_DIR / f"hpo_with_features_{symbol}_h{horizon}.db"
        if legacy_file.exists():
            return legacy_file
    
    # Priority 3: Any cycle format
    pattern = f"hpo_with_features_{symbol}_h{horizon}_c*.db"
    cycle_files = list(HPO_STUDIES_DIR.glob(pattern))
    if cycle_files:
        cycle_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cycle_files[0]
    
    return None


def analyze_best_trial_filters(db_file: Path, symbol: str, horizon: int) -> Dict:
    """Analyze best trial's filter usage and low support splits"""
    result = {
        'symbol': symbol,
        'horizon': horizon,
        'study_db': str(db_file),
        'best_trial_number': None,
        'best_value': None,
        'filter_used': {
            'min_mask_count': None,
            'min_mask_pct': None,
        },
        'split_analysis': [],
        'low_support_splits': [],
        'all_splits_included': True,
        'summary': {}
    }
    
    try:
        study = optuna.load_study(
            study_name=None,
            storage=f"sqlite:///{db_file}"
        )
        
        if study.best_trial is None:
            result['error'] = 'No best trial found'
            return result
        
        best_trial = study.best_trial
        result['best_trial_number'] = best_trial.number
        result['best_value'] = best_trial.value
        
        # Get symbol_metrics from user_attrs
        symbol_metrics = best_trial.user_attrs.get('symbol_metrics', {})
        symbol_key = f"{symbol}_{horizon}d"
        
        if symbol_key not in symbol_metrics:
            result['error'] = f'Symbol {symbol_key} not found in best trial metrics'
            return result
        
        symbol_data = symbol_metrics[symbol_key]
        split_metrics = symbol_data.get('split_metrics', [])
        
        if not split_metrics:
            result['error'] = 'No split metrics found'
            return result
        
        # Analyze each split
        total_splits = len(split_metrics)
        included_splits = 0
        excluded_splits = 0
        low_support_count = 0
        
        min_mask_count_values = set()
        min_mask_pct_values = set()
        
        for split in split_metrics:
            split_idx = split.get('split_index', 0)
            dirhit = split.get('dirhit')
            mask_count = split.get('mask_count', 0)
            mask_pct = split.get('mask_pct', 0.0)
            low_support = split.get('low_support', False)
            min_mask_count = split.get('min_mask_count', 0)
            min_mask_pct = split.get('min_mask_pct', 0.0)
            
            # Collect filter values used
            if min_mask_count is not None:
                min_mask_count_values.add(min_mask_count)
            if min_mask_pct is not None:
                min_mask_pct_values.add(min_mask_pct)
            
            split_info = {
                'split_index': split_idx,
                'dirhit': dirhit,
                'mask_count': mask_count,
                'mask_pct': mask_pct,
                'low_support': low_support,
                'min_mask_count': min_mask_count,
                'min_mask_pct': min_mask_pct,
                'included': dirhit is not None
            }
            
            result['split_analysis'].append(split_info)
            
            if low_support:
                low_support_count += 1
                result['low_support_splits'].append({
                    'split_index': split_idx,
                    'dirhit': dirhit,
                    'mask_count': mask_count,
                    'mask_pct': mask_pct
                })
            
            if dirhit is not None:
                included_splits += 1
            else:
                excluded_splits += 1
        
        # Determine filter values used
        if len(min_mask_count_values) == 1:
            result['filter_used']['min_mask_count'] = list(min_mask_count_values)[0]
        elif len(min_mask_count_values) > 1:
            result['filter_used']['min_mask_count'] = f"Mixed: {sorted(min_mask_count_values)}"
        
        if len(min_mask_pct_values) == 1:
            result['filter_used']['min_mask_pct'] = list(min_mask_pct_values)[0]
        elif len(min_mask_pct_values) > 1:
            result['filter_used']['min_mask_pct'] = f"Mixed: {sorted(min_mask_pct_values)}"
        
        # Summary
        result['summary'] = {
            'total_splits': total_splits,
            'included_splits': included_splits,
            'excluded_splits': excluded_splits,
            'low_support_count': low_support_count,
            'avg_dirhit': None,
            'included_dirhits': []
        }
        
        # Calculate average DirHit from included splits
        included_dirhits = [s['dirhit'] for s in result['split_analysis'] if s['dirhit'] is not None]
        if included_dirhits:
            result['summary']['included_dirhits'] = included_dirhits
            result['summary']['avg_dirhit'] = sum(included_dirhits) / len(included_dirhits)
        
        result['all_splits_included'] = (excluded_splits == 0)
        
    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()
    
    return result


def print_analysis(result: Dict):
    """Print analysis results"""
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        if 'traceback' in result:
            print(result['traceback'])
        return
    
    print("=" * 80)
    print(f"BEST TRIAL FILTER ANALYSIS: {result['symbol']} {result['horizon']}d")
    print("=" * 80)
    print(f"📁 Study DB: {result['study_db']}")
    print(f"🎯 Best Trial: #{result['best_trial_number']}")
    print(f"📊 Best Value (Score): {result['best_value']:.4f}" if result['best_value'] else "N/A")
    print()
    
    print("🔍 Filter Values Used During HPO:")
    print(f"   HPO_MIN_MASK_COUNT: {result['filter_used']['min_mask_count']}")
    print(f"   HPO_MIN_MASK_PCT: {result['filter_used']['min_mask_pct']}")
    print()
    
    print("📈 Split Analysis:")
    summary = result['summary']
    print(f"   Total Splits: {summary['total_splits']}")
    print(f"   Included Splits: {summary['included_splits']}")
    print(f"   Excluded Splits: {summary['excluded_splits']}")
    print(f"   Low Support Splits: {summary['low_support_count']}")
    print()
    
    if summary['avg_dirhit'] is not None:
        print(f"📊 Average DirHit (from included splits): {summary['avg_dirhit']:.2f}%")
        print(f"   DirHit values: {[f'{d:.2f}%' for d in summary['included_dirhits']]}")
        print()
    
    if result['low_support_splits']:
        print("⚠️  Low Support Splits (excluded from average):")
        for split in result['low_support_splits']:
            dirhit_str = f"{split['dirhit']:.2f}%" if split['dirhit'] is not None else "N/A"
            print(f"   Split {split['split_index']}: DirHit={dirhit_str}, "
                  f"mask_count={split['mask_count']}, mask_pct={split['mask_pct']:.1f}%")
        print()
    
    print("📋 Detailed Split Information:")
    for split in result['split_analysis']:
        status = "✅ INCLUDED" if split['included'] else "❌ EXCLUDED"
        low_sup = " (LOW SUPPORT)" if split['low_support'] else ""
        dirhit_str = f"{split['dirhit']:.2f}%" if split['dirhit'] is not None else "N/A"
        print(f"   Split {split['split_index']}: {status}{low_sup}")
        print(f"      DirHit: {dirhit_str}")
        print(f"      Mask Count: {split['mask_count']}, Mask %: {split['mask_pct']:.1f}%")
        print(f"      Filter: min_count={split['min_mask_count']}, min_pct={split['min_mask_pct']}")
    print()
    
    # Recommendation
    print("💡 Recommendation:")
    if result['all_splits_included']:
        print("   ✅ All splits were included in best trial calculation")
        print("   → Best params are valid regardless of filter")
    else:
        print(f"   ⚠️  {summary['excluded_splits']} split(s) were excluded due to low support")
        if result['filter_used']['min_mask_count'] == 10:
            print("   → Best params were found with HPO_MIN_MASK_COUNT=10, HPO_MIN_MASK_PCT=5.0")
            print("   → For consistency, retrain with same filter OR re-run HPO with 0/0.0")
        else:
            print("   → Best params were found with different filter")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze best trial filter usage from HPO study')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., ADEL)')
    parser.add_argument('--horizon', type=int, required=True, help='Horizon (e.g., 1)')
    parser.add_argument('--cycle', type=int, help='Cycle number (default: from state file)')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    # Find study DB
    study_db = find_study_db(args.symbol, args.horizon, args.cycle)
    if not study_db:
        print(f"❌ Study DB not found for {args.symbol} {args.horizon}d")
        if args.cycle:
            print(f"   (searched for cycle {args.cycle})")
        return 1
    
    # Analyze
    result = analyze_best_trial_filters(study_db, args.symbol, args.horizon)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_analysis(result)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

