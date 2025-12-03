#!/usr/bin/env python3
"""
Study dosyasından, belirli bir filtre (10/5.0 veya 0/0.0) uygulandıktan sonra
en iyi olan trial'ı bulur ve params'larını döner.

Kullanım:
    /opt/bist-pattern/venv/bin/python3 scripts/find_best_trial_with_filter.py --symbol EKGYO --horizon 1 --min_mask_count 10 --min_mask_pct 5.0
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
        import json
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


def apply_filter_to_trial(trial, symbol: str, horizon: int, 
                          min_mask_count: int, min_mask_pct: float) -> Optional[float]:
    """Apply filter to trial and return filtered score (DirHit average)
    
    Returns:
        Filtered average DirHit if trial has sufficient splits, None otherwise
    """
    symbol_key = f"{symbol}_{horizon}d"
    symbol_metrics = trial.user_attrs.get('symbol_metrics', {})
    
    if symbol_key not in symbol_metrics:
        return None
    
    split_metrics = symbol_metrics[symbol_key].get('split_metrics', [])
    if not split_metrics:
        return None
    
    # Apply filter: only include splits that meet min_mask_count and min_mask_pct
    filtered_dirhits = []
    
    for split in split_metrics:
        mask_count = split.get('mask_count', 0)
        mask_pct = split.get('mask_pct', 0.0)
        dirhit = split.get('dirhit')
        
        # Check if split meets filter criteria
        if dirhit is not None:
            if min_mask_count > 0 and mask_count < min_mask_count:
                continue  # Exclude this split
            if min_mask_pct > 0.0 and mask_pct < min_mask_pct:
                continue  # Exclude this split
            
            filtered_dirhits.append(dirhit)
    
    # Need at least 1 split to be valid
    if len(filtered_dirhits) == 0:
        return None
    
    # Return average DirHit of filtered splits
    return sum(filtered_dirhits) / len(filtered_dirhits)


def find_best_trial_with_filter(db_file: Path, symbol: str, horizon: int,
                                min_mask_count: int = 0, min_mask_pct: float = 0.0,
                                top_n: int = 5) -> List[Tuple[optuna.Trial, float]]:
    """Find best trials after applying filter
    
    Returns:
        List of (trial, filtered_score) tuples, sorted by filtered_score (best first)
    """
    try:
        study = optuna.load_study(
            study_name=None,
            storage=f"sqlite:///{db_file}"
        )
        
        symbol_key = f"{symbol}_{horizon}d"
        valid_trials = []
        
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            
            # Apply filter and get filtered score
            filtered_score = apply_filter_to_trial(
                trial, symbol, horizon, min_mask_count, min_mask_pct
            )
            
            if filtered_score is not None:
                valid_trials.append((trial, filtered_score))
        
        # Sort by filtered score (descending)
        valid_trials.sort(key=lambda x: x[1], reverse=True)
        
        return valid_trials[:top_n]
    
    except Exception as e:
        print(f"❌ Error finding best trial: {e}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        return []


def get_trial_params(trial: optuna.Trial) -> Dict:
    """Extract parameters from trial"""
    params = trial.params.copy()
    
    # Get features_enabled and feature_params from user_attrs
    features_enabled = trial.user_attrs.get('features_enabled', {})
    feature_params = trial.user_attrs.get('feature_params', {})
    
    if features_enabled:
        params['features_enabled'] = features_enabled
    if feature_params:
        params['feature_params'] = feature_params
    
    return params


def main():
    parser = argparse.ArgumentParser(
        description='Find best trial with filter applied to study'
    )
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--horizon', type=int, required=True, help='Horizon')
    parser.add_argument('--min_mask_count', type=int, default=0,
                       help='Minimum mask count filter (default: 0)')
    parser.add_argument('--min_mask_pct', type=float, default=0.0,
                       help='Minimum mask percentage filter (default: 0.0)')
    parser.add_argument('--top-n', type=int, default=5,
                       help='Show top N trials (default: 5)')
    parser.add_argument('--cycle', type=int, help='Cycle number (default: from state)')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    # Find study DB
    study_db = find_study_db(args.symbol, args.horizon, args.cycle)
    if not study_db:
        print(f"❌ Study DB not found for {args.symbol} {args.horizon}d")
        return 1
    
    print(f"📁 Study DB: {study_db}")
    print(f"🔍 Filter: min_mask_count={args.min_mask_count}, min_mask_pct={args.min_mask_pct}")
    print()
    
    # Find best trials with filter
    best_trials = find_best_trial_with_filter(
        study_db, args.symbol, args.horizon,
        args.min_mask_count, args.min_mask_pct,
        args.top_n
    )
    
    if not best_trials:
        print("❌ No valid trials found with filter applied")
        return 1
    
    print(f"✅ Found {len(best_trials)} valid trial(s) with filter applied")
    print()
    
    if args.json:
        import json
        result = {
            'symbol': args.symbol,
            'horizon': args.horizon,
            'filter': {
                'min_mask_count': args.min_mask_count,
                'min_mask_pct': args.min_mask_pct
            },
            'best_trials': []
        }
        
        for trial, filtered_score in best_trials:
            trial_info = {
                'trial_number': trial.number,
                'filtered_dirhit': filtered_score,
                'original_score': float(trial.value) if trial.value is not None else None,
                'params': get_trial_params(trial)
            }
            result['best_trials'].append(trial_info)
        
        print(json.dumps(result, indent=2))
    else:
        print("=" * 80)
        print(f"TOP {len(best_trials)} TRIALS WITH FILTER APPLIED")
        print("=" * 80)
        print()
        
        for idx, (trial, filtered_score) in enumerate(best_trials, 1):
            print(f"#{idx} Trial #{trial.number}")
            print(f"   Filtered DirHit (after filter): {filtered_score:.2f}%")
            print(f"   Original Score: {trial.value:.4f}" if trial.value is not None else "   Original Score: N/A")
            
            # Get split info
            symbol_key = f"{args.symbol}_{args.horizon}d"
            symbol_metrics = trial.user_attrs.get('symbol_metrics', {})
            if symbol_key in symbol_metrics:
                split_metrics = symbol_metrics[symbol_key].get('split_metrics', [])
                included = sum(1 for s in split_metrics if s.get('dirhit') is not None)
                total = len(split_metrics)
                print(f"   Splits: {included}/{total} included after filter")
            
            print()
        
        # Show best trial params
        best_trial, best_score = best_trials[0]
        print("=" * 80)
        print(f"BEST TRIAL: #{best_trial.number} (Filtered DirHit: {best_score:.2f}%)")
        print("=" * 80)
        print()
        print("📋 Parameters:")
        params = get_trial_params(best_trial)
        for key, value in sorted(params.items()):
            if key not in ('features_enabled', 'feature_params'):
                print(f"   {key}: {value}")
        print()
        print("💡 Use these parameters for training with the same filter!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

