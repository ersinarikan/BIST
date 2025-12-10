#!/usr/bin/env python3
"""
Find fallback best params for symbols with low support (all splits excluded by
filter).

This script finds best params using a 0/0.0 filter (no filter) when the
original filter excludes all splits. This ensures we have valid params even
for low-support symbols.
"""

import sys
from pathlib import Path
from typing import Optional, Dict

sys.path.insert(0, '/opt/bist-pattern')

try:
    import optuna
except ImportError:
    print("❌ optuna not installed", file=sys.stderr)
    sys.exit(1)


def find_fallback_best_params(
    study_db: Path,
    symbol: str,
    horizon: int,
) -> Optional[Dict]:
    """Find best params using 0/0.0 filter (no filter) as fallback

    Args:
        study_db: Path to Optuna study database
        symbol: Stock symbol
        horizon: Prediction horizon

    Returns:
        Dict with best_params, best_trial_number, features_enabled,
        feature_params or None if not found
    """
    try:
        study = optuna.load_study(
            study_name=None,
            storage=f"sqlite:///{study_db}"
        )

        symbol_key = f"{symbol}_{horizon}d"
        best_trial = None
        best_filtered_score = float('-inf')

        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            # Get split metrics
            symbol_metrics = trial.user_attrs.get('symbol_metrics', {})
            if symbol_key not in symbol_metrics:
                continue

            split_metrics = symbol_metrics[symbol_key].get('split_metrics', [])
            if not split_metrics:
                continue

            # Apply 0/0.0 filter (no filter) - include all splits
            filtered_dirhits = []
            for split in split_metrics:
                dirhit = split.get('dirhit')
                if dirhit is not None:
                    filtered_dirhits.append(dirhit)

            # Need at least 1 split
            if len(filtered_dirhits) == 0:
                continue

            # Calculate filtered average DirHit
            filtered_score = sum(filtered_dirhits) / len(filtered_dirhits)

            if filtered_score > best_filtered_score:
                best_filtered_score = filtered_score
                best_trial = trial

        if best_trial is None:
            return None

        # Extract params
        best_params = best_trial.params.copy()

        # Get features_enabled and feature_params from user_attrs
        features_enabled = best_trial.user_attrs.get('features_enabled', {})
        feature_params = best_trial.user_attrs.get('feature_params', {})

        return {
            'best_params': best_params,
            'best_trial_number': best_trial.number,
            'best_value': best_filtered_score,
            'features_enabled': features_enabled,
            'feature_params': feature_params,
            # Fallback filter
            'filter_used': {'min_mask_count': 0, 'min_mask_pct': 0.0},
            'is_fallback': True
        }
    except Exception as e:
        print(f"❌ Error finding fallback best params: {e}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Find fallback best params for low-support symbols'
    )
    parser.add_argument(
        '--study-db',
        type=str,
        required=True,
        help='Path to Optuna study database',
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Stock symbol',
    )
    parser.add_argument(
        '--horizon',
        type=int,
        required=True,
        help='Prediction horizon',
    )

    args = parser.parse_args()

    study_db = Path(args.study_db)
    if not study_db.exists():
        print(f"❌ Study DB not found: {study_db}", file=sys.stderr)
        sys.exit(1)

    result = find_fallback_best_params(study_db, args.symbol, args.horizon)

    if result:
        print(f"✅ Found fallback params for {args.symbol} {args.horizon}d")
        print(f"   Best trial: #{result['best_trial_number']}")
        print(f"   Best value: {result['best_value']:.2f}%")
        print(f"   Filter: {result['filter_used']}")
    else:
        print(f"❌ No fallback params found for {args.symbol} {args.horizon}d")
        sys.exit(1)
