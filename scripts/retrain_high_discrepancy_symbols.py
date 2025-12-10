#!/usr/bin/env python3
"""
Farkların çok olduğu sembolleri tespit edip, study dosyalarından best params
ile düzeltilmiş filtre (low support gating) ile tekrar eğitir.

Kullanım:
    /opt/bist-pattern/venv/bin/python3
    scripts/retrain_high_discrepancy_symbols.py --threshold 30.0
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

try:
    import optuna
except ImportError:
    # Try to use venv python
    venv_python = '/opt/bist-pattern/venv/bin/python3'
    if os.path.exists(venv_python):
        print(
            f"⚠️ Optuna not found, trying venv python: {venv_python}",
            file=sys.stderr
        )
        # Re-execute with venv python
        os.execv(venv_python, [venv_python] + sys.argv)
    else:
        raise

# Add project root to path
sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

# ✅ FIX: Set DATABASE_URL from secret file (same as systemd service)
if 'DATABASE_URL' not in os.environ:
    secret_file = Path('/opt/bist-pattern/.secrets/db_password')
    if secret_file.exists():
        with open(secret_file, 'r') as f:
            db_password = f.read().strip()
        os.environ['DATABASE_URL'] = (
            f"postgresql://bist_user:{db_password}@127.0.0.1:6432/"
            f"bist_pattern_db"
        )
    else:
        print(
            "⚠️ Warning: DATABASE_URL not set and secret file not found",
            file=sys.stderr
        )
else:
    # Fix port if it's 5432 (should be 6432)
    db_url = os.environ.get('DATABASE_URL', '')
    if ':5432/' in db_url:
        os.environ['DATABASE_URL'] = db_url.replace(':5432/', ':6432/')
        print("⚠️ Fixed DATABASE_URL port from 5432 to 6432", file=sys.stderr)

from scripts.continuous_hpo_training_pipeline import (  # noqa: E402
    ContinuousHPOPipeline,
    STATE_FILE
)
from app import app  # noqa: E402

# HPO Studies directory
HPO_STUDIES_DIR = Path('/opt/bist-pattern/hpo_studies')
RESULTS_DIR = Path('/opt/bist-pattern/results/hpo_results')


def load_state() -> Dict:
    """Load pipeline state"""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def find_study_db(
    symbol: str, horizon: int, cycle: Optional[int] = None
) -> Optional[Path]:
    """Find study database file for symbol-horizon"""
    if cycle is None:
        state = load_state()
        cycle = state.get('cycle', 1)

    if not HPO_STUDIES_DIR.exists():
        return None

    # Priority 1: Cycle format
    cycle_file = (
        HPO_STUDIES_DIR /
        f"hpo_with_features_{symbol}_h{horizon}_c{cycle}.db"
    )
    if cycle_file.exists():
        return cycle_file

    # Priority 2: Legacy format (only for cycle 1)
    if cycle == 1:
        legacy_file = (
            HPO_STUDIES_DIR /
            f"hpo_with_features_{symbol}_h{horizon}.db"
        )
        if legacy_file.exists():
            return legacy_file

    # Priority 3: Any cycle format (fallback)
    pattern = f"hpo_with_features_{symbol}_h{horizon}_c*.db"
    cycle_files = list(HPO_STUDIES_DIR.glob(pattern))
    if cycle_files:
        # Get most recent
        cycle_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cycle_files[0]

    return None


def find_best_trial_with_filter_applied(
    db_file: Path,
    symbol: str,
    horizon: int,
    min_mask_count: int,
    min_mask_pct: float
) -> Tuple[Optional[optuna.trial.FrozenTrial], Optional[float]]:
    """Find best trial after applying filter to study (not JSON's best
    trial)"""
    try:
        import optuna
        study = optuna.load_study(
            study_name=None,
            storage=f"sqlite:///{db_file}"
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

            # Apply filter: only include splits that meet criteria
            filtered_dirhits = []
            for split in split_metrics:
                mask_count = split.get('mask_count', 0)
                mask_pct = split.get('mask_pct', 0.0)
                dirhit = split.get('dirhit')

                if dirhit is not None:
                    if min_mask_count > 0 and mask_count < min_mask_count:
                        continue
                    if min_mask_pct > 0.0 and mask_pct < min_mask_pct:
                        continue
                    filtered_dirhits.append(dirhit)

            # ✅ FIX: Need at least 2 splits for reliable DirHit
            # calculation
            # Single split DirHit is statistically unreliable
            if len(filtered_dirhits) < 2:
                continue

            # Calculate filtered average DirHit
            filtered_score = sum(filtered_dirhits) / len(filtered_dirhits)

            if filtered_score > best_filtered_score:
                best_filtered_score = filtered_score
                best_trial = trial

        return best_trial, best_filtered_score if best_trial else None

    except Exception as e:
        print(
            f"❌ Error finding best trial with filter: {e}",
            file=sys.stderr
        )
        return None, None


def get_best_params_from_study(
    db_file: Path,
    symbol: str,
    horizon: int,
    use_filtered: bool = True,
    cycle: Optional[int] = None
) -> Optional[Dict]:
    """Get best parameters from Optuna study, including filter info"""
    try:
        study = optuna.load_study(
            study_name=None,
            storage=f"sqlite:///{db_file}"
        )

        if study.best_trial is None:
            return None

        best_params = study.best_trial.params.copy()
        best_value = study.best_value

        # Get best trial number
        best_trial_number = study.best_trial.number

        # Get features_enabled from best trial user_attrs if available
        features_enabled = {}
        if hasattr(study.best_trial, 'user_attrs'):
            features_enabled = study.best_trial.user_attrs.get(
                'features_enabled', {}
            )

        # Get feature_params from best trial user_attrs if available
        feature_params = {}
        if hasattr(study.best_trial, 'user_attrs'):
            feature_params = study.best_trial.user_attrs.get(
                'feature_params', {}
            )

        # ✅ NEW: Get filter info from split_metrics
        filter_used = {'min_mask_count': None, 'min_mask_pct': None}
        split_analysis = {
            'total_splits': 0,
            'included_splits': 0,
            'excluded_splits': 0
        }

        try:
            symbol_metrics = study.best_trial.user_attrs.get(
                'symbol_metrics', {}
            )
            symbol_key = f"{symbol}_{horizon}d"
            if symbol_key in symbol_metrics:
                split_metrics = symbol_metrics[symbol_key].get(
                    'split_metrics', []
                )
                split_analysis['total_splits'] = len(split_metrics)

                min_mask_count_values = set()
                min_mask_pct_values = set()

                for split in split_metrics:
                    dirhit = split.get('dirhit')
                    if dirhit is not None:
                        split_analysis['included_splits'] += 1
                    else:
                        split_analysis['excluded_splits'] += 1

                    min_mask_count = split.get('min_mask_count')
                    min_mask_pct = split.get('min_mask_pct')
                    if min_mask_count is not None:
                        min_mask_count_values.add(min_mask_count)
                    if min_mask_pct is not None:
                        min_mask_pct_values.add(min_mask_pct)

                if len(min_mask_count_values) == 1:
                    filter_used['min_mask_count'] = list(
                        min_mask_count_values
                    )[0]
                if len(min_mask_pct_values) == 1:
                    filter_used['min_mask_pct'] = list(min_mask_pct_values)[0]
        except Exception:
            pass  # Filter info not critical

        # ✅ CRITICAL: If use_filtered=True, find best trial AFTER applying
        # filter
        # This ensures we use the "real" best params for the filtered splits
        filtered_trial = None
        filtered_score = None

        if use_filtered and filter_used['min_mask_count'] is not None:
            min_count = filter_used['min_mask_count']
            min_pct = (
                filter_used['min_mask_pct']
                if filter_used['min_mask_pct'] is not None else 0.0
            )

            if min_count > 0 or min_pct > 0.0:
                print(
                    f"   🔍 Finding best trial with filter applied "
                    f"(min_count={min_count}, min_pct={min_pct})..."
                )
                filtered_trial, filtered_score = (
                    find_best_trial_with_filter_applied(
                        db_file, symbol, horizon, min_count, min_pct
                    )
                )

                if filtered_trial:
                    print(
                        f"   ✅ Found filtered best trial: "
                        f"#{filtered_trial.number} "
                        f"(filtered DirHit: {filtered_score:.2f}%)"
                    )
                    print(
                        f"   📊 Original best trial: #{best_trial_number} "
                        f"(original score: {best_value:.4f})"
                    )

                    # Use filtered trial's params instead
                    best_params = filtered_trial.params.copy()
                    best_value = filtered_score
                    best_trial_number = filtered_trial.number

                    # Get features from filtered trial
                    if hasattr(filtered_trial, 'user_attrs'):
                        features_enabled = filtered_trial.user_attrs.get(
                            'features_enabled', {}
                        )
                        feature_params = filtered_trial.user_attrs.get(
                            'feature_params', {}
                        )

                    # ✅ CRITICAL: Update JSON file with filtered trial info
                    # for consistency
                    # This ensures future retraining uses the correct best
                    # params
                    json_file = find_hpo_json(symbol, horizon, cycle)
                    if json_file:
                        print(
                            f"   📝 Updating JSON file with filtered best "
                            f"trial: {json_file.name}"
                        )
                        filtered_trial_data = {
                            'best_trial_number': best_trial_number,
                            'best_value': best_value,
                            'best_dirhit': filtered_score,
                            'best_params': best_params,
                            'features_enabled': features_enabled,
                            'feature_params': feature_params,
                            'filter_used': filter_used
                        }
                        if update_hpo_json_with_filtered_trial(
                            json_file, filtered_trial_data
                        ):
                            print("   ✅ JSON file updated successfully")
                        else:
                            print(
                                "   ⚠️  Failed to update JSON file "
                                "(will continue with filtered params)"
                            )
                    else:
                        print(
                            "   ⚠️  JSON file not found, skipping update"
                        )
                else:
                    print(
                        "   ⚠️  No valid trial found with filter, using "
                        "original best trial"
                    )

        return {
            'best_params': best_params,
            'best_value': best_value,
            'best_trial_number': best_trial_number,
            'features_enabled': features_enabled,
            'feature_params': feature_params,
            'filter_used': filter_used,
            'split_analysis': split_analysis,
            'is_filtered': filtered_trial is not None
        }
    except Exception as e:
        print(
            f"❌ Error loading study {db_file}: {e}",
            file=sys.stderr
        )
        return None


def find_hpo_json(
    symbol: str, horizon: int, cycle: Optional[int] = None
) -> Optional[Path]:
    """Find HPO JSON result file"""
    if not RESULTS_DIR.exists():
        return None

    # Look for JSON files matching symbol and horizon
    if cycle:
        # Try cycle-specific pattern first
        pattern = (
            f"optuna_pilot_features_on_h{horizon}_c{cycle}_*{symbol}*.json"
        )
        json_files = list(RESULTS_DIR.glob(pattern))
        if json_files:
            json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return json_files[0]

    # Fallback: any pattern
    pattern = f"*{symbol}*{horizon}d*.json"
    json_files = list(RESULTS_DIR.glob(pattern))

    if not json_files:
        return None

    # Get most recent
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files[0]


def update_hpo_json_with_filtered_trial(
    json_file: Path, filtered_trial_data: Dict
) -> bool:
    """Update HPO JSON file with filtered best trial information"""
    try:
        import json
        from datetime import datetime

        # Read existing JSON
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Update with filtered trial data
        data['best_trial_number'] = filtered_trial_data['best_trial_number']
        data['best_value'] = filtered_trial_data['best_value']
        data['best_dirhit'] = filtered_trial_data.get(
            'best_dirhit', filtered_trial_data['best_value']
        )
        data['best_params'] = filtered_trial_data['best_params']

        # Update best_trial dict if exists
        if 'best_trial' in data:
            data['best_trial']['number'] = filtered_trial_data[
                'best_trial_number'
            ]
            data['best_trial']['value'] = float(
                filtered_trial_data['best_value']
            )

        # Update feature flags and params if available
        if 'features_enabled' in filtered_trial_data:
            data['features_enabled'] = filtered_trial_data['features_enabled']
        if 'feature_params' in filtered_trial_data:
            data['feature_params'] = filtered_trial_data['feature_params']

        # Add metadata about filter application
        data['filter_applied'] = {
            'min_mask_count': filtered_trial_data.get(
                'filter_used', {}
            ).get('min_mask_count'),
            'min_mask_pct': filtered_trial_data.get(
                'filter_used', {}
            ).get('min_mask_pct'),
            'applied_at': datetime.now().isoformat(),
            'note': 'Best trial found after applying filter to study'
        }

        # Write back
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)

        return True
    except Exception as e:
        print(
            f"❌ Error updating JSON file {json_file}: {e}",
            file=sys.stderr
        )
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        return False


def parse_discrepancy_data(
    discrepancy_text: str
) -> List[Tuple[str, int, float, float, float]]:
    """Parse discrepancy data from user input

    Returns: List of (symbol, horizon, hpo_dirhit, training_dirhit, diff)
    """
    discrepancies = []

    for line in discrepancy_text.strip().split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue

        try:
            # Format: SYMBOL_1d: HPO DirHit=X.XX% Training DirHit=Y.YY%
            parts = line.split(':')
            if len(parts) < 2:
                continue

            symbol_horizon = parts[0].strip()
            metrics = parts[1].strip()

            # Parse symbol and horizon
            if '_' in symbol_horizon:
                symbol, horizon_str = symbol_horizon.rsplit('_', 1)
                horizon = int(horizon_str.replace('d', ''))
            else:
                continue

            # Parse metrics
            hpo_dirhit = None
            training_dirhit = None

            if 'HPO DirHit=' in metrics:
                hpo_part = metrics.split('HPO DirHit=')[1].split()[0]
                hpo_dirhit = float(hpo_part.replace('%', ''))

            if 'Training DirHit=' in metrics:
                train_part = metrics.split('Training DirHit=')[1].split()[0]
                if train_part != 'LOW_SUPPORT':
                    training_dirhit = float(train_part.replace('%', ''))

            if hpo_dirhit is not None and training_dirhit is not None:
                diff = hpo_dirhit - training_dirhit
                discrepancies.append(
                    (symbol, horizon, hpo_dirhit, training_dirhit, diff)
                )
        except Exception as e:
            print(
                f"⚠️ Error parsing line '{line}': {e}",
                file=sys.stderr
            )
            continue

    return discrepancies


def retrain_symbol(
    symbol: str,
    horizon: int,
    best_params_data: Dict,
    pipeline: ContinuousHPOPipeline
) -> bool:
    """Retrain symbol with best params using fixed filter"""
    try:
        print(f"🔄 Retraining {symbol} {horizon}d with best params...")

        # Extract best params
        best_params = best_params_data.get('best_params', {})
        if not best_params:
            print(f"❌ {symbol} {horizon}d: No best params found")
            return False

        # Prepare hpo_result dict (for evaluation)
        hpo_result = {
            'best_trial_number': best_params_data.get('best_trial_number'),
            'features_enabled': best_params_data.get('features_enabled', {}),
            'feature_params': best_params_data.get('feature_params', {})
        }

        # Add features_enabled and feature_params to best_params
        best_params_with_metadata = best_params.copy()
        if hpo_result['features_enabled']:
            best_params_with_metadata['features_enabled'] = (
                hpo_result['features_enabled']
            )
        if hpo_result['feature_params']:
            best_params_with_metadata['feature_params'] = (
                hpo_result['feature_params']
            )

        # ✅ CRITICAL: Use the SAME filter that HPO used when finding best
        # params
        # This ensures we're testing the "real" best params that HPO found
        # If HPO used 10/5.0 filter, training should also use 10/5.0 filter
        # If HPO used 0/0.0 filter, training should also use 0/0.0 filter
        # This way we can see if the discrepancy is due to filter mismatch or
        # other issues

        # Filter values are already set in environment from
        # get_best_params_from_study
        # They will be used by _evaluate_training_dirhits in
        # continuous_hpo_training_pipeline.py
        print(
            f"   🔧 Training will use filter: "
            f"HPO_MIN_MASK_COUNT={os.environ.get('HPO_MIN_MASK_COUNT', '0')}, "
            f"HPO_MIN_MASK_PCT={os.environ.get('HPO_MIN_MASK_PCT', '0.0')}"
        )

        result = pipeline.run_training(
            symbol, horizon, best_params_with_metadata, hpo_result=hpo_result
        )

        if result is None:
            print(f"❌ {symbol} {horizon}d: Training failed")
            return False

        # Get DirHit results
        wfv_dirhit = result.get('wfv')
        if wfv_dirhit:
            print(
                f"✅ {symbol} {horizon}d: Training completed - "
                f"WFV DirHit={wfv_dirhit:.2f}%"
            )
        else:
            print(f"✅ {symbol} {horizon}d: Training completed")

        return True
    except Exception as e:
        print(
            f"❌ {symbol} {horizon}d: Error during retraining: {e}",
            file=sys.stderr
        )
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Retrain symbols with high HPO-Training discrepancy'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=30.0,
        help='Minimum DirHit difference threshold (default: 30.0)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Specific symbols to retrain (e.g., ADEL BRSAN)'
    )
    parser.add_argument(
        '--horizons',
        type=int,
        nargs='+',
        default=[1],
        help='Horizons to retrain (default: 1)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run - only show what would be retrained'
    )

    args = parser.parse_args()

    # User-provided discrepancy data
    discrepancy_data = """
   ADEL_1d: HPO DirHit=85.42% Training DirHit=42.21%
   AKENR_1d: HPO DirHit=100.00% Training DirHit=50.00%
   ANSGR_1d: HPO DirHit=84.78% Training DirHit=53.49%
   BRSAN_1d: HPO DirHit=100.00% Training DirHit=64.41%
   EKGYO_1d: HPO DirHit=100.00% Training DirHit=58.18%
   CONSE_1d: HPO DirHit=81.92% Training DirHit=40.00%
   BRKSN_1d: HPO DirHit=73.68% Training DirHit=35.98%
   BINHO_1d: HPO DirHit=76.25% Training DirHit=54.08%
   BLUME_1d: HPO DirHit=74.09% Training DirHit=56.94%
   BNTAS_1d: HPO DirHit=90.91% Training DirHit=76.36%
   BULGS_1d: HPO DirHit=87.50% Training DirHit=63.64%
   CANTE_1d: HPO DirHit=100.00% Training DirHit=76.92%
   CATES_1d: HPO DirHit=81.67% Training DirHit=53.85%
   DGNMO_1d: HPO DirHit=90.00% Training DirHit=54.55%
   DZGYO_1d: HPO DirHit=90.00% Training DirHit=61.11%
   EBEBK_1d: HPO DirHit=80.00% Training DirHit=50.00%
    """

    # Parse discrepancies
    discrepancies = parse_discrepancy_data(discrepancy_data)

    # Filter by threshold
    high_discrepancies = [
        d for d in discrepancies if d[4] >= args.threshold
    ]

    # Filter by symbols if specified
    if args.symbols:
        high_discrepancies = [
            d for d in high_discrepancies if d[0] in args.symbols
        ]

    # Filter by horizons if specified
    if args.horizons:
        high_discrepancies = [
            d for d in high_discrepancies if d[1] in args.horizons
        ]

    if not high_discrepancies:
        print("ℹ️ No symbols found matching criteria")
        return

    print("=" * 80)
    print("HIGH DISCREPANCY SYMBOLS - RETRAINING")
    print("=" * 80)
    print(
        f"📊 Found {len(high_discrepancies)} symbols with discrepancy >= "
        f"{args.threshold}%"
    )
    print()

    # Sort by discrepancy (highest first)
    high_discrepancies.sort(key=lambda x: x[4], reverse=True)

    print("📋 Symbols to retrain:")
    for symbol, horizon, hpo_dirhit, train_dirhit, diff in high_discrepancies:
        print(
            f"   {symbol}_{horizon}d: HPO={hpo_dirhit:.2f}% → "
            f"Training={train_dirhit:.2f}% (diff={diff:.2f}%)"
        )
    print()

    if args.dry_run:
        print("🔍 DRY RUN - No training will be performed")
        return

    # Initialize pipeline
    pipeline = ContinuousHPOPipeline()

    # Load state to get cycle
    state = load_state()
    cycle = state.get('cycle', 1)

    print(f"🔄 Cycle: {cycle}")
    print()

    # Retrain each symbol
    success_count = 0
    fail_count = 0

    with app.app_context():
        for (
            symbol, horizon, hpo_dirhit, train_dirhit, diff
        ) in high_discrepancies:
            print(f"\n{'=' * 80}")
            print(f"Processing: {symbol} {horizon}d (diff={diff:.2f}%)")
            print(f"{'=' * 80}")

            # Find study DB
            study_db = find_study_db(symbol, horizon, cycle)
            if not study_db:
                print(
                    f"⚠️ {symbol} {horizon}d: Study DB not found, trying to "
                    f"find HPO JSON..."
                )
                # Try to find HPO JSON instead
                hpo_json = find_hpo_json(symbol, horizon)
                if hpo_json:
                    print(f"✅ Found HPO JSON: {hpo_json}")
                    try:
                        with open(hpo_json, 'r') as f:
                            json_data = json.load(f)
                            best_params_data = {
                                'best_params': json_data.get(
                                    'best_params', {}
                                ),
                                'best_value': json_data.get('best_value'),
                                'best_trial_number': json_data.get(
                                    'best_trial_number'
                                ),
                                'features_enabled': json_data.get(
                                    'features_enabled', {}
                                ),
                                'feature_params': json_data.get(
                                    'feature_params', {}
                                )
                            }
                            if retrain_symbol(
                                symbol, horizon, best_params_data, pipeline
                            ):
                                success_count += 1
                            else:
                                fail_count += 1
                    except Exception as e:
                        print(
                            f"❌ Error loading HPO JSON: {e}",
                            file=sys.stderr
                        )
                        fail_count += 1
                else:
                    print(
                        f"❌ {symbol} {horizon}d: Neither study DB nor HPO "
                        f"JSON found"
                    )
                    fail_count += 1
                continue

            print(f"✅ Found study DB: {study_db}")

            # Get best params from study
            best_params_data = get_best_params_from_study(
                study_db, symbol, horizon, use_filtered=True, cycle=cycle
            )
            if not best_params_data:
                print(
                    f"❌ {symbol} {horizon}d: Could not load best params "
                    f"from study"
                )
                fail_count += 1
                continue

            # Show filter info
            filter_info = best_params_data.get('filter_used', {})
            split_info = best_params_data.get('split_analysis', {})
            print(
                f"✅ Best trial: {best_params_data.get('best_trial_number')}, "
                f"Best value: {best_params_data.get('best_value', 0):.2f}"
            )
            print(
                f"   Filter used: "
                f"min_count={filter_info.get('min_mask_count', 'N/A')}, "
                f"min_pct={filter_info.get('min_mask_pct', 'N/A')}"
            )
            print(
                f"   Splits: {split_info.get('included_splits', 0)}/"
                f"{split_info.get('total_splits', 0)} included"
            )

            # ⚠️  Warning if best params found with limited splits
            if split_info.get('excluded_splits', 0) > 0:
                print(
                    f"   ⚠️  WARNING: {split_info.get('excluded_splits')} "
                    f"split(s) excluded - best params may not be optimal!"
                )

            # ✅ CRITICAL: Set filter for retraining to EXACTLY match what
            # HPO used
            # This ensures training evaluation uses the same filter as HPO did
            # If HPO excluded low support splits, training should also exclude
            # them
            # This way we test if best params are truly optimal for the
            # filtered splits
            hpo_min_count = filter_info.get('min_mask_count')
            hpo_min_pct = filter_info.get('min_mask_pct')

            if hpo_min_count is not None:
                os.environ['HPO_MIN_MASK_COUNT'] = str(hpo_min_count)
            else:
                # Fallback: if not found, assume 0 (no filter)
                os.environ['HPO_MIN_MASK_COUNT'] = '0'

            if hpo_min_pct is not None:
                os.environ['HPO_MIN_MASK_PCT'] = str(hpo_min_pct)
            else:
                # Fallback: if not found, assume 0.0 (no filter)
                os.environ['HPO_MIN_MASK_PCT'] = '0.0'

            print(
                f"   🔧 Training will use HPO's filter: "
                f"HPO_MIN_MASK_COUNT={os.environ.get('HPO_MIN_MASK_COUNT')}, "
                f"HPO_MIN_MASK_PCT={os.environ.get('HPO_MIN_MASK_PCT')}"
            )

            # ⚠️  Warning if best params found with limited splits
            if split_info.get('excluded_splits', 0) > 0:
                print(
                    f"   ⚠️  WARNING: Best params found with only "
                    f"{split_info.get('included_splits')}/"
                    f"{split_info.get('total_splits')} splits!"
                )
                print(
                    f"   ⚠️  {split_info.get('excluded_splits')} split(s) "
                    f"were excluded due to low support during HPO"
                )
                print(
                    "   ⚠️  Training will also exclude low support splits to "
                    "match HPO"
                )
                print(
                    "   ⚠️  If results are still poor, best params may not "
                    "be optimal - consider re-running HPO with 0/0.0 filter"
                )

            # Retrain
            if retrain_symbol(symbol, horizon, best_params_data, pipeline):
                success_count += 1
            else:
                fail_count += 1

    print()
    print("=" * 80)
    print("RETRAINING SUMMARY")
    print("=" * 80)
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"📊 Total: {len(high_discrepancies)}")
    print("=" * 80)


if __name__ == '__main__':
    main()
