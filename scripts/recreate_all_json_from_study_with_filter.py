#!/usr/bin/env python3
"""
Recreate all JSON files from study databases with 10/5.0 filter applied
This ensures all JSON files have correct best trial, parameters, features, and filter values
"""

import sys
import os
import json
import argparse
import shutil
import signal
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import time

sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    venv_python = '/opt/bist-pattern/venv/bin/python3'
    if os.path.exists(venv_python):
        os.execv(venv_python, [venv_python] + sys.argv)
    else:
        raise

from scripts.continuous_hpo_training_pipeline import STATE_FILE

# ✅ FIX: Define find_study_db and find_best_trial_with_filter_applied here
# to avoid Flask initialization (from app import app in retrain_high_discrepancy_symbols.py)
# These functions don't need Flask, so we'll define them directly

HPO_STUDIES_DIR = Path('/opt/bist-pattern/hpo_studies')

def find_study_db(symbol: str, horizon: int, cycle: Optional[int] = None) -> Optional[Path]:
    """Find study database file for symbol-horizon"""
    if cycle is None:
        state = load_state()
        cycle = state.get('cycle', 1)
    
    if not HPO_STUDIES_DIR.exists():
        return None
    
    # Priority 1: Cycle format (hpo_with_features_SYMBOL_h1_c2.db)
    cycle_files = list(HPO_STUDIES_DIR.glob(f"hpo_with_features_{symbol}_h{horizon}_c{cycle}.db"))
    if cycle_files:
        return cycle_files[0]
    
    # Priority 2: Legacy format (hpo_with_features_SYMBOL_h1.db)
    legacy_files = list(HPO_STUDIES_DIR.glob(f"hpo_with_features_{symbol}_h{horizon}.db"))
    if legacy_files:
        return legacy_files[0]
    
    return None


def find_best_trial_with_filter_applied(db_file: Path, symbol: str, horizon: int,
                                        min_mask_count: int, min_mask_pct: float):
    """Find best trial after applying filter to study (not JSON's best trial)"""
    try:
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
            
            # Need at least 2 splits for reliable DirHit calculation
            # ✅ FIX: If min_mask_count=0 and min_mask_pct=0.0, accept even 1 split (fallback case)
            min_splits_required = 2 if (min_mask_count > 0 or min_mask_pct > 0.0) else 1
            if len(filtered_dirhits) < min_splits_required:
                continue
            
            # Calculate filtered average DirHit
            filtered_score = sum(filtered_dirhits) / len(filtered_dirhits)
            
            if filtered_score > best_filtered_score:
                best_filtered_score = filtered_score
                best_trial = trial
        
        return best_trial, best_filtered_score if best_trial else None
    
    except Exception as e:
        print(f"❌ Error finding best trial with filter: {e}", file=sys.stderr)
        return None, None


def load_state() -> Dict:
    """Load pipeline state"""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def find_existing_json(symbol: str, horizon: int, cycle: int) -> Optional[Path]:
    """Find existing JSON file for symbol"""
    results_dir = Path('/opt/bist-pattern/results')
    pattern = f"optuna_pilot_features_on_h{horizon}_c{cycle}_*.json"
    json_files = list(results_dir.glob(pattern))
    
    if not json_files:
        return None
    
    for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            if symbol in data.get('symbols', []):
                return json_file
        except Exception:
            continue
    
    return None


class TimeoutError(Exception):
    pass

def find_best_trial_with_timeout(db_file: Path, symbol: str, horizon: int,
                                 min_mask_count: int, min_mask_pct: float,
                                 timeout_seconds: int = 300):
    """Find best trial with timeout - using direct call with progress tracking"""
    try:
        # ✅ FIX: Add SQLite timeout and ensure connection is properly closed
        import sqlite3
        import optuna
        
        # Set SQLite timeout before loading study
        db_path = str(db_file)
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.execute('PRAGMA journal_mode=WAL')  # Enable WAL mode for better concurrency
        conn.close()
        
        # Load study with explicit connection management
        # ✅ FIX: Add timeout wrapper for load_study itself
        load_start = time.time()
        try:
            study = optuna.load_study(
                study_name=None,
                storage=f"sqlite:///{db_file}"
            )
        except Exception as e:
            if time.time() - load_start > 30:  # If loading took more than 30s, it might be stuck
                raise TimeoutError(f"optuna.load_study() took too long (>30s) or failed: {e}")
            raise
        
        symbol_key = f"{symbol}_{horizon}d"
        best_trial = None
        best_filtered_score = float('-inf')
        
        # Progress tracking for large studies
        total_trials = len(study.trials)
        if total_trials > 1000:
            print(f"    ⚠️ Large study: {total_trials} trials, this may take a while...", flush=True)
        
        start_time = time.time()
        processed = 0
        
        # ✅ FIX: Check timeout BEFORE starting iteration
        # optuna.load_study() itself might hang, so we check before iterating
        if time.time() - start_time > timeout_seconds:
            print(f"    ⏱️ TIMEOUT: Exceeded {timeout_seconds}s limit before iteration", flush=True)
            del study
            raise TimeoutError(f"find_best_trial_with_filter_applied exceeded {timeout_seconds}s")
        
        # Get trials list first (this might also hang, but we can't avoid it)
        trials_list = list(study.trials)  # Convert to list to avoid repeated DB queries
        
        for trial in trials_list:
            # Check timeout periodically - more frequently
            if processed % 50 == 0:  # Check every 50 trials
                if time.time() - start_time > timeout_seconds:
                    print(f"    ⏱️ TIMEOUT: Exceeded {timeout_seconds}s limit at trial {processed}", flush=True)
                    del study
                    raise TimeoutError(f"find_best_trial_with_filter_applied exceeded {timeout_seconds}s")
            
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            
            processed += 1
            # More frequent progress updates
            if total_trials > 1000:
                if processed % 200 == 0:
                    elapsed = time.time() - start_time
                    print(f"    📊 Processed {processed}/{total_trials} trials... ({elapsed:.1f}s)", flush=True)
            elif total_trials > 500:
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"    📊 Processed {processed}/{total_trials} trials... ({elapsed:.1f}s)", flush=True)
            
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
            
            # Need at least 2 splits for reliable DirHit calculation
            # ✅ FIX: If min_mask_count=0 and min_mask_pct=0.0, accept even 1 split (fallback case)
            min_splits_required = 2 if (min_mask_count > 0 or min_mask_pct > 0.0) else 1
            if len(filtered_dirhits) < min_splits_required:
                continue
            
            # Calculate filtered average DirHit
            filtered_score = sum(filtered_dirhits) / len(filtered_dirhits)
            
            if filtered_score > best_filtered_score:
                best_filtered_score = filtered_score
                best_trial = trial
        
        result_trial = best_trial
        result_score = best_filtered_score if best_trial else None
        
        # ✅ FIX: Explicitly close study connection
        del study
        
        return result_trial, result_score
        
    except TimeoutError:
        raise
    except Exception as e:
        raise Exception(f"Error in find_best_trial_with_timeout: {e}")

def create_json_from_filtered_trial(db_file: Path, symbol: str, horizon: int, cycle: int,
                                    min_mask_count: int = 5, min_mask_pct: float = 2.5,
                                    min_valid_splits: int = 2, dry_run: bool = False,
                                    timeout_seconds: int = 300) -> Optional[Path]:
    """Create JSON file from study database using filtered best trial"""
    start_time = time.time()
    try:
        # ✅ FIX: Add small delay to prevent SQLite lock contention
        time.sleep(0.1)
        
        # Find best trial with filter applied (with timeout)
        print(f"  🔍 Finding best trial with filter {min_mask_count}/{min_mask_pct}...")
        filtered_trial, filtered_score = find_best_trial_with_timeout(
            db_file, symbol, horizon, min_mask_count, min_mask_pct, timeout_seconds
        )
        
        if not filtered_trial:
            print(f"  ❌ No trial found with filter {min_mask_count}/{min_mask_pct}")
            return None
        
        elapsed = time.time() - start_time
        print(f"  ✅ Found filtered best trial: #{filtered_trial.number} (DirHit: {filtered_score:.2f}%) [{elapsed:.1f}s]")
        
        # Get params from filtered trial
        best_params = filtered_trial.params.copy()
        best_trial_number = filtered_trial.number
        
        # Load study once for n_trials and pruned_count (optimization)
        # ✅ FIX: Add SQLite timeout and WAL mode
        import sqlite3
        db_path = str(db_file)
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.close()
        
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_file}")
        completed_trials = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        pruned_trials = len([t for t in study.trials if t.state == TrialState.PRUNED])
        del study  # Explicit cleanup
        
        # Get symbol-specific avg_dirhit
        symbol_key = f"{symbol}_{horizon}d"
        symbol_metrics = filtered_trial.user_attrs.get('symbol_metrics', {})
        symbol_metric = symbol_metrics.get(symbol_key, {}) if isinstance(symbol_metrics, dict) else {}
        symbol_avg_dirhit = symbol_metric.get('avg_dirhit') if isinstance(symbol_metric, dict) else None
        
        if symbol_avg_dirhit is not None:
            best_dirhit = float(symbol_avg_dirhit)
        else:
            best_dirhit = filtered_score
        
        # Get feature flags and params from trial
        feature_flags = {k: v for k, v in best_params.items() if k.startswith('enable_')}
        
        # Get features_enabled and feature_params from user_attrs if available
        features_enabled = filtered_trial.user_attrs.get('features_enabled', {})
        feature_params = filtered_trial.user_attrs.get('feature_params', {})
        
        # Feature params keys
        feature_params_keys = [
            'external_min_days', 'external_smooth_alpha',
            'yolo_min_conf',
            'ml_loss_mse_weight', 'ml_loss_threshold', 'ml_dir_penalty',
            'n_seeds', 'meta_stacking_alpha',
            'smart_consensus_weight', 'smart_performance_weight', 'smart_sigma',
            'smart_weight_xgb', 'smart_weight_lgbm', 'smart_weight_cat',
            'regime_scale_low', 'regime_scale_high',
            'fingpt_confidence_threshold',
        ]
        feature_params_keys += [k for k in best_params.keys() if k.startswith('ml_adaptive_k_') or k.startswith('ml_pattern_weight_scale_')]
        
        if not feature_params:
            feature_params = {k: v for k, v in best_params.items() if k in feature_params_keys}
        
        # Hyperparameters
        hyperparameters = {k: v for k, v in best_params.items() if not k.startswith('enable_') and k not in feature_params_keys}
        
        # Build features_enabled if not available
        if not features_enabled:
            features_enabled = {
                'ENABLE_EXTERNAL_FEATURES': '1' if feature_flags.get('enable_external_features', False) else '0',
                'ENABLE_FINGPT_FEATURES': '1' if feature_flags.get('enable_fingpt_features', False) else '0',
                'ENABLE_YOLO_FEATURES': '1' if feature_flags.get('enable_yolo_features', False) else '0',
                'ML_USE_DIRECTIONAL_LOSS': '1' if feature_flags.get('enable_directional_loss', False) else '0',
                'ENABLE_SEED_BAGGING': '1' if feature_flags.get('enable_seed_bagging', False) else '0',
                'ENABLE_TALIB_PATTERNS': '1' if feature_flags.get('enable_talib_patterns', False) else '0',
                'ML_USE_SMART_ENSEMBLE': '1' if feature_flags.get('enable_smart_ensemble', False) else '0',
                'ML_USE_STACKED_SHORT': '1' if feature_flags.get('enable_stacked_short', False) else '0',
                'ENABLE_META_STACKING': '1' if feature_flags.get('enable_meta_stacking', False) else '0',
                'ML_USE_REGIME_DETECTION': '1' if feature_flags.get('enable_regime_detection', False) else '0',
                'ENABLE_FINGPT': '1' if feature_flags.get('enable_fingpt', False) else '0',
                'ML_USE_ADAPTIVE_LEARNING': '0',
                'ENABLE_XGBOOST': '1' if best_params.get('model_choice') in ('xgb', 'all') else '0',
                'ENABLE_LIGHTGBM': '1' if best_params.get('model_choice') in ('lgbm', 'all') else '0',
                'ENABLE_CATBOOST': '1' if best_params.get('model_choice') in ('cat', 'all') else '0',
            }
        
        # Get best_trial_metrics (symbol-specific)
        best_trial_metrics = {}
        if isinstance(symbol_metrics, dict) and symbol_key in symbol_metrics:
            best_trial_metrics[symbol_key] = symbol_metric
        
        # Build evaluation_spec from filtered trial's split_metrics
        evaluation_spec = {
            'horizon': int(horizon),
            'dirhit_threshold': 0.005,
            'min_mask_count': int(min_mask_count),
            'min_mask_pct': float(min_mask_pct),
            'best_trial_number': int(best_trial_number),
            'best_trial_seed': int(42 + best_trial_number),
            'scoring': {
                'formula': 'score = 0.7*avg_dirhit - k*avg_nrmse',
                'k': 6.0 if horizon in (1, 3, 7) else 4.0,
            },
        }
        
        # Build symbol_specs with splits
        symbol_specs = {}
        if symbol_metric and isinstance(symbol_metric, dict):
            split_metrics = symbol_metric.get('split_metrics', [])
            splits_out = []
            for s in split_metrics:
                splits_out.append({
                    'split_index': int(s.get('split_index')) if s.get('split_index') is not None else None,
                    'train_end_idx': int(s.get('train_end_idx')) if s.get('train_end_idx') is not None else None,
                    'test_end_idx': int(s.get('test_end_idx')) if s.get('test_end_idx') is not None else None,
                    'train_start': s.get('train_start'),
                    'train_end': s.get('train_end'),
                    'test_start': s.get('test_start'),
                    'test_end': s.get('test_end'),
                })
            symbol_specs[symbol_key] = {'splits': splits_out}
        
        evaluation_spec['symbol_specs'] = symbol_specs
        
        # Build JSON structure (matching optuna_hpo_with_feature_flags.py format)
        result = {
            'best_value': float(filtered_score),
            'best_dirhit': float(best_dirhit),
            'best_params': best_params,
            'best_trial': {
                'number': int(best_trial_number),
                'value': float(filtered_trial.value) if filtered_trial.value is not None else 0.0,
                'state': str(filtered_trial.state),
            },
            'best_trial_number': int(best_trial_number),
            'best_trial_seed': int(42 + best_trial_number),
            'n_trials': completed_trials,
            'pruned_count': pruned_trials,
            'avg_trial_time': 0.0,  # Not available from study alone
            'top_k_trials': [],  # Not calculated
            'study_name': f"hpo_with_features_{symbol}_h{horizon}_c{cycle}",
            'symbols': [symbol],
            'horizon': int(horizon),
            'best_model_choice': best_params.get('model_choice'),
            'feature_flags': feature_flags,
            'feature_params': feature_params,
            'hyperparameters': hyperparameters,
            'features_enabled': features_enabled,
            'best_trial_metrics': best_trial_metrics,
            'evaluation_spec': evaluation_spec,
            '_created_at': datetime.now().isoformat(),
            '_created_from': 'study_database',
            '_filter_applied': {'min_mask_count': min_mask_count, 'min_mask_pct': min_mask_pct},
        }
        
        if dry_run:
            print(f"  [DRY-RUN] Would create JSON for {symbol}_{horizon}d")
            print(f"    - Best trial: #{best_trial_number}")
            print(f"    - Best DirHit: {best_dirhit:.2f}%")
            print(f"    - Filter: {min_mask_count}/{min_mask_pct}")
            return None
        
        # Create backup of existing JSON if exists
        existing_json = find_existing_json(symbol, horizon, cycle)
        if existing_json:
            backup_file = existing_json.with_suffix('.json.backup_before_recreate')
            if not backup_file.exists():
                shutil.copy2(existing_json, backup_file)
                print(f"  ✅ Backup created: {backup_file.name}")
        
        # Write new JSON file
        output_file = f"/opt/bist-pattern/results/optuna_pilot_features_on_h{horizon}_c{cycle}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        elapsed = time.time() - start_time
        print(f"  ✅ Created: {Path(output_file).name} [{elapsed:.1f}s]")
        return Path(output_file)
        
    except TimeoutError as e:
        elapsed = time.time() - start_time
        print(f"  ⏱️ TIMEOUT: {symbol} exceeded {timeout_seconds}s limit (elapsed: {elapsed:.1f}s)")
        print(f"  ⚠️ Skipping {symbol} - study file may be too large or have too many trials")
        print(f"  💡 Suggestion: Process this symbol separately or increase timeout")
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ❌ Error processing {symbol}: {e} [{elapsed:.1f}s]")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Recreate all JSON files from study databases with 10/5.0 filter')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Specific symbols to process (default: all completed)')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1],
                       help='Horizons to process (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - show what would be created')
    parser.add_argument('--min-mask-count', type=int, default=5,
                       help='Minimum mask count for filter (default: 5)')
    parser.add_argument('--min-mask-pct', type=float, default=2.5,
                       help='Minimum mask percentage for filter (default: 2.5)')
    
    args = parser.parse_args()
    
    # Load state
    state = load_state()
    current_cycle = state.get('cycle', 1)
    tasks = state.get('state', {})
    
    # Get symbols to process
    if args.symbols:
        symbols_to_process = [(s, h) for s in args.symbols for h in args.horizons]
    else:
        # ✅ FIX: Sadece completed semboller için JSON oluştur
        # Pending/in-progress semboller için HPO henüz tamamlanmamış, JSON oluşturulamaz
        symbols_to_process = []
        study_dir = Path('/opt/bist-pattern/hpo_studies')
        
        # State'ten sadece completed sembolleri al
        completed_symbols = set()
        for key, task in tasks.items():
            if not isinstance(task, dict):
                continue
            if task.get('cycle', 0) == current_cycle:
                status = task.get('status', '')
                if status == 'completed':  # ✅ Sadece completed
                    parts = key.split('_')
                    if len(parts) == 2:
                        symbol = parts[0]
                        try:
                            horizon = int(parts[1].replace('d', ''))
                            if horizon in args.horizons:
                                completed_symbols.add((symbol, horizon))
                        except:
                            continue
        
        # Study dosyası var mı kontrol et
        for symbol, horizon in completed_symbols:
            study_file = study_dir / f"hpo_with_features_{symbol}_h{horizon}_c{current_cycle}.db"
            if study_file.exists():
                symbols_to_process.append((symbol, horizon))
        
        # Remove duplicates
        symbols_to_process = list(set(symbols_to_process))
        symbols_to_process = sorted(symbols_to_process)
    
    print("=" * 80)
    print("JSON DOSYALARINI YENİDEN OLUŞTURMA (STUDY DOSYALARINDAN)")
    print("=" * 80)
    print(f"\n🔄 Cycle: {current_cycle}")
    print(f"🔍 Filter: {args.min_mask_count}/{args.min_mask_pct}")
    print(f"📊 İşlenecek sembol sayısı: {len(symbols_to_process)}")
    
    if args.dry_run:
        print(f"\n⚠️ DRY-RUN MODE - Dosyalar oluşturulmayacak")
    
    created_count = 0
    failed_count = 0
    skipped_count = 0
    timeout_count = 0
    
    total = len(symbols_to_process)
    
    for idx, (symbol, horizon) in enumerate(sorted(symbols_to_process), 1):
        print(f"\n{'='*80}")
        print(f"📊 [{idx}/{total}] {symbol}_{horizon}d")
        print(f"{'='*80}")
        
        try:
            # Find study database
            db_file = find_study_db(symbol, horizon, current_cycle)
            if not db_file:
                print(f"  ❌ Study database not found")
                failed_count += 1
                continue
            
            # Check file size
            size_mb = db_file.stat().st_size / 1024 / 1024
            print(f"  ✅ Study DB: {db_file.name} ({size_mb:.2f} MB)")
            
            if size_mb > 50:
                print(f"  ⚠️ Large study file - may take longer")
            
            # Create JSON from filtered trial
            # Adjust timeout based on file size
            size_mb = db_file.stat().st_size / 1024 / 1024
            if size_mb > 20:
                timeout = 180  # 3 minutes for large files
            elif size_mb > 10:
                timeout = 120  # 2 minutes for medium files
            else:
                timeout = 60   # 1 minute for small files
            
            # ✅ Her sembol için ayrı subprocess başlat, bitince kapat, sıradakine geç
            # Bu şekilde bir sembol takılırsa sadece o timeout olur, diğerleri devam eder
            symbol_start_time = time.time()
            try:
                import tempfile
                import subprocess
                
                # Her sembol için ayrı script oluştur
                script_content = f"""
import sys
sys.path.insert(0, '/opt/bist-pattern')
from scripts.recreate_all_json_from_study_with_filter import create_json_from_filtered_trial
from pathlib import Path

db_file = Path('{db_file}')
symbol = '{symbol}'
horizon = {horizon}
cycle = {current_cycle}
min_mask_count = {args.min_mask_count}
min_mask_pct = {args.min_mask_pct}
timeout_seconds = {timeout}
dry_run = {args.dry_run}

try:
    result = create_json_from_filtered_trial(
        db_file, symbol, horizon, cycle,
        min_mask_count, min_mask_pct,
        min_valid_splits=2,
        dry_run=dry_run,
        timeout_seconds=timeout_seconds
    )
    if result:
        print(f"SUCCESS: {{result}}")
        sys.exit(0)
    else:
        print("FAILED: No result")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(script_content)
                    temp_script = f.name
                
                try:
                    # Her sembol için ayrı subprocess, timeout ile
                    result = subprocess.run(
                        ['/opt/bist-pattern/venv/bin/python3', temp_script],
                        timeout=timeout + 30,  # 30s buffer
                        capture_output=True,
                        text=True
                    )
                    
                    # Print output
                    if result.stdout:
                        print(result.stdout, end='', flush=True)
                    if result.stderr:
                        # Flask uyarılarını filtrele
                        stderr_lines = result.stderr.split('\n')
                        filtered_stderr = [l for l in stderr_lines if 'FLASK_SECRET_KEY' not in l and 'INTERNAL_API_TOKEN' not in l and 'api_patterns blueprint' not in l and 'AUTO_START_CYCLE' not in l and 'Calibration startup' not in l and 'HPO Configuration Check' not in l]
                        if filtered_stderr:
                            print('\n'.join(filtered_stderr), end='', file=sys.stderr, flush=True)
                    
                    if result.returncode == 0:
                        # Check if JSON was created
                        json_file = find_existing_json(symbol, horizon, current_cycle)
                        if json_file and json_file.stat().st_mtime >= symbol_start_time:
                            created_count += 1
                            print(f"  ✅ JSON created successfully")
                        else:
                            failed_count += 1
                            print(f"  ❌ JSON not found after success")
                    else:
                        # ✅ FALLBACK: Eğer 5/2.5 filtresine uyan trial yoksa, 0/0.0 ile dene
                        if "No trial found with filter" in result.stdout or "No trial found with filter" in result.stderr:
                            print(f"  ⚠️ No trial found with filter {args.min_mask_count}/{args.min_mask_pct}, trying fallback 0/0.0...")
                            
                            # Fallback script
                            fallback_script = f"""
import sys
sys.path.insert(0, '/opt/bist-pattern')
from scripts.recreate_all_json_from_study_with_filter import create_json_from_filtered_trial
from pathlib import Path

db_file = Path('{db_file}')
symbol = '{symbol}'
horizon = {horizon}
cycle = {current_cycle}

try:
    result = create_json_from_filtered_trial(
        db_file, symbol, horizon, cycle,
        min_mask_count=0,
        min_mask_pct=0.0,
        min_valid_splits=2,
        dry_run={args.dry_run},
        timeout_seconds={timeout}
    )
    if result:
        print(f"SUCCESS: {{result}}")
        sys.exit(0)
    else:
        print("FAILED: No result even with fallback")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                                f.write(fallback_script)
                                fallback_temp = f.name
                            
                            try:
                                fallback_result = subprocess.run(
                                    ['/opt/bist-pattern/venv/bin/python3', fallback_temp],
                                    timeout=timeout + 30,
                                    capture_output=True,
                                    text=True
                                )
                                
                                if fallback_result.stdout:
                                    print(fallback_result.stdout, end='', flush=True)
                                
                                if fallback_result.returncode == 0:
                                    json_file = find_existing_json(symbol, horizon, current_cycle)
                                    if json_file and json_file.stat().st_mtime >= symbol_start_time:
                                        created_count += 1
                                        print(f"  ✅ JSON created with fallback filter 0/0.0")
                                    else:
                                        failed_count += 1
                                else:
                                    failed_count += 1
                                    print(f"  ❌ Fallback also failed")
                            finally:
                                try:
                                    os.unlink(fallback_temp)
                                except:
                                    pass
                        else:
                            failed_count += 1
                            print(f"  ❌ Subprocess failed with return code {result.returncode}")
                        
                except subprocess.TimeoutExpired:
                    print(f"  ⏱️ TIMEOUT: {symbol} exceeded {timeout+30}s, killing process...")
                    timeout_count += 1
                    failed_count += 1
                finally:
                    # Cleanup temp script
                    try:
                        os.unlink(temp_script)
                    except:
                        pass
                        
            except Exception as e:
                print(f"  ❌ Unexpected error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue
                
        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted by user at {symbol}")
            break
        except Exception as e:
            print(f"  ❌ Unexpected error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 ÖZET")
    print(f"{'='*80}")
    print(f"✅ Oluşturulan: {created_count}")
    print(f"❌ Başarısız: {failed_count}")
    print(f"⏱️  Timeout: {timeout_count}")
    print(f"⏭️  Atlanan: {skipped_count}")
    print(f"📊 Toplam: {len(symbols_to_process)}")
    print(f"📈 İlerleme: {created_count}/{len(symbols_to_process)} ({created_count/len(symbols_to_process)*100:.1f}%)")


if __name__ == '__main__':
    main()

