#!/usr/bin/env python3
"""
Retrain symbols whose trial number changed after filter application

This script directly uses the pipeline's run_training function to retrain
symbols that had their best trial number changed.
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

# Set DATABASE_URL (same as systemd service - read from secret file)
try:
    secret_path = '/opt/bist-pattern/.secrets/db_password'
    if os.path.exists(secret_path):
        with open(secret_path, 'r') as sp:
            _pwd = sp.read().strip()
            if _pwd:
                os.environ['DATABASE_URL'] = f'postgresql://bist_user:{_pwd}@127.0.0.1:6432/bist_pattern_db'
                print(f"✅ DATABASE_URL set from secret file (port 6432)")
            else:
                raise ValueError("Empty password from secret file")
    else:
        raise FileNotFoundError(f"Secret file not found: {secret_path}")
except Exception as e:
    print(f"⚠️  Error reading secret file: {e}")
    # Fallback (but this might not work)
    os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:6432/bist_pattern_db'
    print(f"⚠️  Using fallback DATABASE_URL (may fail)")

from scripts.continuous_hpo_training_pipeline import ContinuousHPOPipeline, STATE_FILE
from app import app

def find_trial_changed_symbols() -> list:
    """Find symbols whose trial number changed"""
    results_dir = Path('/opt/bist-pattern/results')
    json_files = list(results_dir.glob('optuna_pilot_features_on_h*.json'))
    
    changed_symbols = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if updated
            if '_updated_at' not in data:
                continue
            
            # Get best_trial_number from updated data
            new_trial = data.get('best_trial_number')
            
            # Try to get old trial from backup
            backup_file = json_file.with_suffix('.json.backup')
            if backup_file.exists():
                with open(backup_file, 'r') as f:
                    backup_data = json.load(f)
                old_trial = backup_data.get('best_trial_number')
                
                if old_trial and new_trial and old_trial != new_trial:
                    # Get symbol from JSON
                    symbols = data.get('symbols', [])
                    horizon = data.get('horizon')
                    if symbols and horizon:
                        for symbol in symbols:
                            changed_symbols.append((symbol, horizon))
                            print(f"✅ {symbol}_{horizon}d: Trial #{old_trial} → #{new_trial}")
        except Exception as e:
            continue
    
    return list(set(changed_symbols))  # Remove duplicates


def main():
    print("=" * 80)
    print("🔄 Retraining Symbols with Changed Trial Numbers")
    print("=" * 80)
    
    # Find symbols
    changed_symbols = find_trial_changed_symbols()
    
    if not changed_symbols:
        print("⚠️  No symbols with changed trial numbers found")
        return
    
    print(f"\n📊 Found {len(changed_symbols)} symbol-horizon pairs to retrain")
    
    # Initialize pipeline
    with app.app_context():
        pipeline = ContinuousHPOPipeline()
        
        success_count = 0
        fail_count = 0
        
        for symbol, horizon in changed_symbols:
            print(f"\n{'='*80}")
            print(f"🔄 Retraining {symbol} {horizon}d")
            print(f"{'='*80}")
            
            try:
                # Find JSON file
                json_file = None
                results_dir = Path('/opt/bist-pattern/results')
                json_files = sorted(results_dir.glob(f'optuna_pilot_features_on_h{horizon}_*.json'), 
                                  key=lambda x: x.stat().st_mtime, reverse=True)
                
                for jf in json_files:
                    try:
                        with open(jf, 'r') as f:
                            data = json.load(f)
                        if symbol in data.get('symbols', []):
                            json_file = jf
                            break
                    except Exception:
                        continue
                
                if not json_file:
                    print(f"❌ JSON file not found for {symbol} {horizon}d")
                    fail_count += 1
                    continue
                
                # Load HPO data
                with open(json_file, 'r') as f:
                    hpo_data = json.load(f)
                
                best_params = hpo_data.get('best_params', {})
                if not best_params:
                    print(f"❌ No best_params in JSON for {symbol} {horizon}d")
                    fail_count += 1
                    continue
                
                # Prepare hpo_result
                hpo_result = {
                    'best_params': best_params,
                    'best_trial_number': hpo_data.get('best_trial_number'),
                    'best_dirhit': hpo_data.get('best_dirhit'),
                    'json_file': str(json_file),
                    'features_enabled': hpo_data.get('features_enabled', {}),
                    'feature_params': hpo_data.get('feature_params', {}),
                }
                
                # Ensure DATABASE_URL is set before training
                if 'DATABASE_URL' not in os.environ:
                    try:
                        secret_path = '/opt/bist-pattern/.secrets/db_password'
                        if os.path.exists(secret_path):
                            with open(secret_path, 'r') as sp:
                                _pwd = sp.read().strip()
                                if _pwd:
                                    os.environ['DATABASE_URL'] = f'postgresql://bist_user:{_pwd}@127.0.0.1:6432/bist_pattern_db'
                    except Exception:
                        pass
                
                # Run training
                result = pipeline.run_training(symbol, horizon, best_params, hpo_result=hpo_data)
                
                if result:
                    print(f"✅ Training completed for {symbol} {horizon}d")
                    if result.get('wfv'):
                        print(f"   WFV DirHit: {result['wfv']:.2f}%")
                    success_count += 1
                else:
                    print(f"❌ Training failed for {symbol} {horizon}d")
                    fail_count += 1
                    
            except Exception as e:
                print(f"❌ Error retraining {symbol} {horizon}d: {e}")
                import traceback
                print(traceback.format_exc())
                fail_count += 1
        
        print(f"\n{'='*80}")
        print(f"📊 Summary: {success_count} succeeded, {fail_count} failed")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()

