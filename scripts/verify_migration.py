#!/usr/bin/env python3
"""
Verify Migration: Check if best_trial_metrics were added to HPO JSON files
"""

import json
from pathlib import Path

def verify_migration():
    results_dir = Path('/opt/bist-pattern/results')
    json_files = sorted(results_dir.glob('optuna_pilot_features_on_h*.json'))
    
    migrated_count = 0
    unmigrated_count = 0
    
    print("=" * 80)
    print("🔍 Migration Verification")
    print("=" * 80)
    print()
    
    complete_migrated = 0
    partial_migrated = 0
    errors = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            symbol = data.get('symbols', ['UNKNOWN'])[0]
            horizon = data.get('horizon', '?')
            key = f"{symbol}_{horizon}d"
            
            if 'best_trial_metrics' in data and data.get('best_trial_metrics'):
                metrics = data['best_trial_metrics']
                
                # Check if metrics exist for this symbol_horizon
                symbol_metrics = None
                for k, v in metrics.items():
                    if isinstance(v, dict) and k == key:
                        symbol_metrics = v
                        break
                
                if symbol_metrics:
                    has_raw_r2 = 'avg_model_metrics' in symbol_metrics and symbol_metrics.get('avg_model_metrics')
                    has_avg_dirhit = symbol_metrics.get('avg_dirhit') is not None
                    has_split_metrics = 'split_metrics' in symbol_metrics and len(symbol_metrics.get('split_metrics', [])) > 0
                    
                    if has_raw_r2 and has_avg_dirhit and has_split_metrics:
                        complete_migrated += 1
                        if len(json_files) <= 20:  # Only print details for small sets
                            print(f"✅ COMPLETE: {key}")
                    elif has_raw_r2 or has_avg_dirhit:
                        partial_migrated += 1
                        if len(json_files) <= 20:
                            print(f"⚠️  PARTIAL: {key} (missing some metrics)")
                    else:
                        unmigrated_count += 1
                        if len(json_files) <= 20:
                            print(f"❌ INCOMPLETE: {key} (structure exists but empty)")
                else:
                    unmigrated_count += 1
                    if len(json_files) <= 20:
                        print(f"❌ NOT MIGRATED: {key} (no metrics found)")
            else:
                unmigrated_count += 1
                if len(json_files) <= 20:
                    print(f"❌ NOT MIGRATED: {key} (no best_trial_metrics field)")
                
        except Exception as e:
            errors.append((json_file.name, str(e)))
            if len(json_files) <= 20:
                print(f"⚠️  ERROR reading {json_file.name}: {e}")
    
    print()
    print("=" * 80)
    print(f"📊 Migration Summary")
    print("=" * 80)
    print(f"✅ Complete: {complete_migrated} files")
    print(f"⚠️  Partial: {partial_migrated} files")
    print(f"❌ Not Migrated: {unmigrated_count} files")
    if errors:
        print(f"⚠️  Errors: {len(errors)} files")
    print(f"📁 Total: {len(json_files)} files")
    print("=" * 80)
    
    if errors and len(json_files) <= 20:
        print("\n⚠️  Errors encountered:")
        for filename, error in errors:
            print(f"  - {filename}: {error}")

if __name__ == '__main__':
    verify_migration()

