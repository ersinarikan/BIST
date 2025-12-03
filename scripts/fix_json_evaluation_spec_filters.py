#!/usr/bin/env python3
"""
Fix evaluation_spec filter values in HPO JSON files
Reads actual filter values from split_metrics and updates evaluation_spec
"""
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

RESULTS_DIR = Path('/opt/bist-pattern/results')
BACKUP_DIR = RESULTS_DIR / 'json_backups_filter_fix'

def get_filter_from_split_metrics(data: Dict[str, Any]) -> tuple[int, float]:
    """Get filter values from split_metrics (what was actually used during HPO)"""
    best_trial_metrics = data.get('best_trial_metrics', {})
    if not isinstance(best_trial_metrics, dict):
        return 0, 0.0
    
    # Get filter values from first symbol's first split (all should have same filter)
    for sym_key, sym_metrics in best_trial_metrics.items():
        if not isinstance(sym_metrics, dict):
            continue
        split_metrics = sym_metrics.get('split_metrics', [])
        if split_metrics:
            first_split = split_metrics[0]
            min_mc = int(first_split.get('min_mask_count', 0))
            min_mp = float(first_split.get('min_mask_pct', 0.0))
            return min_mc, min_mp
    
    return 0, 0.0

def fix_json_file(json_file: Path, dry_run: bool = False) -> bool:
    """Fix evaluation_spec filter values in a JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ {json_file.name}: Error reading: {e}")
        return False
    
    # Get current evaluation_spec
    eval_spec = data.get('evaluation_spec', {})
    if not isinstance(eval_spec, dict):
        print(f"⚠️ {json_file.name}: No evaluation_spec found")
        return False
    
    # Get current filter values
    current_mc = eval_spec.get('min_mask_count', 0)
    current_mp = eval_spec.get('min_mask_pct', 0.0)
    
    # Get actual filter values from split_metrics
    actual_mc, actual_mp = get_filter_from_split_metrics(data)
    
    # Check if fix is needed
    if current_mc == actual_mc and current_mp == actual_mp:
        return False  # No fix needed
    
    print(f"\n{json_file.name}:")
    print(f"  Current filter: min_mask_count={current_mc}, min_mask_pct={current_mp}")
    print(f"  Actual filter: min_mask_count={actual_mc}, min_mask_pct={actual_mp}")
    
    if dry_run:
        print(f"  [DRY-RUN] Would update evaluation_spec")
        return True
    
    # Create backup
    backup_dir = BACKUP_DIR / json_file.stem
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backup_dir / json_file.name
    shutil.copy2(json_file, backup_file)
    print(f"  ✅ Backup created: {backup_file}")
    
    # Update evaluation_spec
    eval_spec['min_mask_count'] = actual_mc
    eval_spec['min_mask_pct'] = actual_mp
    
    # Save updated JSON
    try:
        # Atomic write
        tmp_file = json_file.with_suffix('.json.tmp')
        with open(tmp_file, 'w') as f:
            json.dump(data, f, indent=2)
        tmp_file.replace(json_file)
        print(f"  ✅ Updated evaluation_spec")
        return True
    except Exception as e:
        print(f"  ❌ Error saving: {e}")
        # Restore backup
        shutil.copy2(backup_file, json_file)
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fix evaluation_spec filter values in HPO JSON files')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without modifying files')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to fix (e.g., ADEL AKENR)')
    args = parser.parse_args()
    
    print("=" * 100)
    print("HPO JSON DOSYALARINDA evaluation_spec FİLTER DEĞERLERİNİ DÜZELTME")
    print("=" * 100)
    
    if args.dry_run:
        print("\n🔍 DRY-RUN MODE: Dosyalar değiştirilmeyecek")
    
    # Find all HPO JSON files
    json_files = list(RESULTS_DIR.glob('optuna_pilot_features_on_h*_c*.json'))
    
    if args.symbols:
        # Filter by symbols
        filtered_files = []
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            symbols = data.get('symbols', [])
            if any(sym in symbols for sym in args.symbols):
                filtered_files.append(json_file)
        json_files = filtered_files
    
    if not json_files:
        print("\n❌ HPO JSON dosyası bulunamadı!")
        return 1
    
    print(f"\n📋 {len(json_files)} JSON dosyası bulundu")
    
    fixed = 0
    skipped = 0
    errors = 0
    
    for json_file in sorted(json_files):
        try:
            if fix_json_file(json_file, dry_run=args.dry_run):
                fixed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"❌ {json_file.name}: Error: {e}")
            errors += 1
    
    print("\n" + "=" * 100)
    print("ÖZET")
    print("=" * 100)
    print(f"  Düzeltilen: {fixed}")
    print(f"  Değişiklik gerekmeyen: {skipped}")
    print(f"  Hatalar: {errors}")
    
    if fixed > 0 and not args.dry_run:
        print(f"\n✅ {fixed} JSON dosyası düzeltildi!")
        print(f"   Backup'lar: {BACKUP_DIR}")
    
    return 0 if errors == 0 else 1

if __name__ == '__main__':
    import sys
    sys.exit(main())

