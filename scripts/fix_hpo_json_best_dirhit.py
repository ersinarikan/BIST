#!/usr/bin/env python3
"""
Fix HPO JSON files: Update best_dirhit from best_trial_metrics.

This script fixes the issue where best_dirhit in HPO JSON files is incorrect
because symbol_metrics was not saved during HPO. The script updates best_dirhit
using the avg_dirhit from best_trial_metrics (which was added by migration script).

Usage:
    python scripts/fix_hpo_json_best_dirhit.py [--dry-run] [--symbols-only SYMBOL1,SYMBOL2,...]
"""

import json
import os
import sys
from pathlib import Path
import argparse


def fix_hpo_json_best_dirhit(json_file: str, dry_run: bool = False) -> bool:
    """Fix best_dirhit in a single HPO JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading {json_file}: {e}")
        return False
    
    # Check if best_trial_metrics exists
    best_trial_metrics = data.get('best_trial_metrics', {})
    if not best_trial_metrics:
        print(f"⚠️  {json_file}: No best_trial_metrics found, skipping")
        return False
    
    # Get symbols from HPO JSON
    symbols = data.get('symbols', [])
    horizon = data.get('horizon', 1)
    
    if not symbols:
        print(f"⚠️  {json_file}: No symbols found, skipping")
        return False
    
    # For each symbol, get avg_dirhit from best_trial_metrics
    updated = False
    old_best_dirhit = data.get('best_dirhit')
    
    # If there's only one symbol, use its avg_dirhit
    if len(symbols) == 1:
        symbol_key = f"{symbols[0]}_{horizon}d"
        if symbol_key in best_trial_metrics:
            new_best_dirhit = best_trial_metrics[symbol_key].get('avg_dirhit')
            if new_best_dirhit is not None:
                if old_best_dirhit != new_best_dirhit:
                    print(f"📝 {json_file}: best_dirhit {old_best_dirhit:.2f}% → {new_best_dirhit:.2f}%")
                    if not dry_run:
                        data['best_dirhit'] = float(new_best_dirhit)
                    updated = True
                else:
                    print(f"✅ {json_file}: best_dirhit already correct ({old_best_dirhit:.2f}%)")
            else:
                print(f"⚠️  {json_file}: avg_dirhit is None in best_trial_metrics")
        else:
            print(f"⚠️  {json_file}: {symbol_key} not found in best_trial_metrics")
    else:
        # Multiple symbols: average across all symbols
        dirhits = []
        for sym in symbols:
            symbol_key = f"{sym}_{horizon}d"
            if symbol_key in best_trial_metrics:
                avg_dirhit = best_trial_metrics[symbol_key].get('avg_dirhit')
                if avg_dirhit is not None:
                    dirhits.append(avg_dirhit)
        
        if dirhits:
            new_best_dirhit = float(sum(dirhits) / len(dirhits))
            if old_best_dirhit != new_best_dirhit:
                print(f"📝 {json_file}: best_dirhit {old_best_dirhit:.2f}% → {new_best_dirhit:.2f}% (avg of {len(dirhits)} symbols)")
                if not dry_run:
                    data['best_dirhit'] = new_best_dirhit
                updated = True
            else:
                print(f"✅ {json_file}: best_dirhit already correct ({old_best_dirhit:.2f}%)")
        else:
            print(f"⚠️  {json_file}: No valid avg_dirhit values found in best_trial_metrics")
    
    # Save if updated
    if updated and not dry_run:
        try:
            # Atomic write
            tmp_file = json_file + '.tmp'
            with open(tmp_file, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_file, json_file)
            print(f"✅ {json_file}: Updated successfully")
            return True
        except Exception as e:
            print(f"❌ {json_file}: Error saving: {e}")
            return False
    
    return updated


def main():
    parser = argparse.ArgumentParser(description='Fix best_dirhit in HPO JSON files')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no changes)')
    parser.add_argument('--symbols-only', type=str, help='Comma-separated list of symbols to process')
    parser.add_argument('--results-dir', type=str, default='/opt/bist-pattern/results',
                       help='Results directory (default: /opt/bist-pattern/results)')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return 1
    
    # Find all HPO JSON files
    json_files = list(results_dir.glob('optuna_pilot_features_on_*.json'))
    
    if not json_files:
        print(f"⚠️  No HPO JSON files found in {results_dir}")
        return 0
    
    print(f"🔍 Found {len(json_files)} HPO JSON files")
    print()
    
    # Filter by symbols if specified
    if args.symbols_only:
        symbols_filter = [s.strip().upper() for s in args.symbols_only.split(',')]
        print(f"📋 Filtering by symbols: {symbols_filter}")
        print()
        
        filtered_files = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                symbols = data.get('symbols', [])
                if any(sym in symbols_filter for sym in symbols):
                    filtered_files.append(json_file)
            except Exception:
                continue
        
        json_files = filtered_files
        print(f"📋 Filtered to {len(json_files)} files")
        print()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print()
    
    updated_count = 0
    skipped_count = 0
    
    for json_file in sorted(json_files):
        if fix_hpo_json_best_dirhit(str(json_file), dry_run=args.dry_run):
            updated_count += 1
        else:
            skipped_count += 1
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Updated: {updated_count}")
    print(f"⚠️  Skipped: {skipped_count}")
    print(f"📊 Total: {len(json_files)}")
    
    if args.dry_run:
        print()
        print("💡 Run without --dry-run to apply changes")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
