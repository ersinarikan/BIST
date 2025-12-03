#!/usr/bin/env python3
"""
Analyze existing HPO JSON files to find symbols with low support warnings

This script:
1. Scans all HPO JSON files
2. Finds symbols with low_support_warnings
3. Lists symbols that need retraining
"""

import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

sys.path.insert(0, '/opt/bist-pattern')

def analyze_hpo_json_files(results_dir: Path) -> Dict[str, List[Tuple[str, int, str]]]:
    """Analyze all HPO JSON files and find low support symbols
    
    Returns:
        Dict mapping horizon -> List of (symbol, horizon, json_file) tuples
    """
    low_support_symbols = defaultdict(list)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return low_support_symbols
    
    # Find all HPO JSON files
    json_files = list(results_dir.glob("optuna_pilot_features_on_h*.json"))
    
    if not json_files:
        print(f"⚠️  No HPO JSON files found in {results_dir}")
        return low_support_symbols
    
    print(f"📊 Analyzing {len(json_files)} HPO JSON files...")
    
    for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check for low_support_warnings
            low_support_warnings = data.get('low_support_warnings', [])
            
            if low_support_warnings:
                # Extract horizon from filename (e.g., "optuna_pilot_features_on_h1_...")
                horizon = None
                for part in json_file.stem.split('_'):
                    if part.startswith('h') and part[1:].isdigit():
                        horizon = int(part[1:])
                        break
                
                if horizon:
                    for symbol_horizon in low_support_warnings:
                        # Parse "SYMBOL_Hd" format
                        parts = symbol_horizon.rsplit('_', 1)
                        if len(parts) == 2:
                            symbol = parts[0]
                            try:
                                h = int(parts[1].replace('d', ''))
                                if h == horizon:  # Verify horizon matches
                                    low_support_symbols[horizon].append((symbol, h, str(json_file)))
                            except Exception:
                                pass
        except Exception as e:
            print(f"⚠️  Error reading {json_file.name}: {e}")
            continue
    
    return low_support_symbols


def analyze_study_dbs(study_dir: Path) -> Dict[str, List[Tuple[str, int, str]]]:
    """Analyze Optuna study databases to find symbols with no valid splits
    
    Returns:
        Dict mapping horizon -> List of (symbol, horizon, db_file) tuples
    """
    low_support_symbols = defaultdict(list)
    
    if not study_dir.exists():
        print(f"⚠️  Study directory not found: {study_dir}")
        return low_support_symbols
    
    # Find all study DB files
    db_files = list(study_dir.glob("**/*.db"))
    
    if not db_files:
        print(f"⚠️  No study DB files found in {study_dir}")
        return low_support_symbols
    
    print(f"📊 Analyzing {len(db_files)} study DB files...")
    
    try:
        import optuna
    except ImportError:
        print("❌ optuna not installed")
        return low_support_symbols
    
    for db_file in sorted(db_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            study = optuna.load_study(
                study_name=None,
                storage=f"sqlite:///{db_file}"
            )
            
            if study.best_trial is None:
                continue
            
            # Get symbol_metrics from best trial
            symbol_metrics = study.best_trial.user_attrs.get('symbol_metrics', {})
            
            if not symbol_metrics:
                continue
            
            # Check each symbol for low_support_warning
            for symbol_key, metrics in symbol_metrics.items():
                if isinstance(metrics, dict) and metrics.get('low_support_warning'):
                    # Parse "SYMBOL_Hd" format
                    parts = symbol_key.rsplit('_', 1)
                    if len(parts) == 2:
                        symbol = parts[0]
                        try:
                            h = int(parts[1].replace('d', ''))
                            low_support_symbols[h].append((symbol, h, str(db_file)))
                        except Exception:
                            pass
        except Exception as e:
            print(f"⚠️  Error reading {db_file.name}: {e}")
            continue
    
    return low_support_symbols


def main():
    results_dir = Path("/opt/bist-pattern/results")
    study_dir = Path("/opt/bist-pattern/results/optuna_studies")
    
    print("=" * 80)
    print("🔍 Analyzing Low Support Symbols")
    print("=" * 80)
    
    # Analyze JSON files
    json_low_support = analyze_hpo_json_files(results_dir)
    
    # Analyze study DBs
    db_low_support = analyze_study_dbs(study_dir)
    
    # Combine results
    all_low_support = defaultdict(set)
    
    for horizon, symbols in json_low_support.items():
        for symbol, h, json_file in symbols:
            all_low_support[horizon].add((symbol, h))
    
    for horizon, symbols in db_low_support.items():
        for symbol, h, db_file in symbols:
            all_low_support[horizon].add((symbol, h))
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 Low Support Symbols Summary")
    print("=" * 80)
    
    if not all_low_support:
        print("✅ No low support symbols found!")
        return
    
    total_count = 0
    for horizon in sorted(all_low_support.keys()):
        symbols = sorted(all_low_support[horizon])
        print(f"\n📈 Horizon {horizon}d: {len(symbols)} symbol(s)")
        for symbol, h in symbols:
            print(f"   - {symbol} {h}d")
            total_count += 1
    
    print(f"\n📊 Total: {total_count} symbol-horizon pairs with low support")
    
    # Generate retraining command
    print("\n" + "=" * 80)
    print("🔄 Retraining Command")
    print("=" * 80)
    print("\nTo retrain these symbols, use:")
    print("\n/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/retrain_high_discrepancy_symbols.py \\")
    print("  --symbols " + ",".join([f"{sym}_{h}d" for h in sorted(all_low_support.keys()) for sym, _ in sorted(all_low_support[h])]))
    
    # Save to file
    output_file = Path("/opt/bist-pattern/results/low_support_symbols.txt")
    with open(output_file, 'w') as f:
        f.write("# Low Support Symbols - Generated by analyze_low_support_symbols.py\n")
        f.write(f"# Total: {total_count} symbol-horizon pairs\n\n")
        for horizon in sorted(all_low_support.keys()):
            symbols = sorted(all_low_support[horizon])
            for symbol, h in symbols:
                f.write(f"{symbol} {h}\n")
    
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == '__main__':
    main()

