#!/usr/bin/env python3
"""
Compare HPO results from Phase 1 and Phase 2
Creates a comprehensive comparison table for all symbols and horizons
"""
import os
import sys
import json
import glob
import pandas as pd
from pathlib import Path

# Set environment
sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

def load_phase1_hpo_results():
    """Load Phase 1 HPO results (features OFF during HPO)."""
    results = {}
    json_files = glob.glob('/opt/bist-pattern/results/optuna_pilot_h*.json')
    # Exclude Phase 2 files
    json_files = [f for f in json_files if 'features_on' not in f]
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            symbol = data.get('symbols', ['UNKNOWN'])[0]
            horizon = data.get('horizon', 1)
            best_value = data.get('best_value', 0.0)
            
            key = f"{symbol}_{horizon}d"
            if key not in results:
                results[key] = {
                    'symbol': symbol,
                    'horizon': f"{horizon}d",
                    'phase1_hpo_dirhit': best_value,
                    'phase1_hpo_json': json_file
                }
        except Exception as e:
            print(f"⚠️ Error loading {json_file}: {e}")
            continue
    
    return results

def load_phase1_training_results():
    """Load Phase 1 training results (best params with all features ON)."""
    results = {}
    
    csv_file = '/opt/bist-pattern/results/train_completed_hpo_dirhits.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            key = f"{row['symbol']}_{row['horizon']}"
            results[key] = {
                'symbol': row['symbol'],
                'horizon': row['horizon'],
                'phase1_training_dirhit': row['dirhit']
            }
    
    return results

def load_phase2_hpo_results():
    """Load Phase 2 HPO results (features ON during HPO)."""
    results = {}
    json_files = glob.glob('/opt/bist-pattern/results/optuna_pilot_features_on_*.json')
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            symbol = data.get('symbols', ['UNKNOWN'])[0]
            horizon = data.get('horizon', 1)
            best_value = data.get('best_value', 0.0)
            
            key = f"{symbol}_{horizon}d"
            if key not in results:
                results[key] = {
                    'symbol': symbol,
                    'horizon': f"{horizon}d",
                    'phase2_hpo_dirhit': best_value,
                    'phase2_hpo_json': json_file
                }
        except Exception as e:
            print(f"⚠️ Error loading {json_file}: {e}")
            continue
    
    return results

def load_phase2_training_results():
    """Load Phase 2 training results (best params with all features ON)."""
    results = {}
    
    csv_file = '/opt/bist-pattern/results/train_completed_hpo_phase2_dirhits.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            key = f"{row['symbol']}_{row['horizon']}"
            results[key] = {
                'symbol': row['symbol'],
                'horizon': row['horizon'],
                'phase2_training_dirhit': row['dirhit']
            }
    
    return results

def create_comparison_table():
    """Create comprehensive comparison table."""
    print("=" * 100)
    print("📊 HPO RESULTS COMPARISON TABLE")
    print("=" * 100)
    print()
    
    # Load all results
    print("Loading Phase 1 HPO results...")
    phase1_hpo = load_phase1_hpo_results()
    print(f"  ✅ Loaded {len(phase1_hpo)} Phase 1 HPO results")
    
    print("Loading Phase 1 training results...")
    phase1_train = load_phase1_training_results()
    print(f"  ✅ Loaded {len(phase1_train)} Phase 1 training results")
    
    print("Loading Phase 2 HPO results...")
    phase2_hpo = load_phase2_hpo_results()
    print(f"  ✅ Loaded {len(phase2_hpo)} Phase 2 HPO results")
    
    print("Loading Phase 2 training results...")
    phase2_train = load_phase2_training_results()
    print(f"  ✅ Loaded {len(phase2_train)} Phase 2 training results")
    print()
    
    # Merge all results
    all_keys = set(phase1_hpo.keys()) | set(phase1_train.keys()) | set(phase2_hpo.keys()) | set(phase2_train.keys())
    
    comparison_data = []
    for key in sorted(all_keys):
        # Extract symbol and horizon from key
        parts = key.rsplit('_', 1)
        if len(parts) == 2:
            symbol = parts[0]
            horizon = parts[1]
        else:
            continue
        
        row = {
            'symbol': symbol,
            'horizon': horizon,
            'phase1_hpo_dirhit': phase1_hpo.get(key, {}).get('phase1_hpo_dirhit'),
            'phase1_training_dirhit': phase1_train.get(key, {}).get('phase1_training_dirhit'),
            'phase2_hpo_dirhit': phase2_hpo.get(key, {}).get('phase2_hpo_dirhit'),
            'phase2_training_dirhit': phase2_train.get(key, {}).get('phase2_training_dirhit'),
        }
        
        # Calculate differences
        if row['phase1_training_dirhit'] is not None and row['phase2_training_dirhit'] is not None:
            row['diff_training'] = row['phase2_training_dirhit'] - row['phase1_training_dirhit']
        else:
            row['diff_training'] = None
        
        if row['phase1_hpo_dirhit'] is not None and row['phase2_hpo_dirhit'] is not None:
            row['diff_hpo'] = row['phase2_hpo_dirhit'] - row['phase1_hpo_dirhit']
        else:
            row['diff_hpo'] = None
        
        comparison_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Filter for Phase 2 symbols (5 symbols: ALKA, ALKIM, ARASE, ARENA, ARSAN)
    phase2_symbols = {'ALKA', 'ALKIM', 'ARASE', 'ARENA', 'ARSAN'}
    df_phase2 = df[df['symbol'].isin(phase2_symbols)].copy()
    
    # Sort by symbol and horizon
    df_phase2['horizon_num'] = df_phase2['horizon'].str.replace('d', '').astype(int)
    df_phase2 = df_phase2.sort_values(['symbol', 'horizon_num'])
    df_phase2 = df_phase2.drop('horizon_num', axis=1)
    
    # Save to CSV
    output_dir = Path('/opt/bist-pattern/results')
    output_dir.mkdir(exist_ok=True)
    
    csv_file = output_dir / 'hpo_comparison_table.csv'
    df_phase2.to_csv(csv_file, index=False)
    print(f"✅ Comparison table saved to: {csv_file}")
    print()
    
    # Print summary table
    print("=" * 100)
    print("📊 COMPARISON TABLE (Phase 2 Symbols Only)")
    print("=" * 100)
    print()
    
    # Format table for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # Format DirHit values
    display_df = df_phase2.copy()
    for col in ['phase1_hpo_dirhit', 'phase1_training_dirhit', 'phase2_hpo_dirhit', 'phase2_training_dirhit', 'diff_training', 'diff_hpo']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    print(display_df.to_string(index=False))
    print()
    
    # Summary statistics
    print("=" * 100)
    print("📈 SUMMARY STATISTICS")
    print("=" * 100)
    print()
    
    # Count available results
    phase1_hpo_count = df_phase2['phase1_hpo_dirhit'].notna().sum()
    phase1_train_count = df_phase2['phase1_training_dirhit'].notna().sum()
    phase2_hpo_count = df_phase2['phase2_hpo_dirhit'].notna().sum()
    phase2_train_count = df_phase2['phase2_training_dirhit'].notna().sum()
    
    print(f"Phase 1 HPO results: {phase1_hpo_count}/{len(df_phase2)}")
    print(f"Phase 1 Training results: {phase1_train_count}/{len(df_phase2)}")
    print(f"Phase 2 HPO results: {phase2_hpo_count}/{len(df_phase2)}")
    print(f"Phase 2 Training results: {phase2_train_count}/{len(df_phase2)}")
    print()
    
    # Average DirHit for each phase
    if phase1_hpo_count > 0:
        avg_phase1_hpo = df_phase2['phase1_hpo_dirhit'].mean()
        print(f"Average Phase 1 HPO DirHit: {avg_phase1_hpo:.2f}%")
    
    if phase1_train_count > 0:
        avg_phase1_train = df_phase2['phase1_training_dirhit'].mean()
        print(f"Average Phase 1 Training DirHit: {avg_phase1_train:.2f}%")
    
    if phase2_hpo_count > 0:
        avg_phase2_hpo = df_phase2['phase2_hpo_dirhit'].mean()
        print(f"Average Phase 2 HPO DirHit: {avg_phase2_hpo:.2f}%")
    
    if phase2_train_count > 0:
        avg_phase2_train = df_phase2['phase2_training_dirhit'].mean()
        print(f"Average Phase 2 Training DirHit: {avg_phase2_train:.2f}%")
    print()
    
    # Compare training results (most important)
    if phase1_train_count > 0 and phase2_train_count > 0:
        df_both = df_phase2[df_phase2['phase1_training_dirhit'].notna() & df_phase2['phase2_training_dirhit'].notna()]
        if len(df_both) > 0:
            avg_diff = df_both['diff_training'].mean()
            better_phase2 = (df_both['diff_training'] > 0).sum()
            better_phase1 = (df_both['diff_training'] < 0).sum()
            equal = (df_both['diff_training'] == 0).sum()
            
            print("=" * 100)
            print("🎯 TRAINING RESULTS COMPARISON (Phase 1 vs Phase 2)")
            print("=" * 100)
            print(f"Average difference (Phase 2 - Phase 1): {avg_diff:+.2f}%")
            print(f"Phase 2 better: {better_phase2} cases")
            print(f"Phase 1 better: {better_phase1} cases")
            print(f"Equal: {equal} cases")
            print()
            
            if avg_diff > 0:
                print("✅ Phase 2 (features ON during HPO) performs BETTER on average")
            elif avg_diff < 0:
                print("✅ Phase 1 (features OFF during HPO) performs BETTER on average")
            else:
                print("➖ Both phases perform EQUALLY on average")
    
    print()
    print("=" * 100)

if __name__ == '__main__':
    create_comparison_table()

