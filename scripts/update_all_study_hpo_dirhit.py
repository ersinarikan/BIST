#!/usr/bin/env python3
"""
Update hpo_dirhit for ALL symbols (completed + in-progress) from study files (with 10/5.0 filter)
This ensures all hpo_dirhit values are filtered (10/5.0) regardless of status
"""

import sys
import os
import json
import subprocess
import re
from pathlib import Path
from typing import Optional, Set, Tuple

sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

from scripts.continuous_hpo_training_pipeline import STATE_FILE
from scripts.retrain_high_discrepancy_symbols import find_best_trial_with_filter_applied


def find_study_db(symbol: str, horizon: int, cycle: int) -> Optional[Path]:
    """Find study database file"""
    study_dir = Path('/opt/bist-pattern/hpo_studies')
    if not study_dir.exists():
        return None
    
    cycle_file = study_dir / f"hpo_with_features_{symbol}_h{horizon}_c{cycle}.db"
    if cycle_file.exists():
        return cycle_file
    
    # Legacy format
    if cycle == 1:
        legacy_file = study_dir / f"hpo_with_features_{symbol}_h{horizon}.db"
        if legacy_file.exists():
            return legacy_file
    
    return None


def get_active_hpo_symbols() -> Set[Tuple[str, int]]:
    """Get symbols with active HPO processes"""
    active = set()
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if 'optuna_hpo_with_feature_flags' in line and '--symbols' in line:
                match = re.search(r'--symbols\s+(\w+).*--horizon\s+(\d+)', line)
                if match:
                    symbol = match.group(1)
                    horizon = int(match.group(2))
                    active.add((symbol, horizon))
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Failed to parse active HPO processes: {e}")
    return active


def get_filtered_best_dirhit_from_study(db_file: Path, symbol: str, horizon: int,
                                       min_mask_count: int = 10, min_mask_pct: float = 5.0) -> Optional[float]:
    """Get best DirHit from study with filter applied"""
    try:
        filtered_trial, filtered_score = find_best_trial_with_filter_applied(
            db_file, symbol, horizon, min_mask_count, min_mask_pct
        )
        
        if not filtered_trial:
            return None
        
        # Get symbol-specific avg_dirhit
        symbol_key = f"{symbol}_{horizon}d"
        symbol_metrics = filtered_trial.user_attrs.get('symbol_metrics', {})
        if symbol_key in symbol_metrics:
            avg_dirhit = symbol_metrics[symbol_key].get('avg_dirhit')
            if avg_dirhit is not None:
                return float(avg_dirhit)
        
        # Fallback: use filtered_score
        return filtered_score
    except Exception:
        return None


def main():
    # Load state
    if not STATE_FILE.exists():
        print("❌ State file not found")
        return
    
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    
    current_cycle = state.get('cycle', 1)
    tasks = state.get('state', {})
    
    # Get active HPO symbols
    active_hpo = get_active_hpo_symbols()
    
    print("=" * 80)
    print("TÜM SEMBOLLER İÇİN HPO_DIRHIT GÜNCELLEME (STUDY DOSYALARINDAN)")
    print("=" * 80)
    print(f"\n🔄 Cycle: {current_cycle}")
    print(f"🔍 Filter: 10/5.0")
    print(f"📊 Aktif HPO: {len(active_hpo)} sembol")
    
    # Get all symbols with study files
    study_dir = Path('/opt/bist-pattern/hpo_studies')
    if not study_dir.exists():
        print("❌ Study directory not found")
        return
    
    # Find all study files for cycle 2, horizon 1
    study_files = list(study_dir.glob(f"hpo_with_features_*_h1_c{current_cycle}.db"))
    
    print(f"\n📊 Study dosyaları: {len(study_files)}")
    
    updated_count = 0
    no_trial_count = 0
    already_correct_count = 0
    
    processed_symbols = set()
    
    for study_file in study_files:
        # Extract symbol from filename: hpo_with_features_SYMBOL_h1_c2.db
        parts = study_file.stem.split('_')
        if len(parts) >= 4:
            symbol = parts[3]
            horizon = 1
            
            if (symbol, horizon) in processed_symbols:
                continue
            processed_symbols.add((symbol, horizon))
            
            key = f"{symbol}_{horizon}d"
            task = tasks.get(key, {})
            
            # Skip if task doesn't exist or is not relevant
            if not task:
                continue
            
            # Get filtered best DirHit from study
            filtered_dirhit = get_filtered_best_dirhit_from_study(study_file, symbol, horizon, 10, 5.0)
            
            if filtered_dirhit is None:
                no_trial_count += 1
                continue
            
            # Update state if needed
            current_hpo_dirhit = task.get('hpo_dirhit')
            if current_hpo_dirhit is None or abs(current_hpo_dirhit - filtered_dirhit) > 0.01:
                task['hpo_dirhit'] = float(filtered_dirhit)
                tasks[key] = task
                updated_count += 1
                status = task.get('status', 'unknown')
                is_active = (symbol, horizon) in active_hpo
                if current_hpo_dirhit is not None:
                    print(f"✅ {symbol}_{horizon}d ({status}{', aktif HPO' if is_active else ''}): {current_hpo_dirhit:.2f}% → {filtered_dirhit:.2f}%")
                else:
                    print(f"✅ {symbol}_{horizon}d ({status}{', aktif HPO' if is_active else ''}): {filtered_dirhit:.2f}% (yeni)")
            else:
                already_correct_count += 1
    
    # Save state
    if updated_count > 0:
        state['state'] = tasks
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"\n✅ {updated_count} sembol güncellendi")
    else:
        print(f"\n✅ Güncellenecek sembol yok")
    
    print(f"\n📊 Özet:")
    print(f"   Güncellenen: {updated_count}")
    print(f"   Zaten doğru: {already_correct_count}")
    print(f"   Best trial bulunamadı: {no_trial_count}")
    print(f"   Aktif HPO: {len(active_hpo)} sembol")


if __name__ == '__main__':
    main()

