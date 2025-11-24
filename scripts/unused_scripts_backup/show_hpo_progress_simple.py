#!/usr/bin/env python3
"""
HPO Pipeline İlerleme Göstergesi - Basit Versiyon
Gerçek zamanlı trial sayılarını gösterir
"""
import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, '/opt/bist-pattern')

STATE_FILE = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
HPO_STUDIES_DIR = Path('/opt/bist-pattern/hpo_studies')

def load_state():
    """Load pipeline state"""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def get_active_hpos():
    """Get active HPO processes from ps"""
    import subprocess
    active_hpos = {}
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if 'optuna_hpo_with_feature_flags' in line and '--symbols' in line:
                try:
                    symbol = line.split('--symbols')[1].split()[0].strip()
                    horizon = int(line.split('--horizon')[1].split()[0].strip())
                    trials = int(line.split('--trials')[1].split()[0].strip()) if '--trials' in line else 10
                    key = f"{symbol}_{horizon}d"
                    active_hpos[key] = {'symbol': symbol, 'horizon': horizon, 'target_trials': trials}
                except:
                    pass
    except:
        pass
    return active_hpos

def get_trial_counts():
    """Get trial counts from study database files"""
    trial_counts = {}
    if not HPO_STUDIES_DIR.exists():
        return trial_counts
    
    for db_file in HPO_STUDIES_DIR.glob('*.db'):
        try:
            # Check if file is recent (last hour)
            file_age = datetime.now().timestamp() - db_file.stat().st_mtime
            if file_age > 3600:
                continue
            
            # Extract symbol and horizon from filename
            # Format: hpo_with_features_{symbol}_h{horizon}_{timestamp}.db
            filename = db_file.stem
            if '_h' in filename:
                parts = filename.split('_h')
                symbol_part = parts[0].replace('hpo_with_features_', '')
                horizon_part = parts[1].split('_')[0]
                symbol = symbol_part
                horizon = int(horizon_part)
                key = f"{symbol}_{horizon}d"
                
                # Get trial counts from database
                conn = sqlite3.connect(str(db_file))
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM trials")
                total = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'")
                complete = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM trials WHERE state='RUNNING'")
                running = cursor.fetchone()[0]
                conn.close()
                
                trial_counts[key] = {
                    'total': total,
                    'complete': complete,
                    'running': running
                }
        except:
            pass
    
    return trial_counts

def main():
    """Main function"""
    print("=" * 100)
    print("📊 HPO PIPELINE İLERLEME RAPORU (Gerçek Zamanlı)")
    print("=" * 100)
    print()
    
    # Load state
    state_data = load_state()
    state = state_data.get('state', {})
    cycle = state_data.get('cycle', 0)
    
    print(f"🔄 Cycle: {cycle}")
    print(f"📋 Toplam Task (State): {len(state)}")
    print()
    
    # Get active HPO processes
    active_hpos = get_active_hpos()
    print(f"🔬 AKTIF HPO PROCESS'LERİ: {len(active_hpos)}")
    print("-" * 100)
    
    # Get trial counts
    trial_counts = get_trial_counts()
    
    # Show active HPOs
    if active_hpos:
        for key in sorted(active_hpos.keys()):
            info = active_hpos[key]
            symbol = info['symbol']
            horizon = info['horizon']
            target = info['target_trials']
            
            # Get trial count
            if key in trial_counts:
                tc = trial_counts[key]
                total = tc['total']
                complete = tc['complete']
                running = tc['running']
                print(f"   {key}: {total}/{target} trials (Running: {running}, Complete: {complete})")
            else:
                print(f"   {key}: Başlatılıyor... (0/{target} trials)")
    else:
        print("   Aktif HPO yok")
    print()
    
    # Group tasks by status from state
    by_status = defaultdict(list)
    for key, task in state.items():
        status = task.get('status', 'unknown')
        by_status[status].append((key, task))
    
    # Show training in progress
    training_in_progress = by_status.get('training_in_progress', [])
    if training_in_progress:
        print("🎯 EĞİTİM YAPILIYOR:")
        print("-" * 100)
        for key, task in training_in_progress:
            hpo_dirhit = task.get('hpo_dirhit', 0.0)
            print(f"   {key} (HPO DirHit: {hpo_dirhit:.2f}%)")
        print()
    
    # Show completed
    completed = by_status.get('completed', [])
    if completed:
        print(f"✅ TAMAMLANAN: {len(completed)} task")
        print("-" * 100)
        for key, task in completed[:10]:
            hpo_dirhit = task.get('hpo_dirhit', 0.0)
            adaptive_dirhit = task.get('adaptive_dirhit', 'N/A')
            if isinstance(adaptive_dirhit, (int, float)):
                print(f"   {key}: HPO={hpo_dirhit:.2f}%, Adaptive={adaptive_dirhit:.2f}%")
            else:
                print(f"   {key}: HPO={hpo_dirhit:.2f}%")
        print()
    
    # Statistics
    print("📊 İSTATİSTİKLER:")
    print("-" * 100)
    print(f"   Pending: {len(by_status.get('pending', []))}")
    print(f"   HPO In Progress (State): {len(by_status.get('hpo_in_progress', []))}")
    print(f"   HPO In Progress (Process): {len(active_hpos)}")
    print(f"   Training In Progress: {len(training_in_progress)}")
    print(f"   Completed: {len(completed)}")
    print(f"   Failed: {len(by_status.get('failed', []))}")
    print(f"   Skipped: {len(by_status.get('skipped', []))}")
    print()
    
    print("=" * 100)
    print(f"📅 Rapor Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

if __name__ == '__main__':
    main()
