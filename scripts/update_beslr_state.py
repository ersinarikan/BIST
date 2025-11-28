#!/usr/bin/env python3
"""
BESLR state dosyasını güncelle - yeni HPO DirHit ve Training DirHit değerleriyle
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, '/opt/bist-pattern')

def load_state():
    """Load pipeline state"""
    state_file = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
    if not state_file.exists():
        return {}
    try:
        with open(state_file, 'r') as f:
            content = f.read().strip()
            if content.count('{') > 1:
                last_brace = content.rfind('}')
                if last_brace > 0:
                    brace_count = 0
                    start_pos = last_brace
                    for i in range(last_brace, -1, -1):
                        if content[i] == '}':
                            brace_count += 1
                        elif content[i] == '{':
                            brace_count -= 1
                            if brace_count == 0:
                                start_pos = i
                                break
                    content = content[start_pos:last_brace+1]
            return json.loads(content)
    except Exception:
        return {}

def save_state(state):
    """Save pipeline state"""
    state_file = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
    try:
        # Write to temp file first, then atomic rename
        temp_file = state_file.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)
        temp_file.replace(state_file)
        return True
    except Exception as e:
        print(f"Error saving state: {e}")
        return False

def main():
    symbol = 'BESLR'
    horizon = 1
    key = f"{symbol}_{horizon}d"
    
    print("=" * 100)
    print(f"🔄 BESLR {horizon}d State Dosyası Güncelleniyor")
    print("=" * 100)
    print()
    
    # Load state
    state = load_state()
    if not state:
        print("❌ State dosyası yüklenemedi!")
        return 1
    
    tasks = state.get('state', {})
    if key not in tasks:
        print(f"❌ {key} task bulunamadı!")
        return 1
    
    task = tasks[key]
    
    print("📊 Mevcut Değerler:")
    print(f"   HPO DirHit: {task.get('hpo_dirhit', 'N/A')}%")
    print(f"   Training DirHit: {task.get('training_dirhit', 'N/A')}%")
    print(f"   Training DirHit (WFV): {task.get('training_dirhit_wfv', 'N/A')}%")
    print(f"   Adaptive DirHit: {task.get('adaptive_dirhit', 'N/A')}%")
    print()
    
    # Update values
    # HPO DirHit: Trial #957'den (73.33% - güvenilir)
    # Training DirHit: Yeni eğitimden (71.43%)
    old_hpo_dirhit = task.get('hpo_dirhit')
    old_training_dirhit = task.get('training_dirhit')
    
    task['hpo_dirhit'] = 73.33  # Trial #957 DirHit (güvenilir)
    task['training_dirhit'] = 71.43  # Yeni training DirHit
    task['training_dirhit_wfv'] = 71.43  # WFV DirHit
    task['training_dirhit_online'] = 71.43  # Online DirHit
    task['adaptive_dirhit'] = 71.43  # Adaptive DirHit
    
    print("📝 Yeni Değerler:")
    print(f"   HPO DirHit: {task['hpo_dirhit']}% (Trial #957 - güvenilir)")
    print(f"   Training DirHit: {task['training_dirhit']}% (yeni eğitim)")
    print(f"   Training DirHit (WFV): {task['training_dirhit_wfv']}%")
    print(f"   Adaptive DirHit: {task['adaptive_dirhit']}%")
    print()
    
    # Save state
    tasks[key] = task
    state['state'] = tasks
    
    if save_state(state):
        print("✅ State dosyası güncellendi!")
        print()
        print("📊 Değişiklikler:")
        print(f"   HPO DirHit: {old_hpo_dirhit}% → {task['hpo_dirhit']}%")
        print(f"   Training DirHit: {old_training_dirhit}% → {task['training_dirhit']}%")
        print()
        return 0
    else:
        print("❌ State dosyası kaydedilemedi!")
        return 1

if __name__ == '__main__':
    sys.exit(main())

