#!/usr/bin/env python3
"""
Recovery mekanizması bug fix
BAYRK ve EKOS gibi durumları düzeltir
"""
import sys
import json
import sqlite3
from pathlib import Path

sys.path.insert(0, '/opt/bist-pattern')

STATE_FILE = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
HPO_STUDIES_DIR = Path('/opt/bist-pattern/hpo_studies')

HPO_TRIALS = 1500
MIN_TRIALS_FOR_RECOVERY = HPO_TRIALS - 10  # 1490


def load_state():
    """Load state file"""
    if not STATE_FILE.exists():
        return {}
    
    try:
        with open(STATE_FILE, 'r') as f:
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
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return {}


def fix_symbol(symbol, horizon, cycle):
    """Fix a symbol that was incorrectly marked as completed"""
    key = f"{symbol}_{horizon}d"
    
    # Check study database
    db_file = HPO_STUDIES_DIR / f"hpo_with_features_{symbol}_h{horizon}_c{cycle}.db"
    if not db_file.exists():
        db_file = HPO_STUDIES_DIR / f"hpo_with_features_{symbol}_h{horizon}.db"
    
    if not db_file.exists():
        print(f"❌ {key}: Study database bulunamadı")
        return False
    
    # Check complete trials
    try:
        conn = sqlite3.connect(str(db_file), timeout=30.0)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'")
        complete_trials = cursor.fetchone()[0]
        conn.close()
    except Exception as e:
        print(f"❌ {key}: Database okuma hatası: {e}")
        return False
    
    print(f"📊 {key}:")
    print(f"   Complete Trials: {complete_trials}/{HPO_TRIALS}")
    print(f"   MIN_TRIALS_FOR_RECOVERY: {MIN_TRIALS_FOR_RECOVERY}")
    
    if complete_trials >= MIN_TRIALS_FOR_RECOVERY:
        print(f"   ✅ HPO tamamlanmış - recovery yapılabilir")
        return False  # No fix needed, recovery should work
    else:
        print(f"   ❌ HPO tamamlanmamış - status 'pending' yapılmalı")
        
        # Load state
        state = load_state()
        tasks = state.get('state', {})
        task = tasks.get(key, {})
        
        if task.get('status') == 'completed' and not task.get('hpo_completed_at'):
            print(f"   🔧 Düzeltiliyor: status 'pending' yapılıyor...")
            
            # Fix: Reset to pending
            task['status'] = 'pending'
            task['training_completed_at'] = None
            task['error'] = None
            task['cycle'] = cycle
            
            # Save state
            tasks[key] = task
            state['state'] = tasks
            
            # Write back
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
            
            print(f"   ✅ Düzeltildi: {key} artık 'pending' durumunda")
            print(f"   ℹ️  HPO devam edecek (warm-start)")
            return True
        else:
            print(f"   ℹ️  Zaten düzgün durumda")
            return False


def main():
    print("=" * 100)
    print("RECOVERY BUG FIX")
    print("=" * 100)
    print()
    
    state = load_state()
    cycle = state.get('cycle', 1)
    
    print(f"Cycle: {cycle}")
    print()
    
    # Fix BAYRK and EKOS
    fixed = []
    for symbol, horizon in [('BAYRK', 1), ('EKOS', 1)]:
        if fix_symbol(symbol, horizon, cycle):
            fixed.append(f"{symbol}_{horizon}d")
        print()
    
    if fixed:
        print(f"✅ {len(fixed)} sembol düzeltildi: {', '.join(fixed)}")
        print()
        print("ℹ️  Bu semboller artık HPO'ya devam edecek (warm-start)")
        print("ℹ️  HPO tamamlandığında training otomatik başlayacak")
    else:
        print("ℹ️  Düzeltilecek sorun bulunamadı")
    print()


if __name__ == '__main__':
    main()

