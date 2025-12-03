#!/usr/bin/env python3
"""
Fix missing hpo_dirhit in state file by reading from study database
"""
import json
import sqlite3
from pathlib import Path

STATE_FILE = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
HPO_STUDIES_DIR = Path('/opt/bist-pattern/hpo_studies')

def load_state():
    """Load pipeline state"""
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
    except Exception:
        return {}

def save_state(state):
    """Save pipeline state"""
    try:
        # Atomic write
        tmp_file = STATE_FILE.with_suffix('.json.tmp')
        with open(tmp_file, 'w') as f:
            json.dump(state, f, indent=2)
        tmp_file.replace(STATE_FILE)
        return True
    except Exception as e:
        print(f"❌ Error saving state: {e}")
        return False

def get_hpo_dirhit_from_study(symbol: str, horizon: int, cycle: int) -> float:
    """Get HPO DirHit from study database (symbol-specific avg_dirhit from best trial)"""
    study_file = HPO_STUDIES_DIR / f"hpo_with_features_{symbol}_h{horizon}_c{cycle}.db"
    
    if not study_file.exists():
        return None
    
    try:
        conn = sqlite3.connect(str(study_file), timeout=30.0)
        cursor = conn.cursor()
        
        # Get best trial
        cursor.execute("""
            SELECT t.number, tv.value
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state='COMPLETE' AND tv.value IS NOT NULL AND tv.value_type='FINITE'
            ORDER BY tv.value DESC
            LIMIT 1
        """)
        best_trial_row = cursor.fetchone()
        
        if not best_trial_row:
            conn.close()
            return None
        
        best_trial_number, best_value = best_trial_row
        
        # ✅ FIX: Get symbol-specific avg_dirhit from symbol_metrics (not all symbols avg)
        symbol_key = f"{symbol}_{horizon}d"
        cursor.execute("""
            SELECT value_json
            FROM trial_user_attributes
            WHERE trial_id = ? AND key = 'symbol_metrics'
        """, (best_trial_number,))
        row = cursor.fetchone()
        
        if row:
            try:
                symbol_metrics = json.loads(row[0])
                if isinstance(symbol_metrics, dict) and symbol_key in symbol_metrics:
                    sym_metrics = symbol_metrics[symbol_key]
                    if isinstance(sym_metrics, dict):
                        symbol_avg_dirhit = sym_metrics.get('avg_dirhit')
                        if symbol_avg_dirhit is not None:
                            conn.close()
                            return float(symbol_avg_dirhit)
            except Exception:
                pass
        
        # Fallback: Try avg_dirhit (all symbols average)
        cursor.execute("""
            SELECT value_json
            FROM trial_user_attributes
            WHERE trial_id = ? AND key = 'avg_dirhit'
        """, (best_trial_number,))
        row = cursor.fetchone()
        
        if row:
            try:
                avg_dirhit = float(json.loads(row[0]))
                conn.close()
                return avg_dirhit
            except Exception:
                pass
        
        # Final fallback: Try dirhit
        cursor.execute("""
            SELECT value_json
            FROM trial_user_attributes
            WHERE trial_id = ? AND key = 'dirhit'
        """, (best_trial_number,))
        row = cursor.fetchone()
        
        if row:
            try:
                dirhit = float(json.loads(row[0]))
                conn.close()
                return dirhit
            except Exception:
                pass
        
        conn.close()
        return None
    except Exception as e:
        print(f"❌ Error reading study {study_file}: {e}")
        return None

def main():
    """Main function"""
    print("=" * 100)
    print("HPO DIRHIT EKSİK SEMBOLLERİ DÜZELTME")
    print("=" * 100)
    
    state = load_state()
    if not state:
        print("❌ State dosyası yüklenemedi!")
        return 1
    
    current_cycle = state.get('cycle', 1)
    tasks = state.get('state', {})
    
    # Find symbols with missing hpo_dirhit but completed status
    missing_hpo_dirhit = []
    for key, task in tasks.items():
        if not isinstance(task, dict):
            continue
        if task.get('status') != 'completed':
            continue
        if task.get('cycle', 0) != current_cycle:
            continue
        if task.get('hpo_dirhit') is None:
            # Check if training was done
            if task.get('training_dirhit_wfv') is not None or task.get('training_completed_at'):
                missing_hpo_dirhit.append(key)
    
    if not missing_hpo_dirhit:
        print("\n✅ HPO DirHit eksik sembol bulunamadı!")
        return 0
    
    print(f"\n📋 {len(missing_hpo_dirhit)} sembol için HPO DirHit eksik:")
    for key in sorted(missing_hpo_dirhit):
        print(f"   {key}")
    
    print("\n🔍 Study dosyalarından okunuyor...")
    
    fixed = 0
    for key in missing_hpo_dirhit:
        parts = key.split('_')
        if len(parts) != 2:
            continue
        
        symbol = parts[0]
        horizon_str = parts[1].replace('d', '')
        try:
            horizon = int(horizon_str)
        except ValueError:
            continue
        
        # Get hpo_dirhit from study
        hpo_dirhit = get_hpo_dirhit_from_study(symbol, horizon, current_cycle)
        
        if hpo_dirhit is not None:
            task = tasks[key]
            old_hpo_dirhit = task.get('hpo_dirhit')
            task['hpo_dirhit'] = hpo_dirhit
            
            print(f"✅ {key}: hpo_dirhit = {hpo_dirhit:.2f}% (study'den okundu)")
            fixed += 1
        else:
            print(f"⚠️ {key}: Study'den hpo_dirhit okunamadı")
    
    if fixed > 0:
        state['state'] = tasks
        if save_state(state):
            print(f"\n✅ {fixed} sembol için hpo_dirhit state'e eklendi!")
            return 0
        else:
            print(f"\n❌ State dosyası kaydedilemedi!")
            return 1
    else:
        print("\n⚠️ Hiçbir sembol için hpo_dirhit bulunamadı!")
        return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

