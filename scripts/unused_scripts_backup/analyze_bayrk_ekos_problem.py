#!/usr/bin/env python3
"""
BAYRK ve EKOS problemi detaylı analizi
Neden HPO tamamlanmadan training yapılmış?
"""
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/opt/bist-pattern')

STATE_FILE = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
HPO_STUDIES_DIR = Path('/opt/bist-pattern/hpo_studies')
RESULTS_DIR = Path('/opt/bist-pattern/results')

# Constants
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


def analyze_symbol(symbol, horizon, cycle=2):
    """Analyze a symbol in detail"""
    print("=" * 100)
    print(f"{symbol}_{horizon}d DETAYLI ANALİZ")
    print("=" * 100)
    print()
    
    # Load state
    state = load_state()
    tasks = state.get('state', {})
    key = f"{symbol}_{horizon}d"
    task = tasks.get(key, {})
    
    print("📋 STATE DOSYASI BİLGİLERİ:")
    print("-" * 100)
    print(f"   Status: {task.get('status')}")
    print(f"   Cycle: {task.get('cycle')}")
    print(f"   HPO Completed At: {task.get('hpo_completed_at')}")
    print(f"   Training Completed At: {task.get('training_completed_at')}")
    print(f"   Best Params File: {task.get('best_params_file')}")
    print(f"   HPO DirHit: {task.get('hpo_dirhit')}")
    print(f"   Training DirHit: {task.get('training_dirhit')}")
    print(f"   Error: {task.get('error')}")
    print(f"   Retry Count: {task.get('retry_count')}")
    print()
    
    # Check study database
    db_file = HPO_STUDIES_DIR / f"hpo_with_features_{symbol}_h{horizon}_c{cycle}.db"
    if not db_file.exists():
        db_file = HPO_STUDIES_DIR / f"hpo_with_features_{symbol}_h{horizon}.db"
    
    print("📊 STUDY DATABASE BİLGİLERİ:")
    print("-" * 100)
    if db_file.exists():
        print(f"   Dosya: {db_file}")
        print(f"   Boyut: {db_file.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"   Son Değişiklik: {datetime.fromtimestamp(db_file.stat().st_mtime).isoformat()}")
        
        try:
            conn = sqlite3.connect(str(db_file), timeout=30.0)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM trials")
            total_trials = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'")
            complete_trials = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trials WHERE state='RUNNING'")
            running_trials = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trials WHERE state='FAIL'")
            failed_trials = cursor.fetchone()[0]
            
            print(f"   Total Trials: {total_trials}")
            print(f"   Complete Trials: {complete_trials}")
            print(f"   Running Trials: {running_trials}")
            print(f"   Failed Trials: {failed_trials}")
            print()
            
            print(f"   ⚠️  MIN_TRIALS_FOR_RECOVERY: {MIN_TRIALS_FOR_RECOVERY}")
            print(f"   ⚠️  Complete Trials: {complete_trials}")
            if complete_trials >= MIN_TRIALS_FOR_RECOVERY:
                print(f"   ✅ HPO tamamlanmış sayılır (>= {MIN_TRIALS_FOR_RECOVERY})")
            else:
                print(f"   ❌ HPO tamamlanmamış (< {MIN_TRIALS_FOR_RECOVERY})")
                print(f"   ⚠️  Recovery mekanizması training başlatmamalı!")
            
            # Get best trial
            cursor.execute("""
                SELECT t.number, tv.value, t.datetime_start, t.datetime_complete
                FROM trials t
                JOIN trial_values tv ON t.trial_id = tv.trial_id
                WHERE t.state='COMPLETE' AND tv.value IS NOT NULL AND tv.value_type='FINITE'
                ORDER BY tv.value DESC
                LIMIT 1
            """)
            best = cursor.fetchone()
            if best:
                print()
                print(f"   Best Trial: #{best[0]}")
                print(f"   Best Score: {best[1]:.4f}")
                print(f"   Start: {best[2]}")
                print(f"   Complete: {best[3]}")
            
            conn.close()
        except Exception as e:
            print(f"   ❌ Hata: {e}")
    else:
        print(f"   ❌ Study database bulunamadı: {db_file}")
    print()
    
    # Check JSON files
    print("📄 JSON DOSYALARI:")
    print("-" * 100)
    json_files = list(RESULTS_DIR.glob(f'optuna_pilot_features_on_h{horizon}_c{cycle}_*.json'))
    json_files.extend(list(RESULTS_DIR.glob(f'optuna_pilot_features_on_h{horizon}_*.json')))
    
    found_json = None
    for json_file in sorted(json_files, key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            if symbol in data.get('symbols', []):
                found_json = json_file
                print(f"   ✅ Bulundu: {json_file.name}")
                print(f"      Best DirHit: {data.get('best_dirhit')}")
                print(f"      Best Trial: #{data.get('best_trial_number')}")
                print(f"      N Trials: {data.get('n_trials')}")
                break
        except Exception:
            continue
    
    if not found_json:
        print(f"   ❌ JSON dosyası bulunamadı")
    print()
    
    # Analyze the problem
    print("🔍 PROBLEM ANALİZİ:")
    print("-" * 100)
    
    issues = []
    
    # Issue 1: HPO not completed but training done
    if task.get('status') == 'completed' and task.get('training_completed_at'):
        if not task.get('hpo_completed_at'):
            issues.append({
                'type': 'HPO_NOT_COMPLETED',
                'severity': 'CRITICAL',
                'message': 'Training yapılmış ama HPO tamamlanmamış'
            })
    
    # Issue 2: No best_params_file
    if task.get('status') == 'completed' and not task.get('best_params_file'):
        issues.append({
            'type': 'NO_BEST_PARAMS',
            'severity': 'CRITICAL',
            'message': 'Training yapılmış ama best_params_file yok'
        })
    
    # Issue 3: Recovery should not have worked
    if db_file.exists():
        try:
            conn = sqlite3.connect(str(db_file), timeout=30.0)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'")
            complete_trials = cursor.fetchone()[0]
            conn.close()
            
            if complete_trials < MIN_TRIALS_FOR_RECOVERY:
                issues.append({
                    'type': 'RECOVERY_SHOULD_NOT_WORK',
                    'severity': 'HIGH',
                    'message': f'Recovery mekanizması çalışmamalıydı ({complete_trials} < {MIN_TRIALS_FOR_RECOVERY})'
                })
        except Exception:
            pass
    
    for issue in issues:
        print(f"   ⚠️  [{issue['severity']}] {issue['type']}: {issue['message']}")
    
    if not issues:
        print("   ✅ Sorun bulunamadı")
    print()
    
    # Possible causes
    print("💡 OLASI NEDENLER:")
    print("-" * 100)
    print("   1. Recovery mekanizmasında bug:")
    print("      - complete_trials < MIN_TRIALS_FOR_RECOVERY olmasına rağmen training başlatılmış")
    print("      - Kod satırı 3376'da kontrol yapılıyor ama belki bir exception olmuş")
    print()
    print("   2. Başka bir script training'i doğrudan çağırmış:")
    print("      - retrain_completed_all.py (ama bu sadece best_params_file olanları işler)")
    print("      - Başka bir script veya manuel çağrı")
    print()
    print("   3. State dosyası manuel olarak değiştirilmiş:")
    print("      - Status 'completed' yapılmış")
    print("      - Training completed_at set edilmiş")
    print()
    print("   4. Recovery mekanizması yanlış çalışmış:")
    print("      - JSON dosyası bulunamadı ama yine de training başlatılmış")
    print("      - Satır 3419'da hpo_result kontrolü yapılıyor ama belki bir bug var")
    print()
    
    # Check recovery logic
    print("🔧 RECOVERY MEKANİZMASI KONTROLÜ:")
    print("-" * 100)
    if db_file.exists():
        try:
            conn = sqlite3.connect(str(db_file), timeout=30.0)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'")
            complete_trials = cursor.fetchone()[0]
            conn.close()
            
            print(f"   Complete Trials: {complete_trials}")
            print(f"   MIN_TRIALS_FOR_RECOVERY: {MIN_TRIALS_FOR_RECOVERY}")
            print()
            
            if task.get('status') == 'completed' and not task.get('best_params_file'):
                print("   Recovery koşulları:")
                print(f"      - task.status in ('failed', 'completed'): {task.get('status') in ('failed', 'completed')}")
                print(f"      - best_params_file is None: {task.get('best_params_file') is None}")
                print()
                
                if complete_trials >= MIN_TRIALS_FOR_RECOVERY:
                    print("   ✅ Recovery çalışmalı (complete_trials >= MIN_TRIALS_FOR_RECOVERY)")
                    print("   ⚠️  Ama JSON dosyası bulunamadı, training başlatılmamalıydı")
                else:
                    print("   ❌ Recovery çalışmamalı (complete_trials < MIN_TRIALS_FOR_RECOVERY)")
                    print("   ⚠️  Ama training yapılmış - BU BİR BUG!")
        except Exception as e:
            print(f"   ❌ Hata: {e}")


def main():
    print("=" * 100)
    print("BAYRK ve EKOS PROBLEMİ DETAYLI ANALİZ")
    print("=" * 100)
    print()
    
    analyze_symbol('BAYRK', 1, cycle=2)
    print()
    print()
    analyze_symbol('EKOS', 1, cycle=2)


if __name__ == '__main__':
    main()

