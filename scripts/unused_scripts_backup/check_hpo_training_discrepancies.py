#!/usr/bin/env python3
"""
Check HPO DirHit and Training DirHit discrepancies for all completed tasks
"""
import sys
import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional, List

sys.path.insert(0, '/opt/bist-pattern')

STATE_FILE = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
HPO_STUDIES_DIR = Path('/opt/bist-pattern/hpo_studies')
RESULTS_DIR = Path('/opt/bist-pattern/results')


def load_state() -> Dict:
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
    except Exception as e:
        print(f"Error loading state: {e}", file=sys.stderr)
        return {}


def find_study_db(symbol: str, horizon: int, cycle: Optional[int] = None) -> Optional[Path]:
    """Find the most recent study database file for symbol-horizon"""
    if cycle is None:
        state = load_state()
        cycle = state.get('cycle', 1)
    
    study_dir = HPO_STUDIES_DIR
    if not study_dir.exists():
        return None
    
    # Priority 1: Cycle format
    cycle_format_file = study_dir / f"hpo_with_features_{symbol}_h{horizon}_c{cycle}.db"
    if cycle_format_file.exists():
        return cycle_format_file
    
    # Priority 2: Legacy format (no cycle) - only if cycle is 1
    if cycle == 1:
        legacy_format_file = study_dir / f"hpo_with_features_{symbol}_h{horizon}.db"
        if legacy_format_file.exists():
            return legacy_format_file
    
    # Priority 3: Old format (with timestamp)
    old_format_pattern = f"hpo_with_features_{symbol}_h{horizon}_*.db"
    old_format_files = list(study_dir.glob(old_format_pattern))
    if old_format_files:
        timestamp_files = [f for f in old_format_files if not f.name.endswith(f'_c{cycle}.db')]
        if timestamp_files:
            timestamp_files = sorted(timestamp_files, key=lambda p: p.stat().st_mtime, reverse=True)
            return timestamp_files[0]
    
    return None


def get_hpo_dirhit_from_db(db_file: Path, symbol: str, horizon: int) -> Optional[Dict]:
    """Get HPO DirHit from study database"""
    try:
        if not db_file.exists():
            return None
        
        conn = sqlite3.connect(str(db_file), timeout=30.0)
        cursor = conn.cursor()
        
        # Get best trial (highest value)
        cursor.execute("""
            SELECT t.number, tv.value, t.state
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state='COMPLETE' AND tv.value IS NOT NULL AND tv.value_type='FINITE'
            ORDER BY tv.value DESC
            LIMIT 1
        """)
        best_trial_row = cursor.fetchone()
        best_trial_number = None
        best_value = None
        if best_trial_row:
            best_trial_number, best_value, _ = best_trial_row
        
        # Get DirHit from user_attrs for best trial
        best_dirhit = None
        if best_trial_number is not None:
            try:
                cursor.execute("""
                    SELECT value_json
                    FROM trial_user_attributes
                    WHERE trial_id = ? AND key = 'dirhit'
                """, (best_trial_number,))
                row = cursor.fetchone()
                if row:
                    best_dirhit = float(json.loads(row[0]))
            except Exception:
                try:
                    cursor.execute("""
                        SELECT value
                        FROM trial_user_attrs
                        WHERE trial_id = ? AND key = 'dirhit'
                    """, (best_trial_number,))
                    row = cursor.fetchone()
                    if row:
                        best_dirhit = float(row[0])
                except Exception:
                    pass
        
        # Get total trials
        cursor.execute("SELECT COUNT(*) FROM trials")
        total_trials = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'")
        complete_trials = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'best_trial_number': best_trial_number,
            'best_value': best_value,
            'best_dirhit': best_dirhit,
            'total_trials': total_trials,
            'complete_trials': complete_trials
        }
    except Exception as e:
        return {'error': str(e)}


def get_hpo_dirhit_from_json(json_file: Path) -> Optional[float]:
    """Get HPO DirHit from JSON file"""
    try:
        if not json_file.exists():
            return None
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Try symbol-specific DirHit first
        symbol = None
        horizon = None
        # Extract from filename or check all symbols
        best_dirhit = data.get('best_dirhit')
        
        # Check best_trial_metrics for symbol-specific DirHit
        best_trial_metrics = data.get('best_trial_metrics', {})
        if best_trial_metrics:
            # Try to find symbol-specific DirHit
            for key, metrics in best_trial_metrics.items():
                if isinstance(metrics, dict):
                    symbol_dirhit = metrics.get('avg_dirhit')
                    if symbol_dirhit is not None:
                        return symbol_dirhit
        
        return best_dirhit
    except Exception as e:
        return None


def check_symbol(symbol: str, horizon: int, task: Dict) -> Dict:
    """Check a single symbol-horizon pair"""
    result = {
        'symbol': symbol,
        'horizon': horizon,
        'key': f"{symbol}_{horizon}d",
        'status': task.get('status'),
        'cycle': task.get('cycle'),
        'state_hpo_dirhit': task.get('hpo_dirhit'),
        'state_training_dirhit': task.get('training_dirhit'),
        'state_adaptive_dirhit': task.get('adaptive_dirhit'),
        'state_training_dirhit_wfv': task.get('training_dirhit_wfv'),
        'state_training_dirhit_online': task.get('training_dirhit_online'),
        'db_hpo_dirhit': None,
        'json_hpo_dirhit': None,
        'db_file': None,
        'json_file': None,
        'issues': []
    }
    
    # Check study database
    cycle = task.get('cycle', 1)
    db_file = find_study_db(symbol, horizon, cycle=cycle)
    if db_file:
        result['db_file'] = str(db_file)
        db_info = get_hpo_dirhit_from_db(db_file, symbol, horizon)
        if db_info:
            result['db_hpo_dirhit'] = db_info.get('best_dirhit')
            result['db_best_trial'] = db_info.get('best_trial_number')
            result['db_total_trials'] = db_info.get('total_trials')
            result['db_complete_trials'] = db_info.get('complete_trials')
    
    # Check JSON file
    json_file_path = task.get('best_params_file')
    if json_file_path:
        json_file = Path(json_file_path)
        result['json_file'] = str(json_file)
        if json_file.exists():
            result['json_hpo_dirhit'] = get_hpo_dirhit_from_json(json_file)
    
    # Identify issues
    if result['state_hpo_dirhit'] is None:
        result['issues'].append("HPO DirHit eksik state dosyasında")
        if result['db_hpo_dirhit'] is not None:
            result['issues'].append(f"Ancak DB'de HPO DirHit var: {result['db_hpo_dirhit']:.2f}%")
        elif result['json_hpo_dirhit'] is not None:
            result['issues'].append(f"Ancak JSON'da HPO DirHit var: {result['json_hpo_dirhit']:.2f}%")
        else:
            result['issues'].append("DB ve JSON'da da HPO DirHit bulunamadı")
    
    # Check for discrepancies
    if result['state_hpo_dirhit'] is not None and result['state_training_dirhit'] is not None:
        diff = abs(result['state_hpo_dirhit'] - result['state_training_dirhit'])
        if diff > 20:  # More than 20% difference
            result['issues'].append(f"Büyük fark: HPO={result['state_hpo_dirhit']:.2f}% vs Training={result['state_training_dirhit']:.2f}% (fark: {diff:.2f}%)")
    
    # Check if DB and state HPO DirHit match
    if result['state_hpo_dirhit'] is not None and result['db_hpo_dirhit'] is not None:
        if abs(result['state_hpo_dirhit'] - result['db_hpo_dirhit']) > 1.0:
            result['issues'].append(f"State ve DB HPO DirHit farklı: State={result['state_hpo_dirhit']:.2f}% vs DB={result['db_hpo_dirhit']:.2f}%")
    
    return result


def main():
    """Main function"""
    state = load_state()
    tasks = state.get('state', {})
    current_cycle = state.get('cycle', 1)
    
    print("=" * 100)
    print("HPO DirHit ve Training DirHit Kontrol Raporu")
    print("=" * 100)
    print(f"Cycle: {current_cycle}")
    print()
    
    # Get all completed tasks
    completed_tasks = []
    for key, task in tasks.items():
        if isinstance(task, dict) and task.get('status') == 'completed':
            task_cycle = task.get('cycle', 0)
            if task_cycle != current_cycle:
                continue
            
            symbol = task.get('symbol', '')
            horizon = task.get('horizon', 0)
            if symbol and horizon:
                completed_tasks.append((symbol, horizon, task))
    
    print(f"Toplam {len(completed_tasks)} tamamlanan görev bulundu")
    print()
    
    # Check each task
    results = []
    for symbol, horizon, task in sorted(completed_tasks):
        result = check_symbol(symbol, horizon, task)
        results.append(result)
    
    # Group by issue type
    missing_hpo = [r for r in results if r['state_hpo_dirhit'] is None]
    large_discrepancies = [r for r in results if r['state_hpo_dirhit'] is not None and r['state_training_dirhit'] is not None and abs(r['state_hpo_dirhit'] - r['state_training_dirhit']) > 20]
    db_state_mismatch = [r for r in results if r['state_hpo_dirhit'] is not None and r['db_hpo_dirhit'] is not None and abs(r['state_hpo_dirhit'] - r['db_hpo_dirhit']) > 1.0]
    
    # Print summary
    print("📊 ÖZET:")
    print("-" * 100)
    print(f"   Toplam kontrol edilen: {len(results)}")
    print(f"   HPO DirHit eksik: {len(missing_hpo)}")
    print(f"   Büyük farklar (>20%): {len(large_discrepancies)}")
    print(f"   State-DB uyumsuzluğu: {len(db_state_mismatch)}")
    print()
    
    # Print missing HPO DirHit
    if missing_hpo:
        print("⚠️  HPO DirHit EKSİK OLANLAR:")
        print("-" * 100)
        for r in sorted(missing_hpo, key=lambda x: x['key']):
            print(f"   {r['key']}:")
            print(f"      Status: {r['status']}, Cycle: {r['cycle']}")
            print(f"      Training DirHit: {r['state_training_dirhit']:.2f}%" if r['state_training_dirhit'] else "      Training DirHit: Yok")
            if r['db_hpo_dirhit']:
                print(f"      DB HPO DirHit: {r['db_hpo_dirhit']:.2f}% (Trial #{r.get('db_best_trial', '?')})")
            if r['json_hpo_dirhit']:
                print(f"      JSON HPO DirHit: {r['json_hpo_dirhit']:.2f}%")
            if r['issues']:
                for issue in r['issues']:
                    print(f"      ⚠️  {issue}")
            print()
    
    # Print large discrepancies
    if large_discrepancies:
        print("⚠️  BÜYÜK FARKLAR (>20%):")
        print("-" * 100)
        for r in sorted(large_discrepancies, key=lambda x: abs(x['state_hpo_dirhit'] - x['state_training_dirhit']), reverse=True):
            diff = abs(r['state_hpo_dirhit'] - r['state_training_dirhit'])
            print(f"   {r['key']}:")
            print(f"      HPO DirHit: {r['state_hpo_dirhit']:.2f}%")
            print(f"      Training DirHit: {r['state_training_dirhit']:.2f}%")
            print(f"      Fark: {diff:.2f}%")
            if r['db_hpo_dirhit']:
                print(f"      DB HPO DirHit: {r['db_hpo_dirhit']:.2f}%")
            if r['issues']:
                for issue in r['issues']:
                    print(f"      ⚠️  {issue}")
            print()
    
    # Print all results with issues
    print("📋 TÜM SONUÇLAR (Sorunlu Olanlar):")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x['key']):
        if r['issues']:
            print(f"   {r['key']}:")
            print(f"      HPO DirHit (State): {r['state_hpo_dirhit']:.2f}%" if r['state_hpo_dirhit'] else "      HPO DirHit (State): Yok")
            print(f"      Training DirHit: {r['state_training_dirhit']:.2f}%" if r['state_training_dirhit'] else "      Training DirHit: Yok")
            if r['db_hpo_dirhit']:
                print(f"      HPO DirHit (DB): {r['db_hpo_dirhit']:.2f}%")
            for issue in r['issues']:
                print(f"      ⚠️  {issue}")
            print()
    
    # Print all results
    print("📋 TÜM SONUÇLAR:")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x['key']):
        print(f"   {r['key']}:", end="")
        if r['state_hpo_dirhit'] is not None:
            print(f" HPO DirHit={r['state_hpo_dirhit']:.2f}%", end="")
        else:
            print(" HPO DirHit=YOK", end="")
        
        training_dirhit = r['state_adaptive_dirhit'] or r['state_training_dirhit_online'] or r['state_training_dirhit_wfv'] or r['state_training_dirhit']
        if training_dirhit is not None:
            print(f" Training DirHit={training_dirhit:.2f}%", end="")
        else:
            print(" Training DirHit=YOK", end="")
        
        if r['issues']:
            print(f" ⚠️ ({len(r['issues'])} sorun)", end="")
        print()


if __name__ == '__main__':
    main()

