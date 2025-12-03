#!/usr/bin/env python3
"""
Comprehensive analysis of HPO vs Training discrepancies
Detects logic errors in HPO and training process
"""
import sys
import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

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
    """Find study database"""
    if cycle is None:
        state = load_state()
        cycle = state.get('cycle', 1)
    
    study_dir = HPO_STUDIES_DIR
    if not study_dir.exists():
        return None
    
    cycle_format_file = study_dir / f"hpo_with_features_{symbol}_h{horizon}_c{cycle}.db"
    if cycle_format_file.exists():
        return cycle_format_file
    
    if cycle == 1:
        legacy_format_file = study_dir / f"hpo_with_features_{symbol}_h{horizon}.db"
        if legacy_format_file.exists():
            return legacy_format_file
    
    return None


def get_study_info(db_file: Path) -> Dict:
    """Get study database information"""
    result = {
        'exists': False,
        'total_trials': 0,
        'complete_trials': 0,
        'best_trial_number': None,
        'best_dirhit': None,
        'best_value': None
    }
    
    try:
        if not db_file.exists():
            return result
        
        result['exists'] = True
        conn = sqlite3.connect(str(db_file), timeout=30.0)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM trials")
        result['total_trials'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'")
        result['complete_trials'] = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT t.number, tv.value
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state='COMPLETE' AND tv.value IS NOT NULL AND tv.value_type='FINITE'
            ORDER BY tv.value DESC
            LIMIT 1
        """)
        best_row = cursor.fetchone()
        if best_row:
            result['best_trial_number'] = best_row[0]
            result['best_value'] = best_row[1]
            
            # Get DirHit
            try:
                cursor.execute("""
                    SELECT value_json
                    FROM trial_user_attributes
                    WHERE trial_id = ? AND key = 'dirhit'
                """, (best_row[0],))
                row = cursor.fetchone()
                if row:
                    result['best_dirhit'] = float(json.loads(row[0]))
            except Exception:
                try:
                    cursor.execute("""
                        SELECT value
                        FROM trial_user_attrs
                        WHERE trial_id = ? AND key = 'dirhit'
                    """, (best_row[0],))
                    row = cursor.fetchone()
                    if row:
                        result['best_dirhit'] = float(row[0])
                except Exception:
                    pass
        
        conn.close()
    except Exception as e:
        result['error'] = str(e)
    
    return result


def analyze_hpo_json(json_file: Path, symbol: str, horizon: int) -> Dict:
    """Analyze HPO JSON file"""
    result = {
        'exists': False,
        'best_dirhit': None,
        'best_trial_number': None,
        'symbols': [],
        'horizon': None,
        'split_metrics': [],
        'low_support': False,
        'high_variance': False
    }
    
    try:
        if not json_file.exists():
            return result
        
        result['exists'] = True
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        result['best_dirhit'] = data.get('best_dirhit')
        result['best_trial_number'] = data.get('best_trial_number')
        result['symbols'] = data.get('symbols', [])
        result['horizon'] = data.get('horizon')
        
        # Get symbol-specific metrics
        best_trial_metrics = data.get('best_trial_metrics', {})
        symbol_key = f"{symbol}_{horizon}d"
        
        if symbol_key in best_trial_metrics:
            symbol_metrics = best_trial_metrics[symbol_key]
            split_metrics = symbol_metrics.get('split_metrics', [])
            
            split_dirhits = []
            split_mask_counts = []
            
            for split in split_metrics:
                dirhit = split.get('dirhit')
                mask_count = split.get('mask_count', 0)
                
                if dirhit is not None:
                    split_dirhits.append(dirhit)
                    split_mask_counts.append(mask_count)
                    
                    if mask_count < 10:
                        result['low_support'] = True
            
            result['split_metrics'] = split_metrics
            
            if len(split_dirhits) > 1:
                min_dirhit = min(split_dirhits)
                max_dirhit = max(split_dirhits)
                if max_dirhit - min_dirhit > 30:
                    result['high_variance'] = True
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def analyze_symbol(symbol: str, horizon: int, task: Dict) -> Dict:
    """Comprehensive analysis of a symbol"""
    key = f"{symbol}_{horizon}d"
    cycle = task.get('cycle', 1)
    
    result = {
        'symbol': symbol,
        'horizon': horizon,
        'key': key,
        'cycle': cycle,
        'status': task.get('status'),
        'hpo_completed_at': task.get('hpo_completed_at'),
        'training_completed_at': task.get('training_completed_at'),
        'best_params_file': task.get('best_params_file'),
        'state_hpo_dirhit': task.get('hpo_dirhit'),
        'state_training_dirhit': task.get('training_dirhit') or task.get('adaptive_dirhit'),
        'error': task.get('error'),
        'logic_errors': [],
        'warnings': [],
        'study_info': None,
        'json_info': None
    }
    
    # Check for logic errors
    if result['status'] == 'completed':
        # Error 1: Training completed but HPO not completed
        if result['hpo_completed_at'] is None and result['training_completed_at'] is not None:
            result['logic_errors'].append({
                'type': 'TRAINING_WITHOUT_HPO',
                'severity': 'CRITICAL',
                'message': f"Training completed ({result['training_completed_at']}) but HPO never completed (hpo_completed_at is null)",
                'explanation': "This should not happen. Training requires HPO to complete first to get best parameters."
            })
        
        # Error 2: Training completed but no best_params_file
        if result['best_params_file'] is None and result['training_completed_at'] is not None:
            result['logic_errors'].append({
                'type': 'TRAINING_WITHOUT_PARAMS',
                'severity': 'CRITICAL',
                'message': "Training completed but best_params_file is null",
                'explanation': "Training should use HPO best parameters. Without best_params_file, training may have used default parameters."
            })
        
        # Error 3: HPO DirHit missing but training completed
        if result['state_hpo_dirhit'] is None and result['state_training_dirhit'] is not None:
            result['logic_errors'].append({
                'type': 'MISSING_HPO_DIRHIT',
                'severity': 'HIGH',
                'message': "HPO DirHit is missing but training DirHit exists",
                'explanation': "Cannot compare HPO vs Training performance without HPO DirHit."
            })
    
    # Check study database
    db_file = find_study_db(symbol, horizon, cycle=cycle)
    if db_file:
        result['study_info'] = get_study_info(db_file)
        
        # Check if study exists but HPO not marked as completed
        if result['study_info']['exists'] and result['hpo_completed_at'] is None:
            if result['study_info']['complete_trials'] >= 1490:  # HPO completed
                result['logic_errors'].append({
                    'type': 'HPO_COMPLETED_BUT_NOT_MARKED',
                    'severity': 'HIGH',
                    'message': f"Study database shows HPO completed ({result['study_info']['complete_trials']} trials) but state file doesn't mark it as completed",
                    'explanation': "State file may not have been updated properly after HPO completion."
                })
    
    # Check JSON file
    json_file_path = task.get('best_params_file')
    if json_file_path:
        json_file = Path(json_file_path)
        if json_file.exists():
            result['json_info'] = analyze_hpo_json(json_file, symbol, horizon)
        else:
            result['warnings'].append({
                'type': 'JSON_FILE_MISSING',
                'message': f"best_params_file specified but file doesn't exist: {json_file_path}"
            })
    elif result['status'] == 'completed' and result['training_completed_at']:
        # Training completed but no JSON file - this is a logic error
        result['logic_errors'].append({
            'type': 'TRAINING_WITHOUT_JSON',
            'severity': 'CRITICAL',
            'message': "Training completed but no JSON file exists",
            'explanation': "Training should have used HPO best parameters from JSON file. Without JSON, training may have used default parameters."
        })
    
    # Check for large discrepancies
    if result['state_hpo_dirhit'] is not None and result['state_training_dirhit'] is not None:
        diff = abs(result['state_hpo_dirhit'] - result['state_training_dirhit'])
        if diff > 20:
            issue = {
                'type': 'LARGE_DISCREPANCY',
                'severity': 'MEDIUM',
                'message': f"Large difference between HPO ({result['state_hpo_dirhit']:.2f}%) and Training ({result['state_training_dirhit']:.2f}%) DirHit: {diff:.2f}%",
                'explanation': "This may indicate overfitting in HPO or different data/parameters used in training."
            }
            
            if result['json_info']:
                if result['json_info'].get('low_support'):
                    issue['explanation'] += " HPO DirHit was calculated with low support (few significant predictions)."
                if result['json_info'].get('high_variance'):
                    issue['explanation'] += " HPO DirHit has high variance across splits."
            
            result['warnings'].append(issue)
    
    return result


def main():
    """Main function"""
    state = load_state()
    tasks = state.get('state', {})
    current_cycle = state.get('cycle', 1)
    
    print("=" * 120)
    print("KAPSAMLI HPO vs TRAINING ANALİZ RAPORU")
    print("=" * 120)
    print(f"Cycle: {current_cycle}")
    print(f"Analiz Tarihi: {datetime.now().isoformat()}")
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
    
    print(f"📊 Toplam {len(completed_tasks)} tamamlanan görev analiz ediliyor...")
    print()
    
    # Analyze all tasks
    all_results = []
    critical_errors = []
    high_severity_errors = []
    warnings = []
    
    for symbol, horizon, task in sorted(completed_tasks):
        result = analyze_symbol(symbol, horizon, task)
        all_results.append(result)
        
        for error in result.get('logic_errors', []):
            if error['severity'] == 'CRITICAL':
                critical_errors.append((result['key'], error))
            elif error['severity'] == 'HIGH':
                high_severity_errors.append((result['key'], error))
        
        warnings.extend([(result['key'], w) for w in result.get('warnings', [])])
    
    # Print critical errors
    if critical_errors:
        print("🚨 KRİTİK MANTIK HATALARI:")
        print("-" * 120)
        for key, error in critical_errors:
            print(f"   {key}:")
            print(f"      Tip: {error['type']}")
            print(f"      Mesaj: {error['message']}")
            print(f"      Açıklama: {error['explanation']}")
            
            # Show additional context
            result = next((r for r in all_results if r['key'] == key), None)
            if result:
                print(f"      Durum:")
                print(f"         Status: {result['status']}")
                print(f"         HPO Completed At: {result['hpo_completed_at'] or 'YOK'}")
                print(f"         Training Completed At: {result['training_completed_at'] or 'YOK'}")
                print(f"         Best Params File: {result['best_params_file'] or 'YOK'}")
                if result['study_info']:
                    print(f"         Study DB: {result['study_info']['complete_trials']} complete trials")
            print()
    
    # Print high severity errors
    if high_severity_errors:
        print("⚠️  YÜKSEK ÖNCELİKLİ SORUNLAR:")
        print("-" * 120)
        for key, error in high_severity_errors:
            print(f"   {key}: {error['message']}")
            print(f"      Açıklama: {error['explanation']}")
            print()
    
    # Print warnings
    if warnings:
        print("⚠️  UYARILAR:")
        print("-" * 120)
        for key, warning in warnings[:20]:  # Limit to first 20
            print(f"   {key}: {warning['message']}")
            if 'explanation' in warning:
                print(f"      Açıklama: {warning['explanation']}")
        if len(warnings) > 20:
            print(f"   ... ve {len(warnings) - 20} uyarı daha")
        print()
    
    # Summary statistics
    print("📊 ÖZET İSTATİSTİKLER:")
    print("-" * 120)
    print(f"   Toplam Görev: {len(completed_tasks)}")
    print(f"   Kritik Hata: {len(critical_errors)}")
    print(f"   Yüksek Öncelikli Hata: {len(high_severity_errors)}")
    print(f"   Uyarı: {len(warnings)}")
    print()
    
    # Detailed analysis for critical cases
    if critical_errors:
        print("🔍 KRİTİK DURUMLARIN DETAYLI ANALİZİ:")
        print("-" * 120)
        for key, error in critical_errors:
            result = next((r for r in all_results if r['key'] == key), None)
            if not result:
                continue
            
            print(f"\n   {key}:")
            print(f"      Hata Tipi: {error['type']}")
            print(f"      Durum: {result['status']}")
            print(f"      HPO Completed: {result['hpo_completed_at'] or 'YOK'}")
            print(f"      Training Completed: {result['training_completed_at'] or 'YOK'}")
            print(f"      Best Params File: {result['best_params_file'] or 'YOK'}")
            print(f"      HPO DirHit: {result['state_hpo_dirhit'] or 'YOK'}")
            print(f"      Training DirHit: {result['state_training_dirhit'] or 'YOK'}")
            
            if result['study_info']:
                print(f"      Study Database:")
                print(f"         Var mı: {result['study_info']['exists']}")
                print(f"         Total Trials: {result['study_info']['total_trials']}")
                print(f"         Complete Trials: {result['study_info']['complete_trials']}")
                print(f"         Best Trial: #{result['study_info']['best_trial_number'] or 'YOK'}")
                print(f"         Best DirHit: {result['study_info']['best_dirhit'] or 'YOK'}")
            
            if result['json_info']:
                print(f"      JSON File:")
                print(f"         Var mı: {result['json_info']['exists']}")
                print(f"         Best DirHit: {result['json_info']['best_dirhit'] or 'YOK'}")
                print(f"         Best Trial: #{result['json_info']['best_trial_number'] or 'YOK'}")
            
            print(f"      Açıklama: {error['explanation']}")
            print()
    
    # Recommendations
    print("💡 ÖNERİLER:")
    print("-" * 120)
    if critical_errors:
        print("   1. KRİTİK: HPO tamamlanmadan training yapılan semboller için:")
        print("      - Bu sembollerin training'i geçersizdir (default parametreler kullanılmış olabilir)")
        print("      - Bu semboller için HPO'yu tamamlayıp yeniden training yapılmalı")
        print("      - State dosyasındaki 'completed' durumu düzeltilmeli")
    
    if high_severity_errors:
        print("   2. YÜKSEK ÖNCELİK: HPO tamamlanmış ama state güncellenmemiş semboller için:")
        print("      - State dosyası manuel olarak güncellenebilir")
        print("      - Veya recovery mekanizması çalıştırılabilir")
    
    if warnings:
        print("   3. UYARI: Büyük farklar için:")
        print("      - HPO DirHit düşük support ile hesaplanmış olabilir (güvenilir değil)")
        print("      - Training DirHit daha güvenilir olabilir")
        print("      - HPO sırasında overfitting olmuş olabilir")
    
    print()


if __name__ == '__main__':
    main()

