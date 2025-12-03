#!/usr/bin/env python3
"""
Generate comprehensive report of HPO vs Training DirHit discrepancies
"""
import sys
import json
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, '/opt/bist-pattern')

STATE_FILE = Path('/opt/bist-pattern/results/continuous_hpo_state.json')


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


def analyze_hpo_json(json_file: Path, symbol: str, horizon: int) -> Dict:
    """Analyze HPO JSON file"""
    result = {
        'split_count': 0,
        'split_dirhits': [],
        'split_mask_counts': [],
        'low_support_splits': 0,
        'high_variance': False
    }
    
    try:
        if not json_file.exists():
            return result
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        best_trial_metrics = data.get('best_trial_metrics', {})
        symbol_key = f"{symbol}_{horizon}d"
        
        if symbol_key in best_trial_metrics:
            symbol_metrics = best_trial_metrics[symbol_key]
            result['split_count'] = symbol_metrics.get('split_count', 0)
            
            split_metrics = symbol_metrics.get('split_metrics', [])
            for split in split_metrics:
                dirhit = split.get('dirhit')
                mask_count = split.get('mask_count', 0)
                
                if dirhit is not None:
                    result['split_dirhits'].append(dirhit)
                    result['split_mask_counts'].append(mask_count)
                    
                    # Check for low support (less than 10 significant predictions)
                    if mask_count < 10:
                        result['low_support_splits'] += 1
            
            # Check for high variance
            if len(result['split_dirhits']) > 1:
                min_dirhit = min(result['split_dirhits'])
                max_dirhit = max(result['split_dirhits'])
                if max_dirhit - min_dirhit > 30:
                    result['high_variance'] = True
    
    except Exception:
        pass
    
    return result


def main():
    """Main function"""
    state = load_state()
    tasks = state.get('state', {})
    current_cycle = state.get('cycle', 1)
    
    print("=" * 100)
    print("HPO vs Training DirHit Discrepancy Raporu")
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
    
    # Categorize issues
    missing_hpo = []
    large_discrepancies = []
    low_support_issues = []
    high_variance_issues = []
    
    for symbol, horizon, task in sorted(completed_tasks):
        hpo_dirhit = task.get('hpo_dirhit')
        training_dirhit = task.get('training_dirhit') or task.get('adaptive_dirhit')
        
        key = f"{symbol}_{horizon}d"
        
        # Check for missing HPO DirHit
        if hpo_dirhit is None:
            missing_hpo.append({
                'key': key,
                'symbol': symbol,
                'horizon': horizon,
                'training_dirhit': training_dirhit,
                'hpo_completed_at': task.get('hpo_completed_at'),
                'best_params_file': task.get('best_params_file')
            })
            continue
        
        # Check for large discrepancies
        if training_dirhit is not None:
            diff = abs(hpo_dirhit - training_dirhit)
            if diff > 20:
                issue = {
                    'key': key,
                    'symbol': symbol,
                    'horizon': horizon,
                    'hpo_dirhit': hpo_dirhit,
                    'training_dirhit': training_dirhit,
                    'diff': diff
                }
                
                # Analyze HPO JSON for additional issues
                json_file_path = task.get('best_params_file')
                if json_file_path:
                    json_file = Path(json_file_path)
                    hpo_analysis = analyze_hpo_json(json_file, symbol, horizon)
                    issue['hpo_analysis'] = hpo_analysis
                    
                    if hpo_analysis['low_support_splits'] > 0:
                        low_support_issues.append(issue)
                    elif hpo_analysis['high_variance']:
                        high_variance_issues.append(issue)
                
                large_discrepancies.append(issue)
    
    # Print report
    print("📊 ÖZET:")
    print("-" * 100)
    print(f"   Toplam tamamlanan görev: {len(completed_tasks)}")
    print(f"   HPO DirHit eksik: {len(missing_hpo)}")
    print(f"   Büyük farklar (>20%): {len(large_discrepancies)}")
    print(f"   - Düşük support sorunları: {len(low_support_issues)}")
    print(f"   - Yüksek varyans sorunları: {len(high_variance_issues)}")
    print()
    
    # Missing HPO DirHit
    if missing_hpo:
        print("⚠️  HPO DirHit EKSİK OLANLAR:")
        print("-" * 100)
        for item in missing_hpo:
            print(f"   {item['key']}:")
            print(f"      Training DirHit: {item['training_dirhit']:.2f}%" if item['training_dirhit'] else "      Training DirHit: Yok")
            print(f"      HPO Completed At: {item['hpo_completed_at'] or 'Yok'}")
            print(f"      Best Params File: {item['best_params_file'] or 'Yok'}")
            print(f"      ⚠️  HPO tamamlanmamış görünüyor ama training yapılmış")
            print()
    
    # Large discrepancies with low support
    if low_support_issues:
        print("⚠️  DÜŞÜK SUPPORT SORUNLARI (HPO'da çok az significant prediction):")
        print("-" * 100)
        for item in sorted(low_support_issues, key=lambda x: x['diff'], reverse=True):
            print(f"   {item['key']}:")
            print(f"      HPO DirHit: {item['hpo_dirhit']:.2f}%")
            print(f"      Training DirHit: {item['training_dirhit']:.2f}%")
            print(f"      Fark: {item['diff']:.2f}%")
            if 'hpo_analysis' in item:
                hpo = item['hpo_analysis']
                print(f"      Split Sayısı: {hpo['split_count']}")
                print(f"      Düşük Support Split Sayısı: {hpo['low_support_splits']}")
                if hpo['split_mask_counts']:
                    print(f"      Mask Count'lar: {hpo['split_mask_counts']}")
            print(f"      ⚠️  HPO DirHit düşük support ile hesaplanmış (güvenilir değil)")
            print()
    
    # Large discrepancies with high variance
    if high_variance_issues:
        print("⚠️  YÜKSEK VARYANS SORUNLARI (Split'ler arasında büyük fark):")
        print("-" * 100)
        for item in sorted(high_variance_issues, key=lambda x: x['diff'], reverse=True):
            print(f"   {item['key']}:")
            print(f"      HPO DirHit: {item['hpo_dirhit']:.2f}%")
            print(f"      Training DirHit: {item['training_dirhit']:.2f}%")
            print(f"      Fark: {item['diff']:.2f}%")
            if 'hpo_analysis' in item:
                hpo = item['hpo_analysis']
                if hpo['split_dirhits']:
                    min_dirhit = min(hpo['split_dirhits'])
                    max_dirhit = max(hpo['split_dirhits'])
                    print(f"      Split DirHit'leri: {min_dirhit:.2f}% - {max_dirhit:.2f}%")
            print(f"      ⚠️  Split'ler arasında yüksek varyans (HPO DirHit güvenilir değil)")
            print()
    
    # Other large discrepancies
    other_issues = [item for item in large_discrepancies 
                   if item not in low_support_issues and item not in high_variance_issues]
    if other_issues:
        print("⚠️  DİĞER BÜYÜK FARKLAR:")
        print("-" * 100)
        for item in sorted(other_issues, key=lambda x: x['diff'], reverse=True):
            print(f"   {item['key']}:")
            print(f"      HPO DirHit: {item['hpo_dirhit']:.2f}%")
            print(f"      Training DirHit: {item['training_dirhit']:.2f}%")
            print(f"      Fark: {item['diff']:.2f}%")
            print()
    
    # Recommendations
    print("💡 ÖNERİLER:")
    print("-" * 100)
    if missing_hpo:
        print("   1. BAYRK_1d ve EKOS_1d için HPO tamamlanmamış. Bu semboller için HPO'yu tamamlayın.")
    if low_support_issues or high_variance_issues:
        print("   2. Düşük support veya yüksek varyans sorunları olan semboller için:")
        print("      - HPO DirHit güvenilir değil, Training DirHit daha güvenilir")
        print("      - HPO sırasında daha fazla split kullanılmalı veya split'ler daha uzun olmalı")
        print("      - Minimum mask_count (significant prediction sayısı) kontrolü yapılmalı")
    if large_discrepancies:
        print("   3. Büyük farklar için:")
        print("      - HPO sırasında overfitting olmuş olabilir")
        print("      - Training sırasında kullanılan veri seti HPO'dan farklı olabilir")
        print("      - HPO ve Training DirHit hesaplama yöntemleri karşılaştırılmalı")
    print()


if __name__ == '__main__':
    main()

