#!/usr/bin/env python3
"""
Detailed analysis of HPO vs Training DirHit discrepancies
"""
import sys
import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional, List

sys.path.insert(0, '/opt/bist-pattern')

STATE_FILE = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
HPO_STUDIES_DIR = Path('/opt/bist-pattern/hpo_studies')


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
    """Analyze HPO JSON file for detailed metrics"""
    result = {
        'json_file': str(json_file),
        'best_dirhit': None,
        'best_trial_number': None,
        'symbol_specific_dirhit': None,
        'split_metrics': [],
        'split_count': 0,
        'split_dirhits': []
    }
    
    try:
        if not json_file.exists():
            return result
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        result['best_dirhit'] = data.get('best_dirhit')
        result['best_trial_number'] = data.get('best_trial_number')
        
        # Get symbol-specific metrics
        best_trial_metrics = data.get('best_trial_metrics', {})
        symbol_key = f"{symbol}_{horizon}d"
        
        if symbol_key in best_trial_metrics:
            symbol_metrics = best_trial_metrics[symbol_key]
            result['symbol_specific_dirhit'] = symbol_metrics.get('avg_dirhit')
            result['split_count'] = symbol_metrics.get('split_count', 0)
            
            # Get split metrics
            split_metrics = symbol_metrics.get('split_metrics', [])
            for split in split_metrics:
                split_info = {
                    'split_index': split.get('split_index'),
                    'dirhit': split.get('dirhit'),
                    'train_days': split.get('train_days'),
                    'test_days': split.get('test_days'),
                    'train_start': split.get('train_start'),
                    'train_end': split.get('train_end'),
                    'test_start': split.get('test_start'),
                    'test_end': split.get('test_end'),
                    'mask_count': split.get('mask_count'),  # Number of significant predictions
                    'mask_pct': split.get('mask_pct')  # Percentage of significant predictions
                }
                result['split_metrics'].append(split_info)
                if split_info['dirhit'] is not None:
                    result['split_dirhits'].append(split_info['dirhit'])
        
        # Also check top_k_trials for additional info
        top_k_trials = data.get('top_k_trials', [])
        if top_k_trials:
            best_trial = top_k_trials[0]
            attrs = best_trial.get('attrs', {})
            symbol_metrics = attrs.get('symbol_metrics', {})
            if symbol_key in symbol_metrics:
                # This might have more detailed split info
                pass
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def analyze_symbol(symbol: str, horizon: int, task: Dict) -> Dict:
    """Detailed analysis of a single symbol"""
    result = {
        'symbol': symbol,
        'horizon': horizon,
        'key': f"{symbol}_{horizon}d",
        'state_hpo_dirhit': task.get('hpo_dirhit'),
        'state_training_dirhit': task.get('training_dirhit'),
        'state_adaptive_dirhit': task.get('adaptive_dirhit'),
        'state_training_dirhit_wfv': task.get('training_dirhit_wfv'),
        'state_training_dirhit_online': task.get('training_dirhit_online'),
        'hpo_analysis': None,
        'issues': []
    }
    
    # Analyze HPO JSON
    json_file_path = task.get('best_params_file')
    if json_file_path:
        json_file = Path(json_file_path)
        if json_file.exists():
            result['hpo_analysis'] = analyze_hpo_json(json_file, symbol, horizon)
    
    # Identify issues
    if result['state_hpo_dirhit'] is None:
        result['issues'].append("HPO DirHit eksik")
    
    if result['state_hpo_dirhit'] is not None and result['state_training_dirhit'] is not None:
        diff = abs(result['state_hpo_dirhit'] - result['state_training_dirhit'])
        if diff > 20:
            result['issues'].append(f"Büyük fark: {diff:.2f}%")
            
            # Check if HPO DirHit is from multiple splits
            if result['hpo_analysis']:
                split_count = result['hpo_analysis'].get('split_count', 0)
                split_dirhits = result['hpo_analysis'].get('split_dirhits', [])
                
                if split_count > 0:
                    result['issues'].append(f"HPO {split_count} split üzerinden ortalama")
                    if split_dirhits:
                        min_split = min(split_dirhits)
                        max_split = max(split_dirhits)
                        result['issues'].append(f"Split DirHit'leri: {min_split:.2f}% - {max_split:.2f}% (ortalama: {sum(split_dirhits)/len(split_dirhits):.2f}%)")
                        
                        # Check if there's high variance in splits
                        if max_split - min_split > 30:
                            result['issues'].append(f"⚠️ Split'ler arasında yüksek varyans: {max_split - min_split:.2f}%")
    
    return result


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python3 detailed_hpo_training_analysis.py SYMBOL [HORIZON]")
        print("Example: python3 detailed_hpo_training_analysis.py ADEL 1")
        print("Example: python3 detailed_hpo_training_analysis.py ALL")
        sys.exit(1)
    
    symbol_arg = sys.argv[1].upper()
    horizon_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    state = load_state()
    tasks = state.get('state', {})
    current_cycle = state.get('cycle', 1)
    
    print("=" * 100)
    print("Detaylı HPO vs Training DirHit Analizi")
    print("=" * 100)
    print(f"Cycle: {current_cycle}")
    print()
    
    # Get tasks to analyze
    tasks_to_analyze = []
    
    if symbol_arg == 'ALL':
        # Analyze all completed tasks with issues
        for key, task in tasks.items():
            if isinstance(task, dict) and task.get('status') == 'completed':
                task_cycle = task.get('cycle', 0)
                if task_cycle != current_cycle:
                    continue
                
                symbol = task.get('symbol', '')
                horizon = task.get('horizon', 0)
                hpo_dirhit = task.get('hpo_dirhit')
                training_dirhit = task.get('training_dirhit') or task.get('adaptive_dirhit')
                
                # Only analyze tasks with issues
                if hpo_dirhit is None or (training_dirhit is not None and abs(hpo_dirhit - training_dirhit) > 20):
                    tasks_to_analyze.append((symbol, horizon, task))
    else:
        # Analyze specific symbol
        key = f"{symbol_arg}_{horizon_arg}d"
        if key in tasks:
            task = tasks[key]
            if isinstance(task, dict):
                tasks_to_analyze.append((symbol_arg, horizon_arg, task))
        else:
            print(f"❌ {key} bulunamadı")
            sys.exit(1)
    
    if not tasks_to_analyze:
        print("Analiz edilecek görev bulunamadı")
        sys.exit(0)
    
    print(f"Toplam {len(tasks_to_analyze)} görev analiz edilecek")
    print()
    
    # Analyze each task
    for symbol, horizon, task in sorted(tasks_to_analyze):
        result = analyze_symbol(symbol, horizon, task)
        
        print(f"📊 {result['key']}:")
        print("-" * 100)
        print(f"   HPO DirHit (State): {result['state_hpo_dirhit']:.2f}%" if result['state_hpo_dirhit'] else "   HPO DirHit (State): Yok")
        
        training_dirhit = result['state_adaptive_dirhit'] or result['state_training_dirhit_online'] or result['state_training_dirhit_wfv'] or result['state_training_dirhit']
        print(f"   Training DirHit: {training_dirhit:.2f}%" if training_dirhit else "   Training DirHit: Yok")
        
        if result['state_hpo_dirhit'] and training_dirhit:
            diff = abs(result['state_hpo_dirhit'] - training_dirhit)
            print(f"   Fark: {diff:.2f}%")
        
        # HPO Analysis
        if result['hpo_analysis']:
            hpo = result['hpo_analysis']
            print()
            print(f"   📈 HPO Analizi:")
            print(f"      JSON Dosyası: {hpo.get('json_file', 'N/A')}")
            print(f"      Best Trial: #{hpo.get('best_trial_number', 'N/A')}")
            print(f"      Best DirHit (JSON): {hpo.get('best_dirhit', 'N/A'):.2f}%" if hpo.get('best_dirhit') else "      Best DirHit (JSON): N/A")
            print(f"      Symbol-Specific DirHit: {hpo.get('symbol_specific_dirhit', 'N/A'):.2f}%" if hpo.get('symbol_specific_dirhit') else "      Symbol-Specific DirHit: N/A")
            print(f"      Split Sayısı: {hpo.get('split_count', 0)}")
            
            if hpo.get('split_metrics'):
                print()
                print(f"      Split Detayları:")
                for split in hpo['split_metrics']:
                    print(f"         Split #{split.get('split_index', '?')}:")
                    print(f"            DirHit: {split.get('dirhit', 'N/A'):.2f}%" if split.get('dirhit') else "            DirHit: N/A")
                    print(f"            Train: {split.get('train_days', '?')} gün ({split.get('train_start', '?')} - {split.get('train_end', '?')})")
                    print(f"            Test: {split.get('test_days', '?')} gün ({split.get('test_start', '?')} - {split.get('test_end', '?')})")
                    if split.get('mask_count') is not None:
                        print(f"            Mask Count: {split.get('mask_count')} ({split.get('mask_pct', 0):.2f}%)")
        
        # Issues
        if result['issues']:
            print()
            print(f"   ⚠️  Sorunlar:")
            for issue in result['issues']:
                print(f"      - {issue}")
        
        print()


if __name__ == '__main__':
    main()

