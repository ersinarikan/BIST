#!/usr/bin/env python3
"""
HPO vs Training DirHit Discrepancy Analyzer

Bu script, HPO DirHit ve Training DirHit arasındaki farkı detaylı analiz eder.
Özellikle BESLR_1d gibi durumlarda neden HPO DirHit=100% ama Training DirHit=40% olduğunu araştırır.

Kullanım:
    python3 scripts/analyze_hpo_training_discrepancy.py BESLR 1
"""

import sys
import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Any
import optuna
import numpy as np

sys.path.insert(0, '/opt/bist-pattern')

RESULTS_DIR = Path('/opt/bist-pattern/results')
HPO_STUDIES_DIR = Path('/opt/bist-pattern/hpo_studies')
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
    except Exception:
        return {}


def find_hpo_json(symbol: str, horizon: int, cycle: Optional[int] = None) -> Optional[Path]:
    """Find HPO JSON file for symbol-horizon"""
    if cycle is None:
        state = load_state()
        cycle = state.get('cycle', 1)
    
    patterns = [
        f'optuna_pilot_features_on_h{horizon}_c{cycle}_*.json',
        f'optuna_pilot_features_on_h{horizon}_*.json'
    ]
    
    json_files = []
    for pattern in patterns:
        found = sorted(
            RESULTS_DIR.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        json_files.extend(found)
        if pattern.startswith(f'optuna_pilot_features_on_h{horizon}_c{cycle}_'):
            if found:
                json_files = found
                break
    
    # Filter by symbol
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            symbols = data.get('symbols', [])
            if symbol in symbols:
                return json_file
        except Exception:
            continue
    
    return None


def find_study_db(symbol: str, horizon: int, cycle: Optional[int] = None) -> Optional[Path]:
    """Find study database file"""
    if cycle is None:
        state = load_state()
        cycle = state.get('cycle', 1)
    
    # Priority 1: Cycle format
    cycle_file = HPO_STUDIES_DIR / f"hpo_with_features_{symbol}_h{horizon}_c{cycle}.db"
    if cycle_file.exists():
        return cycle_file
    
    # Priority 2: Legacy format (only for cycle 1)
    if cycle == 1:
        legacy_file = HPO_STUDIES_DIR / f"hpo_with_features_{symbol}_h{horizon}.db"
        if legacy_file.exists():
            return legacy_file
    
    return None


def analyze_hpo_json(json_file: Path, symbol: str, horizon: int) -> Dict[str, Any]:
    """Analyze HPO JSON file"""
    result = {
        'json_file': str(json_file),
        'best_dirhit': None,
        'best_value': None,
        'best_trial_number': None,
        'n_trials': None,
        'symbols': [],
        'best_trial_metrics': {},
        'symbol_specific_dirhit': None,
        'issue': None
    }
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        result['best_dirhit'] = data.get('best_dirhit')
        result['best_value'] = data.get('best_value')
        result['n_trials'] = data.get('n_trials')
        result['symbols'] = data.get('symbols', [])
        
        best_trial_info = data.get('best_trial', {})
        if isinstance(best_trial_info, dict):
            result['best_trial_number'] = best_trial_info.get('number')
        
        # Check best_trial_metrics
        best_trial_metrics = data.get('best_trial_metrics', {})
        symbol_key = f"{symbol}_{horizon}d"
        if symbol_key in best_trial_metrics:
            symbol_metrics = best_trial_metrics[symbol_key]
            result['symbol_specific_dirhit'] = symbol_metrics.get('avg_dirhit')
            result['best_trial_metrics'] = symbol_metrics
        
        # Identify issue
        if result['symbols'] and len(result['symbols']) > 1:
            if result['best_dirhit'] != result['symbol_specific_dirhit']:
                result['issue'] = f"best_dirhit ({result['best_dirhit']}%) is average across {len(result['symbols'])} symbols, not specific to {symbol}"
        
        if result['best_dirhit'] == 100.0 and result['symbol_specific_dirhit'] != 100.0:
            result['issue'] = f"HPO shows 100% DirHit (average), but {symbol} specific DirHit is {result['symbol_specific_dirhit']}%"
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def analyze_study_db(db_file: Path, symbol: str, horizon: int) -> Dict[str, Any]:
    """Analyze Optuna study database"""
    result = {
        'db_file': str(db_file),
        'best_trial_number': None,
        'best_value': None,
        'best_dirhit': None,
        'total_trials': 0,
        'complete_trials': 0,
        'symbol_metrics': None,
        'low_support_issue': False
    }
    
    try:
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_file}")
        
        result['total_trials'] = len(study.trials)
        result['complete_trials'] = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        
        if study.best_trial:
            result['best_trial_number'] = study.best_trial.number
            result['best_value'] = float(study.best_trial.value) if study.best_trial.value is not None else None
            
            # Get best_dirhit from user_attrs
            best_dirhit = study.best_trial.user_attrs.get('avg_dirhit')
            if best_dirhit is not None:
                result['best_dirhit'] = float(best_dirhit)
            
            # Try to get symbol_metrics
            symbol_metrics = study.best_trial.user_attrs.get('symbol_metrics')
            if symbol_metrics:
                symbol_key = f"{symbol}_{horizon}d"
                if symbol_key in symbol_metrics:
                    result['symbol_metrics'] = symbol_metrics[symbol_key]
                    
                    # Check for low support issue
                    split_metrics = symbol_metrics[symbol_key].get('split_metrics', [])
                    for split in split_metrics:
                        mask_count = split.get('mask_count', 0)
                        mask_pct = split.get('mask_pct', 0.0)
                        dirhit = split.get('dirhit')
                        
                        if dirhit == 100.0 and (mask_count < 10 or mask_pct < 5.0):
                            result['low_support_issue'] = True
                            result['low_support_details'] = {
                                'split': split.get('split_index'),
                                'mask_count': mask_count,
                                'mask_pct': mask_pct,
                                'dirhit': dirhit
                            }
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def analyze_training_state(symbol: str, horizon: int) -> Dict[str, Any]:
    """Analyze training state from state file"""
    result = {
        'hpo_dirhit': None,
        'training_dirhit': None,
        'training_dirhit_wfv': None,
        'training_dirhit_online': None,
        'adaptive_dirhit': None,
        'status': None
    }
    
    state = load_state()
    tasks = state.get('state', {})
    task_key = f"{symbol}_{horizon}d"
    
    if task_key in tasks:
        task = tasks[task_key]
        result['status'] = task.get('status')
        result['hpo_dirhit'] = task.get('hpo_dirhit')
        result['training_dirhit'] = task.get('training_dirhit')
        result['training_dirhit_wfv'] = task.get('training_dirhit_wfv')
        result['training_dirhit_online'] = task.get('training_dirhit_online')
        result['adaptive_dirhit'] = task.get('adaptive_dirhit')
    
    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/analyze_hpo_training_discrepancy.py SYMBOL HORIZON")
        print("Example: python3 scripts/analyze_hpo_training_discrepancy.py BESLR 1")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    horizon = int(sys.argv[2])
    
    print("=" * 100)
    print(f"🔍 HPO vs TRAINING DIRHIT DISCREPANCY ANALİZİ: {symbol}_{horizon}d")
    print("=" * 100)
    print()
    
    # Get cycle
    state = load_state()
    cycle = state.get('cycle', 1)
    print(f"🔄 Cycle: {cycle}")
    print()
    
    # 1. Analyze HPO JSON
    print("📄 1. HPO JSON ANALİZİ")
    print("-" * 100)
    json_file = find_hpo_json(symbol, horizon, cycle)
    if json_file:
        json_analysis = analyze_hpo_json(json_file, symbol, horizon)
        print(f"   JSON File: {json_file.name}")
        print(f"   Best DirHit (JSON): {json_analysis['best_dirhit']}%")
        print(f"   Best Value (Score): {json_analysis['best_value']}")
        print(f"   Best Trial Number: {json_analysis['best_trial_number']}")
        print(f"   Total Trials: {json_analysis['n_trials']}")
        print(f"   Symbols in HPO: {json_analysis['symbols']}")
        
        if json_analysis['symbol_specific_dirhit'] is not None:
            print(f"   {symbol} Specific DirHit (from best_trial_metrics): {json_analysis['symbol_specific_dirhit']}%")
        else:
            print(f"   ⚠️ {symbol} specific DirHit not found in best_trial_metrics")
        
        if json_analysis['issue']:
            print(f"   ⚠️ ISSUE: {json_analysis['issue']}")
        
        if json_analysis.get('best_trial_metrics'):
            metrics = json_analysis['best_trial_metrics']
            print(f"   Split Count: {metrics.get('split_count', 'N/A')}")
            split_metrics = metrics.get('split_metrics', [])
            if split_metrics:
                print(f"   Split Details:")
                for split in split_metrics[:3]:  # Show first 3 splits
                    split_idx = split.get('split_index', 'N/A')
                    dirhit = split.get('dirhit', 'N/A')
                    mask_count = split.get('mask_count', 'N/A')
                    mask_pct = split.get('mask_pct', 'N/A')
                    print(f"      Split {split_idx}: DirHit={dirhit}%, mask_count={mask_count}, mask_pct={mask_pct:.1f}%")
    else:
        print(f"   ⚠️ HPO JSON file not found for {symbol} {horizon}d")
    print()
    
    # 2. Analyze Study Database
    print("💾 2. HPO STUDY DATABASE ANALİZİ")
    print("-" * 100)
    db_file = find_study_db(symbol, horizon, cycle)
    if db_file:
        db_analysis = analyze_study_db(db_file, symbol, horizon)
        print(f"   DB File: {db_file.name}")
        print(f"   Total Trials: {db_analysis['total_trials']}")
        print(f"   Complete Trials: {db_analysis['complete_trials']}")
        print(f"   Best Trial Number: {db_analysis['best_trial_number']}")
        print(f"   Best Value (Score): {db_analysis['best_value']}")
        print(f"   Best DirHit (from user_attrs): {db_analysis['best_dirhit']}%")
        
        if db_analysis.get('symbol_metrics'):
            print(f"   ✅ Symbol metrics found in study")
            symbol_metrics = db_analysis['symbol_metrics']
            print(f"      Avg DirHit: {symbol_metrics.get('avg_dirhit', 'N/A')}%")
            print(f"      Split Count: {symbol_metrics.get('split_count', 'N/A')}")
        else:
            print(f"   ⚠️ Symbol metrics NOT found in study (may have failed to save)")
        
        if db_analysis.get('low_support_issue'):
            print(f"   ⚠️ LOW SUPPORT ISSUE DETECTED:")
            details = db_analysis['low_support_details']
            print(f"      Split {details['split']}: DirHit=100% but mask_count={details['mask_count']}, mask_pct={details['mask_pct']:.1f}%")
            print(f"      This suggests spurious 100% DirHit due to very few significant predictions")
    else:
        print(f"   ⚠️ Study database file not found for {symbol} {horizon}d")
    print()
    
    # 3. Analyze Training State
    print("🎓 3. TRAINING STATE ANALİZİ")
    print("-" * 100)
    training_analysis = analyze_training_state(symbol, horizon)
    print(f"   Status: {training_analysis['status']}")
    print(f"   HPO DirHit (from state): {training_analysis['hpo_dirhit']}%")
    print(f"   Training DirHit (WFV): {training_analysis['training_dirhit_wfv']}%")
    print(f"   Training DirHit (Online): {training_analysis['training_dirhit_online']}%")
    print(f"   Training DirHit (Adaptive): {training_analysis['adaptive_dirhit']}%")
    print(f"   Training DirHit (Legacy): {training_analysis['training_dirhit']}%")
    
    # Calculate discrepancy
    hpo_dirhit = training_analysis['hpo_dirhit']
    training_dirhit = training_analysis['adaptive_dirhit'] or training_analysis['training_dirhit_online'] or training_analysis['training_dirhit_wfv'] or training_analysis['training_dirhit']
    
    if hpo_dirhit is not None and training_dirhit is not None:
        discrepancy = training_dirhit - hpo_dirhit
        print(f"   📊 Discrepancy: {discrepancy:+.2f}% (Training - HPO)")
        if abs(discrepancy) > 20:
            print(f"   ⚠️ LARGE DISCREPANCY DETECTED (>20%)")
    print()
    
    # 4. Summary and Recommendations
    print("📋 4. ÖZET VE ÖNERİLER")
    print("-" * 100)
    
    issues = []
    
    if json_file and json_analysis.get('issue'):
        issues.append(f"JSON Issue: {json_analysis['issue']}")
    
    if json_analysis.get('symbol_specific_dirhit') is None:
        issues.append("Symbol-specific DirHit not found in best_trial_metrics (may need migration script)")
    
    if db_analysis.get('low_support_issue'):
        issues.append("Low support issue: 100% DirHit may be spurious due to very few significant predictions")
    
    if hpo_dirhit is not None and training_dirhit is not None:
        if abs(training_dirhit - hpo_dirhit) > 20:
            issues.append(f"Large discrepancy: HPO={hpo_dirhit}% vs Training={training_dirhit}%")
    
    if issues:
        print("   ⚠️ TESPİT EDİLEN SORUNLAR:")
        for i, issue in enumerate(issues, 1):
            print(f"      {i}. {issue}")
    else:
        print("   ✅ Önemli sorun tespit edilmedi")
    
    print()
    print("   💡 ÖNERİLER:")
    
    if json_analysis.get('symbol_specific_dirhit') is None:
        print("      1. best_trial_metrics eksik - migration script çalıştırılmalı")
        print("         python3 scripts/migrate_hpo_json_add_metrics.py")
    
    if json_analysis.get('best_dirhit') != json_analysis.get('symbol_specific_dirhit'):
        print("      2. best_dirhit güncellenmeli - fix script çalıştırılmalı")
        print("         python3 scripts/fix_hpo_json_best_dirhit.py")
    
    if db_analysis.get('low_support_issue'):
        print("      3. Low support sorunu - HPO_MIN_MASK_COUNT veya HPO_MIN_MASK_PCT ayarlanmalı")
        print("         Örnek: export HPO_MIN_MASK_COUNT=10")
    
    if abs(training_dirhit - hpo_dirhit) > 20 if (hpo_dirhit and training_dirhit) else False:
        print("      4. Büyük fark - aşağıdakiler kontrol edilmeli:")
        print("         - Seed uyumu (HPO best_trial seed vs Training seed)")
        print("         - Feature flags uyumu")
        print("         - Data split uyumu")
        print("         - Parameter uyumu")
    
    print()
    print("=" * 100)


if __name__ == '__main__':
    main()

