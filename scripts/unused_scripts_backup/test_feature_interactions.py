#!/usr/bin/env python3
"""
Feature Interaction Testing - Feature synergy'lerini tespit eder

Bu script, feature'ların birbirleriyle etkileşimlerini (interaction/synergy) tespit eder.
Bir feature tek başına negatif olabilir ama diğer feature'larla birlikte pozitif etkisi olabilir.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from itertools import combinations

sys.path.insert(0, '/opt/bist-pattern')

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import cast

from enhanced_ml_system import EnhancedMLSystem
from bist_pattern.core.config_manager import ConfigManager

# Import from test_feature_combinations
sys.path.insert(0, '/opt/bist-pattern/scripts')
from test_feature_combinations import (
    FEATURES, load_hpo_results, fetch_prices, 
    train_and_evaluate, setup_test_environment
)

logger = logging.getLogger(__name__)

def test_greedy_feature_selection(symbol: str, horizon: int, hpo_data: Dict, 
                                  train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                  test_folder: Path) -> List[Dict]:
    """
    Greedy feature selection - Her adımda en iyi feature'ı seç
    
    Bu yaklaşım feature interaction'ları (synergy) tespit eder.
    """
    print(f"\n{'='*80}")
    print(f"🔬 Greedy Feature Selection: {symbol} {horizon}d")
    print(f"{'='*80}\n")
    
    best_params = hpo_data.get('best_params', {})
    best_params['best_trial_number'] = hpo_data.get('best_trial', {}).get('number', 42)
    hpo_dirhit = hpo_data.get('best_dirhit', 0)
    
    results = []
    selected_features = []
    available_features = list(FEATURES.keys())
    current_config = {f: False for f in FEATURES.keys()}
    current_dirhit = None
    
    # Base (HPO feature seti)
    print(f"📊 Base (HPO feature seti - tüm kapalı):")
    base_result = train_and_evaluate(symbol, horizon, train_df, test_df, best_params, current_config, test_folder)
    current_dirhit = base_result.get('dirhit', 0)
    print(f"   DirHit: {current_dirhit:.2f}% (HPO: {hpo_dirhit:.2f}%)\n")
    results.append({
        'config': current_config.copy(),
        'config_name': 'Base',
        'dirhit': current_dirhit,
        'diff_from_hpo': current_dirhit - hpo_dirhit if current_dirhit else None,
        'selected_features': []
    })
    
    # Greedy selection: Her adımda en iyi feature'ı seç
    iteration = 0
    max_iterations = len(available_features)
    
    while available_features and iteration < max_iterations:
        iteration += 1
        print(f"🔄 Iteration {iteration}: Mevcut DirHit: {current_dirhit:.2f}%")
        print(f"   Seçili feature'lar: {[FEATURES[f] for f in selected_features] if selected_features else 'Yok'}")
        print(f"   Test edilecek: {len(available_features)} feature\n")
        
        best_feature = None
        best_dirhit = current_dirhit
        best_improvement = 0
        
        # Her available feature'ı test et
        for feature in available_features:
            test_config = current_config.copy()
            test_config[feature] = True
            
            print(f"   📊 Test: +{FEATURES[feature]}", end=' ')
            result = train_and_evaluate(symbol, horizon, train_df, test_df, best_params, test_config, test_folder)
            dirhit = result.get('dirhit', 0)
            
            if dirhit:
                improvement = dirhit - current_dirhit
                print(f"→ DirHit: {dirhit:.2f}% ({improvement:+.2f}%)")
                
                if dirhit > best_dirhit:
                    best_dirhit = dirhit
                    best_feature = feature
                    best_improvement = improvement
        
        # En iyi feature'ı seç
        if best_feature and best_improvement > 0:
            selected_features.append(best_feature)
            current_config[best_feature] = True
            available_features.remove(best_feature)
            current_dirhit = best_dirhit
            
            print(f"\n   ✅ SEÇİLDİ: {FEATURES[best_feature]} (+{best_improvement:.2f}%)")
            print(f"   📈 Yeni DirHit: {current_dirhit:.2f}%\n")
            
            results.append({
                'config': current_config.copy(),
                'config_name': f"Base + {' + '.join([FEATURES[f] for f in selected_features])}",
                'dirhit': current_dirhit,
                'diff_from_previous': best_improvement,
                'diff_from_hpo': current_dirhit - hpo_dirhit if current_dirhit else None,
                'selected_features': selected_features.copy()
            })
        else:
            print(f"\n   ⛔ DURDURULDU: Hiçbir feature DirHit'i artırmadı")
            print(f"   📊 Final DirHit: {current_dirhit:.2f}%\n")
            break
    
    return results

def test_pairwise_interactions(symbol: str, horizon: int, hpo_data: Dict,
                               train_df: pd.DataFrame, test_df: pd.DataFrame,
                               test_folder: Path, selected_features: List[str]) -> List[Dict]:
    """
    Seçilen feature'ların ikili kombinasyonlarını test et (synergy tespiti)
    """
    print(f"\n{'='*80}")
    print(f"🔬 Pairwise Interaction Testing: {symbol} {horizon}d")
    print(f"{'='*80}\n")
    
    best_params = hpo_data.get('best_params', {})
    best_params['best_trial_number'] = hpo_data.get('best_trial', {}).get('number', 42)
    hpo_dirhit = hpo_data.get('best_dirhit', 0)
    
    results = []
    
    if len(selected_features) < 2:
        print("   ⚠️ İkili kombinasyon için en az 2 feature gerekli")
        return results
    
    # Base (seçilen feature'lar açık)
    base_config = {f: f in selected_features for f in FEATURES.keys()}
    base_result = train_and_evaluate(symbol, horizon, train_df, test_df, best_params, base_config, test_folder)
    base_dirhit = base_result.get('dirhit', 0)
    print(f"📊 Base (Seçili feature'lar): {', '.join([FEATURES[f] for f in selected_features])}")
    print(f"   DirHit: {base_dirhit:.2f}%\n")
    
    # İkili kombinasyonları test et
    print(f"📊 İkili Kombinasyonlar ({len(list(combinations(selected_features, 2)))} test):\n")
    
    for feat1, feat2 in combinations(selected_features, 2):
        # Her iki feature'ı kapat, tek tek test et
        test_config = base_config.copy()
        test_config[feat1] = False
        test_config[feat2] = False
        
        # Sadece feat1
        test_config[feat1] = True
        result1 = train_and_evaluate(symbol, horizon, train_df, test_df, best_params, test_config, test_folder)
        dirhit1 = result1.get('dirhit', 0)
        
        # Sadece feat2
        test_config[feat1] = False
        test_config[feat2] = True
        result2 = train_and_evaluate(symbol, horizon, train_df, test_df, best_params, test_config, test_folder)
        dirhit2 = result2.get('dirhit', 0)
        
        # İkisi birlikte
        test_config[feat1] = True
        test_config[feat2] = True
        result_both = train_and_evaluate(symbol, horizon, train_df, test_df, best_params, test_config, test_folder)
        dirhit_both = result_both.get('dirhit', 0)
        
        # Synergy hesapla
        expected = (dirhit1 + dirhit2) / 2  # Basit ortalama
        synergy = dirhit_both - expected
        
        print(f"   {FEATURES[feat1]} + {FEATURES[feat2]}:")
        print(f"      Tek başına {FEATURES[feat1]}: {dirhit1:.2f}%")
        print(f"      Tek başına {FEATURES[feat2]}: {dirhit2:.2f}%")
        print(f"      İkisi birlikte: {dirhit_both:.2f}%")
        print(f"      Synergy: {synergy:+.2f}%")
        
        if synergy > 1.0:  # Önemli synergy
            print(f"      ✅ ÖNEMLİ SYNERGY TESPİT EDİLDİ!")
        
        results.append({
            'feature1': feat1,
            'feature2': feat2,
            'dirhit_solo1': dirhit1,
            'dirhit_solo2': dirhit2,
            'dirhit_both': dirhit_both,
            'synergy': synergy
        })
        print()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Feature Interaction Testing')
    parser.add_argument('--symbol', type=str, default='ASELS', help='Symbol to test')
    parser.add_argument('--horizon', type=int, default=7, help='Horizon in days')
    parser.add_argument('--mode', type=str, choices=['greedy', 'pairwise', 'both'], 
                       default='both', help='Test mode')
    args = parser.parse_args()
    
    symbol = args.symbol.upper()
    horizon = args.horizon
    mode = args.mode
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # HPO sonuçlarını yükle
    hpo_data = load_hpo_results(symbol, horizon)
    if not hpo_data:
        print(f"❌ HPO sonuçları bulunamadı: {symbol} {horizon}d")
        return 1
    
    print(f"✅ HPO sonuçları yüklendi: {symbol} {horizon}d")
    print(f"   Best DirHit: {hpo_data.get('best_dirhit', 0):.2f}%")
    
    # Veri yükle
    db_url = 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db'
    engine = create_engine(db_url, poolclass=None)
    df = fetch_prices(engine, symbol, limit=1200)
    
    if df is None or df.empty:
        print(f"❌ Veri bulunamadı: {symbol}")
        return 1
    
    # Train/test split (HPO ile aynı)
    total_days = len(df)
    if total_days >= 240:
        split_idx = total_days - 120
    elif total_days >= 180:
        split_idx = int(total_days * 2 / 3)
    else:
        split_idx = max(1, int(total_days * 2 / 3))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Test environment
    test_folder = setup_test_environment(symbol, horizon, 'feature_interactions')
    
    all_results = {}
    
    # Greedy selection
    if mode in ('greedy', 'both'):
        greedy_results = test_greedy_feature_selection(symbol, horizon, hpo_data, train_df, test_df, test_folder)
        all_results['greedy'] = greedy_results
        
        # Seçilen feature'ları al
        selected_features = []
        if greedy_results:
            last_result = greedy_results[-1]
            selected_features = last_result.get('selected_features', [])
    
    # Pairwise interactions
    if mode in ('pairwise', 'both') and selected_features:
        pairwise_results = test_pairwise_interactions(symbol, horizon, hpo_data, train_df, test_df, test_folder, selected_features)
        all_results['pairwise'] = pairwise_results
    
    # Sonuçları kaydet
    results_file = test_folder / 'results' / f'feature_interactions_{symbol}_{horizon}d.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'symbol': symbol,
            'horizon': horizon,
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
            'hpo_dirhit': hpo_data.get('best_dirhit', 0),
            'results': all_results
        }, f, indent=2)
    
    print(f"\n✅ Sonuçlar kaydedildi: {results_file}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

