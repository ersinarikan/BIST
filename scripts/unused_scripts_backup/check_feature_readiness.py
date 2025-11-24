#!/usr/bin/env python3
"""
Feature Readiness Check - Feature'ların kullanılabilir olup olmadığını kontrol eder

Her feature için gerekli veri/kaynakların mevcut olup olmadığını kontrol eder.
Bu sayede gereksiz test kombinasyonlarından kaçınırız.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, '/opt/bist-pattern')

def check_fingpt_readiness(symbol: str, min_days: int = 30) -> tuple[bool, str]:
    """FinGPT feature'ının kullanılabilir olup olmadığını kontrol et"""
    feature_dir = os.getenv('EXTERNAL_FEATURE_DIR', '/opt/bist-pattern/logs/feature_backfill')
    fingpt_file = os.path.join(feature_dir, 'fingpt', f'{symbol}.csv')
    
    if not os.path.exists(fingpt_file):
        return False, f"FinGPT CSV bulunamadı: {fingpt_file}"
    
    try:
        import pandas as pd
        df = pd.read_csv(fingpt_file)
        if 'date' not in df.columns:
            return False, "FinGPT CSV'de 'date' kolonu yok"
        
        df['date'] = pd.to_datetime(df['date'])
        days = len(df)
        
        if days < min_days:
            return False, f"FinGPT CSV'de yeterli veri yok: {days} gün (minimum: {min_days})"
        
        return True, f"FinGPT hazır: {days} gün veri"
    except Exception as e:
        return False, f"FinGPT CSV okunamadı: {e}"

def check_yolo_readiness(symbol: str, min_days: int = 30) -> tuple[bool, str]:
    """YOLO feature'ının kullanılabilir olup olmadığını kontrol et"""
    feature_dir = os.getenv('EXTERNAL_FEATURE_DIR', '/opt/bist-pattern/logs/feature_backfill')
    yolo_file = os.path.join(feature_dir, 'yolo', f'{symbol}.csv')
    
    if not os.path.exists(yolo_file):
        return False, f"YOLO CSV bulunamadı: {yolo_file}"
    
    try:
        import pandas as pd
        df = pd.read_csv(yolo_file)
        if 'date' not in df.columns:
            return False, "YOLO CSV'de 'date' kolonu yok"
        
        df['date'] = pd.to_datetime(df['date'])
        days = len(df)
        
        if days < min_days:
            return False, f"YOLO CSV'de yeterli veri yok: {days} gün (minimum: {min_days})"
        
        return True, f"YOLO hazır: {days} gün veri"
    except Exception as e:
        return False, f"YOLO CSV okunamadı: {e}"

def check_external_features_readiness(symbol: str, min_days: int = 30) -> tuple[bool, str]:
    """External Features (FinGPT + YOLO) için kontrol"""
    fingpt_ready, fingpt_msg = check_fingpt_readiness(symbol, min_days)
    yolo_ready, yolo_msg = check_yolo_readiness(symbol, min_days)
    
    if not fingpt_ready and not yolo_ready:
        return False, f"External Features hazır değil: {fingpt_msg}, {yolo_msg}"
    elif not fingpt_ready:
        return False, f"External Features hazır değil: {fingpt_msg}"
    elif not yolo_ready:
        return False, f"External Features hazır değil: {yolo_msg}"
    else:
        return True, f"External Features hazır: FinGPT ve YOLO mevcut"

def check_seed_bagging_readiness(train_size: int, min_train_size: int = 200) -> tuple[bool, str]:
    """Seed Bagging için yeterli train verisi var mı?"""
    if train_size < min_train_size:
        return False, f"Seed Bagging için yeterli train verisi yok: {train_size} gün (minimum: {min_train_size})"
    return True, f"Seed Bagging hazır: {train_size} gün train verisi"

def check_adaptive_learning_readiness(test_size: int, min_test_size: int = 60) -> tuple[bool, str]:
    """Adaptive Learning Phase 2 için yeterli test verisi var mı?"""
    # Not: Evaluation mode'da Phase 2 skip ediliyor, ama production için kontrol
    if test_size < min_test_size:
        return False, f"Adaptive Learning için yeterli test verisi yok: {test_size} gün (minimum: {min_test_size})"
    return True, f"Adaptive Learning hazır: {test_size} gün test verisi"

def check_all_features(symbol: str, train_size: int, test_size: int, 
                      min_external_days: int = 30) -> Dict[str, tuple[bool, str]]:
    """Tüm feature'ların hazır olup olmadığını kontrol et"""
    results = {}
    
    # External Features
    results['ENABLE_EXTERNAL_FEATURES'] = check_external_features_readiness(symbol, min_external_days)
    
    # FinGPT Features
    results['ENABLE_FINGPT_FEATURES'] = check_fingpt_readiness(symbol, min_external_days)
    
    # YOLO Features
    results['ENABLE_YOLO_FEATURES'] = check_yolo_readiness(symbol, min_external_days)
    
    # Seed Bagging
    results['ENABLE_SEED_BAGGING'] = check_seed_bagging_readiness(train_size)
    
    # Adaptive Learning
    results['ML_USE_ADAPTIVE_LEARNING'] = check_adaptive_learning_readiness(test_size)
    
    # Diğer feature'lar her zaman hazır (veri bağımlılığı yok)
    always_ready = [
        'ML_USE_DIRECTIONAL_LOSS',
        'ENABLE_TALIB_PATTERNS',
        'ML_USE_SMART_ENSEMBLE',
        'ML_USE_STACKED_SHORT',
        'ENABLE_META_STACKING',
        'ML_USE_REGIME_DETECTION',
        'ENABLE_FINGPT'  # Real-time RSS, CSV'ye bağımlı değil
    ]
    
    for feature in always_ready:
        results[feature] = (True, "Her zaman hazır (veri bağımlılığı yok)")
    
    return results

def get_ready_features(symbol: str, train_size: int, test_size: int, 
                      min_external_days: int = 30) -> List[str]:
    """Hazır olan feature'ları listele"""
    all_checks = check_all_features(symbol, train_size, test_size, min_external_days)
    ready_features = [f for f, (ready, _) in all_checks.items() if ready]
    return ready_features

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Readiness Check')
    parser.add_argument('--symbol', type=str, default='ASELS', help='Symbol to check')
    parser.add_argument('--train-size', type=int, default=433, help='Train data size')
    parser.add_argument('--test-size', type=int, default=120, help='Test data size')
    parser.add_argument('--min-external-days', type=int, default=30, help='Minimum days for external features')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"📊 FEATURE READINESS CHECK: {args.symbol}")
    print(f"{'='*80}\n")
    
    results = check_all_features(args.symbol, args.train_size, args.test_size, args.min_external_days)
    
    ready_count = sum(1 for ready, _ in results.values() if ready)
    total_count = len(results)
    
    print(f"✅ Hazır: {ready_count}/{total_count} feature\n")
    
    for feature, (ready, message) in results.items():
        status = "✅" if ready else "❌"
        print(f"{status} {feature}: {message}")
    
    print(f"\n{'='*80}")
    print(f"💡 HAZIR FEATURE'LAR:")
    print(f"{'='*80}\n")
    ready_features = get_ready_features(args.symbol, args.train_size, args.test_size, args.min_external_days)
    for feature in ready_features:
        print(f"   - {feature}")
    
    print(f"\n{'='*80}")
    print(f"📊 KOMBİNASYON SAYISI:")
    print(f"{'='*80}\n")
    print(f"   Tüm feature'lar: 2^{total_count} = {2**total_count}")
    print(f"   Hazır feature'lar: 2^{len(ready_features)} = {2**len(ready_features)}")
    print(f"   Azalma: {2**total_count - 2**len(ready_features)} kombinasyon ({100 * (1 - 2**len(ready_features) / 2**total_count):.1f}%)")
    print(f"{'='*80}\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

