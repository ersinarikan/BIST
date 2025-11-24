#!/usr/bin/env python3
"""
Feature Impact Analysis Script

Bu script, Phase 3 test sonuçlarını analiz ederek her feature'ın olumlu/olumsuz etkisini ölçer.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

TEST_DIR = Path('/opt/bist-pattern/test_results')

FEATURES = {
    'ENABLE_EXTERNAL_FEATURES': 'External Features (FinGPT+YOLO)',
    'ENABLE_FINGPT_FEATURES': 'FinGPT Features',
    'ENABLE_YOLO_FEATURES': 'YOLO Features',
    'ML_USE_DIRECTIONAL_LOSS': 'Directional Loss',
    'ENABLE_SEED_BAGGING': 'Seed Bagging',
    'ML_USE_ADAPTIVE_LEARNING': 'Adaptive Learning'
}

def load_test_results(symbol: str, horizon: int, mode: str = 'all') -> Optional[List[Dict]]:
    """Test sonuçlarını yükle"""
    results_file = TEST_DIR / symbol / f'{horizon}d' / 'results' / f'feature_test_{symbol}_{horizon}d_{mode}.json'
    
    if not results_file.exists():
        print(f"❌ Sonuç dosyası bulunamadı: {results_file}")
        return None
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        return data.get('results', [])
    except Exception as e:
        print(f"❌ Sonuç dosyası okunamadı: {e}")
        return None

def analyze_feature_impact(results: List[Dict], hpo_dirhit: float) -> Dict:
    """Her feature'ın etkisini analiz et"""
    
    # Valid sonuçları filtrele
    valid_results = [r for r in results if r.get('dirhit') is not None and r.get('dirhit') > 0]
    
    if not valid_results:
        print("❌ Geçerli sonuç bulunamadı!")
        return {}
    
    # Her feature için istatistikler
    feature_stats = {}
    
    for feature in FEATURES.keys():
        # Bu feature açık olan sonuçlar
        with_feature = [r for r in valid_results if r.get('config', {}).get(feature, False)]
        # Bu feature kapalı olan sonuçlar
        without_feature = [r for r in valid_results if not r.get('config', {}).get(feature, False)]
        
        if not with_feature or not without_feature:
            continue
        
        # Ortalama DirHit
        avg_with = np.mean([r.get('dirhit', 0) for r in with_feature])
        avg_without = np.mean([r.get('dirhit', 0) for r in without_feature])
        
        # Medyan DirHit
        median_with = np.median([r.get('dirhit', 0) for r in with_feature])
        median_without = np.median([r.get('dirhit', 0) for r in without_feature])
        
        # En iyi sonuç
        best_with = max([r.get('dirhit', 0) for r in with_feature])
        best_without = max([r.get('dirhit', 0) for r in without_feature])
        
        # En kötü sonuç
        worst_with = min([r.get('dirhit', 0) for r in with_feature])
        worst_without = min([r.get('dirhit', 0) for r in without_feature])
        
        # HPO'dan fark
        avg_diff_from_hpo_with = np.mean([r.get('diff_from_hpo', 0) for r in with_feature if r.get('diff_from_hpo') is not None])
        avg_diff_from_hpo_without = np.mean([r.get('diff_from_hpo', 0) for r in without_feature if r.get('diff_from_hpo') is not None])
        
        # Olumlu/olumsuz etki sayısı
        positive_with = len([r for r in with_feature if r.get('diff_from_hpo', 0) > 0])
        negative_with = len([r for r in with_feature if r.get('diff_from_hpo', 0) < 0])
        positive_without = len([r for r in without_feature if r.get('diff_from_hpo', 0) > 0])
        negative_without = len([r for r in without_feature if r.get('diff_from_hpo', 0) < 0])
        
        feature_stats[feature] = {
            'name': FEATURES[feature],
            'avg_with': avg_with,
            'avg_without': avg_without,
            'avg_impact': avg_with - avg_without,
            'median_with': median_with,
            'median_without': median_without,
            'median_impact': median_with - median_without,
            'best_with': best_with,
            'best_without': best_without,
            'worst_with': worst_with,
            'worst_without': worst_without,
            'avg_diff_from_hpo_with': avg_diff_from_hpo_with,
            'avg_diff_from_hpo_without': avg_diff_from_hpo_without,
            'positive_count_with': positive_with,
            'negative_count_with': negative_with,
            'positive_count_without': positive_without,
            'negative_count_without': negative_without,
            'total_with': len(with_feature),
            'total_without': len(without_feature)
        }
    
    return feature_stats

def print_feature_impact_analysis(feature_stats: Dict, hpo_dirhit: float):
    """Feature etki analizini yazdır"""
    
    print("\n" + "="*100)
    print("📊 FEATURE IMPACT ANALYSIS")
    print("="*100)
    print(f"HPO Baseline DirHit: {hpo_dirhit:.2f}%")
    print("="*100 + "\n")
    
    # Ortalama etkiye göre sırala
    sorted_features = sorted(
        feature_stats.items(),
        key=lambda x: x[1]['avg_impact'],
        reverse=True
    )
    
    print("🎯 ORTALAMA ETKİ (Açık vs Kapalı):")
    print("-"*100)
    print(f"{'Feature':<40} {'Açık Ort.':<12} {'Kapalı Ort.':<12} {'Etki':<12} {'Durum':<15}")
    print("-"*100)
    
    for feature, stats in sorted_features:
        impact = stats['avg_impact']
        status = "✅ OLUMLU" if impact > 0 else "❌ OLUMSUZ" if impact < 0 else "⚪ NÖTR"
        print(f"{stats['name']:<40} {stats['avg_with']:>10.2f}%  {stats['avg_without']:>10.2f}%  {impact:>+10.2f}%  {status:<15}")
    
    print("\n" + "="*100)
    print("📈 MEDYAN ETKİ (Açık vs Kapalı):")
    print("-"*100)
    print(f"{'Feature':<40} {'Açık Med.':<12} {'Kapalı Med.':<12} {'Etki':<12} {'Durum':<15}")
    print("-"*100)
    
    sorted_median = sorted(
        feature_stats.items(),
        key=lambda x: x[1]['median_impact'],
        reverse=True
    )
    
    for feature, stats in sorted_median:
        impact = stats['median_impact']
        status = "✅ OLUMLU" if impact > 0 else "❌ OLUMSUZ" if impact < 0 else "⚪ NÖTR"
        print(f"{stats['name']:<40} {stats['median_with']:>10.2f}%  {stats['median_without']:>10.2f}%  {impact:>+10.2f}%  {status:<15}")
    
    print("\n" + "="*100)
    print("🏆 EN İYİ SONUÇLAR:")
    print("-"*100)
    print(f"{'Feature':<40} {'Açık En İyi':<12} {'Kapalı En İyi':<12} {'Fark':<12}")
    print("-"*100)
    
    sorted_best = sorted(
        feature_stats.items(),
        key=lambda x: x[1]['best_with'] - x[1]['best_without'],
        reverse=True
    )
    
    for feature, stats in sorted_best:
        diff = stats['best_with'] - stats['best_without']
        print(f"{stats['name']:<40} {stats['best_with']:>10.2f}%  {stats['best_without']:>10.2f}%  {diff:>+10.2f}%")
    
    print("\n" + "="*100)
    print("📊 HPO'DAN FARK (Ortalama):")
    print("-"*100)
    print(f"{'Feature':<40} {'Açıkken Ort.':<15} {'Kapalıyken Ort.':<15} {'Fark':<12}")
    print("-"*100)
    
    sorted_hpo_diff = sorted(
        feature_stats.items(),
        key=lambda x: x[1]['avg_diff_from_hpo_with'] - x[1]['avg_diff_from_hpo_without'],
        reverse=True
    )
    
    for feature, stats in sorted_hpo_diff:
        diff = stats['avg_diff_from_hpo_with'] - stats['avg_diff_from_hpo_without']
        print(f"{stats['name']:<40} {stats['avg_diff_from_hpo_with']:>+13.2f}%  {stats['avg_diff_from_hpo_without']:>+13.2f}%  {diff:>+10.2f}%")
    
    print("\n" + "="*100)
    print("📈 OLUMLU/OLUMSUZ ETKİ DAĞILIMI:")
    print("-"*100)
    print(f"{'Feature':<40} {'Açıkken +':<8} {'Açıkken -':<8} {'Kapalıyken +':<10} {'Kapalıyken -':<10}")
    print("-"*100)
    
    for feature, stats in sorted_features:
        print(f"{stats['name']:<40} {stats['positive_count_with']:>6}  {stats['negative_count_with']:>6}  {stats['positive_count_without']:>8}  {stats['negative_count_without']:>8}")
    
    print("\n" + "="*100)
    print("💡 ÖNERİLER:")
    print("-"*100)
    
    # En olumlu etkili feature'lar
    positive_features = [f for f, s in sorted_features if s['avg_impact'] > 0]
    if positive_features:
        print("✅ KULLANILMASI ÖNERİLEN FEATURE'LAR (Ortalama etki > 0):")
        for feature in positive_features[:3]:  # En iyi 3
            stats = feature_stats[feature]
            print(f"   • {stats['name']}: +{stats['avg_impact']:.2f}% (Ortalama)")
    
    # En olumsuz etkili feature'lar
    negative_features = [f for f, s in sorted_features if s['avg_impact'] < 0]
    if negative_features:
        print("\n❌ KULLANILMAMASI ÖNERİLEN FEATURE'LAR (Ortalama etki < 0):")
        for feature in negative_features[:3]:  # En kötü 3
            stats = feature_stats[feature]
            print(f"   • {stats['name']}: {stats['avg_impact']:.2f}% (Ortalama)")
    
    print("\n" + "="*100 + "\n")

def main():
    if len(sys.argv) < 3:
        print("Kullanım: python3 analyze_feature_impact.py <symbol> <horizon> [mode]")
        print("Örnek: python3 analyze_feature_impact.py ASELS 7 all")
        return 1
    
    symbol = sys.argv[1].upper()
    horizon = int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else 'all'
    
    # Sonuçları yükle
    results = load_test_results(symbol, horizon, mode)
    if not results:
        return 1
    
    # HPO DirHit'i bul
    hpo_dirhit = 0
    for r in results:
        if r.get('config_name') == 'HPO Feature Seti (hepsi kapalı)':
            hpo_dirhit = r.get('dirhit', 0)
            break
    
    if hpo_dirhit == 0:
        print("⚠️ HPO DirHit bulunamadı, 0 olarak kullanılacak")
    
    # Feature etkisini analiz et
    feature_stats = analyze_feature_impact(results, hpo_dirhit)
    
    if not feature_stats:
        print("❌ Feature istatistikleri hesaplanamadı!")
        return 1
    
    # Analizi yazdır
    print_feature_impact_analysis(feature_stats, hpo_dirhit)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

