#!/usr/bin/env python3
"""
Önceki HPO sonuçlarını kullanarak training çalıştır
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, '/opt/bist-pattern')

from scripts.test_single_symbol import run_training_test

# HPO sonuçlarını yükle
hpo_file = '/opt/bist-pattern/results/optuna_pilot_features_on_h7_20251110_212652.json'
with open(hpo_file, 'r') as f:
    hpo_data = json.load(f)

best_params = hpo_data.get('best_params', {})
best_params['best_trial_number'] = hpo_data.get('best_trial', {}).get('number', 92)

print(f"📊 HPO Sonuçları:")
print(f"   Best Trial: {best_params.get('best_trial_number')}")
print(f"   Best DirHit: {hpo_data.get('best_dirhit', 'N/A')}%")
print(f"   Best Value: {hpo_data.get('best_value', 'N/A')}")
print()

# Test klasörünü bul veya oluştur
test_folder = Path('/opt/bist-pattern/test_results/ASELS_7d_20251110_223547')
test_folder.mkdir(parents=True, exist_ok=True)
(test_folder / 'logs').mkdir(exist_ok=True)
(test_folder / 'results').mkdir(exist_ok=True)
(test_folder / 'models').mkdir(exist_ok=True)

print(f"📁 Test klasörü: {test_folder}")
print()

# Training'i çalıştır
print("🚀 Training başlatılıyor...")
print()
result = run_training_test('ASELS', 7, best_params, test_folder)

if result:
    print()
    print("✅ Training tamamlandı!")
    print(f"   HPO DirHit: {hpo_data.get('best_dirhit', 'N/A')}%")
    print(f"   WFV DirHit: {result.get('training_dirhit_wfv', 'N/A')}%")
    print(f"   Online DirHit: {result.get('training_dirhit_online', 'N/A')}%")
    print()
    wfv_diff = result.get('training_dirhit_wfv', 0) - hpo_data.get('best_dirhit', 0)
    online_diff = result.get('training_dirhit_online', 0) - hpo_data.get('best_dirhit', 0)
    print(f"   Fark (WFV - HPO): {wfv_diff:.2f}%")
    print(f"   Fark (Online - HPO): {online_diff:.2f}%")
else:
    print()
    print("❌ Training başarısız!")

