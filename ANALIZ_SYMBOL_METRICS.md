# DETAYLI ANALİZ: symbol_metrics NEDEN KAYDEDİLMEDİ?

## ÖZET

Bu belge, HPO sırasında `symbol_metrics`'in neden kaydedilmediğini ve `best_dirhit` ile `best_trial_metrics` arasındaki farkı açıklar.

## 1. HPO SIRASINDA NE OLDU?

### ✅ avg_dirhit Kaydedildi
- **Kod:** `optuna_hpo_with_feature_flags.py` line 872
- **İşlem:** `trial.set_user_attr('avg_dirhit', avg_dirhit)`
- **Sonuç:** Başarılı (basit float değer)
- **İçerik:** Tüm semboller için ortalama DirHit

### ❌ symbol_metrics Kaydedilmedi
- **Kod:** `optuna_hpo_with_feature_flags.py` line 882
- **İşlem:** `trial.set_user_attr('symbol_metrics', trial_symbol_metrics)`
- **Sonuç:** Exception fırlatıldı, sessizce yutuldu
- **Neden:** Büyük dictionary (68 sembol × detaylı metrics) Optuna'nın user_attrs limitini aşmış olabilir

## 2. BEST_DIRHIT NEREDEN GELİYOR?

### HPO JSON Oluşturulurken
- **Kod:** `optuna_hpo_with_feature_flags.py` line 1137-1145
- **İşlem:** `best_dirhit = best_trial.user_attrs.get('avg_dirhit')`
- **Kaynak:** HPO sırasında kaydedilmiş `avg_dirhit` değeri
- **İçerik:** Tüm semboller için ortalama DirHit (tek bir sembol için değil)

### Örnek: AKFGY
- `best_dirhit = 95.83%` (tüm semboller için ortalama)
- `best_trial_metrics['AKFGY_1d']['avg_dirhit'] = 0.0%` (sadece AKFGY için)

## 3. BEST_TRIAL_METRICS NEREDEN GELİYOR?

### Migration Script
- **Dosya:** `migrate_hpo_json_add_metrics.py`
- **İşlem:**
  1. `best_params` ile YENİDEN eğitim yaptı
  2. Her sembol için split'lerde DirHit hesapladı
  3. `best_trial_metrics` olarak JSON'a ekledi
- **Sonuç:** Her sembol için ayrı ayrı `avg_dirhit` hesaplandı

### JSON Yapısı
```json
{
  "best_dirhit": 95.83,  // Tüm semboller için ortalama (HPO sırasında)
  "best_trial_metrics": {
    "AKFGY_1d": {
      "avg_dirhit": 0.0,  // Sadece AKFGY için (migration script)
      "split_count": 5,
      ...
    }
  }
}
```

## 4. NEDEN BAZI SEMBOLLER %100 EŞLEŞİYOR?

### Eşleşen Semboller (6 adet)
- **Örnek:** AKMGY
- `best_dirhit = 100.0%` (tüm semboller ortalaması)
- `best_trial_metrics['AKMGY_1d']['avg_dirhit'] = 100.0%` (sadece AKMGY)
- `training_dirhit = 100.0%` (training'in hesapladığı)

### Açıklama
- Bu sembolün DirHit'i tesadüfen tüm semboller ortalamasına eşit
- Migration script ve Training aynı sonucu verdi
- `best_dirhit` bu sembol için doğru (tesadüfen)

## 5. NEDEN BAZI SEMBOLLER FARKLI?

### Farklı Olan Semboller (62 adet)
- **Örnek:** AKFGY
- `best_dirhit = 95.83%` (tüm semboller ortalaması)
- `best_trial_metrics['AKFGY_1d']['avg_dirhit'] = 0.0%` (sadece AKFGY)
- `training_dirhit = 0.0%` (training'in hesapladığı)

### Açıklama
- `best_dirhit` tüm semboller için ortalama, bu sembol için değil
- Migration script ve Training aynı sonucu verdi (0.0%)
- `best_dirhit` bu sembol için yanlış

## 6. GERÇEK SORUN NEDİR?

### Sorun 1: symbol_metrics Kaydedilmedi
- **Neden:** Exception fırlatıldı, sessizce yutuldu
- **Etki:** `best_trial_metrics` HPO JSON'unda yok
- **Çözüm:** Migration script ile YENİDEN hesaplandı

### Sorun 2: best_dirhit vs best_trial_metrics Farkı
- **best_dirhit:** Tüm semboller için ortalama (HPO sırasında)
- **best_trial_metrics:** Her sembol için ayrı ayrı (migration script)
- **Etki:** `best_dirhit` tek bir sembol için yanlış olabilir
- **Çözüm:** `best_dirhit`'i `best_trial_metrics[symbol]['avg_dirhit']` ile güncelle

## 7. ÇÖZÜM

### fix_hpo_json_best_dirhit.py Script'i
- Her HPO JSON dosyası için:
  1. `best_trial_metrics` içindeki `avg_dirhit` değerini al
  2. `best_dirhit`'i bu değerle güncelle
  3. JSON dosyasını kaydet

### Sonuç
- Her sembol için doğru `best_dirhit` değeri
- HPO DirHit ve Training DirHit eşleşmesi

## 8. KANITLAR

### Study Dosyası Kontrolü
- `avg_dirhit` ✅ VAR (basit float)
- `symbol_metrics` ❌ YOK (Exception fırlatıldı)

### HPO JSON Kontrolü
- `best_dirhit` ✅ VAR (tüm semboller ortalaması)
- `top_k_trials[0]['attrs']` içinde `symbol_metrics` ❌ YOK
- `best_trial_metrics` ✅ VAR (migration script tarafından eklendi)

### State Dosyası Kontrolü
- 6 sembol %100 eşleşiyor (tesadüfen)
- 62 sembol farklı (best_dirhit yanlış)

## SONUÇ

1. ✅ `avg_dirhit` kaydedildi (basit float, tüm semboller için ortalama)
2. ❌ `symbol_metrics` kaydedilmedi (Exception fırlatıldı, sessizce yutuldu)
3. ❓ `best_dirhit` tüm semboller için ortalama (tek bir sembol için yanlış olabilir)
4. ✅ `best_trial_metrics` migration script tarafından YENİDEN hesaplandı
5. 💡 Eşleşen semboller: Tesadüfen eşleşmiş
6. 💡 Farklı olanlar: `best_dirhit` yanlış, `best_trial_metrics` doğru
7. ✅ Çözüm: `fix_hpo_json_best_dirhit.py` script'ini çalıştır

