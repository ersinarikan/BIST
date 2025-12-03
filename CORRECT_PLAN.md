# Geçmiş HPO ve Training Sonuçlarını Düzeltme Planı - Doğrulandı

## ✅ Doğrulanmış Plan

### Adım 1: HPO Study Dosyalarından Filtreye Göre Best Params Bul ✅

**Script:** `update_json_with_filtered_best_params.py` (YENİ)

**Ne Yapar:**
- Tamamlanmış semboller için study DB dosyalarını bulur
- Filtreye göre best params'ı bulur (`find_best_trial_with_filter_applied`)
- JSON dosyalarını günceller (backup alır)

**Kullanım:**
```bash
# Tüm tamamlanmış semboller için
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/update_json_with_filtered_best_params.py \
  --all-completed --dry-run

# Belirli semboller için
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/update_json_with_filtered_best_params.py \
  --symbols EKGYO_1d,BRSAN_3d --dry-run

# Gerçek güncelleme
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/update_json_with_filtered_best_params.py \
  --all-completed
```

**Özellikler:**
- ✅ Backup oluşturur (`.json.backup`)
- ✅ Mevcut JSON yapısını korur
- ✅ Sadece best_params, best_trial_number, best_dirhit güncellenir
- ✅ Update metadata ekler (`_updated_at`, `_updated_reason`)

### Adım 2: JSON Dosyalarını Güncelle ✅

**Ne Yapılır:**
- Study DB'den bulunan filtered best params JSON'a yazılır
- Backup alınır (güvenlik için)
- Mevcut JSON yapısı korunur

**Güncellenen Alanlar:**
- `best_params` → Yeni filtered best params
- `best_trial_number` → Yeni best trial number
- `best_dirhit` → Yeni best DirHit (filtered)
- `features_enabled` → Güncellenir
- `feature_params` → Güncellenir
- `_updated_at` → Update timestamp
- `_updated_reason` → "filtered_best_params_from_study"

### Adım 3: Training Yap ✅

**Script:** `retrain_high_discrepancy_symbols.py` (zaten var)

**Ne Yapar:**
- Güncellenmiş JSON dosyalarından best params'ı okur
- Aynı filtreyi kullanarak training yapar
- Sonuçları kaydeder

**Kullanım:**
```bash
# Tüm güncellenmiş semboller için
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/retrain_high_discrepancy_symbols.py \
  --symbols $(cat /opt/bist-pattern/results/low_support_symbols.txt | awk '{print $1"_"$2"d"}' | tr '\n' ',' | sed 's/,$//')

# Belirli semboller için
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/retrain_high_discrepancy_symbols.py \
  --symbols EKGYO_1d,BRSAN_3d
```

### Adım 4: HPO Servisini Başlat ✅

**Komut:**
```bash
# Servisi başlat
sudo systemctl start bist-pattern-hpo.service

# Durumu kontrol et
sudo systemctl status bist-pattern-hpo.service

# Log'ları izle
sudo journalctl -u bist-pattern-hpo.service -f
```

**Ne Olur:**
- Servis kaldığı yerden devam eder
- State dosyasından tamamlanmış sembolleri atlar
- Yeni semboller için HPO yapar
- Güncellenmiş JSON dosyalarını kullanır

## 🔄 Tam İşlem Akışı

### 1. Hazırlık
```bash
# Mevcut durumu analiz et (opsiyonel)
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/analyze_low_support_symbols.py
```

### 2. JSON Dosyalarını Güncelle
```bash
# Dry-run (test)
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/update_json_with_filtered_best_params.py \
  --all-completed --dry-run

# Gerçek güncelleme
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/update_json_with_filtered_best_params.py \
  --all-completed
```

### 3. Training Yap
```bash
# Belirli semboller için
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/retrain_high_discrepancy_symbols.py \
  --symbols EKGYO_1d,BRSAN_3d
```

### 4. HPO Servisini Başlat
```bash
sudo systemctl start bist-pattern-hpo.service
sudo systemctl status bist-pattern-hpo.service
```

## 📊 Beklenen Sonuçlar

### Önce:
- JSON dosyalarında eski best params (filtre uygulanmadan)
- HPO ve Training DirHit'leri arasında farklar

### Sonra:
- ✅ JSON dosyalarında filtered best params
- ✅ HPO ve Training DirHit'leri daha tutarlı
- ✅ Servis kaldığı yerden devam eder

## ⚠️  Güvenlik

1. **Backup**: Her JSON dosyası için `.json.backup` oluşturulur
2. **Dry-run**: Önce test edilebilir
3. **State File**: State dosyası korunur (servis kaldığı yerden devam eder)

## 🚀 Hızlı Başlangıç

```bash
# 1. JSON'ları güncelle (dry-run)
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/update_json_with_filtered_best_params.py \
  --all-completed --dry-run

# 2. Gerçek güncelleme
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/update_json_with_filtered_best_params.py \
  --all-completed

# 3. Training (örnek)
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/retrain_high_discrepancy_symbols.py \
  --symbols EKGYO_1d

# 4. Servisi başlat
sudo systemctl start bist-pattern-hpo.service
```

