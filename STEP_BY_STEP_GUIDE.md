# Adım Adım Yapılacaklar - Detaylı Rehber

## 🎯 Amaç

Geçmiş HPO sonuçlarını düzeltmek:
1. Study dosyalarından filtreye göre doğru best params'ı bul
2. Mevcut JSON dosyalarını güncelle
3. Güncellenmiş params ile training yap
4. HPO servisini başlat

---

## 📋 ADIM 1: Hazırlık ve Kontrol

### 1.1 Mevcut Durumu Kontrol Et

**Ne Yapıyoruz:**
- Hangi sembollerin tamamlandığını kontrol ediyoruz
- Hangi JSON dosyalarının olduğunu görüyoruz
- Study DB dosyalarının varlığını kontrol ediyoruz

**Komutlar:**
```bash
# State dosyasını kontrol et
cat /opt/bist-pattern/results/continuous_hpo_state.json | jq 'keys | length'

# Tamamlanmış sembolleri listele
cat /opt/bist-pattern/results/continuous_hpo_state.json | jq 'to_entries | map(select(.value.status == "completed")) | length'

# JSON dosyalarını say
ls -1 /opt/bist-pattern/results/optuna_pilot_features_on_h*.json | wc -l

# Study DB dosyalarını kontrol et
ls -1 /opt/bist-pattern/hpo_studies/*.db | wc -l
```

**Beklenen Çıktı:**
- Kaç sembol tamamlanmış
- Kaç JSON dosyası var
- Kaç study DB dosyası var

---

## 📋 ADIM 2: JSON Dosyalarını Güncelle (DRY-RUN)

### 2.1 Test Modunda Çalıştır

**Ne Yapıyoruz:**
- Tüm tamamlanmış semboller için study DB'den filtered best params buluyoruz
- JSON dosyalarında ne değişeceğini görüyoruz
- **AMA HİÇBİR DOSYA DEĞİŞTİRİLMİYOR** (dry-run)

**Komut:**
```bash
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/update_json_with_filtered_best_params.py \
  --all-completed \
  --dry-run
```

**Ne Göreceğiz:**
```
🔄 Processing EKGYO 1d
✅ Found study DB: hpo_with_features_EKGYO_h1_c2.db
✅ Found best params: trial #123, DirHit: 45.23%
✅ Found JSON file: optuna_pilot_features_on_h1_c2_20251202_001529.json
🔍 DRY-RUN: Would update optuna_pilot_features_on_h1_c2_20251202_001529.json for EKGYO 1d
   Current best_trial_number: 100
   New best_trial_number: 123
   Current best_dirhit: 42.50
   New best_value: 45.23
```

**Kontrol Edilecekler:**
- ✅ Her sembol için study DB bulunuyor mu?
- ✅ Best params bulunuyor mu?
- ✅ JSON dosyası bulunuyor mu?
- ✅ Değişiklikler mantıklı mı? (yeni trial number, yeni DirHit)

**Eğer Hata Varsa:**
- Study DB bulunamıyorsa → O sembolü atla
- JSON bulunamıyorsa → O sembolü atla
- Best params bulunamıyorsa → O sembolü atla

---

## 📋 ADIM 3: JSON Dosyalarını Güncelle (GERÇEK)

### 3.1 Gerçek Güncelleme

**Ne Yapıyoruz:**
- Dry-run'da gördüğümüz değişiklikleri gerçekten uyguluyoruz
- Her JSON dosyası için backup oluşturuluyor (`.json.backup`)
- JSON dosyaları güncelleniyor

**Komut:**
```bash
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/update_json_with_filtered_best_params.py \
  --all-completed
```

**Ne Olacak:**
1. Her JSON için backup oluşturulur: `optuna_pilot_features_on_h1_c2_20251202_001529.json.backup`
2. JSON dosyası güncellenir:
   - `best_params` → Yeni filtered best params
   - `best_trial_number` → Yeni trial number
   - `best_dirhit` → Yeni DirHit
   - `_updated_at` → Timestamp eklenir
   - `_updated_reason` → "filtered_best_params_from_study" eklenir

**Çıktı:**
```
🔄 Processing EKGYO 1d
✅ Found study DB: hpo_with_features_EKGYO_h1_c2.db
✅ Found best params: trial #123, DirHit: 45.23%
✅ Found JSON file: optuna_pilot_features_on_h1_c2_20251202_001529.json
✅ Backup created: optuna_pilot_features_on_h1_c2_20251202_001529.json.backup
✅ Updated optuna_pilot_features_on_h1_c2_20251202_001529.json for EKGYO 1d
   Best trial: #123 (was #100)
   Best DirHit: 45.23%
```

**Kontrol:**
```bash
# Backup dosyalarını kontrol et
ls -1 /opt/bist-pattern/results/*.json.backup | wc -l

# Bir JSON dosyasını kontrol et
cat /opt/bist-pattern/results/optuna_pilot_features_on_h1_c2_20251202_001529.json | jq '._updated_at, .best_trial_number, .best_dirhit'
```

---

## 📋 ADIM 4: Training Yap (Seçilen Semboller)

### 4.1 Hangi Sembolleri Retrain Edeceğiz?

**Seçenekler:**
1. **Tüm güncellenmiş semboller** (uzun sürer)
2. **Sadece filtreye takılan semboller** (önerilen)
3. **Belirli semboller** (test için)

### 4.2 Örnek: Belirli Semboller İçin Training

**Komut:**
```bash
# Örnek: EKGYO ve BRSAN için
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/retrain_high_discrepancy_symbols.py \
  --symbols EKGYO_1d,BRSAN_3d
```

**Ne Yapıyoruz:**
1. Güncellenmiş JSON dosyasından best params'ı okur
2. Study DB'den filtre bilgisini alır
3. Aynı filtreyi kullanarak training yapar
4. Sonuçları kaydeder

**Çıktı:**
```
🔄 Retraining EKGYO 1d with best params...
✅ Found study DB: hpo_with_features_EKGYO_h1_c2.db
✅ Best trial: 123, Best value: 45.23
   Filter used: min_count=10, min_pct=5.0
   Splits: 3/4 included
🔧 Training will use filter: HPO_MIN_MASK_COUNT=10, HPO_MIN_MASK_PCT=5.0
🎯 Starting training for EKGYO 1d with best params...
✅ Training completed for EKGYO 1d
📊 EKGYO 1d: WFV DirHit (adaptive OFF) = 44.50%
```

**Kontrol:**
- Training DirHit ile HPO DirHit karşılaştırılır
- Fark azaldı mı kontrol edilir

---

## 📋 ADIM 5: HPO Servisini Başlat

### 5.1 Servis Durumunu Kontrol Et

**Komut:**
```bash
# Servis durumunu kontrol et
sudo systemctl status bist-pattern-hpo.service
```

**Beklenen:**
- Eğer durdurulmuşsa → `inactive (dead)`
- Eğer çalışıyorsa → `active (running)`

### 5.2 Servisi Başlat

**Komut:**
```bash
# Servisi başlat
sudo systemctl start bist-pattern-hpo.service

# Durumu kontrol et
sudo systemctl status bist-pattern-hpo.service
```

**Beklenen:**
```
● bist-pattern-hpo.service - BIST Pattern HPO Service
   Loaded: loaded
   Active: active (running) since ...
```

### 5.3 Log'ları İzle

**Komut:**
```bash
# Son log'ları göster
sudo journalctl -u bist-pattern-hpo.service -n 50

# Canlı log takibi
sudo journalctl -u bist-pattern-hpo.service -f
```

**Kontrol Edilecekler:**
- ✅ Servis başladı mı?
- ✅ State dosyasından tamamlanmış sembolleri atlıyor mu?
- ✅ Yeni semboller için HPO yapıyor mu?
- ✅ Güncellenmiş JSON dosyalarını kullanıyor mu?

---

## 📊 Özet: Tüm Adımlar

```bash
# ============================================
# ADIM 1: Hazırlık
# ============================================
cat /opt/bist-pattern/results/continuous_hpo_state.json | jq 'to_entries | map(select(.value.status == "completed")) | length'

# ============================================
# ADIM 2: Dry-Run (Test)
# ============================================
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/update_json_with_filtered_best_params.py \
  --all-completed --dry-run

# ============================================
# ADIM 3: Gerçek Güncelleme
# ============================================
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/update_json_with_filtered_best_params.py \
  --all-completed

# Kontrol
ls -1 /opt/bist-pattern/results/*.json.backup | wc -l

# ============================================
# ADIM 4: Training (Örnek)
# ============================================
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/retrain_high_discrepancy_symbols.py \
  --symbols EKGYO_1d

# ============================================
# ADIM 5: Servisi Başlat
# ============================================
sudo systemctl start bist-pattern-hpo.service
sudo systemctl status bist-pattern-hpo.service
sudo journalctl -u bist-pattern-hpo.service -f
```

---

## ⚠️  Önemli Notlar

1. **Backup**: Her JSON için backup oluşturulur, güvenli
2. **Dry-Run**: Önce test edin, sonra gerçek güncelleme yapın
3. **State File**: State dosyası korunur, servis kaldığı yerden devam eder
4. **Zaman**: Training uzun sürebilir, sabırlı olun
5. **Hatalar**: Eğer bir sembol için hata varsa, diğerleri devam eder

---

## 🆘 Sorun Giderme

### JSON Güncellenmedi
- Study DB bulunamadı mı? → Kontrol et
- Best params bulunamadı mı? → Filtre uygulanamadı olabilir
- JSON bulunamadı mı? → State'teki path'i kontrol et

### Training Başarısız
- JSON dosyası güncellenmiş mi? → Kontrol et
- Best params doğru mu? → JSON'u kontrol et
- Filtre değerleri doğru mu? → Environment variable'ları kontrol et

### Servis Başlamıyor
- State dosyası okunuyor mu? → Kontrol et
- Log'larda hata var mı? → `journalctl` ile kontrol et
- Permissions sorunu var mı? → `sudo` kullanın

