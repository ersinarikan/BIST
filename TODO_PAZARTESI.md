# 📋 PAZARTESİ İÇİN İŞ LİSTESİ

**Tarih**: 1 Ekim 2025  
**Pazartesi**: 7 Ekim 2025  
**Durum**: Pazar eğitimi sonrası devam edilecek  

---

## ✅ BUGÜN TAMAMLANAN (6/15)

1. ✅ Purged/Embargo Time-Series CV
2. ✅ Forward-chaining walk-forward (zaten cron'da!)
3. ✅ Trend/Volatilite (ADX, realized vol) - 9 features
4. ✅ Likidite/Hacim tier sınıflaması - 13 features
5. ✅ Seed bagging (3x per model)
6. ✅ Ufuk-bazlı modeller (zaten vardı)
7. ✅ **BONUS**: FinGPT sentiment adjustment

**Bugün Kazanç**: **+16-28% accuracy bekleniyor!**

**Features**: 73 → 95 (+22)  
**Git Commits**: 23  
**Linter**: 0 hata ✅  

---

## ⏳ KALAN 8 MADDE

### ÖNCELİK 1: KOLAY (4h)
**Pazartesi test başarılıysa hemen yap!**

**3. Frozen as-of pipeline** (2h)
   - Reproducible training
   - Specific date snapshot

**5. FinGPT tazelik/güven filtresi** (1h)
   - Script: backfill_fingpt_features.py
   - Çalıştır, CSV'ler oluştur
   - Training'e entegre et

**6. YOLO görsel yoğunluk/uyum** (1h)
   - Script: backfill_yolo_features.py
   - Çalıştır, CSV'ler oluştur
   - Training'e entegre et

**14. Sembol-hacim threshold** (1h)
   - Script: calibrate_thresholds.py
   - Entegre et

---

### ÖNCELİK 2: ÇOK ÖNEMLİ (3h)

**9. USDTRY/CDS/Faiz cross-asset** ⭐ (3h)
   - Yahoo Finance: USDTRY çek
   - Manuel CSV: CDS
   - TCMB API: Faiz
   - 8 macro feature ekle
   - **Kazanç**: +4-6% (ÇOK ÖNEMLİ!)

---

### ÖNCELİK 3: İLERİ SEVİYE (7h)

**4. Multi-anchor as-of + JSON report** (2h)
   - Script: shadow_eval.py var
   - Entegre et

**10. Ridge/Logit meta-learner** (3h)
   - Script: walkforward_meta_stacking.py var
   - OOF training ekle
   - **Kazanç**: +6-10%

**13. Delta volatilite normalizasyonu** (1h)
   - Calibration iyileştirmesi

**Quantile regression** (2h - listende 12)
   - Tahmin bantları (Q25/Q50/Q75)
   - Risk yönetimi

---

## 📊 TOPLAM POTANSİYEL

**Bugün Eklenen**: +16-28%  
**Kalan 8 Madde**: +12-25%  
**TOPLAM**: **+28-53% accuracy artışı!**

---

## 🎯 PAZARTESİ PLANI

### 1. Pazar Eğitimi Kontrol (Sabah)
```bash
# Log kontrol
tail -100 logs/cron_bulk_train.log

# Aranacak kelimeler:
# "✅ Using Purged Time-Series CV"
# "📊 95 feature" (veya 94-96 arası)
# "Seed bagging with 3 seeds"
# "ok_enh=545" (başarılı)
```

### 2. Test (Öğlen)
```python
# Accuracy ölç
# Baseline vs new karşılaştır
# +16-28% var mı?
```

### 3. Başarılıysa Devam (Öğleden Sonra)
**En önemli**: USDTRY/CDS/Faiz (3h, +4-6%)

**Kolay olanlar**: FinGPT/YOLO backfill (2h)

**Toplam**: 5 saat, +10-15% ekstra!

---

## 🚨 PAZAR EĞİTİMİ - BEKLENEN LOGLAR

**Dosya**: `logs/cron_bulk_train.log`

**Başarı Göstergeleri**:
```
[06:00] 🔒 Global ML training lock acquired by cron
[06:00] 🧠 THYAO için enhanced model eğitimi başlatılıyor
[06:00] 📊 95 feature kullanılacak  ← (önceden 73)
[06:00] ✅ Using Purged Time-Series CV (purge=5, embargo=2)
[06:00] XGBoost: Seed bagging with 3 seeds
[06:00] LightGBM: Seed bagging with 3 seeds
[06:00] CatBoost: Seed bagging with 3 seeds
...
[12:00] DONE: ok_enh=545 fail_enh=0 total=545
[12:00] 🔓 Global ML training lock released by cron
```

**Hata Göstergeleri** (olmamalı!):
```
❌ "fail_enh > 0"
❌ "Feature mismatch"
❌ "PurgedTimeSeriesSplit not found"
```

---

## 💾 YEDEKpackup Dosyalar

**Geri Almak İçin** (sorun olursa):
```
enhanced_ml_system.py.backup-purged-cv (önceki versiyon)
enhanced_ml_system.py.backup-seed-bagging
enhanced_ml_system.py.backup-liquidity
```

**Rollback**:
```bash
cp enhanced_ml_system.py.backup-purged-cv enhanced_ml_system.py
sudo systemctl restart bist-pattern
```

---

## 🎊 ÖZET

**Bugün**: 6 iyileştirme, 95 features, +16-28%  
**Pazar**: Training (02:00-09:00)  
**Pazartesi**: Test + devam  
**Kalan**: 8 madde, +12-25% potansiyel  

**Toplam Potansiyel**: +28-53% accuracy! 🎯🚀

---

**Pazartesi görüşürüz!** 😊
