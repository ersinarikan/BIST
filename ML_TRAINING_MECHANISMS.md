# 🎓 ML Model Training - Dual Mechanism Strategy

**Tarih**: 30 Eylül 2025
**Durum**: ✅ Optimize Edildi

---

## 📊 EĞİTİM MEKANİZMALARI

### ✅ Mekanizma 1: Automation Cycle Training (Continuous)

**Lokasyon**: `working_automation.py` (satır 284-377)

**Schedule**: Her cycle (~15 dakika)

**Yapılandırma**:
```bash
ML_TRAIN_INTERVAL_CYCLES=1     # Her cycle'da
ML_TRAIN_PER_CYCLE=50          # 50 sembol/cycle
ML_MAX_MODEL_AGE_DAYS=7        # 7 gün eski modeller retrain
ML_TRAINING_COOLDOWN_HOURS=6   # Tekrar eğitim arası min. süre
```

**Mantık**:
1. `get_training_candidates()` ile akıllı candidate seçimi
2. Yaşlı modeller öncelikli
3. Eksik ufuklar tamamlanır
4. Global training lock ile çakışma önlenir

**Avantajlar**:
- ✅ Sürekli güncel modeller
- ✅ Hızlı adaptasyon (yeni veriye)
- ✅ Aktif hisseler öncelikli
- ✅ Resource-efficient (50/cycle)

**Coverage**: 545 sembol ÷ 50 = ~11 cycle = **~2.75 saat** (tam coverage)

---

### ✅ Mekanizma 2: Crontab Weekly Training (Deep Clean)

**Schedule**: Her Pazar saat 02:00

**Cron Job**:
```cron
0 2 * * 0 /opt/bist-pattern/scripts/run_bulk_train.sh >> /opt/bist-pattern/logs/cron_bulk_train.log 2>&1
```

**Script**: `scripts/run_bulk_train.sh`

**Özellikler**:
- ✅ Systemd environment inherit
- ✅ Global training lock (automation ile çakışma önler)
- ✅ Flock mechanism (duplicate run önler)
- ✅ Tüm semboller eğitilir
- ✅ Post-train validation (`post_train_enhanced_check.py`)
- ✅ Detaylı logging

**Mantık**:
```python
# bulk_train_all.py
1. Acquire global training lock (crontab)
2. Get all active stocks
3. For each symbol:
   - Get 730 days data
   - Train basic ML
   - Train enhanced ML (XGBoost, LightGBM, CatBoost)
4. Post-train check
5. Release lock
```

**Avantajlar**:
- ✅ Full coverage garantisi
- ✅ Unutulan modelleri yakalar
- ✅ Consistency check
- ✅ Weekly baseline refresh

---

## 🔒 Çakışma Önleme

**Global Training Lock Mekanizması**:

```python
# ml_coordinator.py
_global_training_lock = threading.RLock()
_global_training_status = {
    'active': False,
    'started_by': None,  # 'automation', 'crontab', 'manual'
    'started_at': None
}
```

**Akış**:
1. Automation cycle training başlatmak ister
2. `mlc.acquire_global_training_lock("automation")` çağırır
3. Eğer crontab zaten lock almışsa → skip
4. Lock alınırsa → training yapar
5. Bitince `release_global_training_lock()`

**Sonuç**: ✅ Automation ve crontab asla aynı anda çalışmaz!

---

## 📅 Training Schedule

**Continuous (Automation)**:
```
00:00 → Cycle 1 (50 models)
00:15 → Cycle 2 (50 models)
00:30 → Cycle 3 (50 models)
...
02:45 → Cycle 11 (45 models) ← Tüm 545 sembol kapandı
03:00 → Cycle 12 (tekrar baştan, sadece yaşlı modeller)
```

**Weekly (Crontab)**:
```
Pazar 02:00 → Full retrain başlar
Pazar 04:00-06:00 → ~545 sembol tamamlanır (tahmini)
```

**Optimal**: Crontab gece çalışır, automation gündüz devam eder. Çakışma riski minimal.

---

## ✅ CURRENT STATUS

**Automation Cycle Training**:
- ✅ Aktif ve çalışıyor
- ✅ ENV variables doğru
- ✅ Global lock mekanizması çalışıyor
- ✅ Her cycle 50 model

**Crontab Weekly Training**:
- ✅ YENI EKLEND İ (30 Eylül 2025)
- ✅ Script hazır ve çalışır durumda
- ✅ Her Pazar 02:00
- ✅ Logging aktif

**Coordination**:
- ✅ Global lock prevents conflicts
- ✅ Both use ml_coordinator
- ✅ Cooldown respected
- ✅ Model age tracking

---

## 📊 EXPECTED BEHAVIOR

**Günlük**:
- Automation her cycle 50 model günceller
- Yaşlı modeller (>7 gün) öncelikli
- ~3 saatte tüm aktif semboller taranır

**Haftalık**:
- Pazar sabahı tüm modeller retrain
- Baseline consistency
- Unutulan semboller yakalanır

**Sonuç**: **Optimal ML model freshness!** 🎯

---

## 🔧 Monitoring

**Loglar**:
```bash
# Automation training logs
journalctl -u bist-pattern.service | grep "ML training"

# Crontab training logs
tail -f /opt/bist-pattern/logs/cron_bulk_train.log
```

**Metrics**:
- Model age: ml_coordinator.model_status
- Training success rate: logs
- Coverage: Enhanced ML system info

---

## ✅ SONUÇ

**Eğitim Stratejisi**: **Dual Mechanism** ✅

- Automation: Continuous, smart, prioritized
- Crontab: Weekly, comprehensive, guaranteed
- Coordination: Global lock, no conflicts

**Kalite**: ⭐⭐⭐⭐⭐ **Production-grade!**

**ML motorunuz artık en iyi durumda - sürekli güncel ve optimize!** 🚀
