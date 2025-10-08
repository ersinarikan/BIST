# CRON Schedule Optimization Guide

## Mevcut Sorunlar

### 1. Redundant Executions
Bazı job'lar hem standalone hem de nightly_master içinde çalışıyor:
- `evaluate_metrics`: 00:05 + 01:00 (nightly_master içinde)
- `calibrate_confidence`: 00:15 + 01:00 (nightly_master içinde)
- `populate_outcomes`: Her 10 dakika + 01:00 (nightly_master içinde)

### 2. Çok Sık Çalışan Job'lar
- `populate_outcomes`: Her 10 dakikada bir (günde 144 kez!)
  - Gereksiz DB yükü
  - Çoğu çalışma boşa gidiyor (predictions 10 dakikada mature olmaz)

## Önerilen Optimizasyon

### Option A: Nightly Master Merkezli (Önerilen)

```bash
# /etc/cron.d/bist-pattern-optimized

# Outcomes her 20 dakikada bir (yeterli)
*/20 * * * * www-data /opt/bist-pattern/scripts/run_populate_outcomes.sh

# Ana maintenance job (gece 02:00)
0 2 * * * www-data /opt/bist-pattern/scripts/nightly_master.sh

# Haftalık full retrain (Pazar 03:00)
0 3 * * 0 www-data /opt/bist-pattern/scripts/run_bulk_train.sh
```

**Nightly Master Environment Flags:**
```bash
# Standalone job'lar nightly_master'da çalışmasın
export RUN_POPULATE_OUTCOMES=0  # Zaten her 20 dk çalışıyor
export RUN_EVALUATE_METRICS=1   # Sadece nightly_master'da
export RUN_CALIBRATE_CONFIDENCE=1  # Sadece nightly_master'da
export RUN_OPTIMIZE_WEIGHTS=1
export RUN_PUBLISH_PARAMS=1
export RUN_DRIFT_CHECK=1
```

### Option B: Standalone Jobs (Alternatif)

```bash
# Her job ayrı zamanlarda
*/20 * * * * www-data /opt/bist-pattern/scripts/run_populate_outcomes.sh
30 0 * * * www-data /opt/bist-pattern/scripts/run_evaluate_metrics.sh
45 0 * * * www-data /opt/bist-pattern/scripts/run_calibrate_confidence.sh
0 1 * * * www-data /opt/bist-pattern/scripts/nightly_master_light.sh
```

**Nightly Master Light (sadece unique job'lar):**
```bash
#!/usr/bin/env bash
# scripts/nightly_master_light.sh
log "optimize_evidence_weights"
"$PY" scripts/optimize_evidence_weights.py

log "publish_params"
/opt/bist-pattern/scripts/publish_params.sh

log "check_drift_and_alert"
"$PY" scripts/check_drift_and_alert.py
```

## Environment Variables İçin Systemd Override

```ini
# /etc/systemd/system/bist-pattern.service.d/99-consolidated.conf

[Service]
# Cron optimization flags
Environment="RUN_POPULATE_OUTCOMES=0"
Environment="RUN_EVALUATE_METRICS=1"
Environment="RUN_CALIBRATE_CONFIDENCE=1"
Environment="RUN_OPTIMIZE_WEIGHTS=1"
Environment="RUN_PUBLISH_PARAMS=1"
Environment="RUN_DRIFT_CHECK=1"

# Timezone for market
Environment="MARKET_TZ_OFFSET_HOURS=3"  # Istanbul UTC+3
Environment="TZ=Europe/Istanbul"
```

## Uygulama

### 1. Yeni Crontab Yükle
```bash
# Option A için:
crontab -e
# Yukarıdaki Option A schedule'ı yapıştır

# Veya dosyadan yükle:
sudo cp /opt/bist-pattern/docs/crontab.optimized /etc/cron.d/bist-pattern
sudo systemctl restart cron
```

### 2. Environment Variables Set Et
```bash
# Systemd override güncelle
sudo systemctl edit bist-pattern
# Environment variables ekle

# Reload
sudo systemctl daemon-reload
sudo systemctl restart bist-pattern
```

### 3. Test Et
```bash
# Manuel test
RUN_POPULATE_OUTCOMES=0 /opt/bist-pattern/scripts/nightly_master.sh

# Log kontrol
tail -f /opt/bist-pattern/logs/nightly_master.log
```

## Beklenen Davranış

**Önceki Durum:**
- populate_outcomes: 144 kez/gün
- evaluate_metrics: 2 kez/gün
- calibrate_confidence: 2 kez/gün
- Toplam: ~148 job execution/gün

**Sonraki Durum (Option A):**
- populate_outcomes: 72 kez/gün (her 20 dk)
- nightly_master: 1 kez/gün (tüm maintenance)
- Toplam: ~73 job execution/gün

**Tasarruf:** %50 azalma

## Monitoring

### Log Kontrol
```bash
# Populate outcomes
tail -f /opt/bist-pattern/logs/populate_outcomes.log

# Nightly master
tail -f /opt/bist-pattern/logs/nightly_master.log

# Calibration
tail -f /opt/bist-pattern/logs/calibrate_confidence.log
```

### Metrics Kontrol
```bash
# Calibration state
cat /opt/bist-pattern/logs/calibration_state.json | jq '.horizons'

# Param store son güncelleme
cat /opt/bist-pattern/logs/param_store.json | jq '.generated_at'
```

