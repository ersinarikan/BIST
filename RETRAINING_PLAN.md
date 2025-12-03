# Geçmiş HPO ve Training Sonuçlarını Düzeltme Planı

## 🎯 Amaç

Şimdiye kadar yapılan HPO ve training sonuçlarını düzeltmek:
1. Filtreye takılan sembolleri tespit et
2. Bu semboller için doğru best params'ı bul (filtre uygulandıktan sonra)
3. Retraining yap

## 📋 Adımlar

### Adım 1: Mevcut Durumu Analiz Et ✅

**Script:** `analyze_low_support_symbols.py`

**Ne Yapar:**
- Tüm HPO JSON dosyalarını tarar
- `low_support_warnings` listesini kontrol eder
- Study DB dosyalarını tarar
- Filtreye takılan sembolleri listeler

**Kullanım:**
```bash
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/analyze_low_support_symbols.py
```

**Çıktı:**
- Hangi sembollerin filtreye takıldığı listesi
- `low_support_symbols.txt` dosyası

### Adım 2: Retraining Yap

**Script:** `retrain_high_discrepancy_symbols.py` (zaten var, güncellendi)

**Ne Yapar:**
- Study DB'den doğru best params'ı bulur (filtre uygulandıktan sonra)
- Aynı filtreyi kullanarak retraining yapar
- Sonuçları kaydeder

**Kullanım:**
```bash
# Tüm low support semboller için
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/retrain_high_discrepancy_symbols.py \
  --symbols $(cat /opt/bist-pattern/results/low_support_symbols.txt | awk '{print $1"_"$2"d"}' | tr '\n' ',' | sed 's/,$//')

# Veya belirli semboller için
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/retrain_high_discrepancy_symbols.py \
  --symbols EKGYO_1d,BRSAN_3d
```

### Adım 3: Sonuçları Doğrula

**Kontrol:**
- Retraining sonuçlarını kontrol et
- HPO ve Training DirHit'leri karşılaştır
- Farkların azaldığını doğrula

## 🔄 İşlem Sırası

1. **Analiz Et** → `analyze_low_support_symbols.py`
2. **Retraining Yap** → `retrain_high_discrepancy_symbols.py`
3. **Doğrula** → Sonuçları kontrol et

## 📊 Beklenen Sonuçlar

### Önce:
- Filtreye takılan semboller için best params optimal olmayabilir
- HPO ve Training DirHit'leri arasında büyük farklar olabilir

### Sonra:
- Filtreye takılan semboller için doğru best params kullanılacak
- HPO ve Training DirHit'leri daha tutarlı olacak
- Uyarılar log'larda görünecek

## ⚠️  Dikkat Edilmesi Gerekenler

1. **Mevcut Modeller**: Retraining mevcut modelleri güncelleyecek
2. **Zaman**: Retraining uzun sürebilir (her sembol için)
3. **Kaynaklar**: CPU/GPU kullanımı artacak

## 🚀 Hızlı Başlangıç

```bash
# 1. Analiz et
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/analyze_low_support_symbols.py

# 2. Retraining yap (dry-run önce)
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/retrain_high_discrepancy_symbols.py \
  --symbols EKGYO_1d --dry-run

# 3. Gerçek retraining
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/retrain_high_discrepancy_symbols.py \
  --symbols EKGYO_1d
```

