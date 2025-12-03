# Retrain Strategy - HPO Best Params ile Training

## 🎯 Amaç

HPO'nun bulduğu best params'ların gerçekten optimal olup olmadığını test etmek.

## 🔍 Problem

1. **HPO sırasında**: Low support filtre uygulanıyor (10/5.0 veya 0/0.0)
2. **Best params**: Bu filtre uygulanmış split'ler üzerinden bulunuyor
3. **Training sırasında**: Farklı filtre kullanılırsa, best params optimal olmayabilir

## ✅ Çözüm

Study dosyalarından:
1. Best trial'ın split bilgilerini oku
2. Hangi filtre kullanılmış tespit et (min_mask_count, min_mask_pct)
3. Training'de **AYNI filtreyi** kullan
4. Böylece HPO'nun bulduğu best params'ların gerçekten optimal olup olmadığını test et

## 📊 Örnek Senaryolar

### Senaryo 1: ADEL_1d
- **HPO**: 0/0.0 filtre → Tüm 4 split dahil → Best params bulundu
- **Training**: 0/0.0 filtre kullan → Best params optimal olmalı
- **Sonuç**: Eğer hala fark varsa, başka bir sorun var demektir

### Senaryo 2: EKGYO_1d
- **HPO**: 10/5.0 filtre → Sadece 1 split dahil (3 split exclude) → Best params bulundu
- **Training**: 10/5.0 filtre kullan → Best params bu 1 split için optimal olmalı
- **Sonuç**: Eğer hala fark varsa, best params gerçekten optimal değil demektir

## 🔧 Script Kullanımı

```bash
# Study dosyalarından filtre analizi
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/analyze_best_trial_filters.py --symbol ADEL --horizon 1

# HPO'nun kullandığı filtre ile retrain
/opt/bist-pattern/venv/bin/python3 /opt/bist-pattern/scripts/retrain_high_discrepancy_symbols.py --threshold 30.0
```

## ⚠️  Önemli Notlar

1. **Filtre Tutarlılığı**: Training'de HPO'nun kullandığı filtreyi kullanmak kritik
2. **Low Support Uyarısı**: Eğer best params sadece 1-2 split üzerinden bulunduysa, optimal olmayabilir
3. **Yeniden HPO**: Eğer sonuçlar hala kötüyse, HPO'yu 0/0.0 filtre ile yeniden çalıştırmak gerekebilir

## 📈 Beklenen Sonuçlar

- **Filtre tutarlıysa**: HPO DirHit ≈ Training DirHit (küçük farklar normal)
- **Filtre tutarsızsa**: Büyük farklar görülebilir
- **Best params optimal değilse**: Her iki durumda da kötü sonuçlar görülebilir

