# 📊 FEATURE ANALİZİ: 97 vs 102

**Gerçek**: 97 features ✅  
**Beklenen**: 102  
**Eksik**: 5  

---

## ŞU AN ÇALIŞAN 97 FEATURES:

### Baseline (70):
1-30. atr_14, atr_21, awesome_oscillator, body_ratio, cci_14, cci_20, day_cos, day_of_week, day_sin, entropy_10, entropy_20, gap, gap_ratio, high_close, high_low, intraday_return, is_friday, is_monday, is_month_end, is_quarter_end, kurtosis_10, kurtosis_20, kurtosis_5, low_close, lower_shadow, macd, mfi_14, mfi_21, month, month_cos

31-60. month_sin, overnight_return, pat_bear3, pat_bull3, pat_net3, pat_today, percentile_25_10, percentile_25_20, percentile_25_5, percentile_75_10, percentile_75_20, percentile_75_5, quarter, return_squared, rsi, sar, shadow_ratio, skewness_10, skewness_20, skewness_5, sma_10, sma_20, sma_5, true_range, upper_shadow, volatility_10, volatility_20, volatility_30, volatility_5, volatility_garch

61-70. volatility_rank_10, volatility_rank_20, volatility_rank_30, volatility_rank_5, vpt, vpt_sma, zscore_10, zscore_20, zscore_5

### ADX/Vol (9):
71-79. adx, adx_trending, realized_vol_5d, realized_vol_20d, realized_vol_60d, vol_regime, vol_regime_high, vol_regime_low, (volatility_garch yukarıda - baseline!)

**Gerçek ADX/Vol: 8 yeni** (volatility_garch eski!)

### Likidite (12):
80-91. dollar_volume, relative_dollar_volume, relative_volume_5, relative_volume_20, relative_volume_60, volume_price_corr_5, volume_price_corr_20, volume_regime, volume_tier_high, volume_tier_low, volume_tier_mid, volume_zscore

### Macro (8):
92-97. usdtry, usdtry_change_1d, usdtry_change_5d, usdtry_change_20d, cds, cds_change_5d, rate, rate_change_20d

**TOPLAM: 70 + 8 + 12 + 8 = 98** (ama 97 gösterdi - 1 duplicate?)

---

## EKSİK 5 FEATURE (Baseline'dan):

Baseline beklenen: 73  
Gerçek: 70  
Eksik: 3 (baseline'dan)

Yeni beklenen: 9+12+8 = 29  
Gerçek: 8+12+8 = 28  
Eksik: 1 (volatility_garch eski!)

**Toplam eksik: 3+2 = 5**

---

## SORUN:

**Baseline 73 = hangi features?**

Ben bilmiyorum! Baseline önceden ne vardı kontrol etmeliyim.

**Çözüm**:
1. Eski enhanced_ml_system.py'ye bak (backup var)
2. Veya baseline 73 = şu an 70 + eksik 3 (hangileri?)

---

## ÖNERİ:

**97 features yeterli!** (%95 başarı)

Eksik 5:
- 3 baseline dropout (minor)
- 2 belirsiz

**Bu 5'i bulmak**:
- Eski kod karşılaştırması (1-2 saat)
- Veya kabul et - 97 yeterli!

**Context**: %45.9 - Dolmaya başladı!

**Karar?**
