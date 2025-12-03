# HPO ve Training Filtre Davranışı Analizi

## 🔍 HPO Sürecinde Filtre Uygulaması

### Kod Yeri: `optuna_hpo_with_feature_flags.py` (satır 800-914)

```python
# 1. Filtre değerleri okunuyor (satır 807-813)
_min_mc = int(os.getenv('HPO_MIN_MASK_COUNT', '0'))  # Default: 0
_min_mp = float(os.getenv('HPO_MIN_MASK_PCT', '0'))   # Default: 0.0

# 2. Her split için kontrol (satır 814-830)
if (_min_mc > 0 and mask_count < _min_mc) or (_min_mp > 0.0 and mask_pct < _min_mp):
    low_support = True  # Split exclude edilir
else:
    split_dirhits.append(dirhit_val)  # Split dahil edilir

# 3. Sembol için ortalama hesaplama (satır 874-885)
if split_dirhits:  # Eğer en az 1 split geçerliyse
    avg_dirhit_value = float(np.mean(split_dirhits))
    dirhits.append(avg_dirhit_value)  # Sembol score'a dahil edilir
else:
    print(f"No valid DirHit from any split")  # Sembol score'a dahil edilmez

# 4. Tüm semboller için score hesaplama (satır 900-914)
if not dirhits:  # Eğer HİÇBİR sembol için geçerli DirHit yoksa
    return 0.0  # Trial başarısız sayılır (score=0.0)

avg_dirhit = float(np.mean(dirhits))  # Geçerli sembollerin ortalaması
score = 0.7 * avg_dirhit - k * avg_nrmse  # Final score
```

### Senaryolar:

#### Senaryo 1: Tüm Split'ler Filtreyi Geçer
- **Durum**: 4 split'in hepsi mask_count >= 10 ve mask_pct >= 5.0
- **Sonuç**: Tüm split'ler dahil edilir, avg_dirhit hesaplanır
- **Score**: Normal hesaplanır

#### Senaryo 2: Bazı Split'ler Filtreyi Geçer
- **Durum**: 4 split'ten 2'si geçer, 2'si exclude
- **Sonuç**: Sadece geçen 2 split'in ortalaması alınır
- **Score**: Geçen split'ler üzerinden hesaplanır

#### Senaryo 3: Hiçbir Split Filtreyi Geçemez (KRİTİK!)
- **Durum**: 4 split'in hiçbiri mask_count >= 10 veya mask_pct >= 5.0 değil
- **Sonuç**: `split_dirhits` boş kalır
- **Sembol için**: `avg_dirhit_value = None`, `dirhits` listesine eklenmez
- **Trial için**: 
  - Eğer diğer semboller varsa ve onlar geçerliyse → Onların ortalaması alınır
  - Eğer TÜM semboller için hiçbir split geçemezse → `return 0.0` (trial başarısız)

## 🔍 Training Sürecinde Filtre Uygulaması

### Kod Yeri: `continuous_hpo_training_pipeline.py` (satır 2432-2479)

```python
# 1. Filtre değerleri okunuyor (satır 2438-2445)
_min_mc = int(os.getenv('HPO_MIN_MASK_COUNT', '0'))  # Default: 0
_min_mp = float(os.getenv('HPO_MIN_MASK_PCT', '0.0'))  # Default: 0.0

# 2. Her split için kontrol (satır 2446-2460)
if (_min_mc > 0 and mask_count < _min_mc) or (_min_mp > 0.0 and mask_pct < _min_mp):
    low_support = True  # Split exclude edilir
else:
    split_dirhits.append(dh)  # Split dahil edilir

# 3. Ortalama hesaplama (satır 2467-2479)
if split_dirhits:  # Eğer en az 1 split geçerliyse
    avg_dirhit = float(np.mean(split_dirhits))
    results['wfv'] = avg_dirhit  # DirHit hesaplanır
else:
    results['wfv'] = None  # DirHit None (hesaplanamaz)
```

### Senaryolar:

#### Senaryo 1: Tüm Split'ler Filtreyi Geçer
- **Sonuç**: Tüm split'ler dahil edilir, avg_dirhit hesaplanır

#### Senaryo 2: Bazı Split'ler Filtreyi Geçer
- **Sonuç**: Sadece geçen split'lerin ortalaması alınır

#### Senaryo 3: Hiçbir Split Filtreyi Geçemez (KRİTİK!)
- **Sonuç**: `results['wfv'] = None` → DirHit hesaplanamaz
- **Model**: Yine de eğitilir (model eğitimi filtreye bağlı değil)
- **Evaluation**: DirHit None olur, "LOW_SUPPORT" olarak işaretlenir

## ⚠️  Kritik Sorun: Filtreye Takılan Durumlar

### HPO'da:
- Eğer bir sembol için tüm split'ler exclude edilirse:
  - O sembol score hesaplamasına dahil edilmez
  - Ama diğer semboller varsa, onların ortalaması alınır
  - Trial devam eder, score hesaplanır

### Training'de:
- Eğer tüm split'ler exclude edilirse:
  - Model yine de eğitilir (eğitim filtreye bağlı değil)
  - DirHit None olur → "LOW_SUPPORT" olarak işaretlenir
  - Model kullanılabilir ama değerlendirilemez

## ✅ Doğrulama: HPO ve Training Aynı Filtreyi Kullanıyor mu?

### HPO (optuna_hpo_with_feature_flags.py):
```python
_min_mc = int(os.getenv('HPO_MIN_MASK_COUNT', '0'))  # Default: 0
_min_mp = float(os.getenv('HPO_MIN_MASK_PCT', '0'))   # Default: 0.0
```

### Training (continuous_hpo_training_pipeline.py):
```python
_min_mc = int(os.getenv('HPO_MIN_MASK_COUNT', '0'))  # Default: 0
_min_mp = float(os.getenv('HPO_MIN_MASK_PCT', '0.0'))  # Default: 0.0
```

**✅ EVET, aynı environment variable'ları kullanıyorlar!**

### Systemd Config:
```bash
Environment=HPO_MIN_MASK_COUNT=10
Environment=HPO_MIN_MASK_PCT=5.0
```

**✅ HPO sırasında 10/5.0 kullanılıyor**
**✅ Training'de de aynı environment variable'lar okunuyor (10/5.0)**

## 🎯 Sonuç

1. **HPO ve Training aynı filtreyi kullanıyor**: ✅ (aynı env var'lar)
2. **Filtreye takılan durumlar**:
   - HPO: Sembol score'a dahil edilmez, ama trial devam eder
   - Training: DirHit None olur, ama model eğitilir
3. **Best params bulunamazsa**: 
   - HPO: Trial score=0.0 döner (tüm semboller için geçersizse)
   - Training: Model eğitilir ama DirHit None

## 🔧 Öneri

Eğer filtreye takılan semboller için de model eğitmek istiyorsak:
- Filtreyi gevşetmek (10/5.0 → 5/3.0)
- Veya 0/0.0 kullanmak (filtre kapalı)

