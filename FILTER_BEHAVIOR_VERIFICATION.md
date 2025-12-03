# HPO ve Training Filtre Davranışı - Kod Doğrulama

## ✅ Doğrulama: HPO ve Training Aynı Filtreyi Kullanıyor mu?

### 1. HPO Süreci (optuna_hpo_with_feature_flags.py)

**Satır 807-813:**
```python
_min_mc = int(os.getenv('HPO_MIN_MASK_COUNT', '0'))  # Default: 0
_min_mp = float(os.getenv('HPO_MIN_MASK_PCT', '0'))   # Default: 0.0
```

**Systemd Config:**
```bash
Environment=HPO_MIN_MASK_COUNT=10
Environment=HPO_MIN_MASK_PCT=5.0
```

**Sonuç**: HPO sırasında `HPO_MIN_MASK_COUNT=10` ve `HPO_MIN_MASK_PCT=5.0` kullanılıyor ✅

### 2. Training Süreci (continuous_hpo_training_pipeline.py)

**Satır 2438-2445 (WFV Evaluation):**
```python
_min_mc = int(os.getenv('HPO_MIN_MASK_COUNT', '0'))  # Default: 0
_min_mp = float(os.getenv('HPO_MIN_MASK_PCT', '0.0'))  # Default: 0.0
```

**⚠️  TUTARSIZLIK BULUNDU!**

**Satır 1705 ve 1846'da farklı default'lar var:**
```python
# Satır 1705 (online evaluation):
_min_mc = int(os.getenv('HPO_MIN_MASK_COUNT', '10'))  # Default: 10 ❌

# Satır 1846 (online evaluation):
_min_mc = int(os.getenv('HPO_MIN_MASK_COUNT', '10'))  # Default: 10 ❌
```

**Sorun**: Online evaluation'da default 10/5.0, WFV evaluation'da default 0/0.0

## 🔍 Filtreye Takılan Durumlar

### Senaryo: 10/5.0 Filtresi, Hiçbir Split Geçemez

#### HPO'da (optuna_hpo_with_feature_flags.py):

**Satır 874-885:**
```python
if split_dirhits:  # Eğer en az 1 split geçerliyse
    avg_dirhit_value = float(np.mean(split_dirhits))
    dirhits.append(avg_dirhit_value)  # Sembol score'a dahil
else:
    print(f"No valid DirHit from any split")  # Sembol score'a dahil edilmez
```

**Satır 900-909:**
```python
if not dirhits:  # Eğer HİÇBİR sembol için geçerli DirHit yoksa
    return 0.0  # Trial başarısız (score=0.0)
```

**Sonuç**:
- Eğer bir sembol için tüm split'ler exclude edilirse → O sembol score'a dahil edilmez
- Ama diğer semboller varsa ve onlar geçerliyse → Onların ortalaması alınır, trial devam eder
- Eğer TÜM semboller için hiçbir split geçemezse → `return 0.0` (trial başarısız)

#### Training'de (continuous_hpo_training_pipeline.py):

**Satır 2467-2479:**
```python
if split_dirhits:  # Eğer en az 1 split geçerliyse
    avg_dirhit = float(np.mean(split_dirhits))
    results['wfv'] = avg_dirhit
else:
    results['wfv'] = None  # DirHit hesaplanamaz
```

**Model Eğitimi (satır 3210):**
```python
result = ml.train_enhanced_models(symbol, df)  # Model YİNE DE eğitilir
```

**Sonuç**:
- Model yine de eğitilir (eğitim filtreye bağlı değil) ✅
- DirHit None olur → "LOW_SUPPORT" olarak işaretlenir
- Model kullanılabilir ama değerlendirilemez

## ⚠️  Bulunan Tutarsızlıklar

1. **Online Evaluation'da farklı default'lar** (satır 1705, 1846)
2. **WFV Evaluation'da doğru default'lar** (satır 2438-2445)

## 🔧 Düzeltme Gerekiyor

Online evaluation'da da WFV ile aynı default'ları kullanmalıyız (0/0.0).

