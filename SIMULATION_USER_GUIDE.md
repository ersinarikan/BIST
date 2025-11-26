# Simülasyon Paneli Kullanım Kılavuzu
## Admin Dashboard - Alım-Satım Simülasyonu

---

## 📊 GENEL BAKIŞ

Simülasyon paneli, geliştirdiğiniz AI modelinin önerilerine göre işlem yapıldığında ne kadar kazançlı çıkılacağını test etmenizi sağlar. Bu, **modelin gerçekten işe yarayıp yaramadığını doğrulamak** için kritik bir araçtır.

---

## 🎛️ PARAMETRELER VE AÇIKLAMALARI

### 1. **Sermaye (Trade Amount)**
**Ne İşe Yarar:** Simülasyonda kullanılacak başlangıç sermayesi

**Önerilen Değer:** 
- Test için: 10,000 - 50,000 TL
- Gerçekçi test için: 100,000 TL

**Nasıl Kullanılır:**
- Düşük sermaye ile hızlı test yapabilirsiniz
- Yüksek sermaye ile daha gerçekçi sonuçlar alırsınız
- Commission etkisi daha net görülür

**Örnek:** 100,000 TL → 10 pozisyon için ortalama 10,000 TL/pozisyon

---

### 2. **Horizon (Ufuk)**
**Ne İşe Yarar:** Modelin öngördüğü zaman dilimi

**Seçenekler:**
- **1d:** 1 gün sonraki fiyat tahmini
- **3d:** 3 gün sonraki fiyat tahmini
- **7d:** 7 gün sonraki fiyat tahmini
- **14d:** 14 gün sonraki fiyat tahmini
- **30d:** 30 gün sonraki fiyat tahmini

**Nasıl Kullanılır:**
- **Kısa vadeli test:** 1d, 3d → Hızlı sonuç, daha fazla işlem
- **Orta vadeli test:** 7d, 14d → Dengeli, gerçekçi
- **Uzun vadeli test:** 30d → Daha az işlem, daha uzun süre

**Önemli:** Model bu horizon'a göre pozisyon tutma süresini belirler. Örneğin 14d seçerseniz, model "14 gün tut" der.

**Örnek Senaryo:**
- Horizon: 14d → Model "14 gün sonra %5 artış" dedi
- Simülasyon: 14 gün boyunca pozisyon tutar (model_test modunda)
- 14. günde otomatik satış yapılır

---

### 3. **Top N**
**Ne İşe Yarar:** Portföyde tutulacak maksimum pozisyon sayısı

**Seçenekler:** 1-10 arası

**Nasıl Kullanılır:**
- **Düşük (1-3):** Konsantre portföy, yüksek risk
- **Orta (4-7):** Dengeli portföy
- **Yüksek (8-10):** Çeşitlendirilmiş portföy, düşük risk

**Örnek:**
- Top N: 5 → En yüksek confidence'lı 5 sembol seçilir
- Her sembole confidence'a göre ağırlıklandırılmış sermaye ayrılır

**İpucu:** Daha fazla pozisyon = daha fazla çeşitlendirme, ama daha fazla commission

---

### 4. **Commission (Komisyon)**
**Ne İşe Yarar:** Her alım-satım işleminde kesilen komisyon oranı

**Önerilen Değer:** 0.0005 (0.05% = BIST standardı)

**Nasıl Kullanılır:**
- **Düşük (0.0001-0.0003):** Düşük maliyet, daha fazla işlem yapılabilir
- **Standart (0.0005):** BIST gerçekçi değeri
- **Yüksek (0.001-0.002):** Yüksek maliyet, daha az işlem yapılmalı

**Hesaplama:**
- Alım: 10,000 TL × 0.0005 = 5 TL komisyon
- Satış: 10,000 TL × 0.0005 = 5 TL komisyon
- Toplam: 10 TL (her round-trip için)

**Önemli:** Yüksek commission, sık işlem yapan stratejileri olumsuz etkiler

---

### 5. **Stop-loss %**
**Ne İşe Yarar:** Zararı sınırlamak için pozisyonun otomatik satılacağı düşüş yüzdesi

**Seçenekler:** 0-20% arası

**Nasıl Kullanılır:**
- **Sıkı (1-3%):** Küçük zararları önler, ama çok sık tetiklenebilir
- **Orta (3-5%):** Dengeli, normal volatilite için uygun
- **Gevşek (5-10%):** Büyük zararlara izin verir, daha az tetiklenir
- **Kapalı (0%):** Stop-loss yok (sadece model_test modunda)

**Örnek Senaryo:**
- Entry: 100 TL
- Stop-loss: 3%
- Stop fiyat: 97 TL
- Fiyat 97 TL'ye düşerse → Otomatik satış

**Önemli:** 
- Model_test modunda stop-loss **devre dışı**
- Hybrid modunda stop-loss **aktif** (risk yönetimi kategorisi)
- Risk_management modunda stop-loss **aktif**

**İpucu:** Volatil semboller için daha yüksek stop-loss kullanın (5-7%)

---

### 6. **Relatif Düşüş % (Relative Drop Threshold)**
**Ne İşe Yarar:** Modelin güven skorunun (confidence) ne kadar düşmesine izin verileceği

**Seçenekler:** 1-50% arası

**Nasıl Kullanılır:**
- **Düşük (10-20%):** Küçük güven düşüşünde satış → Daha sık tetiklenir
- **Orta (20-30%):** Normal güven düşüşünde satış → Dengeli
- **Yüksek (30-50%):** Büyük güven düşüşünde satış → Daha az tetiklenir

**Örnek Senaryo:**
- Entry confidence: 0.8 (80%)
- Relative drop: 20%
- Exit condition: confidence < 0.8 × (1 - 0.20) = 0.64 (64%)
- Yeni confidence: 0.63 → Satış yapılır

**Önemli:**
- Model_test modunda **devre dışı**
- Hybrid modunda **aktif** (risk yönetimi kategorisi)
- Risk_management modunda **aktif**

**İpucu:** Yüksek confidence'lı pozisyonlar için daha düşük threshold kullanın (15-20%)

---

### 7. **Simülasyon Modu** ⭐ YENİ
**Ne İşe Yarar:** Simülasyonun neyi ölçeceğini belirler

**Seçenekler:**

#### A. **Hibrit (Model + Risk Yönetimi)** - Önerilen
**Ne Yapar:**
- Model önerilerini kullanır
- Risk yönetimi mekanizmalarını da aktif eder
- Her iki performansı ayrı ayrı ölçer

**Ne Zaman Kullanılır:**
- Model performansını ve risk yönetimini birlikte test etmek istediğinizde
- Gerçekçi bir strateji testi için
- Hangi mekanizmanın daha etkili olduğunu görmek için

**Nasıl Çalışır:**
- Model "14 gün tut" dedi → 14 güne kadar tutar
- Ama 3. günde stop-loss tetiklenirse → Erken satış (risk yönetimi)
- Sonuçlar iki kategoriye ayrılır:
  - **Model Performansı:** Horizon dolduğunda veya sell signal geldiğinde
  - **Risk Yönetimi:** Stop-loss veya confidence drop ile erken çıkış

**Örnek Sonuç:**
```
Toplam P&L: +2,500 TL
├─ Model Performansı: +3,000 TL (10 işlem, %70 hit rate)
└─ Risk Yönetimi: -500 TL (5 işlem, %40 hit rate)
```

---

#### B. **Model Testi (Sadece Model Performansı)**
**Ne Yapar:**
- Sadece model önerilerini takip eder
- Stop-loss ve confidence drop **devre dışı**
- Modelin önerdiği süre (horizon) dolana kadar pozisyon tutar

**Ne Zaman Kullanılır:**
- Modelin saf performansını ölçmek istediğinizde
- "Model haklı mıydı?" sorusunu cevaplamak için
- Risk yönetimi etkisini hariç tutmak için

**Nasıl Çalışır:**
- Model "14 gün tut" dedi → 14 gün boyunca tutar (stop-loss yok)
- 14. günde otomatik satış
- Veya model "sat" sinyali verirse → Satış

**Örnek Senaryo:**
```
Gün 1: Model "14d sonra %5 artış" → Alım (100 TL)
Gün 3: Fiyat 97 TL'ye düştü → Stop-loss YOK, pozisyon tutulur
Gün 14: Fiyat 105 TL → Otomatik satış → +5% kâr
Sonuç: Model haklıydı! ✅
```

**Önemli:** Bu mod, modelin gerçek performansını ölçer. Risk yönetimi etkisi yoktur.

---

#### C. **Risk Yönetimi (Stop-loss + Confidence Drop)**
**Ne Yapar:**
- Sadece risk yönetimi mekanizmalarını test eder
- Stop-loss ve confidence drop aktif
- Modelin horizon önerisi göz ardı edilir

**Ne Zaman Kullanılır:**
- Risk yönetimi mekanizmalarının ne kadar etkili olduğunu görmek için
- Stop-loss ve confidence drop'un değerini ölçmek için
- Model performansından bağımsız risk yönetimi testi için

**Nasıl Çalışır:**
- Model "14 gün tut" dedi ama stop-loss 3. günde tetiklendi → Satış
- Sonuçlar risk yönetimi kategorisinde

**Örnek Sonuç:**
```
Risk Yönetimi Performansı:
- 15 işlem
- P&L: -1,200 TL
- Hit Rate: %45
- Ortalama zarar: -80 TL/işlem
```

---

## 🎯 KULLANIM SENARYOLARI

### Senaryo 1: Model Performansını Test Etmek
**Amaç:** "Modelim gerçekten işe yarıyor mu?"

**Ayarlar:**
- **Mod:** Model Testi
- **Horizon:** 7d veya 14d (modelin önerdiği horizon)
- **Top N:** 5-10
- **Stop-loss:** 0% (devre dışı)
- **Relatif Düşüş:** 0% (devre dışı)

**Ne Beklenir:**
- Modelin önerdiği süre dolana kadar pozisyonlar tutulur
- Sadece model performansı ölçülür
- Sonuç: Model haklı mıydı?

---

### Senaryo 2: Gerçekçi Strateji Testi
**Amaç:** "Gerçek piyasada nasıl performans gösterir?"

**Ayarlar:**
- **Mod:** Hibrit
- **Horizon:** 14d
- **Top N:** 7-10
- **Stop-loss:** 3-5%
- **Relatif Düşüş:** 20-25%
- **Commission:** 0.0005

**Ne Beklenir:**
- Model önerileri + risk yönetimi birlikte çalışır
- Her iki performans ayrı ayrı ölçülür
- Sonuç: Hangi mekanizma daha etkili?

---

### Senaryo 3: Risk Yönetimi Optimizasyonu
**Amaç:** "Stop-loss ve confidence drop ne kadar etkili?"

**Ayarlar:**
- **Mod:** Risk Yönetimi
- **Stop-loss:** 3%, 5%, 7% (farklı değerlerle test)
- **Relatif Düşüş:** 15%, 20%, 25% (farklı değerlerle test)

**Ne Beklenir:**
- Sadece risk yönetimi performansı ölçülür
- Farklı parametrelerle test edilir
- Sonuç: En iyi stop-loss ve confidence drop değerleri

---

### Senaryo 4: Hızlı Test
**Amaç:** "Hızlıca bir fikir edinmek"

**Ayarlar:**
- **Mod:** Model Testi
- **Horizon:** 1d veya 3d (kısa süre)
- **Top N:** 3-5
- **Sermaye:** 10,000 TL

**Ne Beklenir:**
- Hızlı sonuç (1-3 gün)
- Daha fazla işlem
- Genel bir fikir

---

## 📈 SONUÇLARI YORUMLAMA

### Hibrit Mod Sonuçları

**Örnek Çıktı:**
```
Toplam P&L: +2,500 TL (+2.5%)

Model Performansı:
- İşlem: 10
- P&L: +3,000 TL
- Kârlı: 7
- Hit Rate: 70%

Risk Yönetimi:
- İşlem: 5
- P&L: -500 TL
- Kârlı: 2
- Hit Rate: 40%
```

**Yorumlama:**
- ✅ Model performansı iyi (70% hit rate, +3,000 TL)
- ⚠️ Risk yönetimi zarar veriyor (-500 TL, 40% hit rate)
- 💡 **Öneri:** Stop-loss ve confidence drop parametrelerini optimize et

---

### Model Testi Mod Sonuçları

**Örnek Çıktı:**
```
Toplam P&L: +5,000 TL (+5%)
İşlem: 15
Kârlı: 10
Hit Rate: 66.7%
```

**Yorumlama:**
- ✅ Model performansı iyi (66.7% hit rate, +5%)
- ✅ Model önerileri genel olarak doğru
- 💡 **Öneri:** Model güvenilir, gerçek piyasada kullanılabilir

---

## ⚠️ ÖNEMLİ NOTLAR

### 1. **Horizon ve Pozisyon Tutma Süresi**
- Model "14d sonra %5 artış" dedi → 14 gün tutulur (model_test modunda)
- Stop-loss ile erken çıkış → Model performansına dahil edilmez (hibrit modda risk yönetimi kategorisi)

### 2. **Commission Etkisi**
- Sık işlem yapan stratejiler commission'dan olumsuz etkilenir
- Toplam commission'ı kontrol edin
- Net P&L = Brüt P&L - Commission

### 3. **Confidence Ağırlıklandırması**
- Yüksek confidence'lı sinyaller daha fazla sermaye alır
- Toplam confidence'a göre ağırlıklandırılır
- Örnek: conf=0.8 → %40 sermaye, conf=0.2 → %10 sermaye

### 4. **Simülasyon Süresi**
- Simülasyon, seçilen horizon kadar sürer
- Örnek: Horizon=14d → 14 gün boyunca çalışır
- Her gün 10-15 kez kontrol edilir (automation cycle)

### 5. **Pozisyon Rotasyonu**
- Bir pozisyon satıldığında, boş slot doldurulur
- Yeni sinyaller aranır
- Top N kadar pozisyon tutulur

---

## 🚀 HIZLI BAŞLANGIÇ

### İlk Test İçin Önerilen Ayarlar:

```
Sermaye: 50,000 TL
Horizon: 7d
Top N: 5
Commission: 0.0005
Stop-loss: 3%
Relatif Düşüş: 20%
Mod: Hibrit
```

**Bu ayarlarla:**
- 7 gün sürecek bir test
- 5 pozisyon
- Model + risk yönetimi birlikte test edilir
- Her iki performans ayrı ayrı ölçülür

---

## 📊 SONUÇ TABLOSU

| Parametre | Düşük | Orta | Yüksek | Ne Zaman? |
|-----------|-------|------|--------|-----------|
| **Sermaye** | 10K | 50K | 100K+ | Test → Gerçekçi |
| **Horizon** | 1d-3d | 7d-14d | 30d | Hızlı → Uzun |
| **Top N** | 1-3 | 5-7 | 8-10 | Konsantre → Çeşitli |
| **Stop-loss** | 1-2% | 3-5% | 7-10% | Sıkı → Gevşek |
| **Rel. Drop** | 10-15% | 20-25% | 30-50% | Hassas → Toleranslı |
| **Mod** | Model Test | Hibrit | Risk Mgmt | Model → Gerçekçi → Risk |

---

## ❓ SIK SORULAN SORULAR

### Q: Hangi modu seçmeliyim?
**A:** 
- Model performansını ölçmek için: **Model Testi**
- Gerçekçi strateji testi için: **Hibrit** (önerilen)
- Risk yönetimi optimizasyonu için: **Risk Yönetimi**

### Q: Stop-loss ne kadar olmalı?
**A:** 
- Volatil semboller: 5-7%
- Normal semboller: 3-5%
- Düşük volatil: 2-3%

### Q: Horizon'u nasıl seçmeliyim?
**A:** 
- Modelin hangi horizon için eğitildiğini kullanın
- Genellikle 7d veya 14d en iyi sonuç verir
- Kısa test için 1d-3d, uzun test için 30d

### Q: Top N ne kadar olmalı?
**A:** 
- Küçük sermaye (<50K): 3-5
- Orta sermaye (50K-100K): 5-7
- Büyük sermaye (>100K): 7-10

### Q: Simülasyon ne kadar sürer?
**A:** 
- Seçilen horizon kadar (örn: 14d → 14 gün)
- Her gün 10-15 kez kontrol edilir
- Pozisyonlar güncellenir

---

## 🎓 İPUÇLARI

1. **İlk test için küçük başlayın:** 10K sermaye, 3d horizon, 3 pozisyon
2. **Farklı modları deneyin:** Model testi → Hibrit → Risk yönetimi
3. **Parametreleri optimize edin:** Stop-loss ve confidence drop'u test edin
4. **Sonuçları karşılaştırın:** Model vs risk yönetimi performansını karşılaştırın
5. **Gerçekçi commission kullanın:** 0.0005 (BIST standardı)

---

**Başarılar! 🚀**

