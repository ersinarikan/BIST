# Güvenlik Notları

## 🔐 Internal API Token

### ⚠️ ÖNEMLİ UYARI

Kod ve dokümantasyonlarda bulunan örnek token:
```
IBx_gsmQUL9oxymAgr67PxES7ACfKlk1Ex5F9jCCOFw
```

**Bu token sadece ÖRNEK amaçlıdır!**

**Production'da MUTLAKA değiştirin:**

1. Yeni token oluştur:
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

2. `.env` dosyasına ekle:
```bash
INTERNAL_API_TOKEN=yeni_guvenli_token_buraya
```

3. Sunucuyu restart et:
```bash
sudo systemctl restart bist-pattern
```

### Hardcoded Token'ları Temizle

**user-dashboard.js içinde:**
- Line ~1130: Volume tier API çağrısında hardcoded token var
- Environment variable'dan alınmalı

**Düzeltme:**
```javascript
// ❌ KÖTÜ:
headers: {
  'X-Internal-Token': 'IBx_gsmQUL9...'
}

// ✅ İYİ:
headers: {
  'X-Internal-Token': window.INTERNAL_TOKEN || ''
}
```

### Token Güvenliği

- ✅ 32+ karakter
- ✅ URL-safe karakterler
- ✅ Rastgele üretilmiş
- ✅ .env dosyasında saklanmış
- ✅ Git'e commit edilmemiş (.gitignore'da)
- ❌ Kod içinde hardcoded OLMAMALI
- ❌ Dokümantasyonda gerçek token OLMAMALI

---

**Son Güncelleme:** 08 Ekim 2025
