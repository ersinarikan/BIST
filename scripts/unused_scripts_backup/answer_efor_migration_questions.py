#!/usr/bin/env python3
"""
Answer questions about EFOR migration and HPO service behavior
"""

print("=" * 80)
print("EFOR MİGRASYONU VE HPO SERVİSİ DAVRANIŞI - SORU CEVAP")
print("=" * 80)
print()

print("SORU 1: Eksik sembolleri ekleyip HPO servisini restart edersek,")
print("        servis eksik günleri çekip VT'ye yazıyor mu (backfill)?")
print()
print("CEVAP 1: KISMI OLARAK EVET")
print("-" * 80)
print("""
✅ working_automation servisi (veri çekme):
   - Her sembol için period='auto' kullanıyor
   - Son tarihe bakıp gap'e göre period belirliyor:
     * Gap <= 1 gün  → 5d çeker
     * Gap <= 30 gün → 1mo çeker
     * Gap <= 90 gün → 3mo çeker
     * Gap <= 180 gün → 6mo çeker
     * Gap <= 365 gün → 1y çeker
     * Gap > 365 gün → 2y çeker
   
   ⚠️  ÖNEMLİ: Bu sadece SON TARİHTEN İTİBAREN çeker, eksik günleri doldurmaz!
   - Eğer sembolde 100 gün veri varsa ve son veri 10 gün önceyse,
     sadece son 10 günü çeker (1mo period ile)
   - Ortadaki eksik günleri doldurmaz (örnek: 50. gün eksikse doldurmaz)

✅ unified_collector.save_to_db():
   - Çekilen verileri VT'ye yazarken:
     * Mevcut kayıtları günceller (update)
     * Yeni kayıtları ekler (insert)
     * Tarih bazlı duplicate kontrolü yapar
   
   ✅ SONUÇ: Yeni sembol eklendiğinde:
   - working_automation servisi çalışıyorsa, otomatik olarak veri çekmeye başlar
   - period='auto' ile son tarihten itibaren çeker
   - Eksik günleri tam olarak doldurmaz, sadece son tarihten itibaren çeker
""")
print()

print("SORU 2: HPO servisi iş sırasını VT'deki sembol sırasına göre mi yapıyor?")
print("        Servisi reboot etsem, isim sırasına göre VT'den çekip")
print("        eksik günleri tamamlayıp kaldığı yerden devam eder mi?")
print()
print("CEVAP 2: EVET, AMA DETAYLAR VAR")
print("-" * 80)
print("""
✅ Sembol Sıralaması:
   - get_active_symbols() fonksiyonu:
     * Stock.query.filter_by(is_active=True).order_by(Stock.symbol).all()
     * Yani ALFABETİK SIRAYA göre çeker (A'dan Z'ye)
   
✅ State Dosyası (continuous_hpo_state.json):
   - Her sembol-horizon çifti için durum saklanır
   - Status: pending, in_progress, completed, failed, skipped
   - Cycle numarası saklanır
   - Kaldığı yerden devam eder
   
✅ Restart Sonrası Davranış:
   1. State dosyası yüklenir
   2. Mevcut cycle numarası korunur
   3. Pending/failed task'lar current cycle'a atanır
   4. get_active_symbols() ile VT'den alfabetik sırayla semboller çekilir
   5. Her sembol için state'deki duruma göre:
      - completed → atlanır
      - pending → işlenir
      - failed → pending'e reset edilip işlenir
      - in_progress → pending'e reset edilip işlenir (stale process)
   
✅ Yeni Sembol Eklendiğinde:
   - State dosyasında yoksa → pending olarak eklenir
   - Alfabetik sıraya göre sıraya girer
   - Örnek: ADLVY eklerseniz, A ile başlayan semboller arasında sıraya girer
   
⚠️  ÖNEMLİ NOTLAR:
   - HPO servisi veri çekmez, sadece VT'den okur
   - Veri çekme working_automation servisinin işi
   - HPO servisi restart edilse bile, veri çekme working_automation'da devam eder
   - Eksik günler için backfill yapılmaz, sadece son tarihten itibaren çekilir
""")
print()

print("=" * 80)
print("ÖNERİLER:")
print("=" * 80)
print("""
1. ✅ Eksik sembolleri ekle (25 sembol)
2. ✅ EFOR'u ekle (EFORC'yi sonra sil)
3. ✅ working_automation servisinin çalıştığından emin ol
   - Bu servis otomatik olarak yeni semboller için veri çekmeye başlar
4. ✅ HPO servisini restart et
   - State dosyasından kaldığı yerden devam eder
   - Yeni semboller alfabetik sıraya göre sıraya girer
   - Eksik günler için manuel backfill gerekebilir (opsiyonel)

⚠️  EKSİK GÜNLER İÇİN:
   - working_automation sadece son tarihten itibaren çeker
   - Ortadaki eksik günleri doldurmaz
   - Eğer tam backfill istiyorsanız, manuel script gerekebilir
""")

