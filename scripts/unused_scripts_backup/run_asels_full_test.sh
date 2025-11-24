#!/bin/bash
# ASELSAN Full Test - HPO + Training + Feature Testing
# Bu script arka planda çalışır ve tüm testleri sırayla yapar

set -euo pipefail

cd /opt/bist-pattern

SYMBOL="ASELS"
HORIZON=7
TRIALS=100
LOG_DIR="test_results/asels_full_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "🚀 ASELSAN Full Test Başlatılıyor..."
echo "   Log klasörü: $LOG_DIR"
echo "   Test 1: HPO + Training (test_single_symbol.py)"
echo "   Test 2: Feature Testing - All Combinations (64 kombinasyon)"
echo "   Toplam: 65 DirHit değeri (HPO + 64 kombinasyon)"
echo ""

# Test 1: HPO + Training
echo "📊 Test 1: HPO + Training başlatılıyor..."
echo "$(date): Test 1 başladı" >> "$LOG_DIR/test.log"

venv/bin/python3 scripts/test_single_symbol.py --symbol "$SYMBOL" --horizon "$HORIZON" --trials "$TRIALS" > "$LOG_DIR/test1_hpo_training.log" 2>&1

TEST1_EXIT=$?
echo "$(date): Test 1 tamamlandı (exit: $TEST1_EXIT)" >> "$LOG_DIR/test.log"

if [ $TEST1_EXIT -ne 0 ]; then
    echo "❌ Test 1 başarısız! (exit: $TEST1_EXIT)"
    echo "$(date): Test 1 başarısız (exit: $TEST1_EXIT)" >> "$LOG_DIR/test.log"
    exit 1
fi

echo "✅ Test 1 tamamlandı!"
echo "   HPO sonuçları kontrol ediliyor..."

# HPO JSON dosyasını bekle (maksimum 5 dakika)
MAX_WAIT=300
WAIT_COUNT=0
HPO_JSON=""

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    HPO_JSON=$(find test_results -name "ASELS_${HORIZON}d_*" -type d | head -1)
    if [ -n "$HPO_JSON" ]; then
        HPO_JSON="${HPO_JSON}/results/hpo_${SYMBOL}_${HORIZON}d.json"
        if [ -f "$HPO_JSON" ]; then
            echo "✅ HPO JSON bulundu: $HPO_JSON"
            break
        fi
    fi
    sleep 10
    WAIT_COUNT=$((WAIT_COUNT + 10))
    echo "   HPO JSON bekleniyor... ($WAIT_COUNT/$MAX_WAIT saniye)"
done

if [ ! -f "$HPO_JSON" ]; then
    echo "❌ HPO JSON dosyası bulunamadı!"
    echo "$(date): HPO JSON bulunamadı" >> "$LOG_DIR/test.log"
    exit 1
fi

# 1 dakika bekle
echo "⏳ 1 dakika bekleniyor..."
sleep 60

# Test 2: Feature Testing - All Combinations (64 kombinasyon)
echo ""
echo "📊 Test 2: Feature Testing (All Combinations - 64 kombinasyon) başlatılıyor..."
echo "   6 feature'ın tüm kombinasyonları test edilecek (2^6 = 64)"
echo "$(date): Test 2 başladı" >> "$LOG_DIR/test.log"

venv/bin/python3 scripts/test_feature_combinations.py --symbol "$SYMBOL" --horizon "$HORIZON" --mode all > "$LOG_DIR/test2_all_combinations.log" 2>&1

TEST2_EXIT=$?
echo "$(date): Test 2 tamamlandı (exit: $TEST2_EXIT)" >> "$LOG_DIR/test.log"

if [ $TEST2_EXIT -ne 0 ]; then
    echo "❌ Test 2 başarısız! (exit: $TEST2_EXIT)"
    echo "$(date): Test 2 başarısız (exit: $TEST2_EXIT)" >> "$LOG_DIR/test.log"
    exit 1
else
    echo "✅ Test 2 tamamlandı!"
    echo "   64 kombinasyon test edildi"
fi

echo ""
echo "✅ Tüm testler tamamlandı!"
echo "   Log klasörü: $LOG_DIR"
echo "   Test 1 Sonuçları: test_results/ASELS_${HORIZON}d_*/results/"
echo "   Test 2 Sonuçları: test_results/ASELS_${HORIZON}d_feature_test_all_*/results/"
echo "   Toplam: 65 DirHit değeri (HPO + 64 kombinasyon)"
echo "$(date): Tüm testler tamamlandı" >> "$LOG_DIR/test.log"

