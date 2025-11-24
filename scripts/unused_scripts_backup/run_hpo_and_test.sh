#!/bin/bash
# HPO çalıştır ve tamamlandığında reproducibility test yap
set -e

SYMBOL=${1:-ASELS}
HORIZON=${2:-7}
TRIALS=${3:-100}
TIMEOUT=${4:-28800}  # 8 hours

echo "=" | head -c 80 && echo ""
echo "🚀 HPO + Reproducibility Test Pipeline"
echo "=" | head -c 80 && echo ""
echo "Symbol: $SYMBOL"
echo "Horizon: ${HORIZON}d"
echo "Trials: $TRIALS"
echo "Timeout: ${TIMEOUT}s ($(($TIMEOUT / 3600))h)"
echo ""

# Activate venv
cd /opt/bist-pattern
source venv/bin/activate 2>/dev/null || true

# Step 1: Run HPO
echo "📊 Step 1: HPO başlatılıyor..."
HPO_LOG="/tmp/hpo_${SYMBOL}_$(date +%Y%m%d_%H%M%S).log"
python3 scripts/optuna_hpo_pilot_features_on.py \
    --symbols "$SYMBOL" \
    --horizon "$HORIZON" \
    --trials "$TRIALS" \
    --timeout "$TIMEOUT" \
    2>&1 | tee "$HPO_LOG"

HPO_EXIT_CODE=${PIPESTATUS[0]}

if [ $HPO_EXIT_CODE -ne 0 ]; then
    echo "❌ HPO başarısız (exit code: $HPO_EXIT_CODE)"
    exit 1
fi

echo ""
echo "✅ HPO tamamlandı"
echo ""

# Step 2: Find latest HPO result
echo "📋 Step 2: HPO sonuçları aranıyor..."
LATEST_HPO=$(find results -name "optuna_pilot_features_on_*${SYMBOL}*.json" -type f 2>/dev/null | sort -r | head -1)

if [ -z "$LATEST_HPO" ]; then
    echo "❌ HPO sonuç dosyası bulunamadı"
    exit 1
fi

echo "✅ HPO sonuç dosyası: $LATEST_HPO"

# Extract HPO results
HPO_DIRHIT=$(python3 -c "import json; d=json.load(open('$LATEST_HPO')); print(f\"{d.get('best_dirhit', 0):.2f}\")" 2>/dev/null || echo "0")
HPO_TRIAL=$(python3 -c "import json; d=json.load(open('$LATEST_HPO')); print(d.get('best_trial', {}).get('number', 'N/A'))" 2>/dev/null || echo "N/A")

echo "   Best DirHit: ${HPO_DIRHIT}%"
echo "   Best Trial: $HPO_TRIAL"
echo ""

# Step 3: Run reproducibility test
echo "🔬 Step 3: Reproducibility test başlatılıyor..."
TEST_LOG="/tmp/reproducibility_test_${SYMBOL}_$(date +%Y%m%d_%H%M%S).log"
python3 scripts/test_hpo_best_params_reproducibility.py \
    --hpo-file "$LATEST_HPO" \
    2>&1 | tee "$TEST_LOG"

TEST_EXIT_CODE=${PIPESTATUS[0]}

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "❌ Reproducibility test başarısız (exit code: $TEST_EXIT_CODE)"
    exit 1
fi

echo ""
echo "✅ Reproducibility test tamamlandı"
echo ""

# Step 4: Extract and compare results
echo "📊 Step 4: Sonuçlar karşılaştırılıyor..."
TEST_DIRHIT=$(grep "Test DirHit:" "$TEST_LOG" | tail -1 | awk '{print $3}' | sed 's/%//' || echo "0")
DIFF=$(python3 -c "print(f\"{abs($TEST_DIRHIT - $HPO_DIRHIT):.2f}\")" 2>/dev/null || echo "0")

echo ""
echo "=" | head -c 80 && echo ""
echo "📊 SONUÇLAR"
echo "=" | head -c 80 && echo ""
echo "HPO DirHit:        ${HPO_DIRHIT}%"
echo "Test DirHit:       ${TEST_DIRHIT}%"
echo "Fark:              ${DIFF}%"
echo ""

# Check if difference is acceptable (< 2%)
if (( $(echo "$DIFF < 2.0" | bc -l) )); then
    echo "✅ BAŞARILI: DirHit farkı kabul edilebilir (< 2%)"
    echo ""
    echo "🎯 Sonraki adım: 64 kombinasyon testine geçilebilir"
    exit 0
else
    echo "⚠️ UYARI: DirHit farkı büyük (>= 2%)"
    echo ""
    echo "🔍 Detaylı analiz gerekebilir"
    exit 1
fi

