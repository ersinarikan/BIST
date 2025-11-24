#!/bin/bash
# Training ilerlemesini izleme scripti

LOG_FILE=$(find /opt/bist-pattern/logs -name "train_completed_hpo_*.log" -type f -mmin -30 | sort -r | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ Log dosyası bulunamadı!"
    exit 1
fi

echo "📊 Training İlerlemesi İzleniyor..."
echo "Log dosyası: $LOG_FILE"
echo ""

# İlerleme sayacı
while true; do
    clear
    echo "=========================================="
    echo "📊 TRAINING İLERLEMESİ"
    echo "=========================================="
    echo ""
    
    COMPLETED=$(grep -c "Training completed" "$LOG_FILE" 2>/dev/null || echo "0")
    TOTAL=348
    
    echo "✅ Tamamlanan: $COMPLETED/$TOTAL"
    PERCENT=$(echo "scale=1; $COMPLETED * 100 / $TOTAL" | bc 2>/dev/null || echo "0")
    echo "📈 İlerleme: ${PERCENT}%"
    echo ""
    
    echo "Son 5 başarılı training:"
    grep "Training completed" "$LOG_FILE" 2>/dev/null | tail -5 | sed 's/.*INFO[ ]*//'
    echo ""
    
    echo "Şu anki işlem:"
    tail -3 "$LOG_FILE" 2>/dev/null | grep -E "Training.*for horizons|Training.*d with best" | tail -1 | sed 's/.*INFO[ ]*//' || echo "Bekleniyor..."
    echo ""
    
    echo "Son hata (varsa):"
    tail -100 "$LOG_FILE" 2>/dev/null | grep -i "error\|failed" | tail -1 | sed 's/.*\(ERROR\|WARNING\|ERROR\|failed\).*: //' || echo "Hata yok ✅"
    echo ""
    
    echo "=========================================="
    echo "Güncelleme: $(date '+%H:%M:%S')"
    echo "Çıkmak için Ctrl+C"
    echo ""
    
    sleep 5
done

