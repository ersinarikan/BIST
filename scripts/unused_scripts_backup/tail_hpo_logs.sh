#!/bin/bash
# Cycle 2 (Continuous HPO Pipeline) loglarını canlı takip et

LOG_FILE=$(ls -t /opt/bist-pattern/logs/continuous_hpo_pipeline_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ Log dosyası bulunamadı"
    exit 1
fi

echo "📊 Cycle 2 Log Takibi"
echo "Log dosyası: $LOG_FILE"
echo ""
echo "Filtreler:"
echo "  - Starting HPO"
echo "  - HPO completed"
echo "  - Starting training"
echo "  - Training completed"
echo "  - Task completed"
echo "  - Cycle complete"
echo "  - ERROR"
echo "  - WARNING"
echo "  - ✅ Success"
echo "  - ❌ Failed"
echo "  - ⚠️  Warning"
echo ""
echo "Çıkmak için: Ctrl+C"
echo "=" * 80
echo ""

# Canlı takip - önemli mesajlar
tail -f "$LOG_FILE" | grep --line-buffered -E 'Starting HPO|HPO completed|Starting training|Training completed|Task completed|Cycle.*complete|ERROR|WARNING|✅|❌|⚠️|Progress|failed|success'

