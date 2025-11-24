#!/bin/bash
# HPO Canlı Log Takibi

echo "🔍 HPO CANLI LOG TAKİBİ"
echo "Çıkmak için: CTRL+C"
echo "================================================"
echo ""

# Journald'den canlı logları göster (tail -f gibi)
journalctl -u bist-pattern-hpo.service -f --no-pager \
  | grep --line-buffered -E "Starting HPO|Trial|completed|failed|skipped|✅|❌|⏭️|🔬|🎯|finished"
