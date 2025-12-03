#!/usr/bin/env bash
# HPO Servis Durumu Kontrol Script'i

echo "=========================================="
echo "HPO Servis Durumu Kontrolü"
echo "=========================================="
echo ""

# Systemd servis durumu
echo "📊 Systemd Servis Durumu:"
systemctl status bist-pattern-hpo.service --no-pager | head -15
echo ""

# Aktif HPO process sayısı
HPO_PROCESS_COUNT=$(ps aux | grep "optuna_hpo_with_feature_flags.py" | grep -v grep | wc -l)
echo "🔢 Aktif HPO Process Sayısı: $HPO_PROCESS_COUNT"
echo ""

# MAX_WORKERS değeri
MAX_WORKERS=$(grep "HPO_MAX_WORKERS" /etc/default/bist-pattern 2>/dev/null | cut -d'=' -f2 || echo "NOT_SET")
echo "⚙️  HPO_MAX_WORKERS: $MAX_WORKERS"
echo ""

# Eş zamanlı sembol sayısı (tahmini)
UNIQUE_SYMBOLS=$(ps aux | grep "optuna_hpo_with_feature_flags.py" | grep -v grep | awk '{print $NF}' | sed 's/.*--symbols //' | sed 's/ --.*//' | sort | uniq | wc -l)
echo "📈 Eş Zamanlı İşlenen Sembol Sayısı (tahmini): $UNIQUE_SYMBOLS"
echo ""

# Memory kullanımı
echo "💾 Memory Kullanımı:"
systemctl show bist-pattern-hpo.service -p MemoryCurrent --value | numfmt --to=iec-i --suffix=B
echo ""

# CPU kullanımı
echo "⚡ CPU Kullanımı:"
systemctl show bist-pattern-hpo.service -p CPUUsageNSec --value | numfmt --to=iec-i --suffix=B 2>/dev/null || echo "N/A"
echo ""

echo "=========================================="
echo "Güvenli Durdurma için:"
echo "  sudo systemctl stop bist-pattern-hpo.service"
echo "=========================================="

