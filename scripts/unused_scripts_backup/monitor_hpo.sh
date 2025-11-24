#!/bin/bash
# HPO monitoring script
SYMBOL=${1:-ASELS}
HORIZON=${2:-7}

echo "📊 HPO İzleme - $SYMBOL (${HORIZON}d)"
echo "💡 Çıkmak için Ctrl+C"
echo ""

while true; do
    clear
    echo "📊 HPO İzleme - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Symbol: $SYMBOL, Horizon: ${HORIZON}d"
    echo ""
    
    # Process check
    if ps aux | grep "optuna_hpo_pilot_features_on.*$SYMBOL" | grep -v grep > /dev/null; then
        echo "✅ HPO çalışıyor"
        ps aux | grep "optuna_hpo_pilot_features_on.*$SYMBOL" | grep -v grep | head -1 | awk '{print "   PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "TIME:", $10}'
    else
        echo "⏳ HPO tamamlandı veya durdu"
    fi
    
    echo ""
    echo "📈 Son HPO sonuçları:"
    
    # Find latest HPO result
    LATEST_HPO=$(find /opt/bist-pattern/results -name "optuna_pilot_features_on_*${SYMBOL}*.json" -type f 2>/dev/null | sort -r | head -1)
    
    if [ -n "$LATEST_HPO" ]; then
        echo "   Dosya: $(basename $LATEST_HPO)"
        python3 -c "
import json
try:
    d = json.load(open('$LATEST_HPO'))
    print(f\"   Best DirHit: {d.get('best_dirhit', 0):.2f}%\")
    print(f\"   Best Trial: {d.get('best_trial', {}).get('number', 'N/A')}\")
    print(f\"   Trials: {d.get('n_trials', 0)}\")
except:
    print('   Henüz sonuç yok')
" 2>/dev/null
    else
        echo "   Henüz sonuç dosyası yok"
    fi
    
    echo ""
    echo "📝 Son log satırları (HPO):"
    tail -5 /tmp/hpo_asels_*.log 2>/dev/null | tail -3 || echo "   Henüz log yok"
    
    echo ""
    echo "💡 Çıkmak için Ctrl+C"
    sleep 10
done

