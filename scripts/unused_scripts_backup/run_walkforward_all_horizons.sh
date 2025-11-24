#!/bin/bash
# Walk-forward pilot for all horizons (1/3/7/14/30d)
# Run after current walk-forward completes

cd /opt/bist-pattern

echo "🚀 Starting walk-forward pilot (ALL HORIZONS: 1/3/7/14/30d)"
echo "⏰ Started at: $(date)"
echo "📊 Symbols: GARAN, AKBNK, EREGL"
echo "🎯 Modes: single-anchor + weekly retrain"
echo ""

# Clear old log
cat /dev/null > logs/walkfwd_all_horizons.out

# Start walk-forward
nohup .venv/bin/python scripts/pilot_walkforward_3_symbols.py > logs/walkfwd_all_horizons.out 2>&1 &
PID=$!

echo "✅ Walk-forward started - PID: $PID"
echo "📝 Log: logs/walkfwd_all_horizons.out"
echo "⏱️  Estimated time: 3-4 hours"
echo ""
echo "Monitor with:"
echo "  tail -f logs/walkfwd_all_horizons.out"
echo ""
echo "Check progress:"
echo "  tail -50 logs/walkfwd_all_horizons.out | grep -E 'GARAN|AKBNK|EREGL|✅'"
