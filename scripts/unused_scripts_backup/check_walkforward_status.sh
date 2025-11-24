#!/bin/bash
# Quick status check for walk-forward

echo "📊 WALK-FORWARD STATUS"
echo "====================="
echo ""

# Check if running
if ps aux | grep -v grep | grep "pilot_walkforward" > /dev/null; then
    echo "✅ Status: RUNNING"
    PID=$(ps aux | grep -v grep | grep "pilot_walkforward" | awk '{print $2}' | head -1)
    RUNTIME=$(ps -p $PID -o etime= | tr -d ' ')
    CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
    MEM=$(ps -p $PID -o %mem= | tr -d ' ')
    echo "   PID: $PID"
    echo "   Runtime: $RUNTIME"
    echo "   CPU: ${CPU}%"
    echo "   Memory: ${MEM}%"
else
    echo "❌ Status: NOT RUNNING"
fi

echo ""
echo "📁 Latest CSV:"
ls -lh /opt/bist-pattern/logs/walkforward_3_symbols_*.csv 2>/dev/null | tail -1

echo ""
echo "📝 Last 5 log lines:"
tail -5 /opt/bist-pattern/logs/walkfwd_3sym.out 2>/dev/null || tail -5 /opt/bist-pattern/logs/walkfwd_all_horizons.out 2>/dev/null

echo ""
echo "🔍 Progress (symbols completed):"
if [ -f /opt/bist-pattern/logs/walkforward_3_symbols_*.csv ]; then
    LATEST_CSV=$(ls -t /opt/bist-pattern/logs/walkforward_3_symbols_*.csv 2>/dev/null | head -1)
    if [ -f "$LATEST_CSV" ]; then
        echo "   Total rows: $(wc -l < "$LATEST_CSV")"
        echo "   With metrics: $(grep -v "^symbol" "$LATEST_CSV" | grep -v ",,,," | wc -l)"
    fi
fi
