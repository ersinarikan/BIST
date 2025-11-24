#!/bin/bash
# RSS Health Monitoring Script

echo "==================================="
echo "RSS HEALTH MONITOR"
echo "==================================="
echo ""

# Check automation status
echo "1) Automation Status:"
curl -s http://localhost:5000/api/internal/automation/status | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    status = data.get('scheduler_status', {})
    print(f\"   Running: {status.get('is_running', False)}\")
    print(f\"   Thread: {status.get('thread_alive', False)}\")
except:
    print('   ERROR: Cannot check status')
"
echo ""

# Check recent RSS logs
echo "2) Recent RSS Logs (last 5 minutes):"
journalctl -u bist-pattern.service --since "5 minutes ago" | \
  grep -i "rss check\|feedparser\|rss feed error" | \
  tail -10 || echo "   No RSS logs found"
echo ""

# Check RSS fetch success
echo "3) Recent RSS Fetch:"
journalctl -u bist-pattern.service --since "10 minutes ago" | \
  grep "RSS fetch completed" | \
  tail -3 || echo "   No recent RSS fetch"
echo ""

echo "==================================="
echo "Monitor with: watch -n 5 bash $0"
echo "==================================="
