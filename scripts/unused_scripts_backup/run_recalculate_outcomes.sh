#!/bin/bash
set -euo pipefail

# Load environment from systemd service
SYSTEMCTL="$(command -v systemctl || echo /bin/systemctl)"
ENV_LINE="$($SYSTEMCTL show bist-pattern --property=Environment --value || true)"
if [[ -n "${ENV_LINE:-}" ]]; then
  eval "export ${ENV_LINE}"
fi

export PYTHONPATH=/opt/bist-pattern
cd /opt/bist-pattern

echo "🔄 Starting outcomes recalculation with new threshold-based logic..."
echo ""

venv/bin/python3 scripts/recalculate_outcomes.py

echo ""
echo "✅ Recalculation complete!"
echo ""
echo "📊 Dashboard metrics will be updated after next evaluate_metrics run"
echo "    (scheduled at 00:05 local time)"
