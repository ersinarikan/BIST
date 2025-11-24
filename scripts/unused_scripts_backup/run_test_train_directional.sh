#!/bin/bash
set -euo pipefail

# Load environment from systemd service
SYSTEMCTL="$(command -v systemctl || echo /bin/systemctl)"
ENV_LINE="$($SYSTEMCTL show bist-pattern --property=Environment --value || true)"
if [[ -n "${ENV_LINE:-}" ]]; then
  eval "export ${ENV_LINE}"
fi

export PYTHONPATH=/opt/bist-pattern
export ML_USE_DIRECTIONAL_LOSS=1
export ML_LOSS_MSE_WEIGHT=0.3
export ML_LOSS_THRESHOLD=0.005

cd /opt/bist-pattern

echo "🧪 Starting directional loss test training..."
echo ""

venv/bin/python3 scripts/test_train_directional.py

