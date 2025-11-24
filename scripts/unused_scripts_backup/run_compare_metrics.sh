#!/bin/bash
set -euo pipefail

SYSTEMCTL="$(command -v systemctl || echo /bin/systemctl)"
ENV_LINE="$($SYSTEMCTL show bist-pattern --property=Environment --value || true)"
if [[ -n "${ENV_LINE:-}" ]]; then
  eval "export ${ENV_LINE}"
fi

export PYTHONPATH=/opt/bist-pattern
cd /opt/bist-pattern

venv/bin/python3 scripts/compare_model_metrics.py

