#!/usr/bin/env bash
set -euo pipefail

SYSTEMCTL="$(command -v systemctl || echo /bin/systemctl)"
ENV_LINE="$($SYSTEMCTL show bist-pattern --property=Environment --value || true)"
if [[ -n "${ENV_LINE:-}" ]]; then
  eval "export ${ENV_LINE}"
fi

# Ensure sane PATH even if service env overrides it
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

export BIST_LOG_PATH="${BIST_LOG_PATH:-/opt/bist-pattern/logs}"
mkdir -p "$BIST_LOG_PATH"
cd /opt/bist-pattern

PY="/opt/bist-pattern/venv/bin/python3"
if [ ! -x "$PY" ]; then PY="python3"; fi

DAY="${1:-$(date -I)}"
echo "[evaluate-metrics] $(date -Iseconds) day=$DAY start" >> "$BIST_LOG_PATH/evaluate_metrics.log"
LOCK_FILE="/opt/bist-pattern/logs/evaluate_metrics.lock"
exec flock -n "$LOCK_FILE" -c "nice -n 10 ionice -c2 -n7 \"$PY\" scripts/evaluate_metrics.py --date \"$DAY\" >> \"$BIST_LOG_PATH/evaluate_metrics.log\" 2>&1" || {
  echo "[evaluate-metrics] $(date -Iseconds) busy lock: $LOCK_FILE" >> "$BIST_LOG_PATH/evaluate_metrics.log"; exit 0; }


