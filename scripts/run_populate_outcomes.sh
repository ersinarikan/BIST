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
export PYTHONPATH="/opt/bist-pattern:${PYTHONPATH:-}"
mkdir -p "$BIST_LOG_PATH"
cd /opt/bist-pattern

PY="/opt/bist-pattern/venv/bin/python3"
if [ ! -x "$PY" ]; then PY="python3"; fi

echo "[populate-outcomes] $(date -Iseconds) start" >> "$BIST_LOG_PATH/populate_outcomes.log"
LOCK_FILE="/opt/bist-pattern/logs/populate_outcomes.lock"
exec flock -n "$LOCK_FILE" -c "nice -n 10 ionice -c2 -n7 \"$PY\" scripts/populate_outcomes.py --limit 50000 >> \"$BIST_LOG_PATH/populate_outcomes.log\" 2>&1" || {
  echo "[populate-outcomes] $(date -Iseconds) busy lock: $LOCK_FILE" >> "$BIST_LOG_PATH/populate_outcomes.log"; exit 0; }


