#!/usr/bin/env bash
set -euo pipefail

# Env
export FLASK_SECRET_KEY="daily-cron"
DB_PASS="$(cat /opt/bist-pattern/.secrets/db_password 2>/dev/null || true)"
export DATABASE_URL="postgresql://bist_user:${DB_PASS}@127.0.0.1:5432/bist_pattern_db"
export BIST_LOG_PATH="/opt/bist-pattern/logs"
export ML_MIN_DATA_DAYS="200"

mkdir -p "$BIST_LOG_PATH"
cd /opt/bist-pattern

# Prefer venv python
PY="/opt/bist-pattern/venv/bin/python3"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

echo "[daily-walkforward] $(date -Iseconds) start" >> "$BIST_LOG_PATH/walkforward_cron.log"
exec nice -n 10 ionice -c2 -n7 flock -n /opt/bist-pattern/logs/walkforward.lock \
  "$PY" scripts/daily_walkforward.py >> "$BIST_LOG_PATH/walkforward_cron.log" 2>&1


