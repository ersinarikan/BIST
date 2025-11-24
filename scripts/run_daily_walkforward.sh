#!/usr/bin/env bash
set -euo pipefail

# Inherit environment from systemd service to align with runtime config
SYSTEMCTL="$(command -v systemctl || echo /bin/systemctl)"
ENV_LINE="$($SYSTEMCTL show bist-pattern --property=Environment --value || true)"
if [[ -n "${ENV_LINE:-}" ]]; then
  eval "export ${ENV_LINE}"
fi

# Env (honor systemd-provided values; only set if missing)
export FLASK_SECRET_KEY="${FLASK_SECRET_KEY:-daily-cron}"
if [[ -z "${DATABASE_URL:-}" ]]; then
  DB_PASS="$(cat /opt/bist-pattern/.secrets/db_password 2>/dev/null || true)"
  # ✅ FIX: Use PgBouncer port 6432 instead of direct Postgres 5432
  export DATABASE_URL="postgresql://bist_user:${DB_PASS}@127.0.0.1:6432/bist_pattern_db"
fi
export BIST_LOG_PATH="${BIST_LOG_PATH:-/opt/bist-pattern/logs}"
export ML_MIN_DATA_DAYS="${ML_MIN_DATA_DAYS:-200}"

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


