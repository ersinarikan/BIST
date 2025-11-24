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

# ✅ FIX: Default to previous business day when called from cron (no argument)
# This ensures we process the previous market day's data (skips weekends)
if [ -z "${1:-}" ]; then
    # Get yesterday
    DAY="$(date -d "yesterday" -I)"
    # Check if yesterday is weekend (Saturday=6, Sunday=7)
    # date +%u returns: 1=Monday, 7=Sunday
    DOW=$(date -d "$DAY" +%u)
    # If weekend (Saturday=6 or Sunday=7), go back to previous Friday
    if [ "$DOW" -eq 6 ]; then
        # Yesterday was Saturday, go back to Friday
        DAY="$(date -d "$DAY -1 day" -I)"
    elif [ "$DOW" -eq 7 ]; then
        # Yesterday was Sunday, go back 2 days to Friday
        DAY="$(date -d "$DAY -2 days" -I)"
    fi
    # Final check: ensure DAY is a weekday (1-5)
    FINAL_DOW=$(date -d "$DAY" +%u)
    if [ "$FINAL_DOW" -ge 6 ]; then
        # Still weekend (shouldn't happen), go back one more day
        DAY="$(date -d "$DAY -1 day" -I)"
    fi
else
    DAY="$1"
fi
echo "[evaluate-metrics] $(date -Iseconds) day=$DAY start" >> "$BIST_LOG_PATH/evaluate_metrics.log"
LOCK_FILE="/opt/bist-pattern/logs/evaluate_metrics.lock"
exec flock -n "$LOCK_FILE" -c "nice -n 10 ionice -c2 -n7 \"$PY\" scripts/evaluate_metrics.py --date \"$DAY\" >> \"$BIST_LOG_PATH/evaluate_metrics.log\" 2>&1" || {
  echo "[evaluate-metrics] $(date -Iseconds) busy lock: $LOCK_FILE" >> "$BIST_LOG_PATH/evaluate_metrics.log"; exit 0; }


