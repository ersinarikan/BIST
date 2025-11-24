#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${BIST_LOG_PATH:-/opt/bist-pattern/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/publish_params.log"
OUT_JSON="$LOG_DIR/param_store.json"

# Ensure param_store.json exists (calibrate_confidence should have produced it)
if [[ ! -f "$OUT_JSON" ]]; then
  echo "{}" > "$OUT_JSON"
fi

# Set friendly perms for dashboard
chmod 644 "$OUT_JSON" 2>/dev/null || true
chown www-data:www-data "$OUT_JSON" 2>/dev/null || true

# Touch calibration_state.json perms too (optional)
STATE_JSON="$LOG_DIR/calibration_state.json"
if [[ -f "$STATE_JSON" ]]; then
  chmod 644 "$STATE_JSON" 2>/dev/null || true
  chown www-data:www-data "$STATE_JSON" 2>/dev/null || true
fi

# Log
{
  echo "[publish_params] $(date -Iseconds) ok: $OUT_JSON"
} | tee -a "$LOG_FILE" >/dev/null

exit 0
