#!/usr/bin/env bash
set -euo pipefail

SYSTEMCTL="$(command -v systemctl || echo /bin/systemctl)"
ENV_LINE="$($SYSTEMCTL show bist-pattern --property=Environment --value || true)"
if [[ -n "${ENV_LINE:-}" ]]; then
  eval "export ${ENV_LINE}"
fi

# Ensure PATH in unit files with restricted envs
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

export BIST_LOG_PATH="${BIST_LOG_PATH:-/opt/bist-pattern/logs}"
mkdir -p "$BIST_LOG_PATH"
cd /opt/bist-pattern

PY="/opt/bist-pattern/venv/bin/python3"
if [ ! -x "$PY" ]; then PY="python3"; fi

# Load go_live from calibration_state.json if present
CAL_STATE="/opt/bist-pattern/logs/calibration_state.json"
GO_LIVE_ARG=""
if [ -f "$CAL_STATE" ]; then
  GO_LIVE_VAL="$($PY - <<'PY'
import json
p = "/opt/bist-pattern/logs/calibration_state.json"
try:
    with open(p, "r") as f:
        d = json.load(f) or {}
    v = d.get("go_live", "")
    if v:
        print(v)
except Exception:
    pass
PY
)"
  if [ -n "$GO_LIVE_VAL" ]; then
    GO_LIVE_ARG="--go-live $GO_LIVE_VAL"
  fi
fi

# Defaults can be overridden by env or cli
WINDOW_DAYS="${CALIB_WINDOW_DAYS:-30}"
MIN_SAMPLES="${CALIB_MIN_SAMPLES:-150}"

echo "[calibrate-confidence] $(date -Iseconds) start" >> "$BIST_LOG_PATH/calibrate_confidence.log"
exec nice -n 10 ionice -c2 -n7 flock -n /opt/bist-pattern/logs/calibrate_confidence.lock \
  "$PY" scripts/calibrate_confidence.py --window-days "$WINDOW_DAYS" --min-samples "$MIN_SAMPLES" $GO_LIVE_ARG "$@" >> "$BIST_LOG_PATH/calibrate_confidence.log" 2>&1


