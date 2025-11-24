#!/usr/bin/env bash
set -euo pipefail

SYSTEMCTL="$(command -v systemctl || echo /bin/systemctl)"
ENV_LINE="$($SYSTEMCTL show bist-pattern --property=Environment --value || true)"
if [[ -n "${ENV_LINE:-}" ]]; then
  eval "export ${ENV_LINE}"
fi

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
export BIST_LOG_PATH="${BIST_LOG_PATH:-/opt/bist-pattern/logs}"
mkdir -p "$BIST_LOG_PATH"
cd /opt/bist-pattern

log() { echo "[nightly-master] $(date -Iseconds) $*" | tee -a "$BIST_LOG_PATH/nightly_master.log"; }

PY="/opt/bist-pattern/venv/bin/python3"
if [ ! -x "$PY" ]; then PY="python3"; fi

log "start"

# ✅ FIX: Removed duplicates - these are handled by separate cron jobs:
#   1) Populate Outcomes: 21:00 daily (run_populate_outcomes.sh)
#   2) Evaluate Metrics: 00:05 daily (run_evaluate_metrics.sh)
#   3) Calibrate Confidence: 00:15 daily (run_calibrate_confidence.sh)
# Nightly Master only handles additional optimization tasks below

# 4) optimize evidence weights (lightweight)
if [ "${RUN_OPTIMIZE_WEIGHTS:-1}" != "0" ]; then
  log "optimize_evidence_weights"
  if ! "$PY" scripts/optimize_evidence_weights.py; then
    log "optimize_evidence_weights FAILED"
  fi
else
  log "optimize_evidence_weights SKIPPED"
fi

# 5) publish versioned params
if [ "${RUN_PUBLISH_PARAMS:-1}" != "0" ]; then
  log "publish_params"
  if ! /opt/bist-pattern/scripts/publish_params.sh; then
    log "publish_params FAILED"
  fi
else
  log "publish_params SKIPPED"
fi

# 6) drift check and optional rollback
if [ "${RUN_DRIFT_CHECK:-1}" != "0" ]; then
  log "check_drift_and_alert"
  if ! "$PY" scripts/check_drift_and_alert.py; then
    log "check_drift_and_alert FAILED"
  fi
else
  log "check_drift_and_alert SKIPPED"
fi

log "done"


