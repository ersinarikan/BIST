#!/usr/bin/env bash
set -euo pipefail

LOGDIR="${BIST_LOG_PATH:-/opt/bist-pattern/logs}"
mkdir -p "$LOGDIR"

log() { echo "[start_after_reboot] $(date -Iseconds) $*" | tee -a "$LOGDIR/start_after_reboot.log"; }

# Ensure DATABASE_URL
if [ -z "${DATABASE_URL:-}" ] && [ -f /opt/bist-pattern/.secrets/db_password ]; then
  export DATABASE_URL="postgresql://bist_user:$(cat /opt/bist-pattern/.secrets/db_password)@127.0.0.1:5432/bist_pattern_db"
fi

# Prefer venv python
if [ -f /opt/bist-pattern/venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source /opt/bist-pattern/venv/bin/activate
  PY_BIN="/opt/bist-pattern/venv/bin/python3"
else
  PY_BIN="python3"
fi

# Common env
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export ML_HORIZONS="1,3,7"
export ENABLE_SEED_BAGGING=1
export N_SEEDS=3
export ML_EARLY_STOP_ROUNDS=50
export ML_USE_STACKED_SHORT=1
export ML_USE_SMART_ENSEMBLE=1
export ML_USE_ADAPTIVE_LEARNING=1  # ⚡ Phase 2 incremental learning on test data
export ML_USE_REGIME_DETECTION=1   # Regime-aware feature weighting
export STRICT_HORIZON_FEATURES=1
export ML_ENABLE_INTERNAL_MACRO=1
export ML_ENABLE_CALIBRATION=0

# ⚡ Load HPO parameters from latest optimization (if available)
# This will override any hard-coded values below
source /opt/bist-pattern/scripts/load_hpo_params.sh

# Fallback HPO overrides (used if HPO config not found)
export OPTUNA_XGB_N_ESTIMATORS="${OPTUNA_XGB_N_ESTIMATORS:-973}"
export OPTUNA_XGB_MAX_DEPTH="${OPTUNA_XGB_MAX_DEPTH:-9}"
export OPTUNA_XGB_LEARNING_RATE="${OPTUNA_XGB_LEARNING_RATE:-0.020589728197687916}"
export OPTUNA_XGB_SUBSAMPLE="${OPTUNA_XGB_SUBSAMPLE:-0.5909124836035503}"
export OPTUNA_XGB_COLSAMPLE_BYTREE="${OPTUNA_XGB_COLSAMPLE_BYTREE:-0.5917022549267169}"
export OPTUNA_XGB_REG_ALPHA="${OPTUNA_XGB_REG_ALPHA:-5.472429642032198e-06}"
export OPTUNA_XGB_REG_LAMBDA="${OPTUNA_XGB_REG_LAMBDA:-0.00052821153945323}"
export OPTUNA_XGB_MIN_CHILD_WEIGHT="${OPTUNA_XGB_MIN_CHILD_WEIGHT:-5}"
export OPTUNA_XGB_GAMMA="${OPTUNA_XGB_GAMMA:-2.1371407316372935e-06}"

# Fallback adaptive params (used if HPO config not found)
export ML_ADAPTIVE_K_1D="${ML_ADAPTIVE_K_1D:-2.0}"
export ML_ADAPTIVE_K_3D="${ML_ADAPTIVE_K_3D:-1.8}"
export ML_ADAPTIVE_K_7D="${ML_ADAPTIVE_K_7D:-1.6}"
export ML_PATTERN_WEIGHT_SCALE_1D="${ML_PATTERN_WEIGHT_SCALE_1D:-1.2}"
export ML_PATTERN_WEIGHT_SCALE_3D="${ML_PATTERN_WEIGHT_SCALE_3D:-1.15}"
export ML_PATTERN_WEIGHT_SCALE_7D="${ML_PATTERN_WEIGHT_SCALE_7D:-1.1}"

# Clean previous logs (optional)
: > "$LOGDIR/retrain_short_137.log"
: > "$LOGDIR/threshold_grid_bist30_progress.log"

log "Starting short-horizon retrain (BIST30, 1/3/7d)"
nohup stdbuf -oL -eL "$PY_BIN" -u /opt/bist-pattern/scripts/train_bist30_full.py \
  >> "$LOGDIR/retrain_short_137.log" 2>&1 & echo $! > "$LOGDIR/retrain_short_137.pid"

# Threshold-grid (STRICT off for evaluation scan)
SYMS="AKBNK,ARCLK,ASELS,BIMAS,EKGYO,ENJSA,EREGL,FROTO,GARAN,HEKTS,ISCTR,KCHOL,KOZAL,KOZAA,KRDMD,PETKM,PGSUS,SAHOL,SASA,SISE,TAVHL,TCELL,THYAO,TOASO,TUPRS,VAKBN,VESTL,YKBNK,ODAS,SMRTG"
log "Starting threshold-grid (strict=0)"
nohup env STRICT_HORIZON_FEATURES=0 stdbuf -oL -eL \
  "$PY_BIN" -u /opt/bist-pattern/scripts/threshold_grid_search.py \
  --symbols "$SYMS" --horizons 1,3,7 --lookback-days 700 \
  --thr-grids "1:0.002,0.003,0.004,0.005,0.006,0.007,0.008;3:0.003,0.004,0.005,0.006,0.007,0.008;7:0.004,0.005,0.006,0.007,0.008,0.010" \
  --out "$LOGDIR/threshold_grid_bist30_result.json" \
  >> "$LOGDIR/threshold_grid_bist30_progress.log" 2>&1 & echo $! > "$LOGDIR/threshold_grid.pid"

log "OK: retrain pid=$(cat "$LOGDIR/retrain_short_137.pid" 2>/dev/null || echo -n '?'), thr-grid pid=$(cat "$LOGDIR/threshold_grid.pid" 2>/dev/null || echo -n '?')"
exit 0



