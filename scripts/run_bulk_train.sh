#!/usr/bin/env bash
set -euo pipefail
# ✅ FIX: Set umask for group-writable files (002 = group writable)
umask 0002

ROOT_DIR="/opt/bist-pattern"
cd "$ROOT_DIR"

# Inherit environment from the running systemd service so cron uses identical config
SYSTEMCTL="$(command -v systemctl || echo /bin/systemctl)"
ENV_LINE="$($SYSTEMCTL show bist-pattern --property=Environment --value || true)"
if [[ -n "${ENV_LINE:-}" ]]; then
  eval "export ${ENV_LINE}"
fi

# Ensure core system paths are present even if service PATH overrides them
DEFAULT_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
if [[ -z "${PATH:-}" ]]; then
  export PATH="$DEFAULT_PATH"
else
  # Append defaults if missing
  export PATH="${PATH}:$DEFAULT_PATH"
fi

export PYTHONWARNINGS=ignore
export PYTHONPATH="$ROOT_DIR"
# Default: let coordinator decide; set to 1 only for forced backfills
export FORCE_FULL_RETRAIN=0
export ML_MAX_MODEL_AGE_DAYS=6

# ⚡ Load HPO parameters from latest optimization (if available)
# This allows the bulk training to use optimized hyperparameters
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/load_hpo_params.sh"

# ⚡ Enable all ML improvements for normal training (not HPO)
export ML_USE_ADAPTIVE_LEARNING=1  # Phase 2 incremental learning on test data
export ML_USE_SMART_ENSEMBLE=1     # Smart ensemble (consensus + performance)
export ML_USE_STACKED_SHORT=1      # Meta-stacking for short horizons (1/3/7d)
export ML_USE_REGIME_DETECTION=1  # Regime-aware feature weighting
export STRICT_HORIZON_FEATURES=1  # Horizon-specific features
export ENABLE_SEED_BAGGING=1      # Multi-seed training for robustness
export N_SEEDS=3                  # Number of seeds for bagging
export ML_EARLY_STOP_ROUNDS=50    # Early stopping rounds

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] Verifying environment and imports..."
  echo "[DRY_RUN] Using PYTHON: $(command -v python || true)"
  echo "[DRY_RUN] DATABASE_URL: ${DATABASE_URL:+set}${DATABASE_URL:-missing}"
  echo "[DRY_RUN] YOLO_MODEL_PATH: ${YOLO_MODEL_PATH:-}"
  echo "[DRY_RUN] ENABLE_ENHANCED_ML: ${ENABLE_ENHANCED_ML:-}"
  # shellcheck disable=SC1091
  source "$ROOT_DIR/venv/bin/activate"
  python - <<'PY'
import os, sys
print("ok:python", sys.version)
print("ok:PYTHONPATH", os.getenv("PYTHONPATH"))
print("ok:DATABASE_URL", "set" if os.getenv("DATABASE_URL") else "missing")
import app
print("ok:app_import")
PY
  echo "[DRY_RUN] Done."
  exit 0
fi

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/.cache"
LOG_FILE="$ROOT_DIR/logs/train_$(date +%F_%H%M%S).log"
LOCK_FILE="$ROOT_DIR/.cache/train.lock"

# shellcheck disable=SC1091
if [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
  source "$ROOT_DIR/venv/bin/activate"
fi

# Ensure log directory permissions for web user to read logs produced by root cron
chown -R www-data:www-data "$ROOT_DIR/logs" 2>/dev/null || true

(
  flock -n 9 || { echo "Another training is running. Exiting."; exit 0; }
  echo "[$(date -Is)] Starting training..."
  python -u "$ROOT_DIR/scripts/bulk_train_all.py"
  echo "[$(date -Is)] Training finished."
  echo "[$(date -Is)] Post-train enhanced check (all eligible symbols)..."
  python -u "$ROOT_DIR/scripts/post_train_enhanced_check.py" --all || true
) 9>"$LOCK_FILE" >>"$LOG_FILE" 2>&1

# Make sure log and lock are readable (and lock owned) for www-data after run
chmod 644 "$LOG_FILE" 2>/dev/null || true
chown www-data:www-data "$LOG_FILE" 2>/dev/null || true
chown www-data:www-data "$LOCK_FILE" 2>/dev/null || true

echo "Log: $LOG_FILE"


