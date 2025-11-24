#!/usr/bin/env bash
set -euo pipefail
umask 0022

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
  export PATH="${PATH}:$DEFAULT_PATH"
fi

export PYTHONWARNINGS=ignore
export PYTHONPATH="$ROOT_DIR"

# ⚡ CRITICAL: Unset ML_HORIZONS to avoid conflicts with training script
# The training script will set ML_HORIZONS per horizon dynamically
unset ML_HORIZONS

# ⚡ CRITICAL: Disable prediction during training to prevent horizon features not found errors
# analyze_stock() calls prediction, but training hasn't been done yet, so horizon features don't exist
# WRITE_ENHANCED_DURING_CYCLE=0 sadece JSON yazmayı kontrol ediyor, prediction yapılıyor ama yazılmıyor
# DISABLE_ML_PREDICTION_DURING_TRAINING=1 prediction'ı tamamen devre dışı bırakır
export WRITE_ENHANCED_DURING_CYCLE=0  # Disable prediction write during training
export DISABLE_ML_PREDICTION_DURING_TRAINING=1  # Disable ML prediction during training (prevents horizon features not found)
export ENABLE_AUTO_BACKTEST=0          # Disable backtest during training

# ⚡ Enable ALL features for production-like training (same as run_bulk_train.sh)
export ML_USE_ADAPTIVE_LEARNING=1  # Phase 2 incremental learning on test data
export ML_USE_SMART_ENSEMBLE=1     # Smart ensemble (consensus + performance)
export ML_USE_STACKED_SHORT=1      # Meta-stacking for short horizons (1/3/7d)
export ML_USE_REGIME_DETECTION=1  # Regime-aware feature weighting
export STRICT_HORIZON_FEATURES=1  # Horizon-specific features
export ENABLE_SEED_BAGGING=1      # Multi-seed training for robustness
export N_SEEDS=3                  # Number of seeds for bagging
export ML_EARLY_STOP_ROUNDS=50    # Early stopping rounds
export ENABLE_TALIB_PATTERNS=1    # TA-Lib patterns enabled

# Ensure DATABASE_URL is set
if [[ -z "${DATABASE_URL:-}" ]]; then
  export DATABASE_URL="postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:5432/bist_pattern_db"
fi

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/results"
LOG_FILE="$ROOT_DIR/logs/train_completed_hpo_$(date +%F_%H%M%S).log"

# Activate venv if available
if [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
  source "$ROOT_DIR/venv/bin/activate"
fi

# Ensure log directory permissions for web user to read logs produced by root cron
chown -R www-data:www-data "$ROOT_DIR/logs" 2>/dev/null || true

echo "[$(date -Is)] Starting training for completed HPO symbols with best parameters..."
echo "[$(date -Is)] All features enabled (production-like training)"
echo "[$(date -Is)] Log file: $LOG_FILE"

python -u "$ROOT_DIR/scripts/train_completed_hpo_with_best_params.py" 2>&1 | tee "$LOG_FILE"

echo "[$(date -Is)] Training finished."

# Make sure log is readable for www-data after run
chmod 644 "$LOG_FILE" 2>/dev/null || true
chown www-data:www-data "$LOG_FILE" 2>/dev/null || true

echo "Log: $LOG_FILE"

