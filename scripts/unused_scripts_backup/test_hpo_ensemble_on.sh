#!/usr/bin/env bash
# Test HPO Script: Ensemble açık (30 sembol)
# Bu script şu anki HPO bitince çalıştırılacak
# Amaç: Ensemble açık/kapalı HPO sonuçlarını karşılaştırmak

set -uo pipefail

LOGDIR="${LOGDIR:-/opt/bist-pattern/logs/hpo_test_ensemble_on}"
PY_BIN="${PY_BIN:-/opt/bist-pattern/venv/bin/python3}"
CONCURRENCY="${CONCURRENCY:-24}"  # 24 concurrent jobs
HORIZONS="${HORIZONS:-1,3,7,14,30}"

mkdir -p "$LOGDIR"
SCHED_LOG="$LOGDIR/test_hpo_scheduler.log"

echo "[scheduler] 🧪 TEST HPO: Ensemble AÇIK (30 sembol)" | tee -a "$SCHED_LOG"
echo "[scheduler] start $(date -Iseconds) conc=$CONCURRENCY" | tee -a "$SCHED_LOG"

# Rastgele seçilmiş 30 sembol (reproducible, seed=42)
TEST_SYMBOLS="AKGRT,AKMGY,AKSFA,AKSUE,AYES,BALAT,BESLR,BIZIM,BURCE,DOKTA,EGEPO,EGPRO,EKGYO,EMNIS,ERGLI,GEDZA,KTSVK,KUYAS,MEGAP,ORMA,PRKAB,RAYSG,SERNT,SEYKM,SNICA,TMSN,TRKSH,VAKFN,YGYO,ZKBVR"

SYM_COUNT=$(echo "$TEST_SYMBOLS" | tr ',' '\n' | wc -l)
echo "[scheduler] Test symbols: $SYM_COUNT symbols" | tee -a "$SCHED_LOG"

# Read horizons
IFS=',' read -ra HORIZONS_ARR <<< "$HORIZONS"

declare -a PIDS=()

launch_job() {
  local S="$1" H="$2"; local LOG="$LOGDIR/hpo_${S}_${H}d.log"
  : > "$LOG"
  echo "[hpo] start $S ${H}d -> $LOG" | tee -a "$SCHED_LOG"
  (
    # Ensemble AÇIK HPO için environment variables
    export ML_USE_ADAPTIVE_LEARNING=1
    export ML_USE_SMART_ENSEMBLE=1
    export ML_USE_STACKED_SHORT=1
    export ENABLE_META_STACKING=0  # ML_USE_STACKED_SHORT zaten 1
    export ML_USE_REGIME_DETECTION=1
    export ENABLE_SEED_BAGGING=1
    export N_SEEDS=3
    
    # Results dizini (ayrı dizin, karşılaştırma için)
    # Not: optuna_hpo_pilot.py sabit dizin kullanıyor, sonra taşıyacağız
    mkdir -p "/opt/bist-pattern/results/test_ensemble_on"
    
    # Optional: cap per-process threads if HPO_THREADS is set
    if [[ -n "${HPO_THREADS:-}" ]]; then
      export OMP_NUM_THREADS="${HPO_THREADS}"
      export MKL_NUM_THREADS="${HPO_THREADS}"
      export NUMEXPR_MAX_THREADS="${HPO_THREADS}"
      export OPENBLAS_NUM_THREADS="${HPO_THREADS}"
      export OPTUNA_XGB_NTHREAD="${HPO_THREADS}"
      export LIGHTGBM_NUM_THREADS="${HPO_THREADS}"
      export CATBOOST_THREAD_COUNT="${HPO_THREADS}"
    fi
    
    stdbuf -oL -eL "$PY_BIN" -u /opt/bist-pattern/scripts/optuna_hpo_pilot.py \
    --symbols "$S" --horizon "$H" --trials 100 --timeout 57600 \
    >> "$LOG" 2>&1
    
    EXIT_CODE=$?
    if [[ $EXIT_CODE -eq 0 ]]; then
      echo "[hpo] done  $S ${H}d -> $LOG" | tee -a "$SCHED_LOG"
    else
      echo "[hpo] FAIL  $S ${H}d -> $LOG (exit=$EXIT_CODE)" | tee -a "$SCHED_LOG"
    fi
  ) &
  local PID=$!
  PIDS+=("$PID")
}

# Launch all jobs
for S in $(echo "$TEST_SYMBOLS" | tr ',' ' '); do
  for H in "${HORIZONS_ARR[@]}"; do
    # Wait if we've reached concurrency limit
    while [[ ${#PIDS[@]} -ge $CONCURRENCY ]]; do
      # Check which PIDs are still running
      for i in "${!PIDS[@]}"; do
        if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
          unset 'PIDS[$i]'
        fi
      done
      # Rebuild array to remove gaps
      PIDS=("${PIDS[@]}")
      sleep 1
    done
    launch_job "$S" "$H"
  done
done

# Wait for all jobs to complete
echo "[scheduler] Waiting for all jobs to complete..." | tee -a "$SCHED_LOG"
for PID in "${PIDS[@]}"; do
  wait "$PID"
done

echo "[scheduler] ✅ TEST HPO complete $(date -Iseconds)" | tee -a "$SCHED_LOG"
echo "[scheduler] Results saved in: /opt/bist-pattern/results/test_ensemble_on/" | tee -a "$SCHED_LOG"

