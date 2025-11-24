#!/usr/bin/env bash
# Phase 2 HPO: HPO Features AÇIK (48 test symbols)
# Her sembol için 5 horizon (1d, 3d, 7d, 14d, 30d) = 240 HPO çalıştırması

set -uo pipefail

LOGDIR="${LOGDIR:-/opt/bist-pattern/logs/hpo_phase2_features_on}"
PY_BIN="${PY_BIN:-/opt/bist-pattern/venv/bin/python3}"
CONCURRENCY="${CONCURRENCY:-24}"  # 24 concurrent jobs
HORIZONS="${HORIZONS:-1,3,7,14,30}"

mkdir -p "$LOGDIR"
SCHED_LOG="$LOGDIR/hpo_phase2_scheduler.log"

echo "[scheduler] 🔬 PHASE 2 HPO: Features AÇIK (5 test symbols - en fazla verisi olan)" | tee -a "$SCHED_LOG"
echo "[scheduler] start $(date -Iseconds) conc=$CONCURRENCY" | tee -a "$SCHED_LOG"

# Test grubu: En fazla verisi olan 5 sembol (Phase 1'den)
# Toplam train days: 2220 her biri için (tüm horizon'lar)
TEST_SYMBOLS="ARSAN,ARENA,ARASE,ALKIM,ALKA"

SYM_COUNT=$(echo "$TEST_SYMBOLS" | tr ',' '\n' | wc -l)
echo "[scheduler] Test symbols: $SYM_COUNT symbols" | tee -a "$SCHED_LOG"

# Read horizons
IFS=',' read -ra HORIZONS_ARR <<< "$HORIZONS"

declare -a PIDS=()

launch_job() {
  local S="$1" H="$2"; local LOG="$LOGDIR/hpo_${S}_${H}d.log"
  echo "[scheduler] launching $S ${H}d -> $LOG" | tee -a "$SCHED_LOG"
  (stdbuf -oL -eL "$PY_BIN" -u /opt/bist-pattern/scripts/optuna_hpo_pilot_features_on.py --symbols "$S" --horizon "$H" --trials 100 --timeout 57600 >> "$LOG" 2>&1 && echo "[scheduler] done  $S ${H}d -> $LOG" | tee -a "$SCHED_LOG") &
  PIDS+=($!)
}

wait_for_slot() {
  while [ ${#PIDS[@]} -ge "$CONCURRENCY" ]; do
    for i in "${!PIDS[@]}"; do
      if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
        unset 'PIDS[$i]'
        PIDS=("${PIDS[@]}")
        break
      fi
    done
    sleep 2
  done
}

# Launch all jobs
for S in $(echo "$TEST_SYMBOLS" | tr ',' ' '); do
  for H in "${HORIZONS_ARR[@]}"; do
    wait_for_slot
    launch_job "$S" "$H"
  done
done

# Wait for all jobs
echo "[scheduler] waiting for all jobs to complete..." | tee -a "$SCHED_LOG"
wait

echo "[scheduler] ✅ all jobs completed $(date -Iseconds)" | tee -a "$SCHED_LOG"

