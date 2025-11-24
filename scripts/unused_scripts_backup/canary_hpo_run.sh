#!/usr/bin/env bash
set -euo pipefail

LOGDIR="${LOGDIR:-/opt/bist-pattern/logs}"
PY_BIN="${PY_BIN:-/opt/bist-pattern/venv/bin/python3}"
CONCURRENCY="${CONCURRENCY:-2}"
SYMS=(AKBNK ARCLK ASELS BIMAS EKGYO ENJSA EREGL FROTO GARAN HEKTS ISCTR KCHOL KOZAL KOZAA KRDMD PETKM PGSUS SAHOL SASA SISE TAVHL TCELL THYAO TOASO TUPRS VAKBN VESTL YKBNK ODAS SMRTG)
HORIZONS=(1 3 7)

mkdir -p "$LOGDIR"
SCHED_LOG="$LOGDIR/canary_hpo_scheduler.log"
echo "[scheduler] start $(date -Iseconds) conc=$CONCURRENCY" | tee -a "$SCHED_LOG"

declare -a PIDS=()

launch_job() {
  local S="$1" H="$2"; local LOG="$LOGDIR/hpo_${S}_${H}d.log"
  : > "$LOG"
  echo "[canary-hpo] start $S ${H}d -> $LOG" | tee -a "$SCHED_LOG"
  (stdbuf -oL -eL "$PY_BIN" -u /opt/bist-pattern/scripts/optuna_hpo_pilot.py --symbols "$S" --horizon "$H" --trials 100 --timeout 10800 >> "$LOG" 2>&1 && echo "[canary-hpo] done  $S ${H}d -> $LOG" | tee -a "$SCHED_LOG") &
  PIDS+=("$!")
}

# Iterate tasks
for S in "${SYMS[@]}"; do
  for H in "${HORIZONS[@]}"; do
    # Throttle concurrency
    while [ "${#PIDS[@]}" -ge "$CONCURRENCY" ]; do
      if wait -n 2>/dev/null; then :; fi
      # Purge finished PIDs
      tmp=()
      for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then tmp+=("$pid"); fi
      done
      PIDS=("${tmp[@]}")
    done
    launch_job "$S" "$H"
    sleep 1
  done
done

# Wait remaining
for pid in "${PIDS[@]}"; do
  wait "$pid" || true
done

echo "[scheduler] end $(date -Iseconds)" | tee -a "$SCHED_LOG"


