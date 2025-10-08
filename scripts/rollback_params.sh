#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/bist-pattern"
LOG_DIR="${BIST_LOG_PATH:-$ROOT/logs}"
PARAM_DIR="$LOG_DIR/params"
ACTIVE_LINK="$PARAM_DIR/active.json"
VERS_DIR="$PARAM_DIR/versions"

if [ $# -lt 1 ]; then
  echo "usage: rollback_params.sh <version_json_path_or_basename>" >&2
  exit 1
fi

CAND="$1"
if [ ! -f "$CAND" ]; then
  CAND="$VERS_DIR/$CAND"
fi
if [ ! -f "$CAND" ]; then
  echo "version file not found: $1" >&2
  exit 1
fi

ln -sfn "$CAND" "$ACTIVE_LINK"
echo "rolled_back_to=$CAND"
