#!/usr/bin/env bash
set -euo pipefail

# Paths
ROOT="/opt/bist-pattern"
LOG_DIR="${BIST_LOG_PATH:-$ROOT/logs}"
PARAM_DIR="$LOG_DIR/params"
PARAM_FILE="$LOG_DIR/param_store.json"
ACTIVE_LINK="$PARAM_DIR/active.json"
VERS_DIR="$PARAM_DIR/versions"

# Ensure dirs
mkdir -p "$PARAM_DIR" "$VERS_DIR"

STAMP="$(date -Iseconds | sed 's/[:+]/_/g')"
VER_FILE="$VERS_DIR/param_store_${STAMP}.json"

if [ ! -f "$PARAM_FILE" ]; then
  echo "param_store.json not found at $PARAM_FILE" >&2
  exit 1
fi

# Basic JSON validation and checksum sanity
if ! jq -e . "$PARAM_FILE" >/dev/null 2>&1; then
  echo "Invalid JSON: $PARAM_FILE" >&2
  exit 1
fi

WSUM=$(jq -er '.weights_checksum // empty' "$PARAM_FILE" 2>/dev/null || true)
if [ -n "$WSUM" ]; then
  CALC=$(python3 - "$PARAM_FILE" <<'PY'
import sys, json, hashlib
path = sys.argv[1]
with open(path, 'r') as f:
    d = json.load(f) or {}
weights = d.get('weights', {})
print(hashlib.sha256(json.dumps(weights, sort_keys=True).encode('utf-8')).hexdigest())
PY
)
  if [ "$CALC" != "$WSUM" ]; then
    echo "Checksum mismatch in weights; refusing to publish" >&2
    exit 1
  fi
fi

cp -f "$PARAM_FILE" "$VER_FILE"
ln -sfn "$VER_FILE" "$ACTIVE_LINK"
echo "published=$VER_FILE"
