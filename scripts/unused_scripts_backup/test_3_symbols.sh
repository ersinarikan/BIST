#!/bin/bash
set -euo pipefail

# Load environment
SYSTEMCTL="$(command -v systemctl || echo /bin/systemctl)"
ENV_LINE="$($SYSTEMCTL show bist-pattern --property=Environment --value || true)"
if [[ -n "${ENV_LINE:-}" ]]; then
  eval "export ${ENV_LINE}"
fi

export PYTHONPATH=/opt/bist-pattern

# Directional Loss Settings
export ML_USE_DIRECTIONAL_LOSS=1
export ML_LOSS_MSE_WEIGHT=0.3
export ML_LOSS_THRESHOLD=0.005
export DIRECTION_HIT_THRESHOLD=0.005

# Force specific symbols only
export TRAIN_SYMBOLS_FILTER="AKBNK,EREGL,TUPRS"

cd /opt/bist-pattern

echo "🧪 Testing Directional Loss with 3 symbols: AKBNK, EREGL, TUPRS"
echo ""

# Modify bulk_train_all.py temporarily to filter symbols
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/opt/bist-pattern')
import os
os.environ.setdefault('DATABASE_URL', os.environ.get('DATABASE_URL', ''))

# Read and filter
with open('scripts/bulk_train_all.py', 'r') as f:
    code = f.read()

# Check if already filtered
if 'TEST_FILTER' not in code:
    # Add filter after symbol sorting
    code = code.replace(
        'symbols = sorted([sym for sym in raw_symbols if sym and not denylist.search(sym)])',
        '''symbols = sorted([sym for sym in raw_symbols if sym and not denylist.search(sym)])
        # TEST_FILTER: Only train specific symbols
        test_symbols = os.getenv('TRAIN_SYMBOLS_FILTER', '').split(',')
        if test_symbols and test_symbols[0]:
            symbols = [s for s in symbols if s in test_symbols]
            print(f"🧪 TEST MODE: Training only {len(symbols)} symbols: {symbols}")'''
    )
    
    with open('scripts/bulk_train_all_test.py', 'w') as f:
        f.write(code)
    
    print("✅ Test script created: scripts/bulk_train_all_test.py")
else:
    print("✅ Using existing bulk_train_all.py")
    import shutil
    shutil.copy('scripts/bulk_train_all.py', 'scripts/bulk_train_all_test.py')
PYEOF

echo ""
echo "🚀 Starting test training..."
echo ""

venv/bin/python3 scripts/bulk_train_all_test.py

echo ""
echo "✅ Test complete!"
