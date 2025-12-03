#!/usr/bin/env python3
"""
Retrain symbols using the SAME low support gating filter that HPO used (10/5.0).
This ensures we're using the "real" best params that HPO found.

Kullanım:
    /opt/bist-pattern/venv/bin/python3 scripts/retrain_with_hpo_filter.py --symbols ADEL CONSE
"""

import sys
import os

# Add project root to path
sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

# ✅ CRITICAL: Set HPO filter values BEFORE importing pipeline
# This ensures training uses the SAME filter that HPO used when finding best params
os.environ['HPO_MIN_MASK_COUNT'] = '10'
os.environ['HPO_MIN_MASK_PCT'] = '5.0'

# Now import and run retrain script
from scripts.retrain_high_discrepancy_symbols import main

if __name__ == '__main__':
    print("=" * 80)
    print("RETRAINING WITH HPO FILTER (10/5.0)")
    print("=" * 80)
    print("⚠️  Using HPO's original filter values:")
    print("   HPO_MIN_MASK_COUNT=10")
    print("   HPO_MIN_MASK_PCT=5.0")
    print("=" * 80)
    print()
    
    main()

