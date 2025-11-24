#!/bin/bash
# BIST30 Training - Ana eğitim scriptini kullan, sadece BIST30 sembolleri için

cd /opt/bist-pattern

# BIST30 sembolleri
BIST30_SYMBOLS="AKBNK,ARCLK,ASELS,BIMAS,EKGYO,ENJSA,EREGL,FROTO,GARAN,HEKTS,ISCTR,KCHOL,KOZAL,KOZAA,KRDMD,PETKM,PGSUS,SAHOL,SASA,SISE,TAVHL,TCELL,THYAO,TOASO,TUPRS,VAKBN,VESTL,YKBNK,ODAS,SMRTG"

# Environment setup
export ENABLE_EXTERNAL_FEATURES=${ENABLE_EXTERNAL_FEATURES:-1}
export ENABLE_FINGPT_FEATURES=${ENABLE_FINGPT_FEATURES:-1}
export ML_USE_SMART_ENSEMBLE=${ML_USE_SMART_ENSEMBLE:-1}
export ML_USE_REGIME_DETECTION=${ML_USE_REGIME_DETECTION:-1}
export ML_ADAPTIVE_DEADBAND_MODE=${ML_ADAPTIVE_DEADBAND_MODE:-std}
export ML_ADAPTIVE_K_1D=${ML_ADAPTIVE_K_1D:-2.0}
export ML_ADAPTIVE_K_3D=${ML_ADAPTIVE_K_3D:-1.8}
export ML_ADAPTIVE_K_7D=${ML_ADAPTIVE_K_7D:-1.6}
export ML_PATTERN_WEIGHT_SCALE_1D=${ML_PATTERN_WEIGHT_SCALE_1D:-1.2}
export ML_PATTERN_WEIGHT_SCALE_3D=${ML_PATTERN_WEIGHT_SCALE_3D:-1.15}
export ML_PATTERN_WEIGHT_SCALE_7D=${ML_PATTERN_WEIGHT_SCALE_7D:-1.1}
export ML_CAP_PCTL_3D=${ML_CAP_PCTL_3D:-92.5}
export FORCE_FULL_RETRAIN=1  # Bypass model age gate

# Filter symbols (sadece BIST30)
export TRAIN_SYMBOLS="$BIST30_SYMBOLS"

echo "🚀 BIST30 Training Started"
echo "📊 Symbols: 30 (BIST30)"
echo "🔧 External Features: $ENABLE_EXTERNAL_FEATURES"
echo "🧠 Smart Ensemble: $ML_USE_SMART_ENSEMBLE"
echo "📈 Regime Detection: $ML_USE_REGIME_DETECTION"
echo ""

# Ana eğitim scriptini çağır (BIST30 filtresi ile)
.venv/bin/python << 'PYTHON'
import os
import sys
sys.path.insert(0, '/opt/bist-pattern')

# BIST30 filter
BIST30 = set(os.getenv('TRAIN_SYMBOLS', '').split(','))

from app import app, get_pattern_detector
from enhanced_ml_system import get_enhanced_ml_system

with app.app_context():
    det = get_pattern_detector()
    ml = get_enhanced_ml_system()
    
    success = 0
    failed = 0
    skipped = 0
    
    for i, sym in enumerate(sorted(BIST30), 1):
        print(f"[{i}/{len(BIST30)}] Training {sym}...")
        
        try:
            df = det.get_stock_data(sym, days=0)
            
            if df is None or len(df) < 200:
                print(f"  ⚠️ Insufficient data: {len(df) if df is not None else 0} days")
                skipped += 1
                continue
            
            print(f"  📊 Data: {len(df)} days")
            
            ok = ml.train_enhanced_models(sym, df)
            
            if ok:
                ml.save_enhanced_models(sym)
                success += 1
                print(f"  ✅ Success")
            else:
                failed += 1
                print(f"  ❌ Failed")
                
        except Exception as e:
            failed += 1
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Success: {success}/{len(BIST30)}")
    print(f"❌ Failed: {failed}/{len(BIST30)}")
    print(f"⚠️ Skipped: {skipped}/{len(BIST30)}")
PYTHON

echo ""
echo "✅ BIST30 Training Complete"
