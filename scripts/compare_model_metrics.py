#!/usr/bin/env python3
"""
Compare old vs new model metrics for 3 test symbols
"""
import sys
import os

sys.path.insert(0, '/opt/bist-pattern')
os.environ.setdefault('DATABASE_URL', os.environ.get('DATABASE_URL', ''))

from app import app  # noqa: E402
from enhanced_ml_system import get_enhanced_ml_system  # noqa: E402

TEST_SYMBOLS = ['AKBNK', 'EREGL', 'TUPRS']
HORIZONS = ['1d', '3d', '7d']

with app.app_context():
    enh = get_enhanced_ml_system()
    
    print("=" * 120)
    print("📊 MODEL METRICS COMPARISON (Old vs New with 5x Caps)")
    print("=" * 120)
    print()
    
    for symbol in TEST_SYMBOLS:
        print(f"\n{'='*120}")
        print(f"🎯 {symbol}")
        print(f"{'='*120}\n")
        
        # Try to load models from disk
        try:
            success = enh.load_enhanced_models(symbol)
            if not success:
                print(f"❌ Could not load models for {symbol}")
                continue
        except Exception as e:
            print(f"❌ Error loading {symbol}: {e}")
            continue
        
        print(f"{'Horizon':<8} {'Model':<12} {'R²':<10} {'MAPE':<10} {'Confidence':<12} {'Samples':<10}")
        print("-" * 120)
        
        for horizon in HORIZONS:
            h_key = f"{horizon}"
            
            # XGBoost
            xgb_key = f"{symbol}_{h_key}_xgboost"
            if xgb_key in enh.models:
                model_info = enh.models[xgb_key]
                r2 = model_info.get('r2', model_info.get('train_r2', 'N/A'))
                mape = model_info.get('mape', 'N/A')
                conf = model_info.get('confidence', 'N/A')
                samples = model_info.get('n_samples', 'N/A')
                
                if isinstance(r2, (int, float)):
                    r2_str = f"{r2:.3f}"
                else:
                    r2_str = str(r2)
                
                if isinstance(mape, (int, float)):
                    mape_str = f"{mape:.2f}%"
                else:
                    mape_str = str(mape)
                    
                if isinstance(conf, (int, float)):
                    conf_str = f"{conf:.3f}"
                else:
                    conf_str = str(conf)
                
                print(f"{h_key:<8} {'XGBoost':<12} {r2_str:<10} {mape_str:<10} {conf_str:<12} {samples:<10}")
            
            # LightGBM
            lgb_key = f"{symbol}_{h_key}_lightgbm"
            if lgb_key in enh.models:
                model_info = enh.models[lgb_key]
                r2 = model_info.get('r2', model_info.get('train_r2', 'N/A'))
                mape = model_info.get('mape', 'N/A')
                conf = model_info.get('confidence', 'N/A')
                
                if isinstance(r2, (int, float)):
                    r2_str = f"{r2:.3f}"
                else:
                    r2_str = str(r2)
                
                if isinstance(mape, (int, float)):
                    mape_str = f"{mape:.2f}%"
                else:
                    mape_str = str(mape)
                    
                if isinstance(conf, (int, float)):
                    conf_str = f"{conf:.3f}"
                else:
                    conf_str = str(conf)
                
                print(f"{'':<8} {'LightGBM':<12} {r2_str:<10} {mape_str:<10} {conf_str:<12} {'':<10}")
            
            # CatBoost
            cb_key = f"{symbol}_{h_key}_catboost"
            if cb_key in enh.models:
                model_info = enh.models[cb_key]
                r2 = model_info.get('r2', model_info.get('train_r2', 'N/A'))
                mape = model_info.get('mape', 'N/A')
                conf = model_info.get('confidence', 'N/A')
                
                if isinstance(r2, (int, float)):
                    r2_str = f"{r2:.3f}"
                else:
                    r2_str = str(r2)
                
                if isinstance(mape, (int, float)):
                    mape_str = f"{mape:.2f}%"
                else:
                    mape_str = str(mape)
                    
                if isinstance(conf, (int, float)):
                    conf_str = f"{conf:.3f}"
                else:
                    conf_str = str(conf)
                
                print(f"{'':<8} {'CatBoost':<12} {r2_str:<10} {mape_str:<10} {conf_str:<12} {'':<10}")
            
            print()
    
    print()
    print("=" * 120)
    print()
    print("📊 YORUMLAR:")
    print()
    print("R² (Coefficient of Determination):")
    print("  • 1.0  = Mükemmel (model her şeyi açıklıyor)")
    print("  • 0.0  = Baseline (model ortalamadan iyi değil)")
    print("  • <0.0 = Kötü (model ortalamadan daha kötü)")
    print()
    print("MAPE (Mean Absolute Percentage Error):")
    print("  • <1%  = Çok iyi")
    print("  • 1-2% = İyi")
    print("  • 2-5% = Orta")
    print("  • >5%  = Kötü")
    print()
    print("⚠️  NOT: R² negatifse model zayıf demektir!")
    print("⚠️  Caps artırıldı ama MSE loss kullanıldı (Directional Loss devre dışı)")
    print()
