# flake8: noqa
# pyright: reportMissingImports=false
import os
os.environ.setdefault('PYTHONWARNINGS','ignore')
from app import app, get_pattern_detector
import traceback
from datetime import datetime

with app.app_context():
    from models import Stock
    det = get_pattern_detector()
    ml = det.ml_predictor
    try:
        from enhanced_ml_system import get_enhanced_ml_system
        enh = get_enhanced_ml_system()
    except Exception:
        enh = None

    # Import ML coordinator for global lock management
    try:
        from bist_pattern.core.ml_coordinator import get_ml_coordinator
        mlc = get_ml_coordinator()
        
        # Try to acquire global training lock
        if not mlc.acquire_global_training_lock("crontab", timeout=60):
            print("❌ Could not acquire global ML training lock - another training may be active")
            print("   Check if automation pipeline is running training")
            exit(1)
        print("🔒 Global ML training lock acquired by crontab")
    except Exception as e:
        print(f"⚠️ ML coordinator not available: {e}")
        mlc = None

    try:
        # Optional stop-sentinel file path for graceful halt
        STOP_FILE = os.getenv('TRAIN_STOP_FILE', '/opt/bist-pattern/.cache/STOP_TRAIN')
        symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).all()]
        ok_ml = fail_ml = skipped = 0
        ok_enh = fail_enh = 0
        log_dir = '/opt/bist-pattern/logs'
        error_log_path = os.path.join(log_dir, f'train_errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

        for i, sym in enumerate(symbols, 1):
            # Graceful stop check before each symbol
            try:
                if STOP_FILE and os.path.exists(STOP_FILE):
                    print("⛔ Stop sentinel detected. Aborting bulk training loop gracefully.")
                    break
            except Exception:
                pass
            try:
                # Use full history by passing days<=0
                df = det.get_stock_data(sym, days=0)
                if df is None or getattr(df, 'empty', False):
                    df = det.get_stock_data(sym, days=0)
                try:
                    min_days = int(os.getenv('ML_MIN_DATA_DAYS', os.getenv('ML_MIN_DAYS', '180')))
                except Exception:
                    min_days = 180
                if df is None or len(df) < min_days:
                    skipped += 1
                    continue
                
                # ✨ IMPROVED: Use ml_coordinator logic (like automation)
                # Only train if model is old, missing, or needs update
                if mlc:
                    # Check training gate (age, cooldown, completeness)
                    try:
                        ok_gate, reason = mlc.evaluate_training_gate(sym, len(df))
                        if not ok_gate:
                            skipped += 1
                            if i % 50 == 0:
                                print(f"  [{i}] {sym}: Skipped ({reason})")
                            continue
                    except Exception:
                        pass  # If gate check fails, proceed with training
                
                try:
                    if ml:
                        res = ml.train_models(sym, df)
                        ok_ml += 1 if res else 0
                    else:
                        skipped += 1
                except Exception as e:
                    fail_ml += 1
                    try:
                        os.makedirs(log_dir, exist_ok=True)
                        with open(error_log_path, 'a') as ef:
                            ef.write(f"[{datetime.now().isoformat()}] {sym} ML error: {e}\n")
                            ef.write(traceback.format_exc() + "\n")
                    except Exception:
                        pass
                
                # Enhanced ML (persisted to disk) - use coordinator logic
                # Second stop check before enhanced training
                try:
                    if STOP_FILE and os.path.exists(STOP_FILE):
                        print("⛔ Stop sentinel detected before enhanced training. Aborting.")
                        break
                except Exception:
                    pass

                try:
                    if enh and mlc:
                        # Use coordinator's smart training (respects cooldown, age)
                        res_enh = mlc.train_enhanced_model_if_needed(sym, df)
                        ok_enh += 1 if res_enh else 0
                    elif enh:
                        # Fallback: direct training (old behavior)
                        res_enh = enh.train_enhanced_models(sym, df)
                        ok_enh += 1 if res_enh else 0
                    else:
                        skipped += 1
                except Exception as e:
                    fail_enh += 1
                    try:
                        os.makedirs(log_dir, exist_ok=True)
                        with open(error_log_path, 'a') as ef:
                            ef.write(f"[{datetime.now().isoformat()}] {sym} ENH error: {e}\n")
                            ef.write(traceback.format_exc() + "\n")
                    except Exception:
                        pass
                
                # Simple ML training removed - was using non-existent simple_enhanced_ml module
                
            except Exception as e:
                fail_ml += 1
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    with open(error_log_path, 'a') as ef:
                        ef.write(f"[{datetime.now().isoformat()}] {sym} OUTER error: {e}\n")
                        ef.write(traceback.format_exc() + "\n")
                except Exception:
                    pass
            
            if i % 25 == 0:
                print(f"[{i}/{len(symbols)}] ok_ml={ok_ml} fail_ml={fail_ml} ok_enh={ok_enh} fail_enh={fail_enh} skipped={skipped}")
            
            print(f"DONE: ok_ml={ok_ml} fail_ml={fail_ml} ok_enh={ok_enh} fail_enh={fail_enh} skipped={skipped} total={len(symbols)}")
    
    finally:
        # Always release the global training lock
        if mlc:
            mlc.release_global_training_lock()
            print("🔓 Global ML training lock released by crontab")
