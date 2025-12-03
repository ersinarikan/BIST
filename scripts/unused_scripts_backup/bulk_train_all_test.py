# flake8: noqa
# pyright: reportMissingImports=false
import os
import sys
sys.path.insert(0, '/opt/bist-pattern')
os.environ.setdefault('PYTHONWARNINGS','ignore')
# ✅ FIX: FinGPT cache path (www-data user can't access /root/.cache)
os.environ.setdefault('TRANSFORMERS_CACHE', '/opt/bist-pattern/.cache/huggingface')
os.environ.setdefault('HF_HOME', '/opt/bist-pattern/.cache/huggingface')
from app import app, get_pattern_detector
import logging
from logging.handlers import RotatingFileHandler
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
        # Configure logging format to include PID and module
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(process)d %(name)s:%(levelname)s %(message)s')
        # Add rotating file handler per-process under logs/training
        try:
            import os as _os
            from datetime import datetime as _dt
            _log_root = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            _train_dir = _os.path.join(_log_root, 'training')
            _os.makedirs(_train_dir, exist_ok=True)
            _stamp = _dt.now().strftime('%Y%m%d_%H%M%S')
            _pid = _os.getpid()
            _fname = _os.path.join(_train_dir, f'train_{_stamp}_{_pid}.log')
            _fh = RotatingFileHandler(_fname, maxBytes=10*1024*1024, backupCount=5)
            _fh.setLevel(logging.INFO)
            _fh.setFormatter(logging.Formatter('%(asctime)s %(process)d %(name)s:%(levelname)s %(message)s'))
            root_logger = logging.getLogger()
            root_logger.addHandler(_fh)
            logging.getLogger(__name__).info(f"Training log file: {_fname}")
        except Exception as _log_err:
            logging.getLogger(__name__).warning(f"Training file logger setup failed: {_log_err}")

        # Optional stop-sentinel file path for graceful halt
        STOP_FILE = os.getenv('TRAIN_STOP_FILE', '/opt/bist-pattern/.cache/STOP_TRAIN')
        # Allow forcing a full retrain that bypasses coordinator gating
        FORCE_FULL_RETRAIN = str(os.getenv('FORCE_FULL_RETRAIN', '0')).lower() in ('1', 'true', 'yes')
        # Ensure deterministic A→Z order and equity-only training universe with denylist fallback
        try:
            base_query = Stock.query.filter_by(is_active=True)
            if hasattr(Stock, 'type'):
                base_query = base_query.filter_by(type='EQUITY')
            raw_symbols = [s.symbol for s in base_query.all()]
        except Exception:
            raw_symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).all()]

        import re
        denylist = re.compile(r"USDTR|USDTRY|^XU|^OPX|^F_|VIOP|INDEX", re.IGNORECASE)
        symbols = sorted([sym for sym in raw_symbols if sym and not denylist.search(sym)])
        # TEST_FILTER: Only train specific symbols
        test_symbols = os.getenv('TRAIN_SYMBOLS_FILTER', '').split(',')
        if test_symbols and test_symbols[0]:
            symbols = [s for s in symbols if s in test_symbols]
            print(f"🧪 TEST MODE: Training only {len(symbols)} symbols: {symbols}")
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
                if mlc and not FORCE_FULL_RETRAIN:
                    # Check training gate (age, cooldown, completeness) unless forced
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
                    if enh and mlc and not FORCE_FULL_RETRAIN:
                        # Use coordinator's smart training (respects cooldown, age)
                        res_enh = mlc.train_enhanced_model_if_needed(sym, df)
                        ok_enh += 1 if res_enh else 0
                    elif enh:
                        # Forced or coordinator unavailable: train directly and persist
                        res_enh = enh.train_enhanced_models(sym, df)
                        if res_enh:
                            try:
                                enh.save_enhanced_models(sym)
                            except Exception:
                                pass
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
