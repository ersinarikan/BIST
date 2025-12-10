#!/usr/bin/env python3
# pyright: reportUnusedVariable=false, reportUnusedImport=false
"""
Continuous HPO and Training Pipeline

Sürekli çalışan, otomatik HPO ve eğitim yapan pipeline:
1. Her sembol-horizon çifti için HPO (features açık)
2. Best params ile full training (adaptive learning dahil)
3. Model kaydetme
4. ✅ Symbol-based sequential processing: Her sembol için tüm
   horizonları sırayla işle (1d→3d→7d→14d→30d)
5. Cycle management (tüm semboller bittikten sonra başa dön)
6. Yeni verileri ekleyerek sürekli iyileşme

✅ YENİ YAKLAŞIM:
- Her sembol için tüm horizonları sırayla işle (bir sembol bitmeden
  diğerine geçme)
- MAX_WORKERS=1: Tam sequential (tek sembol, tüm horizonlar)
- MAX_WORKERS=3-4: Hybrid (3-4 sembol paralel, her biri sequential)
- Veritabanı yükü azalır (aynı sembol için veri bir kez çekilir)
- SQLite çakışmaları azalır (bir sembol at a time)
"""
import os
import sys
import json
import time
import logging
import math
import signal
import atexit
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import fcntl
import shutil
import threading
import numpy as np
import pandas as pd

# ✅ FIX: Suppress KeyError from gevent monkey-patched threading module
# This is a known issue where gevent's monkey-patched threading._delete()
# tries to delete a thread from _active dict that's already been removed
# This happens during thread cleanup and doesn't affect functionality
_original_threading_delete = getattr(threading.Thread, "_delete", None)


def _safe_threading_delete(self):
    """Wrapper to safely handle KeyError in thread cleanup (gevent
    compatibility)"""
    try:
        if callable(_original_threading_delete):
            _original_threading_delete(self)  # type: ignore[misc]
    except KeyError:
        # Thread already removed from _active dict - safe to ignore
        # This is a harmless race condition in gevent's monkey-patched
        # threading
        pass


setattr(threading.Thread, "_delete", _safe_threading_delete)

# Set environment
sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'
os.environ.setdefault('PYTHONWARNINGS', 'ignore')

# ⚡ CRITICAL: Disable prediction logging during training
os.environ['DISABLE_PREDICTIONS_LOG'] = '1'
os.environ['DISABLE_ML_PREDICTION_DURING_TRAINING'] = '1'
os.environ['WRITE_ENHANCED_DURING_CYCLE'] = '0'

# Ensure DATABASE_URL is set
if 'DATABASE_URL' not in os.environ:
    # PgBouncer default on localhost:6432 (fallback)
    os.environ['DATABASE_URL'] = (
        'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:6432/'
        'bist_pattern_db'
    )


def _get_best_trial_metrics_entry(
    hpo_result: Optional[Dict[str, Any]], symbol: str, horizon: int
) -> Optional[Dict[str, Any]]:
    if not isinstance(hpo_result, dict):
        return None
    metrics = hpo_result.get('best_trial_metrics')
    if not isinstance(metrics, dict):
        return None
    key = f"{symbol}_{horizon}d"
    entry = metrics.get(key)
    return entry if isinstance(entry, dict) else None


def _extract_reference_historical_r2(
    metrics_entry: Optional[Dict[str, Any]]
) -> Optional[Dict[str, float]]:
    if not isinstance(metrics_entry, dict):
        return None
    avg_metrics = metrics_entry.get('avg_model_metrics')
    if not isinstance(avg_metrics, dict):
        return None
    ref_map: Dict[str, float] = {}
    for model_name, stats in avg_metrics.items():
        if not isinstance(stats, dict):
            continue
        raw = stats.get('raw_r2')
        if isinstance(raw, (int, float)) and math.isfinite(raw):
            ref_map[model_name] = float(raw)
    return ref_map if ref_map else None


def _extract_model_metrics_from_train_result(
    train_result: Any, horizon: int
) -> Dict[str, Dict[str, float]]:
    """Extract model metrics (raw_r2, rmse, mape) from training result.

    Used in:
    - _evaluate_training_dirhits (WFV section, line ~1640)
    - _evaluate_training_dirhits (Online section, line ~2059)
    """
    metrics: Dict[str, Dict[str, float]] = {}
    if not isinstance(train_result, dict):
        return metrics
    horizon_key = f"{horizon}d"
    horizon_models = train_result.get(horizon_key, {})
    if not isinstance(horizon_models, dict):
        return metrics
    for model_name, info in horizon_models.items():
        if not isinstance(info, dict):
            continue
        payload: Dict[str, float] = {}
        for key in ('raw_r2', 'rmse', 'mape'):
            val = info.get(key)
            if isinstance(val, (int, float)) and math.isfinite(val):
                payload[key] = float(val)
        if payload:
            metrics[model_name] = payload
    return metrics


# ⚡ CRITICAL: Configure logging BEFORE importing modules that use logging
# This prevents their loggers from affecting our log file

# ⚡ CRITICAL: Disable root logger handlers to prevent imported modules
# from writing to our log file
# This must be done BEFORE importing app.py which starts
# working_automation
root_logger = logging.getLogger()
root_logger.handlers.clear()  # Clear all root logger handlers
root_logger.propagate = False  # Prevent propagation

# ✅ CRITICAL FIX: Disable working_automation auto-start in HPO service
# HPO service runs in separate process and should not start
# working_automation
# This prevents log mixing and resource conflicts
os.environ['AUTO_START_CYCLE'] = 'False'
os.environ['BIST_PIPELINE_STARTED'] = '1'  # Prevent auto-start

# Import after path setup
from app import app  # noqa: E402

# ⚡ CRITICAL: Clear root logger handlers again AFTER app.py import
# app.py calls logging.basicConfig() which adds handlers to root logger
# Clear handlers added by app.py's basicConfig()
root_logger.handlers.clear()
root_logger.propagate = False  # Ensure propagation is still disabled

from pattern_detector import HybridPatternDetector  # noqa: E402
from enhanced_ml_system import get_enhanced_ml_system  # noqa: E402

# ⚡ CRITICAL FIX: Disable propagation for imported modules' loggers
# This prevents their logs from appearing in our log file
logging.getLogger('pattern_detector').propagate = False
logging.getLogger('enhanced_ml_system').propagate = False
logging.getLogger('bist_pattern').propagate = False
logging.getLogger('fingpt_analyzer').propagate = False
logging.getLogger('yahoo_finance_enhanced').propagate = False
logging.getLogger('rss_news_async').propagate = False
logging.getLogger('working_automation').propagate = False
logging.getLogger('advanced_patterns').propagate = False
logging.getLogger('ml_prediction_system').propagate = False
logging.getLogger('bist_pattern.core.broadcaster').propagate = False
logging.getLogger(
    'bist_pattern.core.unified_collector'
).propagate = False
logging.getLogger(
    'bist_pattern.core.pattern_validator'
).propagate = False
logging.getLogger('yfinance_gevent_native').propagate = False
logging.getLogger('app').propagate = False

# Setup logging
# ✅ FIX: Ensure log directory exists and file handler works
log_file = Path('/opt/bist-pattern/logs/continuous_hpo_pipeline.log')
log_file.parent.mkdir(parents=True, exist_ok=True)
# ✅ FIX: Ensure log directory permissions for shared access
from bist_pattern.utils.file_permissions import (  # noqa: E402
    ensure_directory_permissions,
    ensure_file_permissions
)
ensure_directory_permissions(log_file.parent)

# ⚡ CRITICAL FIX: Use module-specific logger to avoid mixing with other
# services
# Don't use basicConfig as it affects root logger and all child loggers
# Instead, configure only our specific logger

# Get module-specific logger
logger = logging.getLogger('continuous_hpo_pipeline')
logger.setLevel(logging.INFO)
logger.handlers.clear()  # Clear any existing handlers
logger.propagate = False  # Prevent propagation to root logger (CRITICAL!)

# ✅ CRITICAL FIX: Disable ALL WebSocket emissions in HPO process
# HPO service runs in separate process and should not broadcast to
# clients
# This prevents log mixing, data leakage and WebSocket disconnections
try:
    app_socketio = getattr(app, 'socketio', None)
    if app_socketio is not None:  # type: ignore[attr-defined]
        _original_hpo_emit = app_socketio.emit  # type: ignore[attr-defined]

        def _hpo_blocked_emit(event, data=None, *args, **kwargs):
            # ✅ FIX: Block ALL WebSocket events in HPO process to prevent
            # log mixing
            # HPO logs should only go to log file, not to WebSocket
            logger.debug(
                f"🚫 HPO: Blocked WebSocket emit '{event}' "
                f"(HPO service should not broadcast)"
            )
            return  # Block all WebSocket events
        app_socketio.emit = _hpo_blocked_emit  # type: ignore[attr-defined]
        logger.info(
            "✅ HPO: All WebSocket emissions disabled "
            "(prevents log mixing)"
        )

    # ✅ FIX: Also disable broadcast_log function to prevent log mixing
    app_broadcast_log = getattr(app, 'broadcast_log', None)
    if app_broadcast_log is not None:  # type: ignore[attr-defined]
        _original_broadcast_log = (
            app_broadcast_log  # type: ignore[attr-defined]
        )

        def _hpo_blocked_broadcast_log(
            level, message, category='system', service=None
        ):
            # Block all broadcast_log calls in HPO process
            logger.debug(
                f"🚫 HPO: Blocked broadcast_log "
                f"(level={level}, category={category})"
            )
            return  # Block all broadcast_log calls
        setattr(app, 'broadcast_log', _hpo_blocked_broadcast_log)
        logger.info(
            "✅ HPO: broadcast_log disabled (prevents log mixing)"
        )
except Exception as e:
    logger.warning(
        f"⚠️ HPO: Failed to disable WebSocket/broadcast: {e}"
    )

# Create file handler with append mode (only for our logger)
file_handler = logging.FileHandler(
    log_file, mode='a', encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)

# Create stream handler (only for our logger)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)

# Add handlers only to our logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# ✅ CRITICAL FIX: Log HPO configuration after logger is initialized
_hpo_max_workers_env = os.getenv('HPO_MAX_WORKERS', 'NOT_SET')
_hpo_max_slots_env = os.getenv('HPO_MAX_SLOTS', 'NOT_SET')
logger.info(
    f"🔍 HPO Configuration Check: HPO_MAX_WORKERS="
    f"{_hpo_max_workers_env}, HPO_MAX_SLOTS={_hpo_max_slots_env}"
)

# Configuration
HORIZONS = [1, 3, 7, 14, 30]
HORIZON_ORDER = [1, 3, 7, 14, 30]
# ✅ Parallel processing - symbols in parallel, each processing all
# horizons sequentially
# Reduce default for system-wide stability; allow override via
# HPO_MAX_WORKERS
# ✅ CRITICAL FIX: Hardcode to 100 to bypass environment variable issue
# Process environment has 100 but Python reads 36 - likely environment
# override issue
# TODO: Investigate why environment variables are not being read
# correctly
try:
    MAX_WORKERS = (
        int(_hpo_max_workers_env)
        if _hpo_max_workers_env != 'NOT_SET' else 100
    )
    # ✅ TEMPORARY FIX: Force to 100 if environment variable is not set
    # correctly
    if MAX_WORKERS < 50:  # If less than 50, likely wrong value
        logger.warning(
            f"⚠️ HPO_MAX_WORKERS seems incorrect ({MAX_WORKERS}), "
            f"forcing to 100"
        )
        MAX_WORKERS = 100
    logger.info(
        f"🔍 MAX_WORKERS set to: {MAX_WORKERS} "
        f"(from HPO_MAX_WORKERS={_hpo_max_workers_env})"
    )
except Exception as e:
    MAX_WORKERS = 100  # Default to 100 instead of 4
    logger.warning(
        f"⚠️ Failed to parse HPO_MAX_WORKERS="
        f"'{_hpo_max_workers_env}', using 100, error: {e}"
    )
# MAX_WORKERS = 1  # Alternative: Fully sequential (one symbol at a time)
# MAX_WORKERS = 3  # Alternative: 3-4 symbols in parallel (conservative)
# ✅ CRITICAL FIX: Read HPO_TRIALS from environment variable
# (default: 1500)
# 1500 provides ~73% feature flag coverage: 1500/2048 combinations
try:
    HPO_TRIALS = int(os.getenv('HPO_TRIALS', '1500'))
    logger.info(
        f"🔍 HPO_TRIALS set to: {HPO_TRIALS} "
        f"(from environment variable)"
    )
except Exception as e:
    HPO_TRIALS = 1500  # Fallback to default
    logger.warning(
        f"⚠️ Failed to parse HPO_TRIALS, using default: "
        f"{HPO_TRIALS}, error: {e}"
    )

# ✅ CRITICAL FIX: MIN_TRIALS_FOR_RECOVERY should be dynamic based on
# HPO_TRIALS
# Recovery threshold: HPO_TRIALS - 10 (allows some margin for completion
# detection)
MIN_TRIALS_FOR_RECOVERY = max(1, HPO_TRIALS - 10)
logger.info(
    f"🔍 MIN_TRIALS_FOR_RECOVERY set to: "
    f"{MIN_TRIALS_FOR_RECOVERY} (HPO_TRIALS - 10)"
)

STATE_FILE = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
RESULTS_DIR = Path('/opt/bist-pattern/results/continuous_hpo')

# ✅ Global HPO concurrency limiter (cross-process via fcntl locks)

# ✅ NUMA-aware CPU optimization
# 4 NUMA nodes: 0-3, each with 32 CPUs
NUMA_NODES = 4
CPUS_PER_NODE = 32
_numa_counter_file = Path('/opt/bist-pattern/results/.numa_counter')
_numa_counter_lock = threading.Lock()


def _get_numa_node_and_cpus() -> Tuple[int, str]:
    """
    Get NUMA node and CPU list using file-based counter with locking
    (works across processes)
    Returns: (numa_node, cpu_list)
    """
    global _numa_counter_file

    # Use file-based counter with fcntl locking for cross-process coordination
    try:
        _numa_counter_file.parent.mkdir(parents=True, exist_ok=True)

        # Open file for read-write with locking
        with open(_numa_counter_file, 'a+') as f:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)

            try:
                # Read current counter
                f.seek(0)
                content = f.read().strip()
                if content:
                    try:
                        counter = int(content)
                    except ValueError:
                        counter = 0
                else:
                    counter = 0

                # Calculate NUMA node (round-robin)
                numa_node = counter % NUMA_NODES

                # Increment and save counter
                counter += 1
                f.seek(0)
                f.truncate()
                f.write(str(counter))
                f.flush()

            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # CPU range for this NUMA node
            cpu_start = numa_node * CPUS_PER_NODE
            cpu_end = cpu_start + CPUS_PER_NODE - 1
            cpu_list = f"{cpu_start}-{cpu_end}"
            return numa_node, cpu_list
    except Exception as e:
        logger.debug(
            f"NUMA detection failed, using process ID fallback: {e}"
        )
        # Fallback: use process ID for deterministic distribution
        import os
        numa_node = os.getpid() % NUMA_NODES
        cpu_start = numa_node * CPUS_PER_NODE
        cpu_end = cpu_start + CPUS_PER_NODE - 1
        cpu_list = f"{cpu_start}-{cpu_end}"
        return numa_node, cpu_list


def _build_numa_cmd(
    base_cmd: List[str], numa_node: int, cpu_list: str
) -> Tuple[List[str], int, str]:
    """
    Build command with CPU affinity (simplified - no NUMA binding)
    Uses taskset for CPU affinity only (NUMA binding removed - Python/ML
    not NUMA-aware)
    Returns: (command, numa_node, cpu_list)
    """
    # Use taskset for CPU affinity only (simpler and more effective)
    try:
        if shutil.which('taskset'):
            # taskset -c CPU_LIST python ... (no -- separator needed)
            return [
                'taskset',
                '-c', cpu_list,
            ] + base_cmd, numa_node, cpu_list
    except Exception as e:
        logger.debug(f"CPU affinity setup failed: {e}")

    # No affinity tools available, return base command
    return base_cmd, numa_node, cpu_list


def _get_hpo_slots_dir() -> Path:
    d = Path('/opt/bist-pattern/results/hpo_slots')
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.debug(f"Failed to create directory {d}: {e}")
    return d


def _get_hpo_max_slots() -> int:
    # ✅ CRITICAL FIX: Hardcode to 100 to bypass environment variable
    # issue
    # Process environment has 100 but Python reads 36 - likely
    # environment override issue
    # TODO: Investigate why environment variables are not being read
    # correctly
    try:
        max_slots = max(1, int(os.getenv('HPO_MAX_SLOTS', '100')))
        # ✅ TEMPORARY FIX: Force to 100 if environment variable is not
        # set correctly
        if max_slots < 50:  # If less than 50, likely wrong value
            logger.warning(
                f"⚠️ HPO_MAX_SLOTS seems incorrect ({max_slots}), "
                f"forcing to 100"
            )
            max_slots = 100
        return max_slots
    except Exception as e:
        logger.debug(f"Failed to get max_slots, using 100: {e}")
        return 100  # Default to 100 instead of 3


def _normalize_feature_flag_key(key: str) -> str:
    """
    Normalize feature flag key to uppercase environment variable name.
    Handles both lowercase (enable_*) and uppercase (ENABLE_*) keys.

    Args:
        key: Feature flag key (e.g., 'enable_seed_bagging' or
            'ENABLE_SEED_BAGGING')

    Returns:
        Uppercase environment variable name (e.g., 'ENABLE_SEED_BAGGING')
    """
    # If already uppercase, return as-is
    if key.isupper():
        return key

    # Map lowercase enable_* keys to uppercase environment variables
    mapping = {
        'enable_external_features': 'ENABLE_EXTERNAL_FEATURES',
        'enable_fingpt_features': 'ENABLE_FINGPT_FEATURES',
        'enable_yolo_features': 'ENABLE_YOLO_FEATURES',
        'enable_directional_loss': 'ML_USE_DIRECTIONAL_LOSS',
        'enable_seed_bagging': 'ENABLE_SEED_BAGGING',
        'enable_talib_patterns': 'ENABLE_TALIB_PATTERNS',
        'enable_smart_ensemble': 'ML_USE_SMART_ENSEMBLE',
        'enable_stacked_short': 'ML_USE_STACKED_SHORT',
        'enable_meta_stacking': 'ENABLE_META_STACKING',
        'enable_regime_detection': 'ML_USE_REGIME_DETECTION',
        'enable_fingpt': 'ENABLE_FINGPT',
        'enable_xgboost': 'ENABLE_XGBOOST',
        'enable_lightgbm': 'ENABLE_LIGHTGBM',
        'enable_catboost': 'ENABLE_CATBOOST',
    }

    # Return mapped value if exists, otherwise uppercase the key
    return mapping.get(key.lower(), key.upper())


class HPOSlotContext:
    """
    Context manager for HPO slot acquisition.

    Ensures the file descriptor is always closed, even if an exception occurs.
    Use with 'with' statement:

        with HPOSlotContext() as slot:
            slot_idx, slot_fd, slot_path = slot
            # ... use slot ...
            # slot_fd is automatically closed when exiting the 'with' block
    """
    def __init__(self, timeout_seconds: float = 300.0):
        self.timeout_seconds = timeout_seconds
        self.slot_idx = None
        self.slot_fd = None
        self.slot_path = None

    def __enter__(self):
        try:
            idx, f, path = acquire_hpo_slot(self.timeout_seconds)
            self.slot_idx = idx
            self.slot_fd = f
            self.slot_path = path
            return (idx, f, path)
        except TimeoutError:
            # Re-raise TimeoutError so caller can handle it
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.slot_fd is not None:
            release_hpo_slot(self.slot_fd)
            self.slot_fd = None
        return False  # Don't suppress exceptions


def acquire_hpo_slot(timeout_seconds: float = 300.0):
    """
    Acquire one of N slot locks (blocks until a slot becomes available).

    ⚠️ WARNING: Returns an open file descriptor that MUST be closed by
    calling release_hpo_slot().
    For automatic resource management, use HPOSlotContext instead:

        with HPOSlotContext() as slot:
            slot_idx, slot_fd, slot_path = slot
            # ... use slot ...

    Returns (slot_index, file_object, lock_path).

    Args:
        timeout_seconds: Maximum time to wait for a slot (default: 5
            minutes).
            If timeout is reached, raises TimeoutError.

    Returns:
        Tuple of (slot_index, file_object, lock_path). The file_object
        must be closed
        by calling release_hpo_slot(file_object) in a finally block to
        prevent file descriptor leaks.
    """
    slots_dir = _get_hpo_slots_dir()
    max_slots = _get_hpo_max_slots()
    import time as _time
    start_time = _time.time()

    while True:
        # Check timeout
        elapsed = _time.time() - start_time
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Failed to acquire HPO slot after "
                f"{timeout_seconds:.1f} seconds. All slots may be stuck."
            )

        for idx in range(max_slots):
            lock_path = slots_dir / f'hpo_slot_{idx}.lock'
            f = None
            try:
                # ✅ CRITICAL FIX: Check if lock file exists and if PID
                # is still alive
                # This prevents stale slots from blocking new processes
                if lock_path.exists():
                    try:
                        with open(lock_path, 'r') as check_f:
                            content = check_f.read().strip()
                            if content:
                                parts = content.split()
                                if len(parts) >= 1:
                                    try:
                                        lock_pid = int(parts[0])
                                        # Check if process is still alive
                                        try:
                                            # Signal 0: check if process
                                            # exists
                                            os.kill(lock_pid, 0)
                                            # Process exists, slot is active
                                        except (OSError, ProcessLookupError):
                                            # Process doesn't exist - stale
                                            # slot!
                                            # Try to remove stale lock file
                                            try:
                                                lock_path.unlink()
                                                logger.info(
                                                    f"🧹 Cleaned stale slot "
                                                    f"{idx} (PID {lock_pid} "
                                                    f"no longer exists)"
                                                )
                                            except Exception as e:
                                                logger.debug(
                                                    f"Failed to clean stale "
                                                    f"slot {idx}: {e}"
                                                )
                                    except (ValueError, IndexError):
                                        pass
                    except Exception as e:
                        logger.debug(f"Failed to process slot cleanup: {e}")

                f = open(lock_path, 'a+')
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    try:
                        f.seek(0)
                        f.truncate()
                        f.write(
                            f'{os.getpid()} '
                            f'{datetime.now().isoformat()}\n'
                        )
                        f.flush()
                        os.fsync(f.fileno())
                    except Exception as e:
                        logger.debug(f"Failed to fsync lock file: {e}")
                    return idx, f, lock_path
                except BlockingIOError:
                    # Slot is locked by another process, close file and try
                    # next slot
                    if f is not None:
                        try:
                            f.close()
                        except Exception as e:
                            logger.debug(
                                f"Failed to close file after "
                                f"BlockingIOError: {e}"
                            )
                    continue
                except Exception as e:
                    logger.debug(f"Other exception during flock: {e}")
                    # Other exception during flock (e.g., OSError,
                    # PermissionError)
                    # ✅ CRITICAL FIX: Close file handle to prevent file
                    # descriptor leak
                    if f is not None:
                        try:
                            f.close()
                        except Exception as e2:
                            logger.debug(
                                f"Failed to close file after flock "
                                f"exception: {e2}"
                            )
                    continue
            except Exception as e:
                logger.debug(f"Exception during file open: {e}")
                # Exception during file open (e.g., PermissionError,
                # FileNotFoundError)
                # File was not opened, so nothing to close
                continue

        # Check for deadlock: if all slots are held for too long, log
        # warning
        if elapsed > 60.0:  # After 1 minute, check for stuck slots
            try:
                stuck_slots = []
                for idx in range(max_slots):
                    lock_path = slots_dir / f'hpo_slot_{idx}.lock'
                    if lock_path.exists():
                        try:
                            # Try to read lock file to see when it was
                            # last updated
                            with open(lock_path, 'r') as check_f:
                                content = check_f.read().strip()
                                if content:
                                    # Parse PID and timestamp
                                    parts = content.split()
                                    if len(parts) >= 2:
                                        try:
                                            lock_time = (
                                                datetime.fromisoformat(
                                                    parts[1]
                                                )
                                            )
                                            lock_age = (
                                                datetime.now() - lock_time
                                            ).total_seconds()
                                            # If lock is older than 2 hours,
                                            # it's probably stuck
                                            if lock_age > 7200:
                                                stuck_slots.append(
                                                    (idx, lock_age)
                                                )
                                        except Exception as e:
                                            logger.debug(
                                                f"Failed to parse lock time "
                                                f"for slot {idx}: {e}"
                                            )
                        except Exception as e:
                            logger.debug(
                                f"Failed to check lock file for slot "
                                f"{idx}: {e}"
                            )
                if stuck_slots:
                    logger.warning(
                        f"⚠️ Detected potentially stuck HPO slots: "
                        f"{stuck_slots}"
                    )
            except Exception as e:
                # Don't fail slot acquisition due to deadlock detection
                # errors
                logger.debug(f"Deadlock detection failed: {e}")

        _time.sleep(0.25)


def release_hpo_slot(fobj) -> None:
    try:
        if fobj is not None:
            try:
                fcntl.flock(fobj.fileno(), fcntl.LOCK_UN)
            except Exception as e:
                logger.debug(f"Failed to unlock file: {e}")
            try:
                fobj.close()
            except Exception as e:
                logger.debug(f"Failed to close file: {e}")
    except Exception as e:
        logger.debug(f"release_hpo_slot failed: {e}")


@dataclass
class TaskState:
    """Task state for a symbol-horizon pair"""
    symbol: str
    horizon: int
    # 'pending', 'hpo_in_progress', 'training_in_progress', 'completed',
    # 'failed', 'skipped'
    status: str
    hpo_completed_at: Optional[str] = None
    training_completed_at: Optional[str] = None
    best_params_file: Optional[str] = None
    hpo_dirhit: Optional[float] = None
    # backward compatibility (WFV)
    training_dirhit: Optional[float] = None
    training_dirhit_wfv: Optional[float] = None
    training_dirhit_online: Optional[float] = None
    # NEW: Adaptive learning DirHit (online DirHit with adaptive learning
    # enabled)
    adaptive_dirhit: Optional[float] = None
    error: Optional[str] = None
    last_error: Optional[str] = None  # legacy/backfill support
    cycle: int = 0
    retry_count: int = 0  # ✅ FIX: Retry count for failed HPO tasks


class ContinuousHPOPipeline:
    """Continuous HPO and Training Pipeline"""

    def __init__(self):
        self.state_file = STATE_FILE
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(exist_ok=True, parents=True)
        # ✅ FIX: Ensure directory permissions for shared access
        ensure_directory_permissions(self.results_dir)
        # ✅ FIX: Clean up old temp state files on startup
        self._cleanup_temp_state_files()
        self.state: Dict[str, TaskState] = {}
        self.cycle = 0
        self.load_state()
        # Reset any stale in-progress tasks (e.g., after crash/restart)
        self._reset_stale_in_progress()
        # ✅ Graceful shutdown flag
        self.shutdown_requested = False
        self.active_hpo_processes: Dict[str, subprocess.Popen] = {}
        self._hpo_process_lock = threading.Lock()
        # ✅ PID file directory for tracking HPO processes (even with
        # start_new_session=True)
        self.hpo_pid_dir = Path('/opt/bist-pattern/results/hpo_pids')
        self.hpo_pid_dir.mkdir(exist_ok=True, parents=True)
        ensure_directory_permissions(self.hpo_pid_dir)
        # ✅ Register signal handlers for graceful shutdown
        self._register_signal_handlers()

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.info(
                f"🛑 Received {signal_name} signal, initiating "
                f"graceful shutdown..."
            )
            self.shutdown_requested = True
            # Gracefully terminate active HPO processes
            self._graceful_shutdown_hpo_processes()

        # Register handlers for SIGTERM and SIGINT
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Register atexit handler for cleanup
        atexit.register(self._cleanup_on_exit)
        logger.info(
            "✅ Signal handlers registered for graceful shutdown"
        )

    def _graceful_shutdown_hpo_processes(self):
        """Gracefully shutdown all active HPO processes"""
        shutdown_start = time.time()  # ✅ Track shutdown duration
        # ✅ IMPROVED: Track processes via both dict and PID files
        # This works even with start_new_session=True where parent can't
        # track children directly

        # First, get processes from active dict
        processes_from_dict = {}
        with self._hpo_process_lock:
            processes_from_dict = self.active_hpo_processes.copy()

        # Also get processes from PID files (for processes started with
        # start_new_session=True)
        processes_from_pids = {}
        if self.hpo_pid_dir.exists():
            for pid_file in self.hpo_pid_dir.glob("*.pid"):
                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())
                        # Check if process is still running
                        try:
                            os.kill(pid, 0)  # Check if process exists
                            # Extract symbol_horizon from filename (e.g.,
                            # "AGESA_1d.pid")
                            key = pid_file.stem
                            processes_from_pids[key] = pid
                        except (OSError, ProcessLookupError):
                            # Process doesn't exist, remove stale PID file
                            pid_file.unlink()
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to read PID file {pid_file}: {e}"
                    )

        # Combine both sources
        all_processes = {}
        for key, process in processes_from_dict.items():
            if process.poll() is None:  # Process is still running
                all_processes[key] = ('dict', process.pid)

        for key, pid in processes_from_pids.items():
            if key not in all_processes:
                all_processes[key] = ('pid', pid)

        # ✅ CRITICAL FIX: Update state BEFORE killing processes
        # This ensures hpo_in_progress tasks are reset to pending
        # immediately
        if all_processes:
            logger.info(
                f"🔄 Updating state for {len(all_processes)} active HPO "
                f"processes before shutdown..."
            )
            self._update_in_progress_to_pending_for_shutdown(
                list(all_processes.keys())
            )

        if not all_processes:
            logger.info("✅ No active HPO processes to shutdown")
            return

        logger.info(
            f"🛑 Gracefully shutting down {len(all_processes)} active HPO "
            f"processes..."
        )

        # First, send SIGTERM to all processes
        for key, (source, pid) in all_processes.items():
            try:
                logger.info(
                    f"   Sending SIGTERM to {key} (PID: {pid}, "
                    f"source: {source})"
                )
                os.kill(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                logger.debug(
                    f"   Process {key} (PID: {pid}) already terminated"
                )
            except Exception as e:
                logger.warning(
                    f"   Failed to terminate {key} (PID: {pid}): {e}"
                )

        # ✅ OPTIMIZED: Reduced timeout to 10 seconds for faster shutdown
        # First 5 seconds check every 0.5s, then every 1s
        timeout = 10
        start_time = time.time()
        while time.time() - start_time < timeout:
            remaining = []
            for key, (source, pid) in all_processes.items():
                try:
                    os.kill(pid, 0)  # Check if process exists
                    remaining.append((key, pid))
                except (OSError, ProcessLookupError):
                    # Process terminated, remove PID file
                    pid_file = self.hpo_pid_dir / f"{key}.pid"
                    if pid_file.exists():
                        pid_file.unlink()

            if not remaining:
                logger.info("✅ All HPO processes terminated gracefully")
                break

            # First 5 seconds check more frequently, then every 1 second
            elapsed = time.time() - start_time
            if elapsed < 5:
                time.sleep(0.5)
            else:
                time.sleep(1)
        else:
            # Force kill remaining processes
            # Re-check which processes are still running
            still_running = []
            for key, (source, pid) in all_processes.items():
                try:
                    os.kill(pid, 0)  # Check if process exists
                    still_running.append((key, pid))
                except (OSError, ProcessLookupError):
                    pass  # Process already terminated

            if still_running:
                logger.warning(
                    f"⚠️ Force killing {len(still_running)} HPO processes "
                    f"that didn't terminate gracefully"
                )
                for key, pid in still_running:
                    try:
                        os.kill(pid, signal.SIGKILL)
                        logger.info(f"   Force killed {key} (PID: {pid})")
                        # Remove PID file
                        pid_file = self.hpo_pid_dir / f"{key}.pid"
                        if pid_file.exists():
                            pid_file.unlink()
                    except Exception as e:
                        logger.warning(
                            f"   Failed to kill {key} (PID: {pid}): {e}"
                        )

        # Clear active processes dict
        with self._hpo_process_lock:
            self.active_hpo_processes.clear()

        # ✅ Log shutdown duration
        shutdown_duration = time.time() - shutdown_start
        logger.info(
            f"✅ Shutdown completed in {shutdown_duration:.2f} seconds"
        )

    def _update_in_progress_to_pending_for_shutdown(
        self, process_keys: List[str]
    ):
        """Update in-progress tasks to pending state during shutdown"""
        try:
            self.load_state()
            changed = False
            updated_count = 0

            for key in process_keys:
                task = self.state.get(key)
                if task and task.status in (
                    'hpo_in_progress', 'training_in_progress'
                ):
                    task.status = 'pending'
                    task.error = 'Interrupted - shutdown requested'
                    task.cycle = self.cycle
                    self.state[key] = task
                    changed = True
                    updated_count += 1

            if changed:
                self.save_state()
                logger.info(
                    f"🔄 Reset {updated_count} in-progress tasks to pending "
                    f"(shutdown)"
                )
            else:
                logger.info(
                    f"ℹ️ No in-progress tasks to update (shutdown) - all "
                    f"{len(process_keys)} processes already in different "
                    f"state"
                )
        except Exception as e:
            logger.error(f"❌ Failed to update state during shutdown: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _update_all_in_progress_to_pending(self):
        """Update all in-progress tasks to pending state (for cycle
        shutdown)"""
        self.load_state()
        changed = False
        updated_count = 0

        for key, task in list(self.state.items()):
            if task.status in (
                'hpo_in_progress', 'training_in_progress'
            ):
                task.status = 'pending'
                task.error = 'Interrupted - shutdown requested'
                task.cycle = self.cycle
                self.state[key] = task
                changed = True
                updated_count += 1

        if changed:
            self.save_state()
            logger.info(
                f"🔄 Reset {updated_count} in-progress tasks to pending "
                f"(cycle shutdown)"
            )

    def _cleanup_on_exit(self):
        """Cleanup function called on exit"""
        if self.active_hpo_processes:
            logger.info("🧹 Cleaning up active HPO processes on exit...")
            self._graceful_shutdown_hpo_processes()

    def load_state(self):
        """Load pipeline state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    # Shared lock for reading to avoid partial reads while
                    # another process writes
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    except Exception as e:
                        logger.debug(
                            f"Failed to acquire shared lock for state file: "
                            f"{e}"
                        )
                    content = f.read()
                    data = json.loads(content)
                    self.cycle = data.get('cycle', 0)
                    state_dict = data.get('state', {})
                    for key, task_data in state_dict.items():
                        self.state[key] = TaskState(**task_data)
                logger.info(
                    f"✅ Loaded state: {len(self.state)} tasks, "
                    f"cycle {self.cycle}"
                )
            except Exception as e:
                logger.error(f"❌ Error loading state: {e}")
                self.state = {}
        else:
            logger.info("📝 No existing state file, starting fresh")

    def _cleanup_temp_state_files(self):
        """Clean up old temporary state files (from crashed processes)"""
        try:
            # Find all temp state files matching pattern:
            # continuous_hpo_state.json.tmp.*
            temp_pattern = self.state_file.with_suffix('.json.tmp.*')
            temp_files = list(
                self.state_file.parent.glob(temp_pattern.name)
            )

            if temp_files:
                cleaned_count = 0
                for temp_file in temp_files:
                    try:
                        # Check if file is old (older than 1 hour) or if
                        # process is dead
                        file_age = time.time() - temp_file.stat().st_mtime
                        if file_age > 3600:  # Older than 1 hour
                            temp_file.unlink()
                            cleaned_count += 1
                            logger.debug(
                                f"🧹 Cleaned up old temp state file: "
                                f"{temp_file.name} "
                                f"(age: {file_age/3600:.1f}h)"
                            )
                        else:
                            # Check if process is still alive
                            try:
                                # Extract PID from filename: .tmp.{pid}
                                pid_str = (
                                    temp_file.suffixes[-1]
                                    if len(temp_file.suffixes) > 1 else None
                                )
                                if pid_str and pid_str.startswith('.'):
                                    pid = int(pid_str[1:])
                                    # Check if process exists
                                    # Signal 0 just checks if process exists
                                    os.kill(pid, 0)
                                    # Process is alive, keep file
                                else:
                                    # Can't determine PID, delete if old
                                    if file_age > 300:  # Older than 5 min
                                        temp_file.unlink()
                                        cleaned_count += 1
                            except (
                                ValueError, ProcessLookupError, OSError
                            ):
                                # Process doesn't exist, safe to delete
                                temp_file.unlink()
                                cleaned_count += 1
                                logger.debug(
                                    f"🧹 Cleaned up temp state file from "
                                    f"dead process: {temp_file.name}"
                                )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Could not clean up temp file "
                            f"{temp_file}: {e}"
                        )

                if cleaned_count > 0:
                    logger.info(
                        f"🧹 Cleaned up {cleaned_count} old temporary "
                        f"state files"
                    )
        except Exception as e:
            logger.warning(f"⚠️ Error during temp state file cleanup: {e}")

    def save_state(self):
        """Save pipeline state to file (merge-aware for concurrent
        processes)"""
        # ✅ CRITICAL FIX: Hold exclusive lock throughout entire
        # read-modify-write cycle
        # This prevents race conditions where another process modifies state
        # between read and write
        lock_fd = None
        try:
            # Open state file for reading/writing and acquire exclusive lock
            # Lock will be held until the entire read-modify-write cycle
            # completes
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            ensure_directory_permissions(self.state_file.parent)

            # Open file in read-write mode to hold lock across read and
            # write operations
            if self.state_file.exists():
                lock_fd = os.open(self.state_file, os.O_RDWR)
            else:
                # File doesn't exist, create it
                lock_fd = os.open(
                    self.state_file, os.O_CREAT | os.O_RDWR, 0o644
                )

            # Acquire exclusive lock - will be held until explicitly
            # released
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except Exception as e:
                # Lock acquisition failed, continue anyway (best effort)
                logger.debug(f"Lock acquisition failed (best effort): {e}")

            # Read existing state (with lock held)
            merged_state = {}
            existing_data = {}
            try:
                # Seek to beginning and read
                os.lseek(lock_fd, 0, os.SEEK_SET)
                content = os.read(lock_fd, 1024 * 1024)  # Read up to 1MB
                if content:
                    existing_data = json.loads(content.decode('utf-8'))
                    # Merge existing tasks
                    for key, task_data in existing_data.get(
                        'state', {}
                    ).items():
                        merged_state[key] = task_data
            except Exception as e:
                # If read fails, start fresh (but log it)
                logger.warning(
                    f"⚠️ Could not read existing state for merge, using "
                    f"current state only: {e}"
                )

            # Merge current process's updates (overwrite only changed tasks)
            # This happens while lock is still held
            for key, task in self.state.items():
                merged_state[key] = asdict(task)

            # Get version number from existing state (if available) for
            # optimistic locking
            version = (
                existing_data.get('version', 0) if existing_data else 0
            )
            new_version = version + 1

            # ✅ FIX: Test JSON serialization before writing to file
            try:
                json.dumps(merged_state)  # Test serialization
            except Exception as json_err:
                logger.error(
                    f"❌ Error serializing state to JSON: {json_err}"
                )
                raise

            data = {
                # ✅ FIX: Add version number for optimistic locking
                'version': new_version,
                'cycle': self.cycle,
                'state': merged_state,
                'last_updated': datetime.now().isoformat()
            }

            # Atomic write (lock still held)
            # Write to temp file first, then atomic rename
            # ✅ FIX: Use unique temp file name per process to avoid race
            # conditions
            import os as os_module
            process_id = os_module.getpid()
            tmp_path = self.state_file.with_suffix(
                f'.json.tmp.{process_id}'
            )
            try:
                # ✅ FIX: Ensure temp file directory exists and has proper
                # permissions
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                ensure_directory_permissions(tmp_path.parent)

                # ✅ FIX: Write to temp file with explicit error handling
                try:
                    # ✅ DEBUG: Log before attempting to write (use INFO
                    # level so it's visible)
                    logger.info(
                        f"📝 Attempting to write temp file: {tmp_path}"
                    )
                    logger.info(
                        f"📝 Parent dir exists: "
                        f"{tmp_path.parent.exists()}, writable: "
                        f"{os.access(tmp_path.parent, os.W_OK)}"
                    )

                    # ✅ FIX: Test JSON serialization before writing
                    try:
                        test_json_str = json.dumps(data)
                        logger.info(
                            f"📝 JSON serialization test: "
                            f"{len(test_json_str)} bytes"
                        )
                    except Exception as json_err:
                        logger.error(
                            f"❌ JSON serialization failed: {json_err}"
                        )
                        raise

                    # ✅ FIX: Write file step by step with verification
                    file_handle = None
                    try:
                        file_handle = open(tmp_path, 'w')
                        logger.info(
                            f"📝 File handle opened: {tmp_path}"
                        )
                        json.dump(data, file_handle, indent=2)
                        logger.info("📝 JSON dumped to file")
                        file_handle.flush()
                        logger.info("📝 File flushed")
                        os.fsync(file_handle.fileno())
                        logger.info("📝 File synced to disk")
                        file_handle.close()
                        file_handle = None
                        logger.info("📝 File handle closed")
                    except Exception as write_err:
                        if file_handle:
                            try:
                                file_handle.close()
                            except Exception as e2:
                                logger.debug(
                                    f"Failed to cleanup after write error: "
                                    f"{e2}"
                                )
                        logger.error(
                            f"❌ Error during file write: {write_err}"
                        )
                        import traceback
                        logger.error(
                            f"❌ Write error traceback: "
                            f"{''.join(traceback.format_exc())}"
                        )
                        raise

                    # ✅ DEBUG: Verify file exists immediately after write
                    if tmp_path.exists():
                        file_size = tmp_path.stat().st_size
                        logger.info(
                            f"✅ Temp file exists after write: {tmp_path}, "
                            f"size: {file_size}"
                        )
                    else:
                        logger.error(
                            f"❌ Temp file does NOT exist after write: "
                            f"{tmp_path}"
                        )
                except Exception as write_file_err:
                    logger.error(
                        f"❌ Error writing temp file {tmp_path}: "
                        f"{write_file_err}"
                    )
                    import traceback
                    logger.error(
                        f"❌ Write error traceback: "
                        f"{''.join(traceback.format_exc())}"
                    )
                    raise

                # ✅ FIX: Verify temp file was created and has content
                # before renaming
                # Note: File existence was already checked above, but
                # check again in case of race condition
                if not tmp_path.exists():
                    logger.error(
                        f"❌ Temp file does not exist after write attempt: "
                        f"{tmp_path}"
                    )
                    logger.error(
                        f"❌ Parent dir: {tmp_path.parent}, exists: "
                        f"{tmp_path.parent.exists()}"
                    )
                    raise IOError(f"Temp file was not created: {tmp_path}")
                if tmp_path.stat().st_size == 0:
                    logger.error(
                        f"❌ Temp file is empty: {tmp_path}"
                    )
                    raise IOError(f"Temp file is empty: {tmp_path}")

                # ✅ FIX: Atomic rename with error handling
                logger.info(
                    f"📝 Attempting atomic rename: {tmp_path} -> "
                    f"{self.state_file}"
                )
                try:
                    # Atomic rename (lock still held, preventing concurrent
                    # writes)
                    os.replace(tmp_path, self.state_file)
                    logger.info("✅ Atomic rename successful")
                except Exception as rename_err:
                    logger.error(
                        f"❌ Atomic rename failed: {rename_err}"
                    )
                    logger.error(
                        f"❌ Temp file exists: {tmp_path.exists()}, "
                        f"State file exists: {self.state_file.exists()}"
                    )
                    import traceback
                    logger.error(
                        f"❌ Rename error traceback: "
                        f"{''.join(traceback.format_exc())}"
                    )
                    raise
                # ✅ FIX: Ensure file permissions for shared access
                ensure_file_permissions(self.state_file)
            except Exception as write_err:
                logger.error(f"❌ Error writing state file: {write_err}")
                import traceback
                logger.error(f"Traceback: {''.join(traceback.format_exc())}")
                raise  # Re-raise to be caught by outer exception handler
            finally:
                # Clean up temp file if it still exists
                if tmp_path.exists():
                    try:
                        os.remove(tmp_path)
                    except Exception as e:
                        logger.debug(f"Failed to remove tmp_path: {e}")
        except Exception as e:
            logger.error(f"❌ Error saving state: {e}")
        finally:
            # Release lock and close file descriptor
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except Exception as e:
                    logger.debug(f"Failed to unlock lock_fd: {e}")
                try:
                    os.close(lock_fd)
                except Exception as e:
                    logger.debug(f"Failed to close lock_fd: {e}")

    def _reset_stale_in_progress(self):
        """Reset tasks left in 'in_progress' states (resume safety after
        restart)."""
        changed = False
        for key, task in list(self.state.items()):
            if task.status in ('hpo_in_progress', 'training_in_progress'):
                task.status = 'pending'
                task.error = 'Interrupted - resumed after restart'
                # ✅ FIX: Ensure task cycle matches current cycle
                task.cycle = self.cycle
                self.state[key] = task
                changed = True
        if changed:
            self.save_state()
            logger.info("🔄 Reset stale in-progress tasks to pending")

    def get_active_symbols(self) -> List[str]:
        """Get list of active symbols from database"""
        try:
            with app.app_context():
                from models import Stock
                from bist_pattern.utils.symbols import sanitize_symbol
                raw_symbols = [
                    s.symbol for s in Stock.query.filter_by(
                        is_active=True
                    ).order_by(Stock.symbol).all()
                ]

                # ✅ FIX: Clean BOM characters and normalize symbol names
                symbols = []
                seen = set()
                for symbol in raw_symbols:
                    if not symbol:
                        continue
                    # Clean BOM and normalize
                    cleaned = sanitize_symbol(symbol)
                    if cleaned and cleaned not in seen:
                        symbols.append(cleaned)
                        seen.add(cleaned)
                    elif cleaned != symbol:
                        logger.debug(
                            f"🔧 Symbol cleaned: '{symbol}' -> "
                            f"'{cleaned}'"
                        )

                # Filter out invalid symbols
                import re
                # ✅ FIX: Delisted semboller - ABVKS gibi delisted
                # sembolleri denylist'e ekle
                denylist = re.compile(
                    r"USDTR|USDTRY|^XU|^OPX|^F_|VIOP|INDEX|ABVKS",
                    re.IGNORECASE
                )
                symbols = [
                    s for s in symbols
                    if s and not denylist.search(s)
                ]
                logger.info(
                    f"📊 Found {len(symbols)} active symbols (after BOM "
                    f"cleaning and denylist filter)"
                )
                return symbols
        except Exception as e:
            logger.error(f"❌ Error getting active symbols: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_pending_symbols(self) -> List[str]:
        """Get symbols that have pending tasks (not fully completed).
        - Returns symbols that have at least one pending/failed horizon
        - Skips symbols that are already being processed (any horizon in
          progress)
        - Skips symbols that are fully completed for current cycle
        """
        symbols = self.get_active_symbols()
        pending_symbols: List[str] = []

        for symbol in symbols:
            # Check if this symbol has any pending/failed tasks
            has_pending = False
            has_in_progress = False
            all_completed = True

            for horizon in HORIZON_ORDER:
                key = f"{symbol}_{horizon}d"
                task = self.state.get(key)

                if task is None:
                    # New task - pending
                    has_pending = True
                    all_completed = False
                    break

                # Check if any horizon is in progress
                if task.status in (
                    'hpo_in_progress', 'training_in_progress'
                ):
                    has_in_progress = True
                    all_completed = False
                    break

                # Check if task is pending or failed (with retries)
                if task.status == 'pending':
                    has_pending = True
                    all_completed = False
                elif task.status == 'failed' and task.retry_count < 10:
                    has_pending = True
                    all_completed = False
                elif task.status == 'skipped':
                    # Skipped tasks don't count as completed for cycle
                    # purposes
                    all_completed = False
                elif task.status == 'completed':
                    # Only count as completed if it's for current cycle
                    if task.cycle < self.cycle:
                        has_pending = True
                        all_completed = False
                else:
                    all_completed = False

            # Skip if already in progress (another process is handling this
            # symbol)
            if has_in_progress:
                continue

            # Add if has pending tasks and not fully completed
            if has_pending or not all_completed:
                pending_symbols.append(symbol)

        logger.info(
            f"📋 Pending symbols: {len(pending_symbols)} (symbols with "
            f"pending/failed horizons)"
        )
        return pending_symbols

    def get_pending_tasks(self) -> List[Tuple[str, int]]:
        """Get next pending tasks: one horizon per symbol, in order.
        - Picks first pending horizon per symbol according to
          HORIZON_ORDER
        - Skips tasks already in progress/completed for current cycle
        - Includes failed tasks with retries remaining
        - ⚠️ DEPRECATED: Use get_pending_symbols() + process_symbol() for
          sequential processing
        """
        symbols = self.get_active_symbols()
        next_tasks: List[Tuple[str, int]] = []

        for symbol in symbols:
            # Determine next horizon to process for this symbol
            chosen_h: Optional[int] = None
            for horizon in HORIZON_ORDER:
                key = f"{symbol}_{horizon}d"
                task = self.state.get(key)

                if task is None:
                    chosen_h = horizon
                    break
                # Skip if already in progress
                if task.status in ('hpo_in_progress', 'training_in_progress'):
                    chosen_h = None
                    break
                # Completed in previous cycle: schedule in new cycle
                if task.status == 'completed' and task.cycle < self.cycle:
                    chosen_h = horizon
                    break
                # Pending: pick it
                if task.status == 'pending':
                    chosen_h = horizon
                    break
                # Failed with retries left
                if task.status == 'failed':
                    # HPO failed: allow up to 3 retries
                    if task.retry_count < 10:
                        chosen_h = horizon
                        break
                # Skipped: do not pick (insufficient data)
                if task.status == 'skipped':
                    continue

            if chosen_h is not None:
                next_tasks.append((symbol, chosen_h))

        # De-duplicate and limit to a reasonable batch (parent will
        # further cap to MAX_WORKERS)
        # Preserve order by using dict
        unique_ordered = list(dict.fromkeys(next_tasks))
        logger.info(
            f"📋 Next batch candidates: {len(unique_ordered)} (one horizon "
            f"per symbol)"
        )
        return unique_ordered

    def run_hpo(self, symbol: str, horizon: int) -> Optional[Dict]:
        """Run HPO for a symbol-horizon pair"""
        try:
            logger.info(f"🔬 Starting HPO for {symbol} {horizon}d...")

            # ✅ CRITICAL FIX: Record HPO start time for JSON file
            # validation
            import time
            hpo_start_time = time.time()

            # Use subprocess to run HPO script (cleaner isolation)

            # ✅ UPDATED: Use optuna_hpo_with_feature_flags.py for
            # comprehensive optimization
            # This script optimizes: feature flags + feature internal
            # parameters + hyperparameters
            hpo_script = Path(
                '/opt/bist-pattern/scripts/optuna_hpo_with_feature_flags.py'
            )
            if not hpo_script.exists():
                logger.error(f"❌ HPO script not found: {hpo_script}")
                return None

            # Run HPO script (1500 trials for comprehensive optimization)
            # Timeout: 250 hours (10.4 days) - allows for 1500 trials ×
            # ~2-3 min/trial = ~50-75 hours
            # Plus margin for slower trials, feature engineering, etc.
            # ⚡ DRY RUN: If DRY_RUN_TRIALS env var is set, use that
            # instead of HPO_TRIALS
            dry_run_trials = int(os.environ.get('DRY_RUN_TRIALS', '0'))
            trials_to_use = (
                dry_run_trials if dry_run_trials > 0 else HPO_TRIALS
            )
            # 1 hour for dry run, 250h for full
            timeout_to_use = (
                3600 if dry_run_trials > 0 else 900000
            )

            logger.info(
                f"🔬 HPO Configuration: {trials_to_use} trials, "
                f"timeout={timeout_to_use}s "
                f"({'DRY RUN' if dry_run_trials > 0 else 'PRODUCTION'})"
            )

            cmd = [
                sys.executable,
                str(hpo_script),
                '--symbols', symbol,
                '--horizon', str(horizon),
                # 10 for dry run, 1500 for production
                '--trials', str(trials_to_use),
                # 1 hour for dry run, 250 hours for production
                '--timeout', str(timeout_to_use)
            ]

            # Pass DRY_RUN_TRIALS to subprocess environment if set
            env = os.environ.copy()
            if dry_run_trials > 0:
                env['DRY_RUN_TRIALS'] = str(dry_run_trials)
            # ✅ CRITICAL FIX: Pass cycle number to HPO script for study
            # file naming
            # This ensures each cycle gets its own study file, enabling
            # new HPO with new data
            env['HPO_CYCLE'] = str(self.cycle)

            # ✅ Acquire global HPO slot (limits cross-process concurrency)
            # ✅ CRITICAL FIX: Use HPOSlotContext for automatic cleanup
            # This ensures slot is released even if parent process is killed
            slot_fd = None
            slot_context = None
            try:
                slot_context = HPOSlotContext()
                _slot_idx, slot_fd, _slot_path = slot_context.__enter__()
                logger.info(
                    f"🔒 HPO slot acquired for {symbol} {horizon}d"
                )
            except TimeoutError as te:
                logger.error(
                    f"⏱️ Failed to acquire HPO slot for {symbol} "
                    f"{horizon}d: {te}"
                )
                return None

            # ✅ CPU affinity optimization: bind to specific CPU range
            # (round-robin)
            # Simplified: Only CPU affinity, no NUMA binding (Python/ML not
            # NUMA-aware)
            numa_node, cpu_list = _get_numa_node_and_cpus()
            numa_cmd, _, _ = _build_numa_cmd(cmd, numa_node, cpu_list)
            # Log CPU affinity if applied
            if numa_cmd != cmd:
                logger.info(
                    f"🔗 CPU affinity: {cpu_list} (node {numa_node})"
                )

            # Set process priority (higher priority for HPO)
            try:
                # Set nice value to -5 (higher priority)
                os.nice(-5)
            except (OSError, PermissionError):
                # May require root, ignore if fails
                pass

            # Run subprocess regardless of whether setting priority
            # succeeded or failed
            # ✅ UPDATED: Use Popen instead of run() for graceful shutdown
            # support
            result: Optional[subprocess.CompletedProcess[str]] = None
            process_key = f"{symbol}_{horizon}d"
            process: Optional[subprocess.Popen] = None

            # ✅ CRITICAL FIX: Write output to files instead of PIPE to
            # prevent buffer overflow
            # HPO script can run for hours and produce thousands of log
            # lines
            # PIPE buffer (64KB) can overflow, causing process to block
            # Writing to files prevents this issue and allows real-time
            # log monitoring
            hpo_log_dir = Path('/opt/bist-pattern/logs/hpo_outputs')
            hpo_log_dir.mkdir(exist_ok=True, parents=True)
            ensure_directory_permissions(hpo_log_dir)

            stdout_file = hpo_log_dir / f"{symbol}_{horizon}d_stdout.log"
            stderr_file = hpo_log_dir / f"{symbol}_{horizon}d_stderr.log"

            try:
                # Start process with Popen for graceful shutdown support
                # ✅ CRITICAL FIX: Use start_new_session=True to make child
                # process independent
                # This prevents child processes from being killed when
                # parent is killed
                # Child processes will continue running even if parent
                # process is terminated
                # ✅ CRITICAL FIX: Write output to files instead of PIPE
                # to prevent buffer overflow
                with open(stdout_file, 'w', buffering=1) as stdout_f, \
                     open(stderr_file, 'w', buffering=1) as stderr_f:
                    process = subprocess.Popen(
                        numa_cmd,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        text=True,
                        cwd='/opt/bist-pattern',
                        # Pass environment variables including
                        # DRY_RUN_TRIALS
                        env=env,
                        # ✅ CRITICAL: Create new process session (child
                        # independent from parent)
                        start_new_session=True,
                        preexec_fn=(
                            lambda: os.nice(-5)
                            if hasattr(os, 'nice') else None
                        )
                    )

                    # Register process for graceful shutdown
                    with self._hpo_process_lock:
                        self.active_hpo_processes[process_key] = process

                    # ✅ CRITICAL: Save PID to file for tracking (works even
                    # with start_new_session=True)
                    pid_file = self.hpo_pid_dir / f"{process_key}.pid"
                    try:
                        with open(pid_file, 'w') as f:
                            f.write(str(process.pid))
                        ensure_file_permissions(pid_file)
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to save PID file for "
                            f"{process_key}: {e}"
                        )

                    logger.info(
                        f"🔬 HPO process started for {symbol} {horizon}d "
                        f"(PID: {process.pid})"
                    )
                    logger.info(
                        f"📝 HPO output files: stdout={stdout_file.name}, "
                        f"stderr={stderr_file.name}"
                    )

                    # Wait for process to complete (with timeout)
                    # ✅ CRITICAL FIX: Output is written to files, so
                    # wait() just waits for process
                    # No buffer overflow risk, no memory issues
                    try:
                        returncode = process.wait(timeout=timeout_to_use)
                        # Read output from files for error analysis
                        stdout_content = ''
                        stderr_content = ''
                        try:
                            if stdout_file.exists():
                                with open(stdout_file, 'r') as f:
                                    stdout_content = f.read()
                            if stderr_file.exists():
                                with open(stderr_file, 'r') as f:
                                    stderr_content = f.read()
                        except Exception as e:
                            logger.debug(
                                f"Failed to read stderr file: {e}"
                            )

                        result = subprocess.CompletedProcess(
                            numa_cmd,
                            returncode,
                            stdout=stdout_content,
                            stderr=stderr_content
                        )
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            f"⏱️ HPO timeout for {symbol} {horizon}d, "
                            f"terminating process..."
                        )
                        process.terminate()
                        try:
                            # Wait 30s for graceful termination
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            logger.warning(
                                "⚠️ HPO process didn't terminate "
                                "gracefully, force killing..."
                            )
                            process.kill()
                            process.wait()

                        # Read output from files even after timeout
                        stdout_content = ''
                        stderr_content = ''
                        try:
                            if stdout_file.exists():
                                with open(stdout_file, 'r') as f:
                                    stdout_content = f.read()
                            if stderr_file.exists():
                                with open(stderr_file, 'r') as f:
                                    stderr_content = f.read()
                        except Exception as e:
                            logger.debug(
                                f"Failed to read stderr file: {e}"
                            )

                        result = subprocess.CompletedProcess(
                            numa_cmd,
                            -15,  # SIGTERM
                            stdout=stdout_content,
                            stderr=stderr_content
                        )
            except Exception as e:
                logger.error(
                    f"❌ Failed to run HPO subprocess for {symbol} "
                    f"{horizon}d: {e}"
                )
                result = None
            finally:
                # Remove from active processes
                with self._hpo_process_lock:
                    self.active_hpo_processes.pop(process_key, None)

                # ✅ Remove PID file
                pid_file = self.hpo_pid_dir / f"{process_key}.pid"
                if pid_file.exists():
                    try:
                        pid_file.unlink()
                    except Exception as e:
                        logger.debug(
                            f"Failed to remove PID file {pid_file}: {e}"
                        )

                # ✅ CRITICAL FIX: Use context manager for slot release
                # This ensures slot is released even if exception occurs
                if slot_context is not None:
                    try:
                        slot_context.__exit__(None, None, None)
                        logger.info(
                            f"🔓 HPO slot released for {symbol} {horizon}d "
                            f"(via context manager)"
                        )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to release slot via context "
                            f"manager: {e}"
                        )
                        # Fallback to manual release
                        if slot_fd is not None:
                            release_hpo_slot(slot_fd)
                            logger.info(
                                f"🔓 HPO slot released for {symbol} "
                                f"{horizon}d (manual fallback)"
                            )
                elif slot_fd is not None:
                    release_hpo_slot(slot_fd)
                    logger.info(
                        f"🔓 HPO slot released for {symbol} {horizon}d"
                    )

            # Check if subprocess completed successfully
            if result is None:
                logger.error(
                    f"❌ HPO subprocess failed to start for {symbol} "
                    f"{horizon}d"
                )
                return None

            # ✅ FIX: Log subprocess output for debugging (use INFO level
            # to ensure visibility)
            # ⚡ CRITICAL: Filter out web app logs (pattern_detector,
            # unified_collector, etc.)
            # Only log HPO-specific messages ([hpo], Starting HPO, HPO
            # completed, etc.)
            if result.stdout:
                # Filter stdout: only keep HPO-related messages
                filtered_stdout = '\n'.join([
                    line for line in result.stdout.split('\n')
                    if any(keyword in line for keyword in [
                        '[hpo]', 'Starting HPO', 'HPO completed', 'Trial',
                        'Best trial',
                        'DirHit', 'Training completed', 'model in memory',
                        'predict_enhanced'
                    ])
                ])
                if filtered_stdout.strip():
                    logger.info(
                        f"HPO stdout for {symbol} {horizon}d:\n"
                        f"{filtered_stdout[:10000]}"
                    )

            if result.stderr:
                # Filter stderr: only keep HPO-related messages
                filtered_stderr = '\n'.join([
                    line for line in result.stderr.split('\n')
                    if any(keyword in line for keyword in [
                        '[hpo]', 'Starting HPO', 'HPO completed', 'Trial',
                        'Best trial',
                        'DirHit', 'Training completed', 'model in memory',
                        'predict_enhanced',
                        'ERROR', 'WARNING'  # Keep errors and warnings
                    ]) and not any(skip in line for skip in [
                        'pattern_detector', 'unified_collector',
                        'broadcaster',
                        'No news items', 'TA-Lib detected',
                        'Pattern Validation'
                    ])
                ])
                if filtered_stderr.strip():
                    logger.info(
                        f"HPO stderr for {symbol} {horizon}d:\n"
                        f"{filtered_stderr[:10000]}"
                    )

            if result.returncode != 0:
                error_details = []
                if result.stderr:
                    error_details.append(
                        f"stderr: {result.stderr[:500]}"
                    )
                if result.stdout:
                    # Look for error messages in stdout
                    error_lines = [
                        line for line in result.stdout.split('\n')
                        if any(
                            keyword in line.lower()
                            for keyword in [
                                'error', 'failed', 'exception', 'traceback'
                            ]
                        )
                    ]
                    if error_lines:
                        error_details.append(
                            f"stdout errors: "
                            f"{'; '.join(error_lines[:5])}"
                        )

                # ✅ IMPROVED: More descriptive error message for SIGTERM
                if result.returncode == -15:
                    error_msg = (
                        "HPO script terminated by SIGTERM (exit code -15) "
                        "- likely due to service restart or parent process "
                        "kill"
                    )
                elif result.returncode == -9:
                    error_msg = (
                        "HPO script killed by SIGKILL (exit code -9) "
                        "- force kill"
                    )
                else:
                    error_msg = (
                        f"HPO script failed (exit code {result.returncode})"
                    )

                if error_details:
                    error_msg += f" - {', '.join(error_details)}"

                logger.error(
                    f"❌ HPO script failed for {symbol} {horizon}d: "
                    f"{error_msg}"
                )
                # If HPO was terminated by SIGTERM/SIGKILL (likely service
                # restart), mark as retryable
                if result.returncode in (-15, -9):
                    return {
                        'retryable': True,
                        'error': error_msg,
                        'returncode': result.returncode,
                    }
                return None

            # ✅ CRITICAL FIX: Find the generated JSON file with cycle
            # number
            # JSON format:
            # optuna_pilot_features_on_h{horizon}_c{cycle}_{timestamp}.json
            # Also check legacy format (no cycle) for backward
            # compatibility
            json_patterns = [
                # Current format with cycle
                f'optuna_pilot_features_on_h{horizon}_c{self.cycle}_*.json',
                # Legacy format (fallback)
                f'optuna_pilot_features_on_h{horizon}_*.json'
            ]
            json_files = []
            for pattern in json_patterns:
                found_files = sorted(
                    Path('/opt/bist-pattern/results').glob(pattern),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                json_files.extend(found_files)
                # If we found files with cycle number, don't use legacy
                # format
                if pattern.startswith(
                    f'optuna_pilot_features_on_h{horizon}_c{self.cycle}_'
                ):
                    if found_files:
                        # Use only cycle-specific files
                        json_files = found_files
                        break
            # Remove duplicates and sort by mtime
            json_files = sorted(
                set(json_files),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            if not json_files:
                # ✅ CRITICAL FIX: If JSON file not found, try to recover
                # from study file
                # This handles cases where JSON file creation failed but
                # study has 1500+ trials
                logger.warning(
                    f"⚠️ No HPO JSON file found for {symbol} {horizon}d, "
                    f"attempting recovery from study file..."
                )

                # Try to find study file
                study_dirs = [
                    Path('/opt/bist-pattern/results/optuna_studies'),
                    Path('/opt/bist-pattern/hpo_studies'),
                ]
                study_file = None
                for study_dir in study_dirs:
                    if not study_dir.exists():
                        continue
                    # Check current cycle format first
                    cycle_file = (
                        study_dir /
                        f"hpo_with_features_{symbol}_h{horizon}_c"
                        f"{self.cycle}.db"
                    )
                    if cycle_file.exists():
                        study_file = cycle_file
                        break
                    # Check legacy format (only for cycle 1)
                    if self.cycle == 1:
                        legacy_file = (
                            study_dir /
                            f"hpo_with_features_{symbol}_h{horizon}.db"
                        )
                        if legacy_file.exists():
                            study_file = legacy_file
                            break

                if study_file and study_file.exists():
                    try:
                        import optuna
                        study_recovered = optuna.load_study(
                            study_name=None,
                            storage=f"sqlite:///{study_file}"
                        )
                        complete_trials = len([
                            t for t in study_recovered.trials
                            if t.state == optuna.trial.TrialState.COMPLETE
                        ])

                        # ✅ CRITICAL: Only recover if HPO is truly
                        # completed (MIN_TRIALS_FOR_RECOVERY+ trials)
                        # We need HPO_TRIALS trials to find best
                        # parameters - partial progress is not enough
                        # If HPO is incomplete, it should continue from
                        # where it left off (warm-start)
                        if complete_trials >= MIN_TRIALS_FOR_RECOVERY:
                            logger.info(
                                f"✅ Found completed HPO in study file "
                                f"({complete_trials} trials), creating JSON "
                                f"file from study..."
                            )

                            # Create JSON file from study
                            best_trial = study_recovered.best_trial
                            best_params = study_recovered.best_params
                            best_value = (
                                float(study_recovered.best_value)
                                if study_recovered.best_value is not None
                                else 0.0
                            )

                            # Get best_dirhit from user_attrs
                            best_dirhit = None
                            try:
                                _val = best_trial.user_attrs.get(
                                    'avg_dirhit', None
                                )
                                if (
                                    isinstance(_val, (int, float))
                                    and np.isfinite(_val)
                                ):
                                    best_dirhit = float(_val)
                            except Exception as e:
                                logger.debug(
                                    f"Failed to get best_dirhit from trial: "
                                    f"{e}"
                                )
                            if (
                                best_dirhit is None
                                or not np.isfinite(best_dirhit)
                            ):
                                best_dirhit = best_value

                            # Extract feature flags, feature parameters,
                            # and hyperparameters
                            feature_flags = {
                                k: v for k, v in best_params.items()
                                if k.startswith('enable_')
                            }
                            feature_params_keys = [
                                'ml_loss_mse_weight', 'ml_loss_threshold',
                                'ml_dir_penalty', 'n_seeds',
                                'meta_stacking_alpha', 'yolo_min_conf',
                                'smart_consensus_weight',
                                'smart_performance_weight', 'smart_sigma',
                                'fingpt_confidence_threshold',
                                'external_min_days',
                                'external_smooth_alpha',
                                'regime_scale_low', 'regime_scale_high',
                            ]
                            feature_params_keys += [
                                k for k in best_params.keys()
                                if (
                                    k.startswith('ml_adaptive_k_')
                                    or k.startswith(
                                        'ml_pattern_weight_scale_'
                                    )
                                )
                            ]
                            feature_params = {
                                k: v for k, v in best_params.items()
                                if k in feature_params_keys
                            }
                            hyperparameters = {
                                k: v for k, v in best_params.items()
                                if (
                                    not k.startswith('enable_')
                                    and k not in feature_params_keys
                                )
                            }

                            # Create JSON result
                            result_recovered = {
                                'best_value': best_value,
                                'best_dirhit': best_dirhit,
                                'best_params': best_params,
                                'best_trial': {
                                    'number': best_trial.number,
                                    'value': (
                                        float(best_trial.value)
                                        if best_trial.value is not None
                                        else 0.0
                                    ),
                                    'state': str(best_trial.state),
                                },
                                'n_trials': len(study_recovered.trials),
                                'study_name': study_recovered.study_name,
                                'symbols': [symbol],
                                'horizon': horizon,
                                'best_model_choice': best_params.get(
                                    'model_choice'
                                ),
                                'feature_flags': feature_flags,
                                'feature_params': feature_params,
                                'hyperparameters': hyperparameters,
                                'features_enabled': {
                                    'ENABLE_XGBOOST': (
                                        '1' if best_params.get(
                                            'model_choice'
                                        ) in ('xgb', 'all') else '0'
                                    ),
                                    'ENABLE_LIGHTGBM': (
                                        '1' if best_params.get(
                                            'model_choice'
                                        ) in ('lgbm', 'all') else '0'
                                    ),
                                    'ENABLE_CATBOOST': (
                                        '1' if best_params.get(
                                            'model_choice'
                                        ) in ('cat', 'all') else '0'
                                    ),
                                }
                            }

                            # Save recovered JSON file
                            recovered_json_file = (
                                f"/opt/bist-pattern/results/"
                                f"optuna_pilot_features_on_h{horizon}_c"
                                f"{self.cycle}_"
                                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                f"_recovered.json"
                            )
                            with open(recovered_json_file, 'w') as f:
                                json.dump(result_recovered, f, indent=2)

                            logger.info(
                                f"✅ Recovered JSON file created: "
                                f"{recovered_json_file}"
                            )

                            # Return recovered result
                            return {
                                'best_value': best_value,
                                'best_dirhit': best_dirhit,
                                'best_params': best_params,
                                'best_trial_number': best_trial.number,
                                'json_file': recovered_json_file,
                                'n_trials': len(study_recovered.trials),
                                'features_enabled': (
                                    result_recovered['features_enabled']
                                ),
                                'feature_params': feature_params,
                                'feature_flags': feature_flags,
                                'hyperparameters': hyperparameters
                            }
                        else:
                            logger.warning(
                                f"⚠️ Study file found but only "
                                f"{complete_trials} trials completed "
                                f"(need {MIN_TRIALS_FOR_RECOVERY}+ for "
                                f"recovery, will continue HPO from where "
                                f"it left off)"
                            )
                    except Exception as recover_err:
                        logger.warning(
                            f"⚠️ Failed to recover from study file: "
                            f"{recover_err}"
                        )

                error_msg = (
                    f"No HPO JSON file found for {symbol} {horizon}d"
                )
                logger.warning(f"⚠️ {error_msg}")
                # Store error for retry analysis
                return {'error': error_msg}

            # ✅ CRITICAL FIX: Check recent files for our symbol with
            # timestamp validation
            # Also verify that the HPO actually completed with sufficient
            # trials
            # Required keys for validation
            required_keys = [
                'best_params', 'best_value', 'symbols', 'horizon', 'n_trials'
            ]

            # ✅ FIX: Collect all valid JSON files, then select the best
            # one (by DirHit or best_value)
            # This handles cases where multiple valid JSON files exist for
            # the same symbol-horizon
            valid_json_candidates = []

            for json_file in json_files[:50]:
                try:
                    # ✅ CRITICAL FIX: Verify JSON file was created after
                    # HPO started
                    json_mtime = json_file.stat().st_mtime

                    # ✅ FIX: First check if JSON file contains valid HPO
                    # result (symbol, horizon, trials)
                    # If valid, we can use it even if timestamp is outside
                    # normal range (handles restart cases)
                    json_file_valid = False
                    try:
                        with open(json_file, 'r') as f:
                            hpo_data_precheck = json.load(f)
                            # Quick validation: check if this JSON is for
                            # our symbol-horizon
                            # with sufficient trials
                            if (
                                isinstance(hpo_data_precheck, dict)
                                and symbol in hpo_data_precheck.get(
                                    'symbols', []
                                )
                                and hpo_data_precheck.get('horizon') == horizon
                                and isinstance(
                                    hpo_data_precheck.get('n_trials', 0),
                                    int
                                )
                                # HPO must be completed
                                and hpo_data_precheck.get(
                                    'n_trials', 0
                                ) >= MIN_TRIALS_FOR_RECOVERY
                            ):
                                json_file_valid = True
                    except Exception as e:
                        logger.debug(
                            f"Failed to validate HPO JSON file: {e}"
                        )

                    # ✅ FIX: Timestamp validation - flexible for valid
                    # JSON files (handles restart cases)
                    # If JSON file is valid (correct symbol/horizon/trials),
                    # use relaxed timestamp validation
                    # This handles cases where HPO completed before restart
                    # but state file wasn't updated
                    if json_file_valid:
                        # ✅ SECURITY FIX: Even for valid JSON, check if
                        # it's not too old (max 7 days)
                        # This prevents using very old JSON files from
                        # previous cycles
                        current_time = time.time()
                        max_age_seconds = 7 * 24 * 3600  # 7 days
                        if json_mtime < current_time - max_age_seconds:
                            logger.debug(
                                f"⚠️ HPO JSON file {json_file.name} is too "
                                f"old (created "
                                f"{(current_time - json_mtime)/86400:.1f} "
                                f"days ago), skipping even though valid"
                            )
                            continue
                        # For valid JSON files, allow wider timestamp range
                        # (up to 7 days old, or up to 96 hours after HPO
                        # start)
                        # 96 hours (4 days) after HPO start
                        if json_mtime > hpo_start_time + 345600:
                            logger.debug(
                                f"⚠️ HPO JSON file {json_file.name} is too "
                                f"new (created "
                                f"{json_mtime - hpo_start_time:.0f}s after "
                                f"HPO start), skipping"
                            )
                            continue
                    else:
                        # Only do strict timestamp validation if JSON file
                        # is not clearly valid
                        # Allow 5 minutes tolerance before HPO start (in
                        # case of clock skew)
                        if json_mtime < hpo_start_time - 300:
                            logger.debug(
                                f"⚠️ HPO JSON file {json_file.name} is too "
                                f"old (created "
                                f"{time.time() - json_mtime:.0f}s before "
                                f"HPO start), skipping"
                            )
                            continue
                        # For invalid JSON files, also check if too new
                        # (strict validation)
                        # HPO can take up to 72 hours for 1500 trials, so
                        # allow up to 96 hours (4 days) after start
                        # 96 hours (4 days) - increased from 18 hours
                        if json_mtime > hpo_start_time + 345600:
                            logger.debug(
                                f"⚠️ HPO JSON file {json_file.name} is too "
                                f"new (created "
                                f"{json_mtime - hpo_start_time:.0f}s after "
                                f"HPO start), skipping"
                            )
                            continue

                    with open(json_file, 'r') as f:
                        hpo_data = json.load(f)

                    # ✅ CRITICAL FIX: Validate JSON structure
                    if not isinstance(hpo_data, dict):
                        logger.warning(
                            f"⚠️ Invalid HPO JSON structure in "
                            f"{json_file.name}: not a dict"
                        )
                        continue

                    # Check required keys
                    missing_keys = [
                        key for key in required_keys
                        if key not in hpo_data
                    ]
                    if missing_keys:
                        logger.warning(
                            f"⚠️ Invalid HPO JSON: {json_file.name} "
                            f"missing required keys: {missing_keys}"
                        )
                        continue

                    hpo_symbols = hpo_data.get('symbols', [])
                    if not isinstance(hpo_symbols, list):
                        logger.warning(
                            f"⚠️ Invalid HPO JSON: {json_file.name} "
                            f"symbols is not a list"
                        )
                        continue

                    if symbol not in hpo_symbols:
                        continue

                    # ✅ FIX: Verify HPO actually completed with sufficient
                    # trials
                    n_trials = hpo_data.get('n_trials', 0)
                    if not isinstance(n_trials, int) or n_trials < 10:
                        logger.warning(
                            f"⚠️ HPO for {symbol} {horizon}d completed "
                            f"with only {n_trials} trials "
                            f"(expected {HPO_TRIALS}), rejecting result"
                        )
                        continue

                    # ✅ FIX: Validate best_params
                    best_params = hpo_data.get('best_params', {})
                    if (
                        not isinstance(best_params, dict)
                        or len(best_params) == 0
                    ):
                        logger.warning(
                            f"⚠️ Invalid HPO JSON: {json_file.name} "
                            f"best_params is empty or not a dict"
                        )
                        continue

                    # ✅ FIX: Validate horizon matches
                    hpo_horizon = hpo_data.get('horizon', None)
                    if hpo_horizon != horizon:
                        logger.warning(
                            f"⚠️ HPO JSON horizon mismatch: "
                            f"{json_file.name} has horizon {hpo_horizon}, "
                            f"expected {horizon}"
                        )
                        continue

                    # ✅ CRITICAL FIX: Use best_dirhit if available,
                    # otherwise fallback to best_value
                    # Method 2 returns score (DirHit * mask_after_thr / 100),
                    # not DirHit
                    best_dirhit = hpo_data.get('best_dirhit')
                    # Renamed to avoid conflict
                    candidate_best_score = hpo_data.get('best_value', 0)

                    # ✅ FIX: Collect this valid JSON as a candidate (don't
                    # return immediately)
                    # We'll select the best one among all candidates
                    best_trial_number = None
                    best_trial_info = hpo_data.get('best_trial', {})
                    if isinstance(best_trial_info, dict):
                        best_trial_number = best_trial_info.get('number')

                    # ✅ CRITICAL FIX: Check LOW SUPPORT for this candidate
                    # We need to filter out spurious results BEFORE selecting
                    # best
                    has_low_support = False
                    total_mask_count = 0
                    avg_mask_pct = 0.0
                    try:
                        best_trial_metrics = hpo_data.get(
                            'best_trial_metrics', {}
                        )
                        symbol_key_check = f"{symbol}_{horizon}d"
                        if symbol_key_check in best_trial_metrics:
                            symbol_metrics = best_trial_metrics[
                                symbol_key_check
                            ]
                            split_metrics = symbol_metrics.get(
                                'split_metrics', []
                            )
                            if split_metrics:
                                total_mask_count = sum(
                                    s.get('mask_count', 0)
                                    for s in split_metrics
                                )
                                mask_pcts = [
                                    s.get('mask_pct', 0.0)
                                    for s in split_metrics
                                    if s.get('mask_pct') is not None
                                ]
                                avg_mask_pct = (
                                    np.mean(mask_pcts)
                                    if mask_pcts else 0.0
                                )

                                # ✅ FIX: Use same thresholds as HPO (from
                                # environment variables)
                                # Default: 0/0.0 (same as HPO code default) -
                                # environment variable will override
                                try:
                                    # Default: 0 (same as HPO)
                                    _min_mc = int(
                                        os.getenv('HPO_MIN_MASK_COUNT', '0')
                                    )
                                except Exception as e:
                                    logger.debug(
                                        f"Failed to get HPO_MIN_MASK_COUNT, "
                                        f"using 0: {e}"
                                    )
                                    _min_mc = 0
                                try:
                                    # Default: 0.0 (same as HPO)
                                    _min_mp = float(
                                        os.getenv('HPO_MIN_MASK_PCT', '0.0')
                                    )
                                except Exception as e:
                                    logger.debug(
                                        f"Failed to get HPO_MIN_MASK_PCT, "
                                        f"using 0.0: {e}"
                                    )
                                    _min_mp = 0.0

                                if (
                                    total_mask_count < _min_mc
                                    or avg_mask_pct < _min_mp
                                ):
                                    has_low_support = True
                                    logger.debug(
                                        "⚠️ %s %sd: Candidate JSON %s has "
                                        "LOW SUPPORT "
                                        "(mask_count=%s, mask_pct=%.1f%%), "
                                        "will be deprioritized",
                                        symbol,
                                        horizon,
                                        json_file.name,
                                        total_mask_count,
                                        avg_mask_pct,
                                    )
                    except Exception as e:
                        # Non-critical check, continue if it fails
                        logger.debug(
                            f"Could not check low support for candidate "
                            f"{json_file.name}: {e}"
                        )

                    valid_json_candidates.append({
                        'json_file': json_file,
                        'json_mtime': json_mtime,
                        'best_value': hpo_data.get('best_value', 0),
                        'best_dirhit': best_dirhit,
                        'best_mask_after_thr': hpo_data.get(
                            'best_mask_after_thr'
                        ),
                        'best_params': hpo_data.get('best_params', {}),
                        'best_trial_number': best_trial_number,
                        'n_trials': n_trials,
                        'features_enabled': hpo_data.get(
                            'features_enabled', {}
                        ),
                        'feature_params': hpo_data.get(
                            'feature_params', {}
                        ),
                        'feature_flags': hpo_data.get('feature_flags', {}),
                        'hyperparameters': hpo_data.get(
                            'hyperparameters', {}
                        ),
                        # ✅ NEW: Track LOW SUPPORT status
                        'has_low_support': has_low_support,
                        # ✅ NEW: Track mask_count for logging
                        'mask_count': total_mask_count,
                        # ✅ NEW: Track mask_pct for logging
                        'mask_pct': avg_mask_pct
                    })

                    if best_dirhit is not None:
                        logger.debug(
                            f"📋 Candidate JSON for {symbol} {horizon}d: "
                            f"{json_file.name} - DirHit = "
                            f"{best_dirhit:.2f}% (score = "
                            f"{candidate_best_score:.2f}, {n_trials} trials)"
                        )
                    else:
                        logger.debug(
                            f"📋 Candidate JSON for {symbol} {horizon}d: "
                            f"{json_file.name} - Score = "
                            f"{candidate_best_score:.2f} "
                            f"(DirHit not available, {n_trials} trials)"
                        )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Error reading JSON file {json_file}: {e}"
                    )
                    continue

            # ✅ FIX: Select the best JSON from all valid candidates
            # ✅ CRITICAL FIX: Prioritize candidates WITHOUT LOW SUPPORT
            # Priority: 1) No LOW SUPPORT, 2) Highest DirHit, 3) Highest
            # best_value, 4) Most recent
            if valid_json_candidates:
                # First, separate candidates by LOW SUPPORT status
                candidates_with_support = [
                    c for c in valid_json_candidates
                    if not c.get('has_low_support', False)
                ]
                candidates_low_support = [
                    c for c in valid_json_candidates
                    if c.get('has_low_support', False)
                ]

                # ✅ CRITICAL: Prefer candidates with sufficient support
                if candidates_with_support:
                    # Sort candidates WITH support by: DirHit (desc), then
                    # best_value (desc), then mtime (desc)
                    candidates_with_support.sort(
                        key=lambda x: (
                            # DirHit (higher is better)
                            x['best_dirhit']
                            if x['best_dirhit'] is not None else -1,
                            x['best_value'],  # best_value (higher is better)
                            x['json_mtime']  # Most recent
                        ),
                        reverse=True
                    )
                    best_candidate = candidates_with_support[0]

                    # Log if we're skipping LOW SUPPORT candidates
                    if candidates_low_support:
                        logger.info(
                            f"✅ {symbol} {horizon}d: Filtered out "
                            f"{len(candidates_low_support)} LOW SUPPORT "
                            f"candidate(s), selected best with sufficient "
                            f"support: {best_candidate['json_file'].name}"
                        )
                else:
                    # ⚠️ FALLBACK: No candidates with sufficient support,
                    # but we still want to train the model
                    # Use best LOW SUPPORT candidate, but mark HPO DirHit
                    # as unreliable
                    # Training DirHit will be more reliable since it uses
                    # more data
                    logger.warning(
                        f"⚠️ {symbol} {horizon}d: WARNING - No HPO "
                        f"candidates with sufficient support found! "
                        f"All {len(candidates_low_support)} candidate(s) "
                        f"have LOW SUPPORT. "
                        f"Falling back to best LOW SUPPORT candidate. "
                        f"HPO DirHit may be unreliable, but Training "
                        f"DirHit will be more reliable."
                    )

                    # Sort LOW SUPPORT candidates by: DirHit (desc), then
                    # best_value (desc), then mtime (desc)
                    candidates_low_support.sort(
                        key=lambda x: (
                            # DirHit (higher is better)
                            x['best_dirhit']
                            if x['best_dirhit'] is not None else -1,
                            x['best_value'],  # best_value (higher is better)
                            x['json_mtime']  # Most recent
                        ),
                        reverse=True
                    )
                    best_candidate = candidates_low_support[0]

                    # Log details about selected LOW SUPPORT candidate
                    logger.warning(
                        f"⚠️ {symbol} {horizon}d: Selected LOW SUPPORT "
                        f"candidate: {best_candidate['json_file'].name} "
                        f"(mask_count={best_candidate.get('mask_count', 0)}, "
                        f"mask_pct={best_candidate.get('mask_pct', 0.0):.1f}%)"
                    )
                    best_dirhit_val = best_candidate.get('best_dirhit')
                    if best_dirhit_val is not None:
                        logger.warning(
                            f"⚠️ {symbol} {horizon}d: HPO DirHit="
                            f"{best_dirhit_val:.2f}% may be unreliable. "
                            f"Training DirHit will be more reliable."
                        )
                    else:
                        logger.warning(
                            f"⚠️ {symbol} {horizon}d: HPO DirHit not "
                            f"available (LOW SUPPORT). "
                            f"Training DirHit will be more reliable."
                        )

                    # Mark this result as having LOW SUPPORT for later
                    # reference
                    # We'll add a flag to the result dict
                    best_candidate['has_low_support_warning'] = True

                if len(valid_json_candidates) > 1:
                    logger.info(
                        f"⚠️ Found {len(valid_json_candidates)} valid JSON "
                        f"files for {symbol} {horizon}d, selecting best one:"
                    )
                    for i, candidate in enumerate(
                        valid_json_candidates[:3]
                    ):  # Show top 3
                        dirhit_str = (
                            f"DirHit={candidate['best_dirhit']:.2f}%"
                            if candidate['best_dirhit'] is not None
                            else f"Score={candidate['best_value']:.2f}"
                        )
                        logger.info(
                            f"   {i+1}. {candidate['json_file'].name}: "
                            f"{dirhit_str}"
                        )

                best_dirhit = best_candidate['best_dirhit']
                best_score = best_candidate['best_value']

                # ✅ NEW: Validate low support issue in best_trial_metrics
                # Check if best_dirhit is spurious due to very few
                # significant predictions
                try:
                    json_file_path = best_candidate['json_file']
                    with open(json_file_path, 'r') as f:
                        hpo_data_check = json.load(f)
                    best_trial_metrics = hpo_data_check.get(
                        'best_trial_metrics', {}
                    )
                    symbol_key = f"{symbol}_{horizon}d"
                    if symbol_key in best_trial_metrics:
                        symbol_metrics = best_trial_metrics[symbol_key]
                        split_metrics = symbol_metrics.get(
                            'split_metrics', []
                        )
                        if split_metrics:
                            total_mask_count = sum(
                                s.get('mask_count', 0)
                                for s in split_metrics
                            )
                            avg_mask_pct = np.mean([
                                s.get('mask_pct', 0.0)
                                for s in split_metrics
                                if s.get('mask_pct') is not None
                            ])

                            # ✅ FIX: Use same thresholds as HPO (from
                            # environment variables)
                            # This ensures consistency between HPO filtering
                            # and validation
                            # Default: 0/0.0 (same as HPO code default) -
                            # environment variable will override
                            try:
                                # Default: 0 (same as HPO)
                                _min_mc = int(
                                    os.getenv('HPO_MIN_MASK_COUNT', '0')
                                )
                            except Exception as e:
                                logger.debug(
                                    f"Failed to get HPO_MIN_MASK_COUNT, "
                                    f"using 0: {e}"
                                )
                                _min_mc = 0
                            try:
                                # Default: 0.0 (same as HPO)
                                _min_mp = float(
                                    os.getenv('HPO_MIN_MASK_PCT', '0.0')
                                )
                            except Exception as e:
                                logger.debug(
                                    f"Failed to get HPO_MIN_MASK_PCT, "
                                    f"using 0.0: {e}"
                                )
                                _min_mp = 0.0

                            # Low support threshold: use same as HPO
                            if (
                                total_mask_count < _min_mc
                                or avg_mask_pct < _min_mp
                            ):
                                logger.warning(
                                    f"⚠️ {symbol} {horizon}d: LOW SUPPORT "
                                    f"detected in best HPO trial! "
                                    f"DirHit={best_dirhit:.2f}% but "
                                    f"mask_count={total_mask_count} "
                                    f"(min={_min_mc}), "
                                    f"mask_pct={avg_mask_pct:.1f}% "
                                    f"(min={_min_mp}%). "
                                    f"This may be a spurious result. "
                                    f"Training DirHit will be more reliable."
                                )
                except Exception as e:
                    # Non-critical check, just log warning if it fails
                    logger.debug(
                        f"Could not validate low support for {symbol} "
                        f"{horizon}d: {e}"
                    )

                if best_dirhit is not None:
                    json_name = best_candidate['json_file'].name
                    logger.info(
                        f"✅ Selected best HPO result for {symbol} "
                        f"{horizon}d: "
                        f"DirHit = {best_dirhit:.2f}% "
                        f"(score = {best_score:.2f}, "
                        f"{best_candidate['n_trials']} trials) "
                        f"from {json_name}"
                    )
                else:
                    json_name = best_candidate['json_file'].name
                    logger.info(
                        f"✅ Selected best HPO result for {symbol} "
                        f"{horizon}d: "
                        f"Score = {best_score:.2f} "
                        f"(DirHit not available, "
                        f"{best_candidate['n_trials']} trials) "
                        f"from {json_name}"
                    )

                return {
                    'best_value': best_candidate['best_value'],
                    'best_dirhit': best_candidate['best_dirhit'],
                    'best_mask_after_thr': best_candidate.get(
                        'best_mask_after_thr'
                    ),
                    'best_params': best_candidate['best_params'],
                    'best_trial_number': (
                        best_candidate['best_trial_number']
                    ),
                    'json_file': str(best_candidate['json_file']),
                    'n_trials': best_candidate['n_trials'],
                    'features_enabled': best_candidate['features_enabled'],
                    'feature_params': best_candidate['feature_params'],
                    'feature_flags': best_candidate['feature_flags'],
                    'hyperparameters': best_candidate['hyperparameters'],
                    # ✅ NEW: Flag for LOW SUPPORT warning
                    'has_low_support_warning': best_candidate.get(
                        'has_low_support_warning', False
                    )
                }

            logger.warning(
                f"⚠️ HPO result not found for {symbol} {horizon}d "
                f"(or insufficient trials)"
            )
            return None

        except subprocess.TimeoutExpired:
            error_msg = f"HPO timeout for {symbol} {horizon}d"
            logger.error(f"❌ {error_msg}")
            return {'error': error_msg}
        except Exception as e:
            error_msg = f"HPO exception: {str(e)}"
            logger.error(f"❌ HPO error for {symbol} {horizon}d: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': error_msg}

    @staticmethod
    def _compute_returns(df: pd.DataFrame, horizon: int) -> np.ndarray:
        if df is None or len(df) < horizon:
            return np.array([])
        return (df['close'].shift(-horizon) / df['close'] - 1.0).values

    @staticmethod
    def _dirhit(
        y_true: np.ndarray, y_pred: np.ndarray, thr: float = 0.005
    ) -> float:
        """Compute directional hit rate (same as HPO).

        ✅ FIX: Use same logic as HPO - only significant predictions are
        evaluated.
        This ensures HPO DirHit and Training DirHit are comparable.
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        yt = np.sign(y_true)
        yp = np.sign(y_pred)
        # ✅ FIX: Use same logic as HPO - only significant predictions
        # HPO uses: m = (np.abs(y_true) > thr) & (np.abs(y_pred) > thr)
        m = (np.abs(y_true) > thr) & (np.abs(y_pred) > thr)
        if m.sum() == 0:
            return 0.0
        return float(np.mean(yt[m] == yp[m]) * 100.0)

    @staticmethod
    def _set_feature_flag_from_key(
        key: str, value: Any,
        original_env_vars: Dict[str, Optional[str]]
    ) -> None:
        """Helper to set a feature flag environment variable from
        enable_* key."""
        import os
        flag_mapping = {
            'enable_external_features': 'ENABLE_EXTERNAL_FEATURES',
            'enable_fingpt_features': 'ENABLE_FINGPT_FEATURES',
            'enable_yolo_features': 'ENABLE_YOLO_FEATURES',
            'enable_directional_loss': 'ML_USE_DIRECTIONAL_LOSS',
            'enable_seed_bagging': 'ENABLE_SEED_BAGGING',
            'enable_talib_patterns': 'ENABLE_TALIB_PATTERNS',
            'enable_smart_ensemble': 'ML_USE_SMART_ENSEMBLE',
            'enable_stacked_short': 'ML_USE_STACKED_SHORT',
            'enable_meta_stacking': 'ENABLE_META_STACKING',
            'enable_regime_detection': 'ML_USE_REGIME_DETECTION',
            'enable_fingpt': 'ENABLE_FINGPT',
        }
        env_key = flag_mapping.get(key)
        if env_key:
            if env_key not in original_env_vars:
                original_env_vars[env_key] = os.environ.get(env_key)
            os.environ[env_key] = '1' if value else '0'

    def _prepare_evaluation_splits(
        self,
        symbol: str,
        horizon: int,
        total_days: int,
        hpo_result: Optional[Dict],
    ) -> List[Tuple[int, int]]:
        """Prepare walk-forward validation splits for evaluation."""
        import os
        from scripts.optuna_hpo_with_feature_flags import (
            generate_walkforward_splits,
            calculate_dynamic_split,
        )

        wfv_splits = generate_walkforward_splits(
            total_days, horizon, n_splits=4
        )

        # If evaluation_spec present in HPO JSON, override splits and
        # thresholds
        eval_spec = None
        try:
            if isinstance(hpo_result, dict):
                eval_spec = hpo_result.get('evaluation_spec')
        except Exception as e:
            logger.debug(
                f"Failed to get eval_spec from hpo_result: {e}"
            )
            eval_spec = None

        if isinstance(eval_spec, dict):
            # Set DirHit threshold from spec
            try:
                dirhit_thr = float(
                    eval_spec.get('dirhit_threshold', 0.005)
                )
                os.environ['DIRHIT_THRESHOLD'] = str(dirhit_thr)
            except Exception as e:
                logger.debug(f"Failed to set DIRHIT_THRESHOLD: {e}")

            # Set mask thresholds
            try:
                if 'min_mask_count' in eval_spec:
                    os.environ['HPO_MIN_MASK_COUNT'] = str(
                        int(eval_spec['min_mask_count'])
                    )
                if 'min_mask_pct' in eval_spec:
                    os.environ['HPO_MIN_MASK_PCT'] = str(
                        float(eval_spec['min_mask_pct'])
                    )
            except Exception as e:
                logger.debug(f"Failed to set mask thresholds: {e}")

            # Override WFV splits using indices if provided
            try:
                symbol_key = f"{symbol}_{horizon}d"
                sym_specs = eval_spec.get('symbol_specs', {})
                sym_spec = sym_specs.get(symbol_key)
                if isinstance(sym_spec, dict):
                    splits_spec = sym_spec.get('splits', [])
                    splits_idx: list[tuple[int, int]] = []
                    for s in splits_spec:
                        tei = s.get('train_end_idx')
                        pei = s.get('test_end_idx')
                        if tei is not None and pei is not None:
                            try:
                                splits_idx.append((int(tei), int(pei)))
                            except Exception as e:
                                logger.debug(
                                    f"Failed to process split_idx: {e}"
                                )
                                continue
                    if splits_idx:
                        wfv_splits = splits_idx
                        logger.info(
                            f"🔁 Using evaluation_spec splits for "
                            f"{symbol} {horizon}d: {len(wfv_splits)} splits"
                        )
            except Exception as e:
                logger.debug(
                    f"Failed to process splits_indices: {e}"
                )

        if not wfv_splits:
            # Fallback to single split if multiple splits not possible
            split_idx = calculate_dynamic_split(total_days, horizon)
            if split_idx < 1:
                logger.warning(
                    f"⚠️ {symbol} {horizon}d: Invalid split, skipping "
                    f"evaluation"
                )
                return []
            wfv_splits = [(split_idx, total_days)]

        logger.info(
            f"📊 {symbol} {horizon}d: Generated {len(wfv_splits)} "
            f"walk-forward splits for evaluation"
        )
        return wfv_splits

    def _setup_best_params(
        self, symbol: str, horizon: int, best_params: Optional[Dict],
        original_env_vars: Optional[Dict[str, Optional[str]]] = None
    ) -> None:
        """Setup best HPO parameters for evaluation.

        Args:
            symbol: Stock symbol
            horizon: Prediction horizon
            best_params: Best HPO parameters
            original_env_vars: Optional dict to track original env var
                values for restoration
        """
        import os
        if best_params is None:
            logger.warning(
                f"⚠️ {symbol} {horizon}d: Best params not provided! "
                f"Evaluation may use default params."
            )
            return

        from scripts.train_completed_hpo_with_best_params import (
            set_hpo_params_as_env
        )
        set_hpo_params_as_env(best_params, horizon)

        # Set feature flags from best_params
        if (
            isinstance(best_params, dict)
            and 'features_enabled' in best_params
        ):
            features_enabled = best_params['features_enabled']
            for key, value in features_enabled.items():
                # Normalize feature flag key to uppercase environment
                # variable name
                env_key = _normalize_feature_flag_key(key)
                # Track original value if original_env_vars is provided
                if (
                    original_env_vars is not None
                    and env_key not in original_env_vars
                ):
                    original_env_vars[env_key] = os.environ.get(env_key)
                os.environ[env_key] = str(value)
            logger.info(
                f"🔧 {symbol} {horizon}d: Feature flags set from "
                f"best_params: {len(features_enabled)} flags"
            )

        # Set feature flags from best_params keys (enable_*)
        feature_flag_keys = [
            k for k in best_params.keys()
            if k.startswith('enable_')
        ]
        for key in feature_flag_keys:
            value = best_params[key]
            # Use original_env_vars if provided, otherwise empty dict
            tracking_dict = (
                original_env_vars
                if original_env_vars is not None else {}
            )
            self._set_feature_flag_from_key(key, value, tracking_dict)

        # Log key params for verification
        xgb_params = {
            k: v for k, v in best_params.items()
            if k.startswith('xgb_')
        }
        if xgb_params:
            logger.info(
                f"⚙️ {symbol} {horizon}d: Best HPO params set for "
                f"evaluation: "
                f"n_est={xgb_params.get('xgb_n_estimators', 'N/A')}, "
                f"max_depth={xgb_params.get('xgb_max_depth', 'N/A')}, "
                f"lr={xgb_params.get('xgb_learning_rate', 'N/A')}"
            )
            env_n_est = os.environ.get(
                'OPTUNA_XGB_N_ESTIMATORS', 'NOT_SET'
            )
            env_max_depth = os.environ.get(
                'OPTUNA_XGB_MAX_DEPTH', 'NOT_SET'
            )
            env_lr = os.environ.get(
                'OPTUNA_XGB_LEARNING_RATE', 'NOT_SET'
            )
            logger.info(
                f"🔍 {symbol} {horizon}d: Environment vars - "
                f"OPTUNA_XGB_N_ESTIMATORS={env_n_est}, "
                f"OPTUNA_XGB_MAX_DEPTH={env_max_depth}, "
                f"OPTUNA_XGB_LEARNING_RATE={env_lr}"
            )
        else:
            logger.info(
                f"⚙️ {symbol} {horizon}d: Best HPO params set for "
                f"evaluation: {len(best_params)} params"
            )

    def _evaluate_wfv_splits(
        self,
        symbol: str,
        horizon: int,
        df: pd.DataFrame,
        wfv_splits: List[Tuple[int, int]],
        ml_eval: Any,
        reference_r2_map: Optional[Dict[str, float]],
        eval_seed: int,
        best_trial_number: Optional[int]
    ) -> Tuple[List[float], List[float], List[float]]:
        """Evaluate walk-forward validation splits and return metrics."""
        import os
        import math
        from enhanced_ml_system import EnhancedMLSystem

        split_dirhits = []
        split_nrmses = []
        split_scores = []

        for split_idx, (train_end_idx, test_end_idx) in enumerate(
            wfv_splits, 1
        ):
            train_df_split = df.iloc[:train_end_idx].copy()
            test_df_split = df.iloc[train_end_idx:test_end_idx].copy()

            logger.info(
                f"📊 {symbol} {horizon}d WFV Split "
                f"{split_idx}/{len(wfv_splits)}: "
                f"train={len(train_df_split)} days, "
                f"test={len(test_df_split)} days"
            )
            logger.info(
                f"🔍 [eval-debug] {symbol} {horizon}d Split {split_idx}: "
                f"Train period: {train_df_split.index.min()} to "
                f"{train_df_split.index.max()}"
            )
            logger.info(
                f"🔍 [eval-debug] {symbol} {horizon}d Split {split_idx}: "
                f"Test period: {test_df_split.index.min()} to "
                f"{test_df_split.index.max()}"
            )

            y_true_split = self._compute_returns(test_df_split, horizon)
            min_test_days = horizon + 10
            if len(test_df_split) < min_test_days:
                logger.warning(
                    f"⚠️ {symbol} {horizon}d Split {split_idx}: "
                    f"Insufficient test data ({len(test_df_split)} days, "
                    f"need {min_test_days})"
                )
                continue

            if split_idx == 1:
                logger.info(
                    f"🔧 Eval env (WFV): adaptive=0, seed_bagging="
                    f"{os.environ.get('ENABLE_SEED_BAGGING', '0')}, "
                    f"directional_loss="
                    f"{os.environ.get('ML_USE_DIRECTIONAL_LOSS', '0')}, "
                    f"smart="
                    f"{os.environ.get('ML_USE_SMART_ENSEMBLE', '1')}, "
                    f"stacked="
                    f"{os.environ.get('ML_USE_STACKED_SHORT', '1')}, "
                    f"regime="
                    f"{os.environ.get('ML_USE_REGIME_DETECTION', '1')}"
                )
                logger.info(
                    f"🔄 {symbol} {horizon}d WFV: Model'i train_df ile "
                    f"yeniden eğitiyoruz (best params, adaptive OFF, data "
                    f"leakage önleme, NEW instance)"
                )

            train_result = ml_eval.train_enhanced_models(
                symbol, train_df_split
            )
            if not train_result:
                logger.warning(
                    f"⚠️ {symbol} {horizon}d WFV Split {split_idx}: "
                    f"Model eğitimi başarısız, split atlanıyor"
                )
                continue

            if reference_r2_map:
                actual_metrics = _extract_model_metrics_from_train_result(
                    train_result, horizon
                )
                if actual_metrics:
                    for model_name, ref_val in reference_r2_map.items():
                        actual_val = actual_metrics.get(
                            model_name, {}
                        ).get('raw_r2')
                        if (
                            actual_val is None
                            or not math.isfinite(actual_val)
                        ):
                            continue
                        delta = actual_val - ref_val
                        logger.info(
                            f"📏 {symbol} {horizon}d WFV Split {split_idx}: "
                            f"{model_name} raw_r2 "
                            f"HPO={ref_val:.4f}, train={actual_val:.4f}, "
                            f"delta={delta:+.4f}"
                        )

            preds = np.full(len(test_df_split), np.nan, dtype=float)
            for t in range(len(test_df_split) - horizon):
                try:
                    cur = pd.concat(
                        [train_df_split, test_df_split.iloc[:t + 1]],
                        axis=0
                    ).copy()
                    p = ml_eval.predict_enhanced(symbol, cur)
                    if isinstance(p, dict):
                        key = f"{horizon}d"
                        obj = p.get(key)
                        if isinstance(obj, dict):
                            pred_price = obj.get('ensemble_prediction')
                            if (
                                isinstance(pred_price, (int, float))
                                and not np.isnan(pred_price)
                            ):
                                last_close = float(cur['close'].iloc[-1])
                                if last_close > 0:
                                    preds[t] = (
                                        float(pred_price) / last_close - 1.0
                                    )
                except Exception as e:
                    logger.debug(
                        f"Failed to process prediction: {e}"
                    )
                    continue

            valid_mask = ~np.isnan(preds) & ~np.isnan(y_true_split)
            valid_count = valid_mask.sum()

            if valid_count > 0:
                thr_env_val = 0.005
                try:
                    _thr_env = os.getenv('DIRHIT_THRESHOLD')
                    if _thr_env is not None:
                        thr_env_val = float(_thr_env)
                except Exception as e:
                    logger.debug(
                        f"Failed to get DIRHIT_THRESHOLD: {e}"
                    )

                dh = self._dirhit(y_true_split, preds, thr=thr_env_val)
                y_true_valid = y_true_split[valid_mask]
                preds_valid = preds[valid_mask]
                rmse = np.sqrt(
                    np.mean((y_true_valid - preds_valid) ** 2)
                )
                mape = (
                    np.mean(
                        np.abs(
                            (y_true_valid - preds_valid) /
                            (y_true_valid + 1e-8)
                        )
                    ) * 100
                )

                nrmse_val = float('inf')
                try:
                    std_y = (
                        float(np.std(y_true_valid))
                        if y_true_valid.size > 1 else 0.0
                    )
                    if std_y > 0:
                        nrmse_val = float(rmse / std_y)
                except Exception as e:
                    logger.debug(f"Failed to calculate NRMSE: {e}")

                score_val = EnhancedMLSystem._calculate_score(
                    dh, nrmse_val, horizon
                )

                thr = 0.005
                try:
                    _thr_env2 = os.getenv('DIRHIT_THRESHOLD')
                    if _thr_env2 is not None:
                        thr = float(_thr_env2)
                except Exception as e:
                    logger.debug(f"Failed to get MASK_THRESHOLD: {e}")

                mask_count = (
                    (
                        np.abs(y_true_valid) > thr
                    ) & (np.abs(preds_valid) > thr)
                ).sum()
                mask_pct = (
                    (mask_count / valid_count) * 100
                    if valid_count > 0 else 0
                )

                if mask_count == 0 and valid_count > 0:
                    pred_abs_max = (
                        float(np.abs(preds_valid).max())
                        if len(preds_valid) > 0 else 0.0
                    )
                    pred_abs_mean = (
                        float(np.abs(preds_valid).mean())
                        if len(preds_valid) > 0 else 0.0
                    )
                    y_true_abs_max = (
                        float(np.abs(y_true_valid).max())
                        if len(y_true_valid) > 0 else 0.0
                    )
                    y_true_abs_mean = (
                        float(np.abs(y_true_valid).mean())
                        if len(y_true_valid) > 0 else 0.0
                    )
                    logger.warning(
                        f"⚠️ {symbol} {horizon}d WFV Split {split_idx}: "
                        f"mask_count=0 (threshold={thr}) - "
                        f"pred_abs_max={pred_abs_max:.6f}, "
                        f"pred_abs_mean={pred_abs_mean:.6f}, "
                        f"y_true_abs_max={y_true_abs_max:.6f}, "
                        f"y_true_abs_mean={y_true_abs_mean:.6f}"
                    )

                direction_matches = (
                    np.sign(y_true_valid) == np.sign(preds_valid)
                ).sum()
                direction_pct = (
                    (direction_matches / valid_count) * 100
                    if valid_count > 0 else 0
                )

                if split_idx == 1:
                    logger.info(
                        f"🔍 [eval-debug] {symbol} {horizon}d WFV: METRICS:"
                    )
                    logger.info(
                        f"🔍 [eval-debug]   Valid predictions: "
                        f"{valid_count}/{len(preds)}"
                    )
                    logger.info(
                        f"🔍 [eval-debug]   DirHit (threshold={thr}): "
                        f"{dh:.2f}% (mask_count={mask_count}, "
                        f"mask_pct={mask_pct:.1f}%)"
                    )
                    logger.info(
                        f"🔍 [eval-debug]   Direction match (all): "
                        f"{direction_pct:.2f}% "
                        f"({direction_matches}/{valid_count})"
                    )
                    logger.info(
                        f"🔍 [eval-debug]   RMSE: {rmse:.6f}"
                    )
                    logger.info(
                        f"🔍 [eval-debug]   MAPE: {mape:.2f}%"
                    )
                    logger.info(
                        f"🔍 [eval-debug]   nRMSE: {nrmse_val:.3f}"
                    )
                    logger.info(
                        f"🔍 [eval-debug]   Score: {score_val:.2f}"
                    )
                    logger.info(
                        f"🔍 [eval-debug]   Seed = {eval_seed} "
                        f"(best_trial={best_trial_number})"
                    )

                low_support = False
                _min_mc = 0
                _min_mp = 0.0
                try:
                    _min_mc = int(os.getenv('HPO_MIN_MASK_COUNT', '0'))
                except Exception as e:
                    logger.debug(
                        f"Failed to get HPO_MIN_MASK_COUNT, using 0: {e}"
                    )
                    _min_mc = 0
                try:
                    _min_mp = float(os.getenv('HPO_MIN_MASK_PCT', '0.0'))
                except Exception as e:
                    logger.debug(
                        f"Failed to get HPO_MIN_MASK_PCT, using 0.0: {e}"
                    )
                    _min_mp = 0.0

                if (
                    (_min_mc > 0 and mask_count < _min_mc)
                    or (_min_mp > 0.0 and mask_pct < _min_mp)
                ):
                    low_support = True
                    logger.info(
                        f"⚠️ {symbol} {horizon}d WFV Split {split_idx}: "
                        f"LOW_SUPPORT → exclude from avg "
                        f"(DirHit={dh:.2f}%, mask_count={mask_count}, "
                        f"mask_pct={mask_pct:.1f}%, "
                        f"min_mc={_min_mc}, min_mp={_min_mp}%)"
                    )

                if not np.isnan(dh) and not low_support:
                    split_dirhits.append(dh)
                    split_nrmses.append(nrmse_val)
                    split_scores.append(score_val)
                    logger.info(
                        f"✅ {symbol} {horizon}d WFV Split {split_idx}: "
                        f"DirHit={dh:.2f}%, nRMSE={nrmse_val:.3f}, "
                        f"Score={score_val:.2f} "
                        f"(valid={valid_count}/{len(preds)}, "
                        f"RMSE={rmse:.6f}, MAPE={mape:.2f}%)"
                    )
                else:
                    logger.warning(
                        f"⚠️ {symbol} {horizon}d WFV Split {split_idx}: "
                        f"DirHit calculation returned NaN"
                    )
            else:
                logger.warning(
                    f"⚠️ {symbol} {horizon}d WFV Split {split_idx}: "
                    f"No valid predictions!"
                )

        return split_dirhits, split_nrmses, split_scores

    # NOTE: This function is intentionally complex due to evaluation
    # requirements.
    # TODO: Consider refactoring into smaller helper functions to reduce
    # complexity.
    def _evaluate_training_dirhits(
        self, symbol: str, horizon: int, df: pd.DataFrame,
        best_params: Optional[Dict] = None,
        hpo_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Evaluate DirHit after training using two modes:
        - wfv: adaptive OFF (no leakage)
        - online: adaptive OFF (HPO ile tutarlılık - hibrit yaklaşım)

        Args:
            symbol: Stock symbol
            horizon: Prediction horizon
            df: Full dataframe
            best_params: Best HPO parameters (must be provided to ensure
                correct evaluation)
            hpo_result: Full HPO result dict (includes best_trial_number,
                features_enabled, etc.)
        """
        # ✅ FIX: Import os at function level to avoid scope issues
        import os  # type: ignore[unused-import]
        results: Dict[str, Any] = {'wfv': None, 'online': None}
        total_days = len(df)

        # ✅ FIX: Dynamic minimum data requirement based on horizon
        # Minimum: enough for train (80%) + test (20%) with at least
        # (horizon + 10) test days
        # Formula: min_total = (horizon + 10) / 0.2 = (horizon + 10) * 5
        # For horizon=1: min_total = 11 * 5 = 55 days
        # For horizon=30: min_total = 40 * 5 = 200 days
        # Use a reasonable minimum: 100 days (allows evaluation for most
        # horizons)
        min_total_days = max(100, (horizon + 10) * 5)

        if total_days < min_total_days:
            logger.warning(
                f"⚠️ {symbol} {horizon}d: Insufficient data for evaluation "
                f"({total_days} days, need {min_total_days})"
            )
            return results

        # Prepare evaluation splits
        wfv_splits = self._prepare_evaluation_splits(
            symbol, horizon, total_days, hpo_result
        )
        if not wfv_splits:
            return results

        # 🔍 DEBUG: Log data source and statistics (same format as HPO)
        logger.info(
            f"🔍 [eval-debug] {symbol} {horizon}d: DATA SOURCE = df "
            f"parameter (from run_training)"
        )
        logger.info(
            f"🔍 [eval-debug] {symbol} {horizon}d: Total data: "
            f"{total_days} days"
        )

        # 1) WFV (adaptive OFF) - Data leakage önleme: Model'i train_df
        # ile yeniden eğit
        # ✅ CRITICAL FIX: Initialize original_env_vars BEFORE any setup
        # to ensure it's always available
        original_env_vars: dict = {}
        original_adaptive = None
        original_seed_bag = None
        original_directional = None
        # ✅ CRITICAL FIX: Save original smart ensemble parameters for
        # restoration (same as online evaluation)
        original_smart_consensus_weight: Optional[str] = None
        original_smart_performance_weight: Optional[str] = None
        original_smart_sigma: Optional[str] = None
        original_smart_weight_xgb: Optional[str] = None
        original_smart_weight_lgb: Optional[str] = None
        original_smart_weight_cat: Optional[str] = None

        # Setup best params (track changes in original_env_vars for proper
        # restoration)
        self._setup_best_params(
            symbol, horizon, best_params, original_env_vars
        )
        try:
            # ✅ CRITICAL FIX: Don't use default value - we need to
            # distinguish between "unset" and "set to '0'"
            # None if not set
            original_adaptive = os.environ.get('ML_USE_ADAPTIVE_LEARNING')
            os.environ['ML_USE_ADAPTIVE_LEARNING'] = '0'  # Adaptive OFF
            # ✅ CRITICAL FIX: Capture original values BEFORE modifying
            # them (needed for both if and else paths)
            # Don't use default values - we need to distinguish between
            # "unset" and "set to default"
            # None if not set
            original_seed_bag = os.environ.get('ENABLE_SEED_BAGGING')
            # None if not set
            original_directional = os.environ.get('ML_USE_DIRECTIONAL_LOSS')
            # ✅ CRITICAL FIX: Set feature flags from best_params to match
            # HPO best trial exactly
            # This ensures evaluation uses the same feature flags as HPO
            if best_params and isinstance(best_params, dict):
                features_enabled = best_params.get('features_enabled', {})
                if features_enabled:
                    # ✅ CRITICAL FIX: Save original values before setting
                    # new ones (same as online evaluation)
                    # ✅ CRITICAL FIX: Normalize feature flag keys to
                    # uppercase environment variable names
                    for key, value in features_enabled.items():
                        # Map lowercase to uppercase if needed
                        env_key = _normalize_feature_flag_key(key)
                        if env_key not in original_env_vars:
                            # Save original value (None if not set)
                            original_env_vars[env_key] = (
                                os.environ.get(env_key)
                            )
                        os.environ[env_key] = str(value)
                    logger.info(
                        f"🔧 {symbol} {horizon}d WFV: Feature flags set "
                        f"from best_params: {len(features_enabled)} flags"
                    )
                # Also set smart-ensemble params from feature_params if
                # available (parity with HPO)
                # ✅ CRITICAL FIX: Save original values before setting new
                # ones (same as online evaluation)
                try:
                    fp = (
                        best_params.get('feature_params', {})
                        if isinstance(best_params, dict) else {}
                    )
                    if isinstance(fp, dict) and fp:
                        if 'smart_consensus_weight' in fp:
                            original_smart_consensus_weight = (
                                os.environ.get('ML_SMART_CONSENSUS_WEIGHT')
                            )
                            os.environ['ML_SMART_CONSENSUS_WEIGHT'] = str(
                                fp['smart_consensus_weight']
                            )
                        if 'smart_performance_weight' in fp:
                            original_smart_performance_weight = (
                                os.environ.get('ML_SMART_PERFORMANCE_WEIGHT')
                            )
                            os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = (
                                str(fp['smart_performance_weight'])
                            )
                        if 'smart_sigma' in fp:
                            original_smart_sigma = os.environ.get(
                                'ML_SMART_SIGMA'
                            )
                            os.environ['ML_SMART_SIGMA'] = str(
                                fp['smart_sigma']
                            )
                        if 'smart_weight_xgb' in fp:
                            original_smart_weight_xgb = os.environ.get(
                                'ML_SMART_WEIGHT_XGB'
                            )
                            os.environ['ML_SMART_WEIGHT_XGB'] = str(
                                fp['smart_weight_xgb']
                            )
                        if (
                            'smart_weight_lgbm' in fp
                            or 'smart_weight_lgb' in fp
                        ):
                            original_smart_weight_lgb = (
                                os.environ.get('ML_SMART_WEIGHT_LGB')
                            )
                            os.environ['ML_SMART_WEIGHT_LGB'] = str(
                                fp.get(
                                    'smart_weight_lgbm',
                                    fp.get('smart_weight_lgb')
                                )
                            )
                        if 'smart_weight_cat' in fp:
                            original_smart_weight_cat = os.environ.get(
                                'ML_SMART_WEIGHT_CAT'
                            )
                            os.environ['ML_SMART_WEIGHT_CAT'] = str(
                                fp['smart_weight_cat']
                            )
                except Exception as e:
                    logger.debug(
                        f"Failed to set smart_weight env vars: {e}"
                    )
                # Also check enable_* keys in best_params
                # ✅ CRITICAL FIX: Save original values before setting new
                # ones (fallback path, same as online evaluation)
                feature_flag_keys = [
                    k for k in best_params.keys()
                    if k.startswith('enable_')
                ]
                for key in feature_flag_keys:
                    value = best_params[key]
                    self._set_feature_flag_from_key(
                        key, value, original_env_vars
                    )
            else:
                # Fallback: Align with HPO - disable seed bagging and
                # directional loss during evaluation
                # ✅ CRITICAL FIX: Original values already captured above
                # (before if/else), just set to disabled values
                os.environ['ENABLE_SEED_BAGGING'] = '0'
                # ✅ CRITICAL FIX: Align with HPO - disable directional
                # loss (use MSE loss)
                # MSE loss (same as HPO)
                os.environ['ML_USE_DIRECTIONAL_LOSS'] = '0'
            # ✅ CRITICAL FIX: Don't use singleton, create new instance
            # (same as HPO)
            # import enhanced_ml_system
            # enhanced_ml_system._enhanced_ml_system = None
            # Not needed, we create new instance

            # ✅ FIX: Clear ConfigManager cache to ensure best params are read
            try:
                from bist_pattern.core.config_manager import ConfigManager
                ConfigManager.clear_cache()
            except Exception as e:
                logger.debug(f"Failed to clear ConfigManager cache: {e}")
            # ✅ CRITICAL FIX: Use best trial's seed for evaluation
            # (match HPO best trial). If best_trial_number is missing,
            # fallback to 42. Try best_params first, then hpo_result.
            best_trial_number = None
            if isinstance(best_params, dict):
                best_trial_number = best_params.get('best_trial_number')
            if best_trial_number is None and hpo_result:
                best_trial_number = hpo_result.get('best_trial_number')
            # HPO uses seed = 42 + trial.number
            try:
                eval_seed = (
                    42 + int(best_trial_number)
                    if best_trial_number is not None
                    else 42
                )
            except Exception as e:
                logger.debug(f"Failed to calculate eval_seed, using 42: {e}")
                eval_seed = 42
            try:
                import random
                random.seed(eval_seed)
                np.random.seed(eval_seed)
                os.environ['PYTHONHASHSEED'] = str(eval_seed)
                # Hint for model libraries (read by EnhancedMLSystem if
                # supported)
                os.environ.setdefault(
                    'OPTUNA_XGB_RANDOM_STATE', str(eval_seed)
                )
                os.environ.setdefault(
                    'OPTUNA_LGB_RANDOM_STATE', str(eval_seed)
                )
                os.environ.setdefault(
                    'OPTUNA_CAT_RANDOM_STATE', str(eval_seed)
                )
                os.environ.setdefault('XGB_SEED', str(eval_seed))
                os.environ.setdefault('LIGHTGBM_SEED', str(eval_seed))
                os.environ.setdefault('CATBOOST_RANDOM_SEED', str(eval_seed))
                logger.info(
                    f"🔧 {symbol} {horizon}d WFV: Using seed={eval_seed} "
                    f"(best_trial={best_trial_number}) for evaluation"
                )
            except Exception as e:
                logger.debug("Failed to set seed env vars: %s", e)

            # ✅ CRITICAL FIX: Create NEW instance (same as HPO) to avoid
            # singleton cache pollution. HPO creates new EnhancedMLSystem() for
            # each trial, we should do the same for evaluation.
            from enhanced_ml_system import EnhancedMLSystem
            ml_eval = EnhancedMLSystem()  # New instance, not singleton!
            ml_eval.prediction_horizons = [horizon]
            # ✅ CRITICAL FIX: Set base_seeds to match HPO best trial.
            # HPO uses: ml.base_seeds = [42 + trial.number]
            # Evaluation should use: ml_eval.base_seeds = [42 +
            # best_trial_number]. If best_trial_number is None, use fallback
            # seed (42) but log warning.
            if best_trial_number is None:
                logger.warning(
                    f"⚠️ {symbol} {horizon}d WFV: "
                    f"best_trial_number not found, "
                    f"using fallback seed=42 "
                    f"(may cause alignment issues)"
                )
            # eval_seed already equals 42 + best_trial_number (or 42 fallback)
            ml_eval.base_seeds = [eval_seed]
            # ✅ FIX: Ensure enable_seed_bagging matches HPO best trial
            # (env var). base_seeds single seed disables bagging, but keep flag
            # consistent.
            logger.info(
                "🔧 %s %sd WFV: Set base_seeds=%s, "
                "enable_seed_bagging=%s (HPO trial %s)",
                symbol,
                horizon,
                ml_eval.base_seeds,
                ml_eval.enable_seed_bagging,
                best_trial_number,
            )

            reference_entry = _get_best_trial_metrics_entry(
                hpo_result, symbol, horizon
            )
            reference_r2_map = _extract_reference_historical_r2(
                reference_entry
            )
            reference_key = f"{symbol}_{horizon}d"
            if reference_r2_map:
                try:
                    ml_eval.reference_historical_r2[reference_key] = (
                        reference_r2_map
                    )
                    ml_eval.use_reference_historical_r2 = True
                    logger.info(
                        "📊 %s %sd: Using HPO historical R² for weighting "
                        "(models: %s)",
                        symbol,
                        horizon,
                        ", ".join(reference_r2_map.keys()),
                    )
                except Exception as ref_err:
                    logger.warning(
                        "⚠️ %s %sd: Failed to inject reference R²: %s",
                        symbol,
                        horizon,
                        ref_err,
                    )
                    ml_eval.use_reference_historical_r2 = False
            else:
                ml_eval.use_reference_historical_r2 = False
                ml_eval.reference_historical_r2.pop(reference_key, None)

            # ⚡ NEW: Evaluate on all splits and average DirHit, nRMSE, Score
            split_dirhits, split_nrmses, split_scores = (
                self._evaluate_wfv_splits(
                    symbol,
                    horizon,
                    df,
                    wfv_splits,
                    ml_eval,
                    reference_r2_map,
                    eval_seed,
                    best_trial_number,
                )
            )

            # Average DirHit, nRMSE, and Score across all splits
            # ✅ FIX: Require at least 2 splits for reliable DirHit (HPO parity)
            _min_mc = 0
            _min_mp = 0.0
            try:
                _min_mc = int(os.getenv('HPO_MIN_MASK_COUNT', '0'))
            except Exception:
                _min_mc = 0
            try:
                _min_mp = float(os.getenv('HPO_MIN_MASK_PCT', '0.0'))
            except Exception:
                _min_mp = 0.0

            if len(split_dirhits) >= 2:
                avg_dirhit = float(np.mean(split_dirhits))
                avg_nrmse = (
                    float(np.mean(split_nrmses))
                    if split_nrmses
                    else float('inf')
                )
                avg_score = (
                    float(np.mean(split_scores)) if split_scores else 0.0
                )
                logger.info(
                    "✅ %s %sd WFV: Avg over %d splits: DirHit=%.2f%%, "
                    "nRMSE=%.3f, Score=%.2f",
                    symbol,
                    horizon,
                    len(split_dirhits),
                    avg_dirhit,
                    avg_nrmse,
                    avg_score,
                )
                results['wfv'] = avg_dirhit
                results['wfv_nrmse'] = avg_nrmse
                results['wfv_score'] = avg_score
            elif len(split_dirhits) == 1:
                logger.warning(
                    "⚠️ %s %sd WFV: Only 1 split passed "
                    "(DirHit=%.2f%%) - excluded (need >=2 for reliability)",
                    symbol,
                    horizon,
                    split_dirhits[0],
                )
                logger.warning(
                    "⚠️ WARNING: %s %sd: Only 1 split passed "
                    "(min_count=%s, min_pct=%s%%) - DirHit excluded "
                    "(need >=2 for statistical reliability)",
                    symbol,
                    horizon,
                    _min_mc,
                    _min_mp,
                )
                results['wfv'] = None
                results['wfv_nrmse'] = None
                results['wfv_score'] = None
                results['low_support_warning'] = True
            else:
                logger.warning(
                    "⚠️ %s %sd WFV: No valid DirHit from any split",
                    symbol,
                    horizon,
                )
                # ✅ WARNING: All splits excluded by filter - best params may
                # not be optimal for this symbol
                logger.warning(
                    "⚠️ WARNING: %s %sd: All splits excluded "
                    "(min_count=%s, min_pct=%s%%) - best params may not be "
                    "optimal",
                    symbol,
                    horizon,
                    _min_mc,
                    _min_mp,
                )
                results['wfv'] = None
                results['wfv_nrmse'] = None
                results['wfv_score'] = None
                results['low_support_warning'] = True

                # ✅ FALLBACK: Try to find best params with 0/0.0 filter
                # (no filter). Ensures we have valid params even for
                # low-support symbols.
                if hpo_result and 'json_file' in hpo_result:
                    try:
                        from scripts.find_fallback_best_params import (
                            find_fallback_best_params
                        )
                        from scripts.retrain_high_discrepancy_symbols import (
                            find_study_db
                        )

                        # Find study DB for fallback
                        study_db = find_study_db(symbol, horizon)

                        if study_db and study_db.exists():
                            logger.info(
                                f"🔄 {symbol} {horizon}d: "
                                "Attempting fallback - finding best params "
                                "with 0/0.0 filter..."
                            )
                            fallback_params = find_fallback_best_params(
                                study_db, symbol, horizon
                            )

                            if fallback_params:
                                logger.info(
                                    f"✅ {symbol} {horizon}d: "
                                    f"Found fallback best params "
                                    f"(trial #"
                                    f"{fallback_params['best_trial_number']}, "
                                    f"DirHit: "
                                    f"{fallback_params['best_value']:.2f}%)"
                                )
                                results['fallback_best_params'] = (
                                    fallback_params
                                )  # type: ignore[assignment]
                                results['fallback_available'] = True
                            else:
                                logger.warning(
                                    f"⚠️ {symbol} {horizon}d: "
                                    f"No fallback best params found"
                                )
                                results['fallback_available'] = False
                        else:
                            logger.debug(
                                f"⚠️ {symbol} {horizon}d: "
                                f"Study DB not found for fallback"
                            )
                            results['fallback_available'] = False
                    except Exception as fallback_err:
                        logger.debug(
                            f"⚠️ {symbol} {horizon}d: "
                            f"Fallback mechanism failed: {fallback_err}"
                        )
                        results['fallback_available'] = False
                else:
                    results['fallback_available'] = False
        except Exception as e:
            logger.error(
                f"❌ WFV evaluation error for {symbol} {horizon}d: {e}"
            )
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # ✅ CRITICAL FIX: Restore adaptive learning - remove if it wasn't
            # set before (same logic as smart ensemble params)
            if original_adaptive is not None:
                os.environ['ML_USE_ADAPTIVE_LEARNING'] = original_adaptive
            elif (
                'ML_USE_ADAPTIVE_LEARNING' in os.environ
                and original_adaptive is None
            ):
                # Was not set before, remove it
                os.environ.pop('ML_USE_ADAPTIVE_LEARNING', None)
            # ✅ CRITICAL FIX: Restore seed bagging - remove if it wasn't set
            # before (same logic as smart ensemble params)
            if original_seed_bag is not None:
                os.environ['ENABLE_SEED_BAGGING'] = original_seed_bag
            elif (
                'ENABLE_SEED_BAGGING' in os.environ
                and original_seed_bag is None
            ):
                # Was not set before, remove it
                os.environ.pop('ENABLE_SEED_BAGGING', None)
            # ✅ CRITICAL FIX: Restore directional loss - remove if it wasn't
            # set before (same logic as smart ensemble params)
            if original_directional is not None:
                os.environ['ML_USE_DIRECTIONAL_LOSS'] = original_directional
            elif (
                'ML_USE_DIRECTIONAL_LOSS' in os.environ
                and original_directional is None
            ):
                # Was not set before, remove it
                os.environ.pop('ML_USE_DIRECTIONAL_LOSS', None)
            # ✅ CRITICAL FIX: Restore smart ensemble parameters (same as online
            # evaluation)
            if original_smart_consensus_weight is not None:
                os.environ['ML_SMART_CONSENSUS_WEIGHT'] = (
                    original_smart_consensus_weight
                )
            elif (
                'ML_SMART_CONSENSUS_WEIGHT' in os.environ
                and original_smart_consensus_weight is None
            ):
                # Was not set before, remove it
                os.environ.pop('ML_SMART_CONSENSUS_WEIGHT', None)
            if original_smart_performance_weight is not None:
                os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = (
                    original_smart_performance_weight
                )
            elif (
                'ML_SMART_PERFORMANCE_WEIGHT' in os.environ
                and original_smart_performance_weight is None
            ):
                os.environ.pop('ML_SMART_PERFORMANCE_WEIGHT', None)
            if original_smart_sigma is not None:
                os.environ['ML_SMART_SIGMA'] = original_smart_sigma
            elif (
                'ML_SMART_SIGMA' in os.environ
                and original_smart_sigma is None
            ):
                os.environ.pop('ML_SMART_SIGMA', None)
            if original_smart_weight_xgb is not None:
                os.environ['ML_SMART_WEIGHT_XGB'] = original_smart_weight_xgb
            elif (
                'ML_SMART_WEIGHT_XGB' in os.environ
                and original_smart_weight_xgb is None
            ):
                os.environ.pop('ML_SMART_WEIGHT_XGB', None)
            if original_smart_weight_lgb is not None:
                os.environ['ML_SMART_WEIGHT_LGB'] = original_smart_weight_lgb
            elif (
                'ML_SMART_WEIGHT_LGB' in os.environ
                and original_smart_weight_lgb is None
            ):
                os.environ.pop('ML_SMART_WEIGHT_LGB', None)
            if original_smart_weight_cat is not None:
                os.environ['ML_SMART_WEIGHT_CAT'] = original_smart_weight_cat
            elif (
                'ML_SMART_WEIGHT_CAT' in os.environ
                and original_smart_weight_cat is None
            ):
                os.environ.pop('ML_SMART_WEIGHT_CAT', None)
            # ✅ CRITICAL FIX: Restore all dynamically set environment variables
            # from features_enabled and enable_* keys (same as online
            # evaluation)
            for key, original_value in original_env_vars.items():
                if original_value is not None:
                    os.environ[key] = original_value
                elif key in os.environ:
                    # Was not set before, remove it
                    os.environ.pop(key, None)

        # 2) Online (adaptive OFF) - HİBRİT YAKLAŞIM: Model'i train_df ile
        # yeniden eğit (adaptive OFF - HPO ile tutarlılık)
        # Initialize for finally safety
        original_adaptive2: Optional[str] = None
        original_seed_bag2: Optional[str] = None
        original_directional2: Optional[str] = None
        # ✅ CRITICAL FIX: Save original smart ensemble parameters for
        # restoration
        original_smart_consensus_weight2: Optional[str] = None
        original_smart_performance_weight2: Optional[
            str
        ] = None
        original_smart_sigma2: Optional[str] = None
        original_smart_weight_xgb2: Optional[str] = None
        original_smart_weight_lgb2: Optional[str] = None
        original_smart_weight_cat2: Optional[str] = None
        # ✅ CRITICAL FIX: Save all dynamically set environment variables from
        # features_enabled
        original_env_vars2: dict = {}
        try:
            # ✅ CRITICAL FIX: Don't use default value - we need to distinguish
            # between "unset" and "set to '0'"
            original_adaptive2 = os.environ.get(
                'ML_USE_ADAPTIVE_LEARNING'
            )  # None if not set
            # ✅ HİBRİT YAKLAŞIM: Adaptive OFF (HPO ile tutarlılık)
            os.environ['ML_USE_ADAPTIVE_LEARNING'] = '0'

            # ✅ CRITICAL FIX: Set feature flags from best_params to match HPO
            # best trial exactly (including seed bagging & directional loss).
            # Don't use defaults; preserve "unset" vs explicit zeros.
            original_seed_bag2 = os.environ.get(
                'ENABLE_SEED_BAGGING'
            )  # None if not set
            original_directional2 = os.environ.get(
                'ML_USE_DIRECTIONAL_LOSS'
            )  # None if not set

            if best_params and isinstance(best_params, dict):
                # Set feature flags from features_enabled dict (HPO best trial)
                features_enabled = best_params.get('features_enabled', {})
                if features_enabled:
                    # ✅ CRITICAL FIX: Save original values before setting new
                    # ones; normalize keys to uppercase env names
                    for key, value in features_enabled.items():
                        env_key = _normalize_feature_flag_key(key)
                        if env_key not in original_env_vars2:
                            original_env_vars2[env_key] = os.environ.get(
                                env_key
                            )
                        os.environ[env_key] = str(value)
                    logger.info(
                        "🔧 %s %sd Online: Feature flags set from "
                        "best_params: %d flags",
                        symbol,
                        horizon,
                        len(features_enabled),
                    )
                    seed_bag_val = os.environ.get(
                        'ENABLE_SEED_BAGGING', 'NOT_SET'
                    )
                    dir_loss_val = os.environ.get(
                        'ML_USE_DIRECTIONAL_LOSS', 'NOT_SET'
                    )
                    logger.info(
                        "🔧 %s %sd Online: ENABLE_SEED_BAGGING=%s, "
                        "ML_USE_DIRECTIONAL_LOSS=%s",
                        symbol,
                        horizon,
                        seed_bag_val,
                        dir_loss_val,
                    )
                else:
                    # Fallback: Also check enable_* keys in best_params
                    feature_flag_keys = [
                        k for k in best_params.keys()
                        if k.startswith('enable_')
                    ]
                    for key in feature_flag_keys:
                        value = best_params[key]
                        # ✅ CRITICAL FIX: Save original values before setting
                        # new ones (fallback path)
                        if key == 'enable_external_features':
                            env_key = 'ENABLE_EXTERNAL_FEATURES'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(
                                    env_key
                                )
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_fingpt_features':
                            env_key = 'ENABLE_FINGPT_FEATURES'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(
                                    env_key
                                )
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_yolo_features':
                            env_key = 'ENABLE_YOLO_FEATURES'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(
                                    env_key
                                )
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_directional_loss':
                            env_key = 'ML_USE_DIRECTIONAL_LOSS'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(
                                    env_key
                                )
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_seed_bagging':
                            env_key = 'ENABLE_SEED_BAGGING'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(
                                    env_key
                                )
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_talib_patterns':
                            env_key = 'ENABLE_TALIB_PATTERNS'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(
                                    env_key
                                )
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_smart_ensemble':
                            env_key = 'ML_USE_SMART_ENSEMBLE'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(
                                    env_key
                                )
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_stacked_short':
                            env_key = 'ML_USE_STACKED_SHORT'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(
                                    env_key
                                )
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_meta_stacking':
                            env_key = 'ENABLE_META_STACKING'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(
                                    env_key
                                )
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_regime_detection':
                            env_key = 'ML_USE_REGIME_DETECTION'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(
                                    env_key
                                )
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_fingpt':
                            env_key = 'ENABLE_FINGPT'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(
                                    env_key
                                )
                            os.environ[env_key] = '1' if value else '0'
                    logger.info(
                        "🔧 %s %sd Online: Feature flags set from enable_* "
                        "keys: %d flags",
                        symbol,
                        horizon,
                        len(feature_flag_keys),
                    )
                # Also set smart-ensemble params from feature_params if
                # available (parity with HPO)
                try:
                    fp = (
                        best_params.get('feature_params', {})
                        if isinstance(best_params, dict)
                        else {}
                    )
                    if isinstance(fp, dict) and fp:
                        # ✅ CRITICAL FIX: Save original values before setting
                        # new ones
                        if 'smart_consensus_weight' in fp:
                            original_smart_consensus_weight2 = os.environ.get(
                                'ML_SMART_CONSENSUS_WEIGHT'
                            )
                            os.environ['ML_SMART_CONSENSUS_WEIGHT'] = str(
                                fp['smart_consensus_weight']
                            )
                        if 'smart_performance_weight' in fp:
                            original_smart_performance_weight2 = (
                                os.environ.get('ML_SMART_PERFORMANCE_WEIGHT')
                            )
                            os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = str(
                                fp['smart_performance_weight']
                            )
                        if 'smart_sigma' in fp:
                            original_smart_sigma2 = os.environ.get(
                                'ML_SMART_SIGMA'
                            )
                            os.environ['ML_SMART_SIGMA'] = str(
                                fp['smart_sigma']
                            )
                        if 'smart_weight_xgb' in fp:
                            original_smart_weight_xgb2 = os.environ.get(
                                'ML_SMART_WEIGHT_XGB'
                            )
                            os.environ['ML_SMART_WEIGHT_XGB'] = str(
                                fp['smart_weight_xgb']
                            )
                        if (
                            'smart_weight_lgbm' in fp
                            or 'smart_weight_lgb' in fp
                        ):
                            original_smart_weight_lgb2 = os.environ.get(
                                'ML_SMART_WEIGHT_LGB'
                            )
                            os.environ['ML_SMART_WEIGHT_LGB'] = str(
                                fp.get(
                                    'smart_weight_lgbm',
                                    fp.get('smart_weight_lgb'),
                                )
                            )
                        if 'smart_weight_cat' in fp:
                            original_smart_weight_cat2 = os.environ.get(
                                'ML_SMART_WEIGHT_CAT'
                            )
                            os.environ['ML_SMART_WEIGHT_CAT'] = str(
                                fp['smart_weight_cat']
                            )
                except Exception as e:
                    logger.debug("Failed to set smart_weight env vars: %s", e)
            else:
                # Fallback: Only disable if best_params not available (should
                # not happen in normal flow)
                logger.warning(
                    "⚠️ %s %sd Online: best_params not available, "
                    "fallback disables seed_bagging & dir_loss",
                    symbol,
                    horizon,
                )
                os.environ['ENABLE_SEED_BAGGING'] = '0'
                # MSE loss (fallback)
                os.environ['ML_USE_DIRECTIONAL_LOSS'] = '0'
            # ✅ CRITICAL FIX: Don't use singleton, create new instance
            # (same as HPO). Avoid shared caches.

            # ✅ FIX: Clear ConfigManager cache to ensure best params are read
            try:
                from bist_pattern.core.config_manager import ConfigManager
                ConfigManager.clear_cache()
            except Exception as e:
                logger.debug(f"Failed to clear ConfigManager cache: {e}")
            # ✅ CRITICAL FIX: Use best trial's seed for evaluation (match HPO
            # best trial). If best_trial_number is not available, fallback to
            # 42. Try to get best_trial_number from best_params first, then
            # from hpo_result.
            best_trial_number = None
            if isinstance(best_params, dict):
                best_trial_number = best_params.get('best_trial_number')
            if best_trial_number is None and hpo_result:
                best_trial_number = hpo_result.get('best_trial_number')
            # Use the same logic as WFV: seed = 42 + best_trial_number
            try:
                eval_seed = (
                    42 + int(best_trial_number)
                    if best_trial_number is not None
                    else 42
                )
            except Exception as e:
                logger.debug("Failed to calculate eval_seed, using 42: %s", e)
                eval_seed = 42
            try:
                import random
                random.seed(eval_seed)
                np.random.seed(eval_seed)
                os.environ['PYTHONHASHSEED'] = str(eval_seed)
                os.environ.setdefault(
                    'OPTUNA_XGB_RANDOM_STATE', str(eval_seed)
                )
                os.environ.setdefault(
                    'OPTUNA_LGB_RANDOM_STATE', str(eval_seed)
                )
                os.environ.setdefault(
                    'OPTUNA_CAT_RANDOM_STATE', str(eval_seed)
                )
                os.environ.setdefault('XGB_SEED', str(eval_seed))
                os.environ.setdefault('LIGHTGBM_SEED', str(eval_seed))
                os.environ.setdefault('CATBOOST_RANDOM_SEED', str(eval_seed))
                logger.info(
                    "🔧 %s %sd Online: Using seed=%s (best_trial=%s) for "
                    "evaluation",
                    symbol,
                    horizon,
                    eval_seed,
                    best_trial_number,
                )
            except Exception as e:
                logger.debug(f"Failed to set seed env vars: {e}")

            # ✅ CRITICAL FIX: Create NEW instance (same as HPO) to avoid
            # singleton cache pollution
            from enhanced_ml_system import EnhancedMLSystem
            ml_online = EnhancedMLSystem()  # New instance, not singleton!
            ml_online.prediction_horizons = [horizon]
            # ✅ CRITICAL FIX: Set base_seeds to match HPO best trial
            # HPO uses: ml.base_seeds = [42 + trial.number]
            # Evaluation should use: ml_online.base_seeds = [42 +
            # best_trial_number]
            # ✅ FIX: If best_trial_number is None, use fallback seed (42) but
            # log warning
            if best_trial_number is None:
                logger.warning(
                    "⚠️ %s %sd Online: best_trial_number not found, "
                    "using fallback seed=42 (may cause alignment issues)",
                    symbol,
                    horizon,
                )
            # Align base_seeds with eval_seed (already 42 + best_trial_number)
            ml_online.base_seeds = [eval_seed]
            # ✅ FIX: Ensure enable_seed_bagging matches HPO best trial (from
            # environment variable). base_seeds single seed disables bagging,
            # but keep flag consistent.
            logger.info(
                "🔧 %s %sd Online: base_seeds=%s, "
                "enable_seed_bagging=%s (HPO trial %s)",
                symbol,
                horizon,
                ml_online.base_seeds,
                ml_online.enable_seed_bagging,
                best_trial_number,
            )
            if reference_r2_map:
                try:
                    ml_online.reference_historical_r2[reference_key] = (
                        reference_r2_map
                    )
                    ml_online.use_reference_historical_r2 = True
                except Exception as ref_err:
                    logger.warning(
                        "⚠️ %s %sd Online: Failed to inject reference R²: %s",
                        symbol,
                        horizon,
                        ref_err,
                    )
                    ml_online.use_reference_historical_r2 = False
            else:
                ml_online.use_reference_historical_r2 = False
                ml_online.reference_historical_r2.pop(reference_key, None)

            # ✅ CRITICAL FIX: Evaluation mode - skip Phase 2 to match HPO data
            # usage. HPO uses full train_df (adaptive OFF). Online adaptive ON
            # would split train_df; skipping Phase 2 keeps data volume aligned.
            os.environ['ML_SKIP_ADAPTIVE_PHASE2'] = '1'

            # ⚡ NEW: Evaluate on all splits and average DirHit (same as HPO
            # and WFV)
            split_dirhits_online = []
            for split_idx, (train_end_idx, test_end_idx) in enumerate(
                wfv_splits, 1
            ):
                train_df_split = df.iloc[:train_end_idx].copy()
                test_df_split = df.iloc[train_end_idx:test_end_idx].copy()

                logger.info(
                    "📊 %s %sd Online Split %s/%s: train=%s days, "
                    "test=%s days",
                    symbol,
                    horizon,
                    split_idx,
                    len(wfv_splits),
                    len(train_df_split),
                    len(test_df_split),
                )
                logger.info(
                    "🔍 [eval-debug] %s %sd Online Split %s: "
                    "Train %s → %s",
                    symbol,
                    horizon,
                    split_idx,
                    train_df_split.index.min(),
                    train_df_split.index.max(),
                )
                logger.info(
                    "🔍 [eval-debug] %s %sd Online Split %s: "
                    "Test %s → %s",
                    symbol,
                    horizon,
                    split_idx,
                    test_df_split.index.min(),
                    test_df_split.index.max(),
                )

                y_true_split = self._compute_returns(test_df_split, horizon)
                min_test_days = horizon + 10
                if len(test_df_split) < min_test_days:
                    logger.warning(
                        "⚠️ %s %sd Online Split %s: Insufficient test data "
                        "(%s days, need %s)",
                        symbol,
                        horizon,
                        split_idx,
                        len(test_df_split),
                        min_test_days,
                    )
                    continue

                # ✅ HİBRİT YAKLAŞIM: Model'i train_df ile yeniden eğit
                # (best params, adaptive OFF). Evaluation'da model'i train_df
                # ile yeniden eğiterek data leakage'ı önlüyoruz. Phase 2'yi
                # atlayarak HPO ile aynı veri miktarını kullanıyoruz.
                # Log evaluation environment summary (only for first split)
                if split_idx == 1:
                    logger.info(
                        "🔧 Eval env (Online): adaptive=0, skip_phase2=1, "
                        "seed_bagging=%s, directional_loss=%s, smart=%s, "
                        "stacked=%s, regime=%s",
                        os.environ.get('ENABLE_SEED_BAGGING', '0'),
                        os.environ.get('ML_USE_DIRECTIONAL_LOSS', '0'),
                        os.environ.get('ML_USE_SMART_ENSEMBLE', '1'),
                        os.environ.get('ML_USE_STACKED_SHORT', '1'),
                        os.environ.get('ML_USE_REGIME_DETECTION', '1'),
                    )
                    logger.info(
                        "🔄 %s %sd Online: Model'i train_df ile yeniden "
                        "eğitiyoruz (best params, adaptive OFF, skip P2, "
                        "new instance)",
                        symbol,
                        horizon,
                    )
                    logger.info(
                        "   ✅ HİBRİT YAKLAŞIM: ML_USE_ADAPTIVE_LEARNING=0, "
                        "ML_SKIP_ADAPTIVE_PHASE2=1 → HPO ile aynı veri ve "
                        "ayarlar"
                    )

                train_result2 = ml_online.train_enhanced_models(
                    symbol, train_df_split
                )
                if not train_result2:
                    logger.warning(
                        "⚠️ %s %sd Online Split %s: Model eğitimi başarısız, "
                        "split atlanıyor",
                        symbol,
                        horizon,
                        split_idx,
                    )
                    continue

                # ✅ HİBRİT YAKLAŞIM: enhanced_ml_system.py adaptive OFF
                # Phase 1: Tüm train_df ile eğitim (HPO ile aynı)
                # Phase 2: SKIP (ML_SKIP_ADAPTIVE_PHASE2=1)
                if split_idx == 1:
                    logger.info(
                        "✅ %s %sd Online: Model eğitimi tamamlandı "
                        "(adaptive OFF - HPO ile tutarlılık)",
                        symbol,
                        horizon,
                    )

                preds2 = np.full(len(test_df_split), np.nan, dtype=float)
                if reference_r2_map:
                    actual_metrics_online = (
                        _extract_model_metrics_from_train_result(
                            train_result2, horizon
                        )
                    )
                    if actual_metrics_online:
                        for model_name, ref_val in reference_r2_map.items():
                            actual_val = actual_metrics_online.get(
                                model_name, {}
                            ).get('raw_r2')
                            if (
                                actual_val is None
                                or not math.isfinite(actual_val)
                            ):
                                continue
                            delta = actual_val - ref_val
                            logger.info(
                                "📏 %s %sd Online Split %s: %s raw_r2 "
                                "HPO=%.4f, train=%.4f, delta=%+.4f",
                                symbol,
                                horizon,
                                split_idx,
                                model_name,
                                ref_val,
                                actual_val,
                                delta,
                            )
                valid_predictions2 = 0
                for t in range(len(test_df_split) - horizon):
                    try:
                        cur = pd.concat(
                            [train_df_split, test_df_split.iloc[:t + 1]],
                            axis=0,
                        ).copy()
                        p = ml_online.predict_enhanced(symbol, cur)
                        if isinstance(p, dict):
                            key = f"{horizon}d"
                            obj = p.get(key)
                            if isinstance(obj, dict):
                                pred_price = obj.get('ensemble_prediction')
                                if (
                                    isinstance(pred_price, (int, float))
                                    and not np.isnan(pred_price)
                                ):
                                    last_close = float(cur['close'].iloc[-1])
                                    if last_close > 0:
                                        preds2[t] = (
                                            float(pred_price)
                                            / last_close
                                            - 1.0
                                        )
                                        valid_predictions2 += 1
                    except Exception as e:
                        logger.debug(
                            "Failed to process prediction (online): %s", e
                        )
                        continue

                # 🔍 DEBUG: Calculate detailed metrics (same as HPO)
                valid_mask2 = ~np.isnan(preds2) & ~np.isnan(y_true_split)
                valid_count2 = valid_mask2.sum()

                if valid_count2 > 0:
                    # DirHit threshold (from evaluation_spec if provided)
                    thr2 = 0.005
                    try:
                        _thr_env_online = os.getenv('DIRHIT_THRESHOLD')
                        if _thr_env_online is not None:
                            thr2 = float(_thr_env_online)
                    except Exception as e:
                        logger.debug(
                            "Failed to get DIRHIT_THRESHOLD (online): %s", e
                        )
                    # DirHit
                    dh2 = self._dirhit(y_true_split, preds2, thr=thr2)

                    # RMSE and MAPE
                    y_true_valid2 = y_true_split[valid_mask2]
                    preds_valid2 = preds2[valid_mask2]
                    rmse2 = np.sqrt(
                        np.mean((y_true_valid2 - preds_valid2) ** 2)
                    )
                    mape2 = np.mean(
                        np.abs(
                            (y_true_valid2 - preds_valid2)
                            / (y_true_valid2 + 1e-8)
                        )
                    ) * 100

                    # Threshold mask statistics
                    mask_count2 = (
                        (np.abs(y_true_valid2) > thr2)
                        & (np.abs(preds_valid2) > thr2)
                    ).sum()
                    mask_pct2 = (
                        (mask_count2 / valid_count2) * 100
                        if valid_count2 > 0
                        else 0
                    )

                    # ✅ FIX: Log prediction magnitude statistics for
                    # mask_count=0 debugging
                    if mask_count2 == 0 and valid_count2 > 0:
                        pred_abs_max2 = (
                            float(np.abs(preds_valid2).max())
                            if len(preds_valid2) > 0
                            else 0.0
                        )
                        pred_abs_mean2 = (
                            float(np.abs(preds_valid2).mean())
                            if len(preds_valid2) > 0
                            else 0.0
                        )
                        y_true_abs_max2 = (
                            float(np.abs(y_true_valid2).max())
                            if len(y_true_valid2) > 0
                            else 0.0
                        )
                        y_true_abs_mean2 = (
                            float(np.abs(y_true_valid2).mean())
                            if len(y_true_valid2) > 0
                            else 0.0
                        )
                        logger.warning(
                            "⚠️ %s %sd Online Split %s: mask_count=0 "
                            "(threshold=%.3f) - pred_abs_max=%.6f, "
                            "pred_abs_mean=%.6f, y_true_abs_max=%.6f, "
                            "y_true_abs_mean=%.6f",
                            symbol,
                            horizon,
                            split_idx,
                            thr2,
                            pred_abs_max2,
                            pred_abs_mean2,
                            y_true_abs_max2,
                            y_true_abs_mean2,
                        )

                    # Direction statistics
                    direction_matches2 = (
                        np.sign(y_true_valid2) == np.sign(preds_valid2)
                    ).sum()
                    direction_pct2 = (
                        (direction_matches2 / valid_count2) * 100
                        if valid_count2 > 0
                        else 0
                    )

                    # 🔍 DEBUG: Log detailed metrics (only for first split)
                    if split_idx == 1:
                        logger.info(
                            "🔍 [eval-debug] %s %sd Online: METRICS:",
                            symbol,
                            horizon,
                        )
                        logger.info(
                            "🔍 [eval-debug]   Valid predictions: %s/%s",
                            valid_count2,
                            len(preds2),
                        )
                        logger.info(
                            "🔍 [eval-debug]   DirHit (thr=%.3f): %.2f%% "
                            "(mask_count=%s, mask_pct=%.1f%%)",
                            thr2,
                            dh2,
                            mask_count2,
                            mask_pct2,
                        )
                        logger.info(
                            "🔍 [eval-debug]   Direction match (all): "
                            "%.2f%% (%s/%s)",
                            direction_pct2,
                            direction_matches2,
                            valid_count2,
                        )
                        logger.info("🔍 [eval-debug]   RMSE: %.6f", rmse2)
                        logger.info("🔍 [eval-debug]   MAPE: %.2f%%", mape2)
                        logger.info(
                            "🔍 [eval-debug]   Seed = %s (best_trial=%s)",
                            eval_seed,
                            best_trial_number,
                        )
                        logger.info(
                            "🔍 [eval-debug]   Adaptive Learning: OFF "
                            "(skip_phase2=1, full train_df - HPO parity)"
                        )

                    # ✅ FIX: Low-support gating (same as HPO)
                    # Use same default values as HPO to ensure consistency
                    low_support2 = False
                    _min_mc2 = 0
                    _min_mp2 = 0.0
                    try:
                        _min_mc2 = int(
                            os.getenv('HPO_MIN_MASK_COUNT', '0')
                        )
                    except Exception as e:
                        logger.debug(
                            "Failed to get HPO_MIN_MASK_COUNT (online), "
                            "using 0: %s",
                            e,
                        )
                        _min_mc2 = 0
                    try:
                        _min_mp2 = float(
                            os.getenv('HPO_MIN_MASK_PCT', '0.0')
                        )
                    except Exception as e:
                        logger.debug(
                            "Failed to get HPO_MIN_MASK_PCT (online), "
                            "using 0.0: %s",
                            e,
                        )
                        _min_mp2 = 0.0
                    if (
                        (_min_mc2 > 0 and mask_count2 < _min_mc2)
                        or (_min_mp2 > 0.0 and mask_pct2 < _min_mp2)
                    ):
                        low_support2 = True
                        logger.info(
                            "⚠️ %s %sd Online Split %s: LOW_SUPPORT → "
                            "exclude from avg (DirHit=%.2f%%, "
                            "mask_count=%s, mask_pct=%.1f%%, "
                            "min_mc=%s, min_mp=%s%%)",
                            symbol,
                            horizon,
                            split_idx,
                            dh2,
                            mask_count2,
                            mask_pct2,
                            _min_mc2,
                            _min_mp2,
                        )
                    if not np.isnan(dh2) and not low_support2:
                        split_dirhits_online.append(dh2)
                        logger.info(
                            "✅ %s %sd Online Split %s: DirHit=%.2f%% "
                            "(valid=%s/%s, mask=%s, RMSE=%.6f, MAPE=%.2f%%)",
                            symbol,
                            horizon,
                            split_idx,
                            dh2,
                            valid_count2,
                            len(preds2),
                            mask_count2,
                            rmse2,
                            mape2,
                        )
                    else:
                        logger.warning(
                            "⚠️ %s %sd Online Split %s: DirHit calculation "
                            "returned NaN",
                            symbol,
                            horizon,
                            split_idx,
                        )
                else:
                    logger.warning(
                        "⚠️ %s %sd Online Split %s: No valid predictions!",
                        symbol,
                        horizon,
                        split_idx,
                    )

            # Average DirHit across all splits
            if split_dirhits_online:
                avg_dirhit_online = float(np.mean(split_dirhits_online))
                logger.info(
                    "✅ %s %sd Online: Avg DirHit across %s splits: %.2f%% "
                    "(splits=%s)",
                    symbol,
                    horizon,
                    len(split_dirhits_online),
                    avg_dirhit_online,
                    split_dirhits_online,
                )
                results['online'] = avg_dirhit_online
            else:
                logger.warning(
                    "⚠️ %s %sd Online: No valid DirHit from any split",
                    symbol,
                    horizon,
                )
                results['online'] = None
        except Exception as e:
            logger.error(
                "❌ Online evaluation error for %s %sd: %s", symbol, horizon, e
            )
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # ✅ CRITICAL FIX: Restore all modified environment variables
            # ✅ Restore adaptive learning - remove if it wasn't set before
            if original_adaptive2 is not None:
                os.environ['ML_USE_ADAPTIVE_LEARNING'] = original_adaptive2
            elif (
                'ML_USE_ADAPTIVE_LEARNING' in os.environ
                and original_adaptive2 is None
            ):
                # Was not set before, remove it
                os.environ.pop('ML_USE_ADAPTIVE_LEARNING', None)
            # ✅ Restore seed bagging - remove if it wasn't set before
            if original_seed_bag2 is not None:
                os.environ['ENABLE_SEED_BAGGING'] = original_seed_bag2
            elif (
                'ENABLE_SEED_BAGGING' in os.environ
                and original_seed_bag2 is None
            ):
                # Was not set before, remove it
                os.environ.pop('ENABLE_SEED_BAGGING', None)
            # ✅ Restore directional loss - remove if it wasn't set before
            if original_directional2 is not None:
                os.environ['ML_USE_DIRECTIONAL_LOSS'] = original_directional2
            elif (
                'ML_USE_DIRECTIONAL_LOSS' in os.environ
                and original_directional2 is None
            ):
                # Was not set before, remove it
                os.environ.pop('ML_USE_DIRECTIONAL_LOSS', None)
            # ✅ CRITICAL FIX: Restore smart ensemble parameters
            if original_smart_consensus_weight2 is not None:
                os.environ['ML_SMART_CONSENSUS_WEIGHT'] = (
                    original_smart_consensus_weight2
                )
            elif (
                'ML_SMART_CONSENSUS_WEIGHT' in os.environ
                and original_smart_consensus_weight2 is None
            ):
                # Was not set before, remove it
                os.environ.pop('ML_SMART_CONSENSUS_WEIGHT', None)
            if original_smart_performance_weight2 is not None:
                os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = (
                    original_smart_performance_weight2
                )
            elif (
                'ML_SMART_PERFORMANCE_WEIGHT' in os.environ
                and original_smart_performance_weight2 is None
            ):
                os.environ.pop('ML_SMART_PERFORMANCE_WEIGHT', None)
            if original_smart_sigma2 is not None:
                os.environ['ML_SMART_SIGMA'] = original_smart_sigma2
            elif (
                'ML_SMART_SIGMA' in os.environ
                and original_smart_sigma2 is None
            ):
                os.environ.pop('ML_SMART_SIGMA', None)
            if original_smart_weight_xgb2 is not None:
                os.environ['ML_SMART_WEIGHT_XGB'] = original_smart_weight_xgb2
            elif (
                'ML_SMART_WEIGHT_XGB' in os.environ
                and original_smart_weight_xgb2 is None
            ):
                os.environ.pop('ML_SMART_WEIGHT_XGB', None)
            if original_smart_weight_lgb2 is not None:
                os.environ['ML_SMART_WEIGHT_LGB'] = original_smart_weight_lgb2
            elif (
                'ML_SMART_WEIGHT_LGB' in os.environ
                and original_smart_weight_lgb2 is None
            ):
                os.environ.pop('ML_SMART_WEIGHT_LGB', None)
            if original_smart_weight_cat2 is not None:
                os.environ['ML_SMART_WEIGHT_CAT'] = original_smart_weight_cat2
            elif (
                'ML_SMART_WEIGHT_CAT' in os.environ
                and original_smart_weight_cat2 is None
            ):
                os.environ.pop('ML_SMART_WEIGHT_CAT', None)
            # ✅ CRITICAL FIX: Restore all dynamically set environment
            # variables from features_enabled
            for key, original_value in original_env_vars2.items():
                try:
                    if original_value is not None:
                        os.environ[key] = original_value
                    elif key in os.environ:
                        # Was not set before, remove it
                        os.environ.pop(key, None)
                except Exception as e:
                    logger.debug(f"Failed to restore env var {key}: {e}")
            try:
                # ✅ Clean up evaluation mode flag
                os.environ.pop('ML_SKIP_ADAPTIVE_PHASE2', None)
            except Exception as e:
                logger.debug("Failed to pop ML_SKIP_ADAPTIVE_PHASE2: %s", e)

        # Mini-CV bloğu kaldırıldı (bakım ve lint sadeleştirme)

            # ✅ DEBUG DUMP: Save evaluation summary for deep analysis
            # (HPO parity checks). Note: Overwrites previous results -
            # historical results are preserved in state.json.
        try:
            from pathlib import Path as _Path
            dbg_dir = _Path('/opt/bist-pattern/results/eval_debug')
            dbg_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                'symbol': symbol,
                'horizon': int(horizon),
                'timestamp': datetime.now().isoformat(),
                'wfv_dirhit': results.get('wfv'),
                'online_dirhit': results.get('online'),
            }
            dbg_path = dbg_dir / f"{symbol}_{horizon}d_eval.json"
            with open(dbg_path, 'w') as _f:
                json.dump(payload, _f)
            logger.info("🧪 Eval debug written: %s", dbg_path)
        except Exception as _e:
            logger.debug("Eval debug dump failed: %s", _e)

        return results

    def run_training(
        self,
        symbol: str,
        horizon: int,
        best_params: Dict,
        hpo_result: Optional[Dict] = None
    ) -> Optional[Dict[str, Optional[float]]]:
        """Run training with best params for a symbol-horizon pair

        NOTE: Runs with adaptive learning ENABLED (full training with test
        data)

        Args:
            symbol: Stock symbol
            horizon: Prediction horizon
            best_params: Best hyperparameters dict
            hpo_result: Full HPO result dict (includes features_enabled,
                feature_params, etc.)
        """
        try:
            logger.info(
                "🎯 Starting training for %s %sd with best params...",
                symbol,
                horizon,
            )

            # Set parameters as environment variables
            from scripts.train_completed_hpo_with_best_params import (
                set_hpo_params_as_env,
            )
            set_hpo_params_as_env(best_params, horizon)

            # ✅ CRITICAL FIX: Set feature flags from hpo_result (if available)
            # hpo_result contains 'features_enabled' with flags from best trial
            if hpo_result and 'features_enabled' in hpo_result:
                features_enabled = hpo_result['features_enabled']
                for key, value in features_enabled.items():
                    os.environ[key] = str(value)
                logger.info(
                    "🔧 %s %sd: Feature flags set from hpo_result: %s flags",
                    symbol,
                    horizon,
                    len(features_enabled),
                )
                # Smart-ensemble params from feature_params (HPO best trial)
                try:
                    fp = (
                        best_params.get('feature_params')
                        or hpo_result.get('feature_params')
                        or {}
                    )
                    if isinstance(fp, dict) and fp:
                        if 'smart_consensus_weight' in fp:
                            os.environ['ML_SMART_CONSENSUS_WEIGHT'] = str(
                                fp['smart_consensus_weight']
                            )
                        if 'smart_performance_weight' in fp:
                            os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = str(
                                fp['smart_performance_weight']
                            )
                        if 'smart_sigma' in fp:
                            os.environ['ML_SMART_SIGMA'] = str(
                                fp['smart_sigma']
                            )
                        if 'smart_weight_xgb' in fp:
                            os.environ['ML_SMART_WEIGHT_XGB'] = str(
                                fp['smart_weight_xgb']
                            )
                        if (
                            'smart_weight_lgbm' in fp
                            or 'smart_weight_lgb' in fp
                        ):
                            os.environ['ML_SMART_WEIGHT_LGB'] = str(
                                fp.get(
                                    'smart_weight_lgbm',
                                    fp.get('smart_weight_lgb'),
                                )
                            )
                        if 'smart_weight_cat' in fp:
                            os.environ['ML_SMART_WEIGHT_CAT'] = str(
                                fp['smart_weight_cat']
                            )
                except Exception as e:
                    logger.debug(f"Failed to set smart_weight env vars: {e}")
            # Also set feature flags from best_params keys (enable_*)
            elif isinstance(best_params, dict):
                feature_flag_keys = [
                    k for k in best_params.keys() if k.startswith('enable_')
                ]
                for key in feature_flag_keys:
                    value = best_params[key]
                    if key == 'enable_external_features':
                        os.environ['ENABLE_EXTERNAL_FEATURES'] = (
                            '1' if value else '0'
                        )
                    elif key == 'enable_fingpt_features':
                        os.environ['ENABLE_FINGPT_FEATURES'] = (
                            '1' if value else '0'
                        )
                    elif key == 'enable_yolo_features':
                        os.environ['ENABLE_YOLO_FEATURES'] = (
                            '1' if value else '0'
                        )
                    elif key == 'enable_directional_loss':
                        os.environ['ML_USE_DIRECTIONAL_LOSS'] = (
                            '1' if value else '0'
                        )
                    elif key == 'enable_seed_bagging':
                        os.environ['ENABLE_SEED_BAGGING'] = (
                            '1' if value else '0'
                        )
                    elif key == 'enable_talib_patterns':
                        os.environ['ENABLE_TALIB_PATTERNS'] = (
                            '1' if value else '0'
                        )
                    elif key == 'enable_smart_ensemble':
                        os.environ['ML_USE_SMART_ENSEMBLE'] = (
                            '1' if value else '0'
                        )
                    elif key == 'enable_stacked_short':
                        os.environ['ML_USE_STACKED_SHORT'] = (
                            '1' if value else '0'
                        )
                    elif key == 'enable_meta_stacking':
                        os.environ['ENABLE_META_STACKING'] = (
                            '1' if value else '0'
                        )
                    elif key == 'enable_regime_detection':
                        os.environ['ML_USE_REGIME_DETECTION'] = (
                            '1' if value else '0'
                        )
                    elif key == 'enable_fingpt':
                        os.environ['ENABLE_FINGPT'] = '1' if value else '0'
                if feature_flag_keys:
                    logger.info(
                        "🔧 %s %sd: Feature flags set from best_params: %s "
                        "flags",
                        symbol,
                        horizon,
                        len(feature_flag_keys),
                    )

            # Set horizon
            os.environ['ML_HORIZONS'] = str(horizon)

            # ✅ UPDATED: Set feature flags from HPO best_params
            # (features_enabled dict). Use HPO-optimized flags, but always
            # enable adaptive learning for training.
            features_enabled = best_params.get('features_enabled', {})
            if features_enabled:
                # Set feature flags from HPO results
                os.environ['ENABLE_EXTERNAL_FEATURES'] = features_enabled.get(
                    'ENABLE_EXTERNAL_FEATURES', '1'
                )
                os.environ['ENABLE_FINGPT_FEATURES'] = features_enabled.get(
                    'ENABLE_FINGPT_FEATURES', '1'
                )
                os.environ['ENABLE_YOLO_FEATURES'] = features_enabled.get(
                    'ENABLE_YOLO_FEATURES', '1'
                )
                os.environ['ML_USE_DIRECTIONAL_LOSS'] = features_enabled.get(
                    'ML_USE_DIRECTIONAL_LOSS', '1'
                )
                os.environ['ENABLE_SEED_BAGGING'] = features_enabled.get(
                    'ENABLE_SEED_BAGGING', '1'
                )
                os.environ['ENABLE_TALIB_PATTERNS'] = features_enabled.get(
                    'ENABLE_TALIB_PATTERNS', '1'
                )
                os.environ['ML_USE_SMART_ENSEMBLE'] = features_enabled.get(
                    'ML_USE_SMART_ENSEMBLE', '1'
                )
                os.environ['ML_USE_STACKED_SHORT'] = features_enabled.get(
                    'ML_USE_STACKED_SHORT', '1'
                )
                os.environ['ENABLE_META_STACKING'] = features_enabled.get(
                    'ENABLE_META_STACKING', '1'
                )
                os.environ['ML_USE_REGIME_DETECTION'] = (
                    features_enabled.get('ML_USE_REGIME_DETECTION', '1')
                )
                os.environ['ENABLE_FINGPT'] = features_enabled.get(
                    'ENABLE_FINGPT', '1'
                )
                # ✅ Set model enable flags from HPO results (model_choice).
                # Ensures training uses only models HPO recommended.
                os.environ['ENABLE_XGBOOST'] = features_enabled.get(
                    'ENABLE_XGBOOST', '0'
                )
                os.environ['ENABLE_LIGHTGBM'] = features_enabled.get(
                    'ENABLE_LIGHTGBM', '0'
                )
                os.environ['ENABLE_CATBOOST'] = features_enabled.get(
                    'ENABLE_CATBOOST', '0'
                )
                xgb_flag = os.environ.get('ENABLE_XGBOOST')
                lgb_flag = os.environ.get('ENABLE_LIGHTGBM')
                cat_flag = os.environ.get('ENABLE_CATBOOST')
                logger.info(
                    "✅ %s %sd: HPO flags (XGB=%s, LGB=%s, CAT=%s)",
                    symbol,
                    horizon,
                    xgb_flag,
                    lgb_flag,
                    cat_flag,
                )
                # Smart-ensemble params from feature_params (for training)
                try:
                    fp = best_params.get('feature_params', {})
                    if isinstance(fp, dict) and fp:
                        if 'smart_consensus_weight' in fp:
                            os.environ['ML_SMART_CONSENSUS_WEIGHT'] = str(
                                fp['smart_consensus_weight']
                            )
                        if 'smart_performance_weight' in fp:
                            os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = str(
                                fp['smart_performance_weight']
                            )
                        if 'smart_sigma' in fp:
                            os.environ['ML_SMART_SIGMA'] = str(
                                fp['smart_sigma']
                            )
                        if 'smart_weight_xgb' in fp:
                            os.environ['ML_SMART_WEIGHT_XGB'] = str(
                                fp['smart_weight_xgb']
                            )
                        if (
                            'smart_weight_lgbm' in fp
                            or 'smart_weight_lgb' in fp
                        ):
                            os.environ['ML_SMART_WEIGHT_LGB'] = str(
                                fp.get(
                                    'smart_weight_lgbm',
                                    fp.get('smart_weight_lgb'),
                                )
                            )
                        if 'smart_weight_cat' in fp:
                            os.environ['ML_SMART_WEIGHT_CAT'] = str(
                                fp['smart_weight_cat']
                            )
                except Exception as e:
                    logger.debug(f"Failed to set smart_weight env vars: {e}")
            else:
                # Fallback: enable all features if features_enabled missing
                os.environ['ENABLE_EXTERNAL_FEATURES'] = '1'
                os.environ['ENABLE_FINGPT_FEATURES'] = '1'
                os.environ['ENABLE_YOLO_FEATURES'] = '1'
                os.environ['ML_USE_DIRECTIONAL_LOSS'] = '1'
                os.environ['ENABLE_SEED_BAGGING'] = '1'
                os.environ['ENABLE_TALIB_PATTERNS'] = '1'
                os.environ['ML_USE_SMART_ENSEMBLE'] = '1'
                os.environ['ML_USE_STACKED_SHORT'] = '1'
                os.environ['ENABLE_META_STACKING'] = '1'
                os.environ['ML_USE_REGIME_DETECTION'] = '1'
                os.environ['ENABLE_FINGPT'] = '1'
                logger.warning(
                    "⚠️ %s %sd: features_enabled missing; using all features "
                    "(fallback)",
                    symbol,
                    horizon,
                )

            # ✅ HİBRİT YAKLAŞIM: Training'de adaptive learning KAPALI
            # (HPO ile tutarlılık). Plan: HPO ve Training aynı veri
            # miktarını kullanmalı; cycle zaten incremental learning etkisi
            # yaratıyor (yeni verilerle yeniden HPO).
            os.environ['ML_USE_ADAPTIVE_LEARNING'] = '0'
            # Phase 2 skip (HPO ile aynı)
            os.environ['ML_SKIP_ADAPTIVE_PHASE2'] = '1'

            # Set feature internal parameters from HPO (if available in
            # best_params). These are already set by set_hpo_params_as_env,
            # but we ensure they're set here too.
            if 'feature_params' in best_params:
                feature_params = best_params['feature_params']
                if 'n_seeds' in feature_params:
                    os.environ['N_SEEDS'] = str(feature_params['n_seeds'])
                if 'meta_stacking_alpha' in feature_params:
                    os.environ['ML_META_STACKING_ALPHA'] = str(
                        feature_params['meta_stacking_alpha']
                    )
                if 'yolo_min_conf' in feature_params:
                    os.environ['YOLO_MIN_CONF'] = str(
                        feature_params['yolo_min_conf']
                    )
                if 'smart_consensus_weight' in feature_params:
                    os.environ['ML_SMART_CONSENSUS_WEIGHT'] = str(
                        feature_params['smart_consensus_weight']
                    )
                if 'smart_performance_weight' in feature_params:
                    os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = str(
                        feature_params['smart_performance_weight']
                    )
                if 'smart_sigma' in feature_params:
                    os.environ['ML_SMART_SIGMA'] = str(
                        feature_params['smart_sigma']
                    )
                    # ✅ NEW: Per-model prior weights for smart ensemble
                    if 'smart_weight_xgb' in feature_params:
                        os.environ['ML_SMART_WEIGHT_XGB'] = str(
                            feature_params['smart_weight_xgb']
                        )
                    if 'smart_weight_lgbm' in feature_params:
                        os.environ['ML_SMART_WEIGHT_LGB'] = str(
                            feature_params['smart_weight_lgbm']
                        )
                    if 'smart_weight_cat' in feature_params:
                        os.environ['ML_SMART_WEIGHT_CAT'] = str(
                            feature_params['smart_weight_cat']
                        )
                if 'fingpt_confidence_threshold' in feature_params:
                    os.environ['FINGPT_CONFIDENCE_THRESHOLD'] = str(
                        feature_params['fingpt_confidence_threshold']
                    )
                if 'ml_loss_mse_weight' in feature_params:
                    os.environ['ML_LOSS_MSE_WEIGHT'] = str(
                        feature_params['ml_loss_mse_weight']
                    )
                if 'ml_loss_threshold' in feature_params:
                    os.environ['ML_LOSS_THRESHOLD'] = str(
                        feature_params['ml_loss_threshold']
                    )
                if 'ml_dir_penalty' in feature_params:
                    os.environ['ML_DIR_PENALTY'] = str(
                        feature_params['ml_dir_penalty']
                    )

            os.environ['STRICT_HORIZON_FEATURES'] = '1'   # Horizon-specific
            logger.info(
                "✅ %s %sd: Training with HPO-optimized feature flags "
                "(adaptive OFF - HPO tutarlılığı)",
                symbol,
                horizon,
            )

            # Clear ConfigManager cache
            try:
                from bist_pattern.core.config_manager import ConfigManager
                ConfigManager.clear_cache()
            except Exception as e:
                logger.debug(f"Failed to clear ConfigManager cache: {e}")

            # ✅ CRITICAL FIX: Clear singleton using thread-safe function
            from enhanced_ml_system import clear_enhanced_ml_system
            clear_enhanced_ml_system()

            with app.app_context():
                # ✅ Use fetch_prices (cache bypass) so training matches HPO
                # data source (direct DB, no cache).
                from sqlalchemy import create_engine
                from scripts.optuna_hpo_with_feature_flags import fetch_prices

                # ✅ Read password from secret file (systemd parity); env may
                # be wrong.
                secret_file = Path('/opt/bist-pattern/.secrets/db_password')
                if secret_file.exists():
                    with open(secret_file, 'r') as f:
                        db_password = f.read().strip()
                    db_url = (
                        "postgresql://bist_user:"
                        f"{db_password}@127.0.0.1:6432/bist_pattern_db"
                    )
                else:
                    # Fallback: try environment variable
                    db_url = os.getenv(
                        'DATABASE_URL',
                        "postgresql://bist_user:5ex5chan5GE5*"
                        "@127.0.0.1:6432/bist_pattern_db",
                    )
                logger.warning(
                    "⚠️ %s %sd: Secret file missing; using env/fallback",
                    symbol,
                    horizon,
                )
                db_url = db_url.strip()
                # Force PgBouncer port if somehow 5432 got in
                if ':5432/' in db_url:
                    db_url = db_url.replace(':5432/', ':6432/')
                    logger.warning(
                        "⚠️ %s %sd: Fixed DATABASE_URL port 5432→6432",
                        symbol,
                        horizon,
                    )
                # Set in environment for downstream use
                os.environ['DATABASE_URL'] = db_url
                engine = create_engine(
                    db_url, pool_pre_ping=True, pool_recycle=3600
                )
                try:
                    df = fetch_prices(engine, symbol, limit=1200)
                finally:
                    # ✅ FIX: Dispose engine to release database connections
                    # back to pool
                    engine.dispose()

                if df is None or len(df) < 50:
                    logger.warning(
                        "❌ %s: Insufficient data (%s days)",
                        symbol,
                        len(df) if df is not None else 0,
                    )
                    # ✅ FIX: Return special value to indicate insufficient data
                    # (for skip logic)
                    return None

                ml = get_enhanced_ml_system()

                # Train with adaptive learning DISABLED (HPO ile tutarlılık)
                result = ml.train_enhanced_models(symbol, df)

                if result:
                    # Save model
                    try:
                        ml.save_enhanced_models(symbol)
                        # ✅ CRITICAL FIX: Verify model was saved successfully
                        if not ml.has_trained_models(symbol):
                            logger.error(
                                "❌ Model kaydetme başarısız: %s %sd - "
                                "dosya yok",
                                symbol,
                                horizon,
                            )
                            return None
                        logger.info(
                            "✅ Model kaydedildi ve doğrulandı: %s %sd",
                            symbol,
                            horizon,
                        )
                    except Exception as save_err:
                        logger.error(
                            "❌ Model kaydetme hatası %s %sd: %s",
                            symbol,
                            horizon,
                            save_err,
                        )
                        import traceback
                        logger.error(traceback.format_exc())
                        return None

                    logger.info(
                        "✅ %s %sd: Training completed (adaptive OFF - HPO "
                        "parity)",
                        symbol,
                        horizon,
                    )

                    # ✅ HİBRİT: Evaluate WFV (adaptive OFF) and Online
                    # (adaptive OFF) DirHit. Pass best_params/hpo_result so
                    # evaluation matches HPO.
                    ev = self._evaluate_training_dirhits(
                        symbol,
                        horizon,
                        df,
                        best_params=best_params,
                        hpo_result=hpo_result,
                    )

                    # Return WFV (adaptive OFF) and Online (adaptive OFF)
                    # DirHit for comparison
                    wfv_dirhit = ev.get('wfv')
                    adaptive_dirhit = ev.get('online')

                    if wfv_dirhit is not None:
                        logger.info(
                            "📊 %s %sd: WFV DirHit (adaptive OFF) = %.2f%%",
                            symbol,
                            horizon,
                            wfv_dirhit,
                        )
                    else:
                        logger.warning(
                            "⚠️ %s %sd: WFV DirHit could not be evaluated",
                            symbol,
                            horizon,
                        )

                    if adaptive_dirhit is not None:
                        logger.info(
                            "📊 %s %sd: Online DirHit (adaptive OFF) = %.2f%%",
                            symbol,
                            horizon,
                            adaptive_dirhit,
                        )
                    else:
                        logger.warning(
                            "⚠️ %s %sd: Adaptive DirHit could not be "
                            "evaluated",
                            symbol,
                            horizon,
                        )

                    # Compare with HPO DirHit
                    if wfv_dirhit is not None:
                        # Get HPO DirHit from task state
                        task_key = f"{symbol}_{horizon}d"
                        task = self.state.get(task_key)
                        if task and task.hpo_dirhit is not None:
                            hpo_dirhit = task.hpo_dirhit
                            diff = wfv_dirhit - hpo_dirhit
                            logger.info(
                                "🔍 %s %sd: WFV=%.2f%% vs HPO=%.2f%% "
                                "(Δ=%+.2f%%)",
                                symbol,
                                horizon,
                                wfv_dirhit,
                                hpo_dirhit,
                                diff,
                            )
                            if abs(diff) < 1.0:
                                logger.info(
                                    "✅ %s %sd: WFV DirHit matches HPO "
                                    "DirHit (difference < 1%%)",
                                    symbol,
                                    horizon,
                                )
                            else:
                                logger.warning(
                                    "⚠️ %s %sd: WFV DirHit differs from HPO "
                                    "by %.2f%%",
                                    symbol,
                                    horizon,
                                    abs(diff),
                                )

                    return {
                        'wfv_dirhit': wfv_dirhit,
                        'adaptive_dirhit': adaptive_dirhit
                    }
                else:
                    logger.warning(
                        "⚠️ Training failed for %s %sd", symbol, horizon
                    )
                    return None

        except Exception as e:
            logger.error(
                "❌ Training error for %s %sd: %s", symbol, horizon, e
            )
            import traceback
            logger.error(traceback.format_exc())
            return None

    def process_task(self, symbol: str, horizon: int) -> bool:
        """Process a single task: HPO + Training

        NOTE: This runs in a separate process, so state updates are file-based.
        """
        key = f"{symbol}_{horizon}d"

        try:
            # ✅ CRITICAL FIX: Preserve cycle before loading state
            # If cycle was explicitly set (e.g., from
            # process_task_standalone), preserve it.
            # Otherwise, load from state file.
            preserved_cycle = self.cycle if self.cycle > 0 else None

            # ✅ CRITICAL FIX: Helper to load state while preserving cycle
            def load_state_preserve_cycle():
                """Preserve cycle if it was explicitly set."""
                self.load_state()
                if preserved_cycle is not None and preserved_cycle > 0:
                    self.cycle = preserved_cycle

            # Load current state
            load_state_preserve_cycle()

            # ✅ RACE CONDITION FIX: Double-check if task already running
            task = self.state.get(key)
            if task and task.status in (
                'hpo_in_progress', 'training_in_progress'
            ):
                logger.info(
                    "⏭️ Skipping %s %sd (already in progress by another "
                    "process)",
                    symbol,
                    horizon,
                )
                return False

            # ✅ CRITICAL FIX: Skip data check if HPO study already exists
            skip_data_check = False
            study_dirs = [
                Path('/opt/bist-pattern/results/optuna_studies'),
                Path('/opt/bist-pattern/hpo_studies'),
            ]
            for study_dir in study_dirs:
                if not study_dir.exists():
                    continue
                cycle_file = (
                    study_dir
                    / f"hpo_with_features_{symbol}_h{horizon}_c{self.cycle}.db"
                )
                if cycle_file.exists():
                    # Check if study has any trials (HPO already started)
                    try:
                        import sqlite3
                        conn = sqlite3.connect(str(cycle_file), timeout=10.0)
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM trials")
                        trial_count = cursor.fetchone()[0]
                        conn.close()
                        if trial_count > 0:
                            skip_data_check = True
                            logger.info(
                                "⏭️ Skip data quality for %s %sd "
                                "(HPO started, %s trials)",
                                symbol,
                                horizon,
                                trial_count,
                            )
                            break
                    except Exception as e:
                        logger.debug(f"Failed to check HPO trial count: {e}")

            # ✅ CRITICAL FIX: Check if HPO completed but state not updated.
            # If JSON missing, attempt recovery. If incomplete, continue
            # from existing study (warm-start).
            if task and task.cycle == self.cycle:
                # Check if HPO completed but best_params_file is missing
                if task.status in ('failed', 'completed') and (
                    task.best_params_file is None
                    or not Path(task.best_params_file).exists()
                ):
                    # Try to find HPO study file
                    study_dirs = [
                        Path('/opt/bist-pattern/results/optuna_studies'),
                        Path('/opt/bist-pattern/hpo_studies'),
                    ]
                    study_file = None
                    for study_dir in study_dirs:
                        if not study_dir.exists():
                            continue
                        # Check current cycle format first
                        cycle_file = (
                            study_dir
                            / f"hpo_with_features_{symbol}_h{horizon}_c"
                              f"{self.cycle}.db"
                        )
                        if cycle_file.exists():
                            study_file = cycle_file
                            break
                        # Check legacy format (only for cycle 1)
                        if self.cycle == 1:
                            legacy_file = (
                                study_dir
                                / f"hpo_with_features_{symbol}_h{horizon}.db"
                            )
                            if legacy_file.exists():
                                study_file = legacy_file
                                break

                    if study_file and study_file.exists():
                        # Check HPO progress in study file
                        try:
                            import sqlite3
                            conn = sqlite3.connect(
                                str(study_file), timeout=30.0
                            )
                            cursor = conn.cursor()
                            cursor.execute(
                                "SELECT COUNT(*) FROM trials "
                                "WHERE state='COMPLETE'"
                            )
                            complete_trials = cursor.fetchone()[0]
                            conn.close()

                            # ✅ CRITICAL: Only recover if HPO truly
                            # completed (MIN_TRIALS_FOR_RECOVERY+ trials).
                            # Partial progress is not enough.
                            if complete_trials >= MIN_TRIALS_FOR_RECOVERY:
                                logger.info(
                                    "✅ Completed HPO for %s %sd "
                                    "(%s trials), resuming training",
                                    symbol,
                                    horizon,
                                    complete_trials,
                                )
                                # Try to find JSON file
                                json_files = sorted(
                                    Path('/opt/bist-pattern/results').glob(
                                        f'optuna_pilot_features_on_h'
                                        f'{horizon}_c{self.cycle}_*.json'
                                    ),
                                    key=lambda p: p.stat().st_mtime,
                                    reverse=True,
                                )
                                # Also check legacy format (no cycle in
                                # filename)
                                if not json_files:
                                    json_files = sorted(
                                        Path('/opt/bist-pattern/results').glob(
                                            f'optuna_pilot_features_on_h'
                                            f'{horizon}_*.json'
                                        ),
                                        key=lambda p: p.stat().st_mtime,
                                        reverse=True,
                                    )

                                # Find JSON file for this symbol
                                hpo_result = None
                                for json_file in json_files[:20]:
                                    try:
                                        with open(json_file, 'r') as f:
                                            hpo_data = json.load(f)
                                        symbols = hpo_data.get('symbols', [])
                                        horizon_val = hpo_data.get('horizon')
                                        if (
                                            symbol in symbols
                                            and horizon_val == horizon
                                        ):
                                            # Found valid JSON file
                                            best_dirhit = hpo_data.get(
                                                'best_dirhit'
                                            )
                                            best_value = hpo_data.get(
                                                'best_value', 0
                                            )
                                            best_trial_num = hpo_data.get(
                                                'best_trial', {}
                                            ).get('number')
                                            hpo_result = {
                                                'best_value': best_value,
                                                'best_dirhit': best_dirhit,
                                                'best_params': hpo_data.get(
                                                    'best_params', {}
                                                ),
                                                'best_trial_number': (
                                                    best_trial_num
                                                ),
                                                'json_file': str(json_file),
                                                'n_trials': hpo_data.get(
                                                    'n_trials', 0
                                                ),
                                                'features_enabled': (
                                                    hpo_data.get(
                                                    'features_enabled', {}
                                                    )
                                                ),
                                                'feature_params': (
                                                    hpo_data.get(
                                                    'feature_params', {}
                                                    )
                                                ),
                                                'feature_flags': (
                                                    hpo_data.get(
                                                    'feature_flags', {}
                                                    )
                                                ),
                                                'hyperparameters': (
                                                    hpo_data.get(
                                                    'hyperparameters', {}
                                                    )
                                                ),
                                            }
                                            logger.info(
                                                "✅ Found HPO JSON file: %s",
                                                json_file.name,
                                            )
                                            break
                                    except Exception as e:
                                        logger.debug(
                                            "Failed to process HPO JSON "
                                            "file: %s",
                                            e,
                                        )
                                        continue

                                if hpo_result:
                                    # Update state and proceed to training
                                    task.hpo_completed_at = (
                                        datetime.now().isoformat()
                                    )
                                    task.hpo_dirhit = (
                                        hpo_result.get('best_dirhit')
                                        or hpo_result.get('best_value', 0)
                                    )
                                    task.best_params_file = hpo_result[
                                        'json_file'
                                    ]
                                    task.status = 'training_in_progress'
                                    task.error = None  # Clear previous error
                                    self.state[key] = task
                                    self.save_state()

                                    # Proceed directly to training
                                    best_params_with_trial = (
                                        hpo_result['best_params'].copy()
                                    )
                                    best_params_with_trial[
                                        'best_trial_number'
                                    ] = hpo_result.get('best_trial_number')
                                    best_params_with_trial[
                                        'features_enabled'
                                    ] = hpo_result.get('features_enabled', {})
                                    best_params_with_trial[
                                        'feature_params'
                                    ] = hpo_result.get('feature_params', {})
                                    best_params_with_trial[
                                        'feature_flags'
                                    ] = hpo_result.get('feature_flags', {})
                                    best_params_with_trial[
                                        'hyperparameters'
                                    ] = hpo_result.get('hyperparameters', {})

                                    training_result = self.run_training(
                                        symbol,
                                        horizon,
                                        best_params_with_trial,
                                        hpo_result=hpo_result,
                                    )

                                    if training_result is None:
                                        load_state_preserve_cycle()
                                        task = self.state.get(key)
                                        if task:
                                            task.status = 'failed'
                                            task.error = (
                                                'Training failed after HPO '
                                                'recovery'
                                            )
                                            task.retry_count = (
                                                task.retry_count + 1
                                            )
                                            self.state[key] = task
                                            self.save_state()
                                        return False

                                    # Training completed
                                    load_state_preserve_cycle()
                                    task = self.state.get(key)
                                    if task:
                                        task.status = 'completed'
                                        task.training_completed_at = (
                                            datetime.now().isoformat()
                                        )
                                        if isinstance(training_result, dict):
                                            task.adaptive_dirhit = (
                                                training_result.get(
                                                    'adaptive_dirhit'
                                                )
                                            )
                                            task.training_dirhit_online = (
                                                training_result.get(
                                                    'adaptive_dirhit'
                                                )
                                            )
                                            task.training_dirhit = (
                                                training_result.get(
                                                    'adaptive_dirhit'
                                                )
                                            )
                                            task.training_dirhit_wfv = (
                                                training_result.get(
                                                    'wfv_dirhit'
                                                )
                                            )
                                        else:
                                            task.training_dirhit = (
                                                training_result
                                            )
                                            task.adaptive_dirhit = (
                                                training_result
                                            )
                                        task.cycle = self.cycle
                                        self.state[key] = task
                                        self.save_state()
                                    logger.info(
                                        "✅ Recovered and completed %s %sd",
                                        symbol,
                                        horizon,
                                    )
                                    return True
                                else:
                                    logger.warning(
                                        "⚠️ HPO completed for %s %sd but JSON "
                                        "file not found; re-running HPO",
                                        symbol,
                                        horizon,
                                    )
                            else:
                                # ✅ CRITICAL: HPO incomplete (< 1490 trials)
                                # continue from where it left off. Warm-start
                                # uses existing trials.
                                logger.info(
                                    "🔄 HPO incomplete for %s %sd "
                                    "(%s/%s trials); warm-start",
                                    symbol,
                                    horizon,
                                    complete_trials,
                                    HPO_TRIALS,
                                )
                                # Reset task to pending so HPO can continue
                                task.status = 'pending'
                                task.error = None  # Clear error
                                task.cycle = self.cycle
                                self.state[key] = task
                                self.save_state()
                                logger.info(
                                    "✅ Reset %s %sd to 'pending' to continue "
                                    "HPO from %s/1500 trials",
                                    symbol,
                                    horizon,
                                    complete_trials,
                                )
                                # Continue with HPO (warm-start).
                                # Return False so task can be picked up again.
                                return False
                        except Exception as e:
                            logger.warning(
                                "⚠️ Error checking study file for %s %sd: %s",
                                symbol,
                                horizon,
                                e,
                            )

            # ✅ RETRY FIX: Analyze failure reason before retrying
            # ✅ UPDATED: Retry count increased from 3 to 10.
            # ✅ CRITICAL: Skip retry if task reset to pending (HPO incomplete).
            task_was_reset = False
            if task and task.status == 'failed':
                study_dirs = [
                    Path('/opt/bist-pattern/results/optuna_studies'),
                    Path('/opt/bist-pattern/hpo_studies'),
                ]
                for study_dir in study_dirs:
                    if not study_dir.exists():
                        continue
                    cycle_file = (
                        study_dir
                        / f"hpo_with_features_{symbol}_h{horizon}_c"
                          f"{self.cycle}.db"
                    )
                    if cycle_file.exists():
                        try:
                            import sqlite3
                            conn = sqlite3.connect(
                                str(cycle_file), timeout=10.0
                            )
                            cursor = conn.cursor()
                            cursor.execute(
                                "SELECT COUNT(*) FROM trials "
                                "WHERE state='COMPLETE'"
                            )
                            complete_trials = cursor.fetchone()[0]
                            conn.close()
                            # If HPO progress incomplete (< threshold),
                            # task was reset to pending.
                            if 0 < complete_trials < MIN_TRIALS_FOR_RECOVERY:
                                task_was_reset = True
                                logger.info(
                                    "⏭️ Skipping retry for %s %sd "
                                    "(HPO incomplete with %s trials, "
                                    "will continue from where it left off)",
                                    symbol,
                                    horizon,
                                    complete_trials,
                                )
                                # Reset to pending if not already
                                if task.status == 'failed':
                                    task.status = 'pending'
                                    task.error = None
                                    task.cycle = self.cycle
                                    self.state[key] = task
                                    self.save_state()
                                break
                        except Exception as e:
                            logger.debug(
                                "Failed to process task recovery: %s", e
                            )

            MAX_RETRY_COUNT = 10
            if (
                task
                and task.status == 'failed'
                and task.retry_count < MAX_RETRY_COUNT
                and not task_was_reset
            ):
                error_msg = task.error or ''
                error_lower = error_msg.lower()

                # Permanent failures - don't retry, mark as skipped
                permanent_failures = [
                    'insufficient data',
                    'no hpo json file found',
                    'symbol not found',
                    'data not available',
                    'delisted',
                    'inactive',
                    # ✅ NEW: LOW SUPPORT failure
                    'no hpo candidates with sufficient support',
                    # ✅ NEW: Alternative wording
                    'all trials had low support'
                ]

                # Check if this is a permanent failure
                is_permanent = any(
                    permanent in error_lower
                    for permanent in permanent_failures
                )

                if is_permanent:
                    logger.info(
                        "⏭️ Skipping %s %sd (permanent failure: %s)",
                        symbol,
                        horizon,
                        error_msg[:50],
                    )
                    task.status = 'skipped'
                    task.error = f'Permanent failure: {error_msg}'
                    task.cycle = self.cycle
                    self.state[key] = task
                    self.save_state()
                    return False

                # Temporary failures - retry (timeout, network, subprocess)
                logger.info(
                    "🔄 Retrying %s %sd (retry %s/%s, error: %s)",
                    symbol,
                    horizon,
                    task.retry_count + 1,
                    MAX_RETRY_COUNT,
                    error_msg[:100],
                )
                task.status = 'pending'
                task.error = None  # Clear previous error for retry
                task.cycle = self.cycle  # Ensure cycle is current
                self.state[key] = task
                self.save_state()

            # ✅ CRITICAL FIX: Data quality check BEFORE HPO
            # Skip if HPO already started (study file exists with trials).
            if not skip_data_check:
                # Check if symbol has sufficient data for this horizon
                try:
                    with app.app_context():
                        det = HybridPatternDetector()
                        df = det.get_stock_data(symbol, days=0)

                        # ✅ FIX: df=None could be connection error or missing
                        # data. Proceed with HPO so it can handle fetching.
                        if df is None:
                            # Could be connection error; let HPO handle.
                            logger.warning(
                                "⚠️ %s %sd: get_stock_data returned None. "
                                "Possible DB issue or missing data. "
                                "Proceeding with HPO (handles fetching)...",
                                symbol,
                                horizon,
                            )
                            # Don't skip - let HPO handle it
                        else:
                            # Minimum data requirements per horizon
                            # ✅ USER REQUEST: All horizons require minimum
                            # 100 days
                            min_required = {
                                1: 100,   # 1d needs 100 days
                                3: 100,   # 3d needs 100 days
                                7: 100,   # 7d needs 100 days
                                14: 100,  # 14d needs 100 days
                                30: 100,  # 30d needs 100 days
                            }

                            min_days = min_required.get(horizon, 100)
                            actual_days = len(df) if df is not None else 0

                            if actual_days < min_days:
                                # Insufficient data - skip this task
                                logger.info(
                                    "⏭️ Skipping %s %sd: insufficient data "
                                    "(%s < %s days required)",
                                    symbol,
                                    horizon,
                                    actual_days,
                                    min_days,
                                )
                                task = self.state.get(
                                    key,
                                    TaskState(
                                        symbol=symbol,
                                        horizon=horizon,
                                        status='pending',
                                        cycle=self.cycle,
                                    ),
                                )
                                task.status = 'skipped'
                                task.error = (
                                    f'Insufficient data: {actual_days}/'
                                    f'{min_days} days'
                                )
                                task.cycle = self.cycle
                                self.state[key] = task
                                self.save_state()
                                return False
                except Exception as e:
                    # ✅ FIX: If data fetch fails (e.g., DB connection
                    # error), proceed with HPO
                    # HPO has its own data fetching mechanism that might
                    # work better
                    logger.warning(
                        "⚠️ Data quality check failed for %s %sd: %s. "
                        "Likely DB issue. Proceeding with HPO "
                        "(it will handle data fetching)...",
                        symbol,
                        horizon,
                        e,
                    )
                    # Don't skip - let HPO handle it

            # Update state: HPO in progress
            task = self.state.get(
                key,
                TaskState(
                    symbol=symbol,
                    horizon=horizon,
                    status='pending',
                    cycle=self.cycle,
                ),
            )
            task.status = 'hpo_in_progress'
            task.cycle = self.cycle  # ✅ FIX: Ensure cycle is set
            self.state[key] = task
            self.save_state()

            # Step 1: Run HPO
            hpo_result = self.run_hpo(symbol, horizon)

            # Retryable failures (e.g., SIGTERM on restart) should be
            # re-queued as pending
            if isinstance(hpo_result, dict) and hpo_result.get('retryable'):
                load_state_preserve_cycle()
                task = self.state.get(key)
                if task:
                    task.status = 'pending'
                    task.error = hpo_result.get('error')
                    task.retry_count = task.retry_count + 1
                    task.cycle = self.cycle
                    self.state[key] = task
                    self.save_state()
                return False

            if not hpo_result:
                load_state_preserve_cycle()
                task = self.state.get(key)
                if task:
                    # ✅ FIX: Extract error message if hpo_result has 'error'
                    error_msg = 'HPO failed'
                    if isinstance(hpo_result, dict) and 'error' in hpo_result:
                        error_msg = hpo_result['error']
                    elif isinstance(hpo_result, str):
                        error_msg = hpo_result

                    task.status = 'failed'
                    task.error = error_msg
                    task.last_error = error_msg
                    task.retry_count = (
                        task.retry_count + 1
                    )  # ✅ FIX: Increment retry count
                    self.state[key] = task
                    self.save_state()
                return False

            # Update state: HPO completed
            load_state_preserve_cycle()
            task = self.state.get(key)
            if task:
                task.hpo_completed_at = datetime.now().isoformat()
                # ✅ CRITICAL FIX: Use symbol-specific avg_dirhit from
                # best_trial_metrics.
                symbol_key = f"{symbol}_{horizon}d"
                best_trial_metrics = hpo_result.get(
                    'best_trial_metrics', {}
                )
                symbol_metrics = (
                    best_trial_metrics.get(symbol_key, {})
                    if isinstance(best_trial_metrics, dict) else {}
                )
                symbol_avg_dirhit = (
                    symbol_metrics.get('avg_dirhit')
                    if isinstance(symbol_metrics, dict) else None
                )

                if symbol_avg_dirhit is not None and isinstance(
                    symbol_avg_dirhit, (int, float)
                ):
                    task.hpo_dirhit = float(symbol_avg_dirhit)
                    logger.info(
                        "✅ %s %sd: HPO DirHit from best_trial_metrics: "
                        "%.2f%%",
                        symbol,
                        horizon,
                        task.hpo_dirhit,
                    )
                else:
                    # Fallback: Use best_dirhit (tüm sembollerin ortalaması)
                    best_dirhit = hpo_result.get('best_dirhit')
                    if best_dirhit is not None:
                        task.hpo_dirhit = best_dirhit
                        logger.warning(
                            "⚠️ %s %sd: Symbol-specific DirHit not found, "
                            "using best_dirhit (all symbols avg): %.2f%%",
                            symbol,
                            horizon,
                            task.hpo_dirhit,
                        )
                    else:
                        # Final fallback: Use best_value (score) but log
                        # warning
                        task.hpo_dirhit = hpo_result.get('best_value', 0)
                        logger.warning(
                            "⚠️ %s %sd: HPO DirHit not available, "
                            "using score instead: %.2f",
                            symbol,
                            horizon,
                            task.hpo_dirhit,
                        )
                task.best_params_file = hpo_result['json_file']
                task.status = 'training_in_progress'
                self.state[key] = task
                self.save_state()

            # Step 2: Run Training
            # ✅ UPDATED: Include all HPO results in best_params for training
            best_params_with_trial = hpo_result['best_params'].copy()
            best_params_with_trial['best_trial_number'] = hpo_result.get(
                'best_trial_number'
            )
            # Include features_enabled and feature_params for training
            best_params_with_trial['features_enabled'] = hpo_result.get(
                'features_enabled', {}
            )
            best_params_with_trial['feature_params'] = hpo_result.get(
                'feature_params', {}
            )
            best_params_with_trial['feature_flags'] = hpo_result.get(
                'feature_flags', {}
            )
            best_params_with_trial['hyperparameters'] = hpo_result.get(
                'hyperparameters', {}
            )
            training_result = self.run_training(
                symbol, horizon, best_params_with_trial, hpo_result=hpo_result
            )

            if training_result is None:
                load_state_preserve_cycle()
                task = self.state.get(key)
                if task:
                    # ✅ FIX: Training failed - Check for insufficient data.
                    error_msg = 'Training failed'
                    if (
                        'Insufficient data' in str(training_result)
                        or training_result is None
                    ):
                        # Check if it's really insufficient data
                        try:
                            with app.app_context():
                                det = HybridPatternDetector()
                                df = det.get_stock_data(symbol, days=0)
                                if df is None or len(df) < 50:
                                    task.status = 'skipped'
                                    error_msg = 'Insufficient data - skipped'
                                    days_val = (
                                        len(df) if df is not None else 0
                                    )
                                    logger.info(
                                        "⏭️ Skipping %s %sd "
                                        "(insufficient data: %s days)",
                                        symbol,
                                        horizon,
                                        days_val,
                                    )
                                else:
                                    task.status = 'failed'
                                    task.retry_count = task.retry_count + 1
                        except Exception as e:
                            logger.debug(
                                "Failed to handle task error: %s", e
                            )
                            task.status = 'failed'
                            task.retry_count = task.retry_count + 1
                    else:
                        task.status = 'failed'
                        task.retry_count = task.retry_count + 1
                    task.error = error_msg
                    self.state[key] = task
                    self.save_state()
                return False

            # Update state: Completed
            load_state_preserve_cycle()
            task = self.state.get(key)
            if task:
                task.status = 'completed'
                task.training_completed_at = datetime.now().isoformat()
                # ✅ HİBRİT: Save WFV (adaptive OFF) and online
                # (adaptive OFF) DirHit
                if isinstance(training_result, dict):
                    # New format: both wfv_dirhit and adaptive_dirhit
                    wfv_dirhit = training_result.get('wfv_dirhit')
                    adaptive_dirhit = training_result.get('adaptive_dirhit')
                    task.adaptive_dirhit = adaptive_dirhit
                    # Backward compatibility: also set training_dirhit_online
                    task.training_dirhit_online = adaptive_dirhit
                    task.training_dirhit = adaptive_dirhit  # Backcompat
                    # ✅ NEW: Save WFV DirHit for comparison with HPO
                    task.training_dirhit_wfv = wfv_dirhit
                else:
                    # Legacy format (should not happen with new code)
                    task.training_dirhit = training_result
                    task.adaptive_dirhit = training_result
                task.cycle = self.cycle
                self.state[key] = task
                self.save_state()

            logger.info(
                "✅ Task completed: %s %sd (Cycle %s)",
                symbol,
                horizon,
                self.cycle,
            )
            return True

        except Exception as e:
            logger.error(
                "❌ Task error for %s %sd: %s", symbol, horizon, e
            )
            import traceback
            logger.error(traceback.format_exc())
            # Preserve cycle even in exception handler.
            # Use preserved_cycle from try block; avoid inner helper.
            preserved_cycle_exc = self.cycle if self.cycle > 0 else None
            self.load_state()
            if preserved_cycle_exc is not None and preserved_cycle_exc > 0:
                self.cycle = preserved_cycle_exc
            task = self.state.get(key)
            if task:
                task.status = 'failed'
                task.error = str(e)
                self.state[key] = task
                self.save_state()
            return False

    def process_symbol(self, symbol: str) -> bool:
        """Process horizons for a symbol sequentially (1d→3d→7d→14d→30d).

        ✅ NEW APPROACH: Symbol-based sequential processing
        - Processes all horizons for one symbol before moving to next
        - Reduces database load (data fetched once per symbol per cycle)
        - Reduces SQLite conflicts (one symbol at a time)
        - More predictable and easier to debug

        Returns True if all horizons completed successfully, False otherwise.
        """
        logger.info(
            "🔄 Processing %s: horizons sequentially", symbol
        )

        completed = 0
        failed = 0
        skipped = 0

        # Process each horizon in order
        for horizon in HORIZON_ORDER:
            key = f"{symbol}_{horizon}d"

            # Check if already completed for current cycle
            self.load_state()
            task = self.state.get(key)
            if (
                task
                and task.status == 'completed'
                and task.cycle == self.cycle
            ):
                logger.info(
                    "⏭️ Skipping %s %sd (already completed in cycle %s)",
                    symbol,
                    horizon,
                    self.cycle,
                )
                completed += 1
                continue

            # Process this horizon
            logger.info(
                "📊 Processing %s %sd (%s/%s)",
                symbol,
                horizon,
                completed + failed + skipped + 1,
                len(HORIZON_ORDER),
            )
            success = self.process_task(symbol, horizon)

            if success:
                completed += 1
                logger.info(
                    "✅ %s %sd completed (%s/%s horizons)",
                    symbol,
                    horizon,
                    completed,
                    len(HORIZON_ORDER),
                )
            else:
                # Check if it was skipped (insufficient data)
                self.load_state()
                task = self.state.get(key)
                if task and task.status == 'skipped':
                    skipped += 1
                    logger.info(
                        "⏭️ %s %sd skipped (insufficient data)",
                        symbol,
                        horizon,
                    )
                else:
                    failed += 1
                    logger.warning(
                        "❌ %s %sd failed (%s failed, %s completed)",
                        symbol,
                        horizon,
                        failed,
                        completed,
                    )

                    # Option: continue even if one fails (allows partial
                    # completion). To stop on first failure, uncomment below.
                    # logger.error(
                    #     f"❌ Stopping {symbol} due to failure at {horizon}d"
                    # )
                    # return False

        # Summary
        total = completed + failed + skipped
        logger.info(
            "📊 %s processing complete: %s completed, %s failed, %s skipped "
            "(%s/%s)",
            symbol,
            completed,
            failed,
            skipped,
            total,
            len(HORIZON_ORDER),
        )

        # Return True if at least some horizons completed
        # (partial success is acceptable)
        return completed > 0

    def cleanup_old_cycle_files(self, keep_cycles: int = 1):
        """Clean up SQLite study files from old cycles

        Args:
            keep_cycles: Number of recent cycles to keep
                (default: 1 = keep only current cycle)
        """
        try:
            study_dirs = [
                Path('/opt/bist-pattern/results/optuna_studies'),
                Path('/opt/bist-pattern/hpo_studies'),
            ]
            existing_dirs = [d for d in study_dirs if d.exists()]
            if not existing_dirs:
                return

            # Get current cycle start time
            cycle_start = None
            log_file = Path(
                '/opt/bist-pattern/logs/continuous_hpo_pipeline.log'
            )
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            if (
                                f"Starting cycle {self.cycle}" in line
                                or f"🔄 Starting cycle {self.cycle}" in line
                            ):
                                parts = line.split(' - ', 1)
                                if parts:
                                    timestamp_str = parts[0].strip()
                                    try:
                                        dt_str = timestamp_str.split(',')[0]
                                        cycle_start = datetime.strptime(
                                            dt_str, '%Y-%m-%d %H:%M:%S'
                                        )
                                        break
                                    except Exception as e:
                                        logger.debug(
                                            "Failed to parse cycle_start "
                                            "from timestamp: %s",
                                            e,
                                        )
                except Exception as e:
                    logger.debug(
                        "Failed to read cycle_start from state file: %s", e
                    )

            # If cycle start not found, use state file mtime
            if cycle_start is None:
                state_file = Path(
                    '/opt/bist-pattern/results/continuous_hpo_state.json'
                )
                if state_file.exists():
                    try:
                        cycle_start = datetime.fromtimestamp(
                            state_file.stat().st_mtime
                        )
                    except Exception as e:
                        logger.debug(
                            "Failed to get cycle_start from file mtime: %s",
                            e,
                        )
                        cycle_start = datetime.now()
                else:
                    cycle_start = datetime.now()

            # ✅ CRITICAL FIX: Clean up by cycle number, not timestamp
            # Delete all study files from cycles < current_cycle
            # Keep only files from current cycle (c{current_cycle}) and
            # legacy files (no cycle number, only for cycle 1)
            deleted_count = 0
            deleted_size = 0
            for study_dir in existing_dirs:
                for db_file in study_dir.glob('*.db'):
                    try:
                        # Extract cycle number from filename
                        name = db_file.name.replace('.db', '')
                        parts = name.split('_')
                        file_cycle = None

                        # Check if cycle number is in filename
                        if len(parts) >= 6 and parts[-1].startswith('c'):
                            try:
                                file_cycle = int(parts[-1][1:])
                            except (ValueError, IndexError):
                                pass

                        # Delete if:
                        # 1. File has cycle number < current_cycle
                        # 2. No cycle number AND current_cycle > 1 (legacy)
                        should_delete = False
                        if file_cycle is not None:
                            if file_cycle < self.cycle:
                                should_delete = True
                        else:
                            # Legacy file (no cycle number) - only keep if
                            # current_cycle is 1
                            if self.cycle > 1:
                                should_delete = True

                        if should_delete:
                            file_size = db_file.stat().st_size
                            db_file.unlink()
                            deleted_count += 1
                            deleted_size += file_size
                            logger.debug(
                                "🧹 Deleted old cycle file: %s "
                                "(cycle %s < %s)",
                                db_file.name,
                                file_cycle if file_cycle else 'legacy',
                                self.cycle,
                            )
                    except Exception as e:
                        logger.warning(
                            "⚠️ Could not delete old SQLite file %s: %s",
                            db_file,
                            e,
                        )

            if deleted_count > 0:
                freed_mb = deleted_size / 1024 / 1024
                logger.info(
                    "🧹 Cleaned up %s old SQLite study files "
                    "(%.2f MB freed)",
                    deleted_count,
                    freed_mb,
                )
        except Exception as e:
            logger.warning(
                "⚠️ Error during cleanup of old cycle files: %s", e
            )

    def run_cycle(self):
        """Run one complete cycle

        ✅ NEW APPROACH: Horizon-First processing (USER REQUEST)
        - Processes ALL symbols for each horizon before moving to next
        - Phase 1: All symbols for 1d → Phase 2: All symbols for 3d → ...
        - Incremental value delivery: 1d ready for all symbols first!
        - MAX_WORKERS: Symbols processed in parallel within each horizon phase
        """
        # ✅ CRITICAL FIX: Only increment cycle if current cycle is complete
        # This prevents cycle number from incrementing on every restart
        self.load_state()
        current_cycle = self.cycle

        # Check if current cycle has any incomplete tasks
        has_incomplete = False
        for key, task in self.state.items():
            if task.cycle == current_cycle and task.status not in (
                'completed', 'skipped'
            ):
                has_incomplete = True
                break

        # Only increment cycle if current cycle is complete (or no tasks exist)
        if not has_incomplete and current_cycle > 0:
            # Check if there are any tasks at all
            has_tasks = any(
                task.cycle == current_cycle for task in self.state.values()
            )
            if has_tasks:
                # Current cycle is complete, increment for new cycle
                self.cycle += 1
                logger.info(
                    "🔄 Current cycle %s complete, starting new cycle %s",
                    current_cycle,
                    self.cycle,
                )
            else:
                # No tasks yet, start with cycle 1
                if current_cycle == 0:
                    self.cycle = 1
                    logger.info("🔄 Starting first cycle %s", self.cycle)
                else:
                    # Keep current cycle
                    logger.info("🔄 Resuming cycle %s", self.cycle)
        else:
            if has_incomplete:
                logger.info(
                    "🔄 Resuming cycle %s (incomplete tasks found)",
                    self.cycle,
                )
            else:
                # First cycle
                if current_cycle == 0:
                    self.cycle = 1
                    logger.info("🔄 Starting first cycle %s", self.cycle)
                else:
                    logger.info("🔄 Resuming cycle %s", self.cycle)

        logger.info("🔄 Starting cycle %s", self.cycle)
        logger.info("   🎯 Mode: HORIZON-FIRST (all symbols per horizon)")
        logger.info(
            "   🚀 Incremental delivery: 1d ready for all symbols → 3d → ..."
        )
        logger.info(
            "   🔥 Parallelism: %s symbols in parallel per horizon phase",
            MAX_WORKERS,
        )
        logger.info(
            "   ✅ HİBRİT: Yeni verilerle HPO (incremental effect)"
        )
        logger.info(
            "   ✅ Cycle %s: Yeni veriler, split değerleri güncellenecek",
            self.cycle,
        )

        # ✅ Save cycle number to state file immediately after determining it.
        # Ensures cycle number persists across restarts.
        self.save_state()

        # ✅ FIX: Clean up old cycle files before starting new cycle
        # Keep only current cycle's files, delete older ones
        if self.cycle > 1:  # Don't clean on first cycle
            logger.info(
                "🧹 Cleaning old cycle files (keeping current cycle only)..."
            )
            self.cleanup_old_cycle_files(keep_cycles=1)

        # ✅ Ensure pending tasks use current cycle; reset failed from
        # previous cycles.
        # ⚠️ Do not lose current cycle when reloading state.
        saved_cycle = self.cycle
        self.load_state()
        self.cycle = saved_cycle
        changed = False
        reset_failed_count = 0
        for key, task in list(self.state.items()):
            # Reset failed tasks from previous cycles to pending (recovery)
            if task.status == 'failed' and task.cycle < self.cycle:
                task.status = 'pending'
                task.cycle = self.cycle
                task.retry_count = 0  # Reset retry count for new cycle
                task.error = None  # Clear error message
                self.state[key] = task
                changed = True
                reset_failed_count += 1
            # Update pending tasks to current cycle
            elif task.status == 'pending' and task.cycle != self.cycle:
                task.cycle = self.cycle
                self.state[key] = task
                changed = True
        if changed:
            self.save_state()
            if reset_failed_count > 0:
                logger.info(
                    "🔄 Reset %s failed tasks to pending for cycle %s "
                    "(recovery)",
                    reset_failed_count,
                    self.cycle,
                )
            updated = sum(
                1 for t in self.state.values() if t.cycle == self.cycle
            )
            logger.info(
                "🔄 Updated %s tasks to cycle %s", updated, self.cycle
            )

        # ✅ NEW: Horizon-First processing
        # Process all symbols for each horizon before moving to next horizon
        total_completed = 0
        total_failed = 0

        for horizon in HORIZON_ORDER:
            # ✅ Check for shutdown before each horizon phase
            if self.shutdown_requested:
                logger.info("🛑 Shutdown requested, stopping cycle...")
                # Update in-progress tasks to pending
                self._update_all_in_progress_to_pending()
                break

            logger.info("=" * 80)
            logger.info(
                "🎯 HORIZON PHASE: %sd - Processing ALL symbols", horizon
            )
            logger.info("=" * 80)

            phase_completed = 0
            phase_failed = 0
            phase_skipped = 0

            # Get all active symbols
            try:
                symbols_all = self.get_active_symbols()
            except Exception as e:
                logger.error(
                    "❌ Failed to get active symbols for %sd: %s",
                    horizon,
                    e,
                )
                continue

            # ✅ Process all symbols in parallel without blocking batches
            logger.info(
                "🌊 Processing %s symbols for %sd in parallel "
                "(max_workers=%s)",
                len(symbols_all),
                horizon,
                MAX_WORKERS,
            )

            # Create executor once for all symbols (non-blocking)
            executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
            future_to_symbol = {}

            try:
                # Submit all tasks at once (non-blocking)
                for symbol in symbols_all:
                    # ✅ Check for shutdown before submitting each task
                    if self.shutdown_requested:
                        logger.info(
                            "🛑 Shutdown requested, stopping task submission..."
                        )
                        break

                    key = f"{symbol}_{horizon}d"

                    # Check if already completed
                    self.load_state()
                    task = self.state.get(key)
                    if (
                        task
                        and task.status == 'completed'
                        and task.cycle == self.cycle
                    ):
                        # Completed and params file exists -> skip
                        if (
                            task.best_params_file
                            and Path(task.best_params_file).exists()
                        ):
                            logger.info(
                                "⏭️ Skipping %s %sd (already completed)",
                                symbol,
                                horizon,
                            )
                            phase_completed += 1
                            continue
                        else:
                            # Completed but file missing - recover
                            logger.warning(
                                "⚠️ %s %sd completed but best_params_file "
                                "missing, attempting recovery",
                                symbol,
                                horizon,
                            )
                            # Continue to process_task() for recovery

                    # Use standalone function with explicit cycle so child
                    # process gets correct cycle.
                    future = executor.submit(
                        process_task_standalone,
                        symbol,
                        horizon,
                        self.cycle,
                    )
                    future_to_symbol[future] = symbol

                # Collect results as they complete (non-blocking)
                completed_count = 0
                for future in as_completed(future_to_symbol):
                    # ✅ Check for shutdown during result collection
                    if self.shutdown_requested:
                        logger.info(
                            "🛑 Shutdown requested, cancelling remaining "
                            "tasks..."
                        )
                        # Cancel remaining futures
                        cancelled_count = 0
                        for f in future_to_symbol:
                            if not f.done():
                                f.cancel()
                                cancelled_count += 1
                        logger.info(
                            "🛑 Cancelled %s pending tasks", cancelled_count
                        )
                        # Update in-progress tasks to pending
                        self._update_all_in_progress_to_pending()
                        break

                    symbol = future_to_symbol[future]
                    try:
                        success = future.result()
                        completed_count += 1
                        if success:
                            phase_completed += 1
                            logger.info(
                                "✅ %s %sd completed (%s/%s, %s/%s futures)",
                                symbol,
                                horizon,
                                phase_completed,
                                len(symbols_all),
                                completed_count,
                                len(future_to_symbol),
                            )
                        else:
                            # Check if skipped
                            self.load_state()
                            task = self.state.get(f"{symbol}_{horizon}d")
                            if task and task.status == 'skipped':
                                phase_skipped += 1
                                logger.info(
                                    "⏭️ %s %sd skipped (insufficient data)",
                                    symbol,
                                    horizon,
                                )
                            elif task and task.status == 'pending':
                                # Re-queued (e.g., restart); don't count fail
                                logger.info(
                                    "🔁 %s %sd requeued (pending)",
                                    symbol,
                                    horizon,
                                )
                            else:
                                phase_failed += 1
                                logger.warning(
                                    "❌ %s %sd failed", symbol, horizon
                                )
                    except Exception as e:
                        logger.error(
                            "❌ Exception for %s %sd: %s", symbol, horizon, e
                        )
                        phase_failed += 1
            finally:
                # ✅ Shutdown executor based on shutdown request
                if self.shutdown_requested:
                    logger.info(
                        "🛑 Shutting down executor without waiting "
                        "(shutdown requested)..."
                    )
                    executor.shutdown(wait=False)
                else:
                    executor.shutdown(wait=True)

            # Phase summary
            logger.info("=" * 80)
            logger.info("✅ HORIZON PHASE %sd COMPLETE:", horizon)
            logger.info(
                "   Completed: %s/%s", phase_completed, len(symbols_all)
            )
            logger.info("   Failed: %s", phase_failed)
            logger.info("   Skipped: %s", phase_skipped)
            logger.info("=" * 80)

            total_completed += phase_completed
            total_failed += phase_failed

        logger.info(
            "✅ Cycle %s complete: %s tasks completed, %s tasks failed",
            self.cycle,
            total_completed,
            total_failed,
        )

    def run_continuous(self):
        """Run pipeline continuously"""
        logger.info("🚀 Starting Continuous HPO and Training Pipeline")
        logger.info("   Max workers: %s", MAX_WORKERS)
        logger.info("   Horizons: %s", HORIZONS)
        logger.info("   HPO trials: %s", HPO_TRIALS)

        try:
            while not self.shutdown_requested:
                # Run one cycle
                self.run_cycle()

                # Check for shutdown request before waiting
                if self.shutdown_requested:
                    logger.info("🛑 Shutdown requested, stopping pipeline...")
                    break

                # Wait before next cycle (check for shutdown periodically)
                wait_hours = 24  # Wait 24 hours before next cycle
                logger.info(
                    "⏳ Waiting %s hours before next cycle...", wait_hours
                )

                # Sleep in smaller chunks to check shutdown flag periodically
                sleep_interval = 300  # Check every 5 minutes
                total_sleep = wait_hours * 3600
                slept = 0
                while slept < total_sleep and not self.shutdown_requested:
                    time.sleep(min(sleep_interval, total_sleep - slept))
                    slept += sleep_interval

                if self.shutdown_requested:
                    logger.info(
                        "🛑 Shutdown requested during wait, stopping..."
                    )
                    break

        except KeyboardInterrupt:
            logger.info("⛔ Pipeline stopped by user")
            self.shutdown_requested = True
        except Exception as e:
            logger.error("❌ Pipeline error: %s", e)
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Ensure graceful shutdown on exit
            logger.info("🧹 Performing final cleanup...")
            self._graceful_shutdown_hpo_processes()
            self.save_state()
            logger.info("✅ Pipeline shutdown complete")


def process_task_standalone(symbol: str, horizon: int, cycle: int) -> bool:
    """Process one task in a separate process with explicit cycle."""
    # Create a new pipeline instance for this process
    pipeline = ContinuousHPOPipeline()
    # ✅ CRITICAL FIX: Set cycle BEFORE load_state() in __init__
    # But __init__ already calls load_state(), so we need to set it after
    # However, process_task() will preserve the cycle we set here
    if cycle > 0:
        pipeline.cycle = cycle  # Explicitly set cycle from parent
    # If cycle is 0, it will be loaded from state file in process_task()
    return pipeline.process_task(symbol, horizon)


def process_symbol_standalone(symbol: str, cycle: int) -> bool:
    """Process all horizons for a symbol in a separate process."""
    # Create a new pipeline instance for this process
    pipeline = ContinuousHPOPipeline()
    pipeline.cycle = cycle
    return pipeline.process_symbol(symbol)


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(
        description='Continuous HPO and Training Pipeline'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process one symbol-horizon pair',
    )
    parser.add_argument(
        '--symbol', type=str, default='ALKA', help='Symbol for test mode'
    )
    parser.add_argument(
        '--horizon', type=int, default=7, help='Horizon for test mode'
    )
    parser.add_argument(
        '--skip-hpo',
        action='store_true',
        help='Skip HPO, use existing best params',
    )
    args = parser.parse_args()

    pipeline = ContinuousHPOPipeline()

    if args.test:
        logger.info(
            "🧪 TEST MODE: Processing %s %sd", args.symbol, args.horizon
        )
        logger.info("=" * 80)

        if args.skip_hpo:
            logger.info("⏭️ Skipping HPO, using existing best params...")
            # Find existing HPO result for this symbol-horizon
            json_files = sorted(
                Path('/opt/bist-pattern/results').glob(
                    f'optuna_pilot_features_on_h{args.horizon}_*.json'
                ),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            hpo_data = None
            found_json = None

            # Check all files for our symbol
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    if args.symbol in data.get('symbols', []):
                        hpo_data = data
                        found_json = json_file
                        break
                except Exception as e:
                    logger.warning(
                        "⚠️ Error reading %s: %s", json_file, e
                    )
                    continue

            if hpo_data and found_json:
                logger.info("✅ Found existing HPO result: %s", found_json)
                best_params = hpo_data.get('best_params', {}) or {}
                # Ensure features_enabled present
                if (
                    'features_enabled' not in best_params
                    and 'features_enabled' in hpo_data
                ):
                    best_params['features_enabled'] = hpo_data[
                        'features_enabled'
                    ]
                # Ensure best_trial_number present
                try:
                    if 'best_trial_number' not in hpo_data:
                        _btn = None
                        if isinstance(hpo_data.get('best_trial'), dict):
                            _btn = hpo_data['best_trial'].get('number')
                        if _btn is not None:
                            hpo_data['best_trial_number'] = _btn
                            if isinstance(best_params, dict):
                                best_params['best_trial_number'] = _btn
                    elif (
                        isinstance(best_params, dict)
                        and 'best_trial_number' not in best_params
                    ):
                        best_params['best_trial_number'] = hpo_data.get(
                            'best_trial_number'
                        )
                except Exception as e:
                    logger.debug("Failed to set best_trial_number: %s", e)
                # Run training only (pass full hpo_result)
                result = pipeline.run_training(
                    args.symbol, args.horizon, best_params, hpo_result=hpo_data
                )
                if result is not None:
                    logger.info(
                        "✅ Test completed: %s %sd", args.symbol, args.horizon
                    )
                else:
                    logger.error(
                        "❌ Test failed: %s %sd", args.symbol, args.horizon
                    )
            else:
                logger.error(
                    "❌ No HPO result found for %s %sd",
                    args.symbol,
                    args.horizon,
                )
                logger.info(
                    "   Searched %s JSON files for horizon %sd",
                    len(json_files),
                    args.horizon,
                )
        else:
            # Run full pipeline: HPO + Training
            success = pipeline.process_task(args.symbol, args.horizon)
            if success:
                logger.info(
                    "✅ Test completed: %s %sd", args.symbol, args.horizon
                )
            else:
                logger.error(
                    "❌ Test failed: %s %sd", args.symbol, args.horizon
                )
    else:
        pipeline.run_continuous()


if __name__ == '__main__':
    main()
