#!/usr/bin/env python3
"""
Continuous HPO and Training Pipeline

Sürekli çalışan, otomatik HPO ve eğitim yapan pipeline:
1. Her sembol-horizon çifti için HPO (features açık)
2. Best params ile full training (adaptive learning dahil)
3. Model kaydetme
4. ✅ Symbol-based sequential processing: Her sembol için tüm horizonları sırayla işle (1d→3d→7d→14d→30d)
5. Cycle management (tüm semboller bittikten sonra başa dön)
6. Yeni verileri ekleyerek sürekli iyileşme

✅ YENİ YAKLAŞIM:
- Her sembol için tüm horizonları sırayla işle (bir sembol bitmeden diğerine geçme)
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
    os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:6432/bist_pattern_db'


def _get_best_trial_metrics_entry(hpo_result: Optional[Dict[str, Any]], symbol: str, horizon: int) -> Optional[Dict[str, Any]]:
    if not isinstance(hpo_result, dict):
        return None
    metrics = hpo_result.get('best_trial_metrics')
    if not isinstance(metrics, dict):
        return None
    key = f"{symbol}_{horizon}d"
    entry = metrics.get(key)
    return entry if isinstance(entry, dict) else None


def _extract_reference_historical_r2(metrics_entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
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


def _extract_model_metrics_from_train_result(train_result: Any, horizon: int) -> Dict[str, Dict[str, float]]:
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

# ⚡ CRITICAL: Disable root logger handlers to prevent imported modules from writing to our log file
# This must be done BEFORE importing app.py which starts working_automation
root_logger = logging.getLogger()
root_logger.handlers.clear()  # Clear all root logger handlers
root_logger.propagate = False  # Prevent propagation

# Import after path setup
from app import app  # noqa: E402

# ⚡ CRITICAL: Clear root logger handlers again AFTER app.py import
# app.py calls logging.basicConfig() which adds handlers to root logger
root_logger.handlers.clear()  # Clear handlers added by app.py's basicConfig()
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
logging.getLogger('bist_pattern.core.unified_collector').propagate = False
logging.getLogger('bist_pattern.core.pattern_validator').propagate = False
logging.getLogger('yfinance_gevent_native').propagate = False
logging.getLogger('app').propagate = False

# Setup logging
# ✅ FIX: Ensure log directory exists and file handler works
log_file = Path('/opt/bist-pattern/logs/continuous_hpo_pipeline.log')
log_file.parent.mkdir(parents=True, exist_ok=True)
# ✅ FIX: Ensure log directory permissions for shared access
from bist_pattern.utils.file_permissions import ensure_directory_permissions, ensure_file_permissions  # noqa: E402
ensure_directory_permissions(log_file.parent)

# ⚡ CRITICAL FIX: Use module-specific logger to avoid mixing with other services
# Don't use basicConfig as it affects root logger and all child loggers
# Instead, configure only our specific logger

# Get module-specific logger
logger = logging.getLogger('continuous_hpo_pipeline')
logger.setLevel(logging.INFO)
logger.handlers.clear()  # Clear any existing handlers
logger.propagate = False  # Prevent propagation to root logger (CRITICAL!)

# Create file handler with append mode (only for our logger)
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Create stream handler (only for our logger)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add handlers only to our logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Configuration
HORIZONS = [1, 3, 7, 14, 30]
HORIZON_ORDER = [1, 3, 7, 14, 30]
# ✅ Parallel processing - symbols in parallel, each processing all horizons sequentially
# Reduce default for system-wide stability; allow override via HPO_MAX_WORKERS
try:
    MAX_WORKERS = int(os.getenv('HPO_MAX_WORKERS', '4'))
except Exception:
    MAX_WORKERS = 4
# MAX_WORKERS = 1  # Alternative: Fully sequential (one symbol at a time)
# MAX_WORKERS = 3  # Alternative: 3-4 symbols in parallel (conservative)
HPO_TRIALS = 1500  # Number of Optuna trials per HPO (1500 provides ~73% feature flag coverage: 1500/2048 combinations)
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
    Get NUMA node and CPU list using file-based counter with locking (works across processes)
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
    except Exception:
        # Fallback: use process ID for deterministic distribution
        import os
        numa_node = os.getpid() % NUMA_NODES
        cpu_start = numa_node * CPUS_PER_NODE
        cpu_end = cpu_start + CPUS_PER_NODE - 1
        cpu_list = f"{cpu_start}-{cpu_end}"
        return numa_node, cpu_list


def _build_numa_cmd(base_cmd: List[str], numa_node: int, cpu_list: str) -> Tuple[List[str], int, str]:
    """
    Build command with CPU affinity (simplified - no NUMA binding)
    Uses taskset for CPU affinity only (NUMA binding removed - Python/ML not NUMA-aware)
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
    except Exception:
        pass
    
    # No affinity tools available, return base command
    return base_cmd, numa_node, cpu_list


def _get_hpo_slots_dir() -> Path:
    d = Path('/opt/bist-pattern/results/hpo_slots')
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d

 
def _get_hpo_max_slots() -> int:
    try:
        return max(1, int(os.getenv('HPO_MAX_SLOTS', '3')))
    except Exception:
        return 3


def _normalize_feature_flag_key(key: str) -> str:
    """
    Normalize feature flag key to uppercase environment variable name.
    Handles both lowercase (enable_*) and uppercase (ENABLE_*) keys.
    
    Args:
        key: Feature flag key (e.g., 'enable_seed_bagging' or 'ENABLE_SEED_BAGGING')
    
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
    
    ⚠️ WARNING: Returns an open file descriptor that MUST be closed by calling release_hpo_slot().
    For automatic resource management, use HPOSlotContext instead:
    
        with HPOSlotContext() as slot:
            slot_idx, slot_fd, slot_path = slot
            # ... use slot ...
    
    Returns (slot_index, file_object, lock_path).
    
    Args:
        timeout_seconds: Maximum time to wait for a slot (default: 5 minutes).
                        If timeout is reached, raises TimeoutError.
    
    Returns:
        Tuple of (slot_index, file_object, lock_path). The file_object must be closed
        by calling release_hpo_slot(file_object) in a finally block to prevent file descriptor leaks.
    """
    slots_dir = _get_hpo_slots_dir()
    max_slots = _get_hpo_max_slots()
    import time as _time
    start_time = _time.time()
    
    while True:
        # Check timeout
        elapsed = _time.time() - start_time
        if elapsed >= timeout_seconds:
            raise TimeoutError(f"Failed to acquire HPO slot after {timeout_seconds:.1f} seconds. All slots may be stuck.")
        
        for idx in range(max_slots):
            lock_path = slots_dir / f'hpo_slot_{idx}.lock'
            f = None
            try:
                f = open(lock_path, 'a+')
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    try:
                        f.seek(0)
                        f.truncate()
                        f.write(f'{os.getpid()} {datetime.now().isoformat()}\n')
                        f.flush()
                        os.fsync(f.fileno())
                    except Exception:
                        pass
                    return idx, f, lock_path
                except BlockingIOError:
                    # Slot is locked by another process, close file and try next slot
                    if f is not None:
                        try:
                            f.close()
                        except Exception:
                            pass
                    continue
                except Exception:
                    # Other exception during flock (e.g., OSError, PermissionError)
                    # ✅ CRITICAL FIX: Close file handle to prevent file descriptor leak
                    if f is not None:
                        try:
                            f.close()
                        except Exception:
                            pass
                    continue
            except Exception:
                # Exception during file open (e.g., PermissionError, FileNotFoundError)
                # File was not opened, so nothing to close
                continue
        
        # Check for deadlock: if all slots are held for too long, log warning
        if elapsed > 60.0:  # After 1 minute, check for stuck slots
            try:
                stuck_slots = []
                for idx in range(max_slots):
                    lock_path = slots_dir / f'hpo_slot_{idx}.lock'
                    if lock_path.exists():
                        try:
                            # Try to read lock file to see when it was last updated
                            with open(lock_path, 'r') as check_f:
                                content = check_f.read().strip()
                                if content:
                                    # Parse PID and timestamp
                                    parts = content.split()
                                    if len(parts) >= 2:
                                        try:
                                            lock_time = datetime.fromisoformat(parts[1])
                                            lock_age = (datetime.now() - lock_time).total_seconds()
                                            # If lock is older than 2 hours, it's probably stuck
                                            if lock_age > 7200:
                                                stuck_slots.append((idx, lock_age))
                                        except Exception:
                                            pass
                        except Exception:
                            pass
                if stuck_slots:
                    logger.warning(f"⚠️ Detected potentially stuck HPO slots: {stuck_slots}")
            except Exception:
                pass  # Don't fail slot acquisition due to deadlock detection errors
        
        _time.sleep(0.25)


def release_hpo_slot(fobj) -> None:
    try:
        if fobj is not None:
            try:
                fcntl.flock(fobj.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                fobj.close()
            except Exception:
                pass
    except Exception:
        pass


@dataclass
class TaskState:
    """Task state for a symbol-horizon pair"""
    symbol: str
    horizon: int
    status: str  # 'pending', 'hpo_in_progress', 'training_in_progress', 'completed', 'failed', 'skipped'
    hpo_completed_at: Optional[str] = None
    training_completed_at: Optional[str] = None
    best_params_file: Optional[str] = None
    hpo_dirhit: Optional[float] = None
    training_dirhit: Optional[float] = None  # backward compatibility (WFV)
    training_dirhit_wfv: Optional[float] = None
    training_dirhit_online: Optional[float] = None
    adaptive_dirhit: Optional[float] = None  # NEW: Adaptive learning DirHit (online DirHit with adaptive learning enabled)
    error: Optional[str] = None
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
        self.state: Dict[str, TaskState] = {}
        self.cycle = 0
        self.load_state()
        # Reset any stale in-progress tasks (e.g., after crash/restart)
        self._reset_stale_in_progress()
    
    def load_state(self):
        """Load pipeline state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    # Shared lock for reading to avoid partial reads while another process writes
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    except Exception:
                        pass
                    content = f.read()
                    data = json.loads(content)
                    self.cycle = data.get('cycle', 0)
                    state_dict = data.get('state', {})
                    for key, task_data in state_dict.items():
                        self.state[key] = TaskState(**task_data)
                logger.info(f"✅ Loaded state: {len(self.state)} tasks, cycle {self.cycle}")
            except Exception as e:
                logger.error(f"❌ Error loading state: {e}")
                self.state = {}
        else:
            logger.info("📝 No existing state file, starting fresh")
    
    def save_state(self):
        """Save pipeline state to file (merge-aware for concurrent processes)"""
        # ✅ CRITICAL FIX: Hold exclusive lock throughout entire read-modify-write cycle
        # This prevents race conditions where another process modifies state between read and write
        lock_fd = None
        try:
            # Open state file for reading/writing and acquire exclusive lock
            # Lock will be held until the entire read-modify-write cycle completes
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            ensure_directory_permissions(self.state_file.parent)
            
            # Open file in read-write mode to hold lock across read and write operations
            if self.state_file.exists():
                lock_fd = os.open(self.state_file, os.O_RDWR)
            else:
                # File doesn't exist, create it
                lock_fd = os.open(self.state_file, os.O_CREAT | os.O_RDWR, 0o644)
            
            # Acquire exclusive lock - will be held until explicitly released
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except Exception:
                pass  # Lock acquisition failed, continue anyway (best effort)
            
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
                    for key, task_data in existing_data.get('state', {}).items():
                        merged_state[key] = task_data
            except Exception:
                # If read fails, start fresh (but log it)
                logger.warning("⚠️ Could not read existing state for merge, using current state only")
            
            # Merge current process's updates (overwrite only changed tasks)
            # This happens while lock is still held
            for key, task in self.state.items():
                merged_state[key] = asdict(task)
            
            # Get version number from existing state (if available) for optimistic locking
            version = existing_data.get('version', 0) if existing_data else 0
            new_version = version + 1
            
            data = {
                'version': new_version,  # ✅ FIX: Add version number for optimistic locking
                'cycle': self.cycle,
                'state': merged_state,
                'last_updated': datetime.now().isoformat()
            }
            
            # Atomic write (lock still held)
            # Write to temp file first, then atomic rename
            tmp_path = self.state_file.with_suffix('.json.tmp')
            try:
                with open(tmp_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Atomic rename (lock still held, preventing concurrent writes)
                os.replace(tmp_path, self.state_file)
                # ✅ FIX: Ensure file permissions for shared access
                ensure_file_permissions(self.state_file)
            finally:
                # Clean up temp file if it still exists
                if tmp_path.exists():
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"❌ Error saving state: {e}")
        finally:
            # Release lock and close file descriptor
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except Exception:
                    pass
                try:
                    os.close(lock_fd)
                except Exception:
                    pass

    def _reset_stale_in_progress(self):
        """Reset tasks left in 'in_progress' states (resume safety after restart)."""
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
                raw_symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).order_by(Stock.symbol).all()]
                
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
                        logger.debug(f"🔧 Symbol cleaned: '{symbol}' -> '{cleaned}'")
                
                # Filter out invalid symbols
                import re
                # ✅ FIX: Delisted semboller - ABVKS gibi delisted sembolleri denylist'e ekle
                denylist = re.compile(r"USDTR|USDTRY|^XU|^OPX|^F_|VIOP|INDEX|ABVKS", re.IGNORECASE)
                symbols = [s for s in symbols if s and not denylist.search(s)]
                logger.info(f"📊 Found {len(symbols)} active symbols (after BOM cleaning and denylist filter)")
                return symbols
        except Exception as e:
            logger.error(f"❌ Error getting active symbols: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_pending_symbols(self) -> List[str]:
        """Get symbols that have pending tasks (not fully completed).
        - Returns symbols that have at least one pending/failed horizon
        - Skips symbols that are already being processed (any horizon in progress)
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
                if task.status in ('hpo_in_progress', 'training_in_progress'):
                    has_in_progress = True
                    all_completed = False
                    break
                
                # Check if task is pending or failed (with retries)
                if task.status == 'pending':
                    has_pending = True
                    all_completed = False
                elif task.status == 'failed' and task.retry_count < 3:
                    has_pending = True
                    all_completed = False
                elif task.status == 'skipped':
                    # Skipped tasks don't count as completed for cycle purposes
                    all_completed = False
                elif task.status == 'completed':
                    # Only count as completed if it's for current cycle
                    if task.cycle < self.cycle:
                        has_pending = True
                        all_completed = False
                else:
                    all_completed = False

            # Skip if already in progress (another process is handling this symbol)
            if has_in_progress:
                continue
            
            # Add if has pending tasks and not fully completed
            if has_pending or not all_completed:
                pending_symbols.append(symbol)

        logger.info(f"📋 Pending symbols: {len(pending_symbols)} (symbols with pending/failed horizons)")
        return pending_symbols
    
    def get_pending_tasks(self) -> List[Tuple[str, int]]:
        """Get next pending tasks: one horizon per symbol, in order.
        - Picks first pending horizon per symbol according to HORIZON_ORDER
        - Skips tasks already in progress/completed for current cycle
        - Includes failed tasks with retries remaining
        - ⚠️ DEPRECATED: Use get_pending_symbols() + process_symbol() for sequential processing
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
                    if task.retry_count < 3:
                        chosen_h = horizon
                        break
                # Skipped: do not pick (insufficient data)
                if task.status == 'skipped':
                    continue

            if chosen_h is not None:
                next_tasks.append((symbol, chosen_h))

        # De-duplicate and limit to a reasonable batch (parent will further cap to MAX_WORKERS)
        # Preserve order by using dict
        unique_ordered = list(dict.fromkeys(next_tasks))
        logger.info(f"📋 Next batch candidates: {len(unique_ordered)} (one horizon per symbol)")
        return unique_ordered
    
    def run_hpo(self, symbol: str, horizon: int) -> Optional[Dict]:
        """Run HPO for a symbol-horizon pair"""
        try:
            logger.info(f"🔬 Starting HPO for {symbol} {horizon}d...")
            
            # ✅ CRITICAL FIX: Record HPO start time for JSON file validation
            import time
            hpo_start_time = time.time()
            
            # Use subprocess to run HPO script (cleaner isolation)
            
            # ✅ UPDATED: Use optuna_hpo_with_feature_flags.py for comprehensive optimization
            # This script optimizes: feature flags + feature internal parameters + hyperparameters
            hpo_script = Path('/opt/bist-pattern/scripts/optuna_hpo_with_feature_flags.py')
            if not hpo_script.exists():
                logger.error(f"❌ HPO script not found: {hpo_script}")
                return None
            
            # Run HPO script (1500 trials for comprehensive optimization)
            # Timeout: 250 hours (10.4 days) - allows for 1500 trials × ~2-3 min/trial = ~50-75 hours
            # Plus margin for slower trials, feature engineering, etc.
            # ⚡ DRY RUN: If DRY_RUN_TRIALS env var is set, use that instead of HPO_TRIALS
            dry_run_trials = int(os.environ.get('DRY_RUN_TRIALS', '0'))
            trials_to_use = dry_run_trials if dry_run_trials > 0 else HPO_TRIALS
            timeout_to_use = 3600 if dry_run_trials > 0 else 900000  # 1 hour for dry run, 250h for full
            
            logger.info(f"🔬 HPO Configuration: {trials_to_use} trials, timeout={timeout_to_use}s ({'DRY RUN' if dry_run_trials > 0 else 'PRODUCTION'})")
            
            cmd = [
                sys.executable,
                str(hpo_script),
                '--symbols', symbol,
                '--horizon', str(horizon),
                '--trials', str(trials_to_use),  # 10 for dry run, 1500 for production
                '--timeout', str(timeout_to_use)  # 1 hour for dry run, 250 hours for production
            ]
            
            # Pass DRY_RUN_TRIALS to subprocess environment if set
            env = os.environ.copy()
            if dry_run_trials > 0:
                env['DRY_RUN_TRIALS'] = str(dry_run_trials)
            # ✅ CRITICAL FIX: Pass cycle number to HPO script for study file naming
            # This ensures each cycle gets its own study file, enabling new HPO with new data
            env['HPO_CYCLE'] = str(self.cycle)
            
            # ✅ Acquire global HPO slot (limits cross-process concurrency)
            slot_fd = None
            try:
                _slot_idx, slot_fd, _slot_path = acquire_hpo_slot()
                logger.info(f"🔒 HPO slot acquired for {symbol} {horizon}d")
            except TimeoutError as te:
                logger.error(f"⏱️ Failed to acquire HPO slot for {symbol} {horizon}d: {te}")
                return None
                
                # ✅ CPU affinity optimization: bind to specific CPU range (round-robin)
                # Simplified: Only CPU affinity, no NUMA binding (Python/ML not NUMA-aware)
                numa_node, cpu_list = _get_numa_node_and_cpus()
                numa_cmd, _, _ = _build_numa_cmd(cmd, numa_node, cpu_list)
                # Log CPU affinity if applied
                if numa_cmd != cmd:
                    logger.info(f"🔗 CPU affinity: {cpu_list} (node {numa_node})")
                
                # Set process priority (higher priority for HPO)
                try:
                    # Set nice value to -5 (higher priority)
                    os.nice(-5)
                except (OSError, PermissionError):
                    # May require root, ignore if fails
                    pass
                
                result = subprocess.run(
                    numa_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_to_use,  # 1 hour for dry run, 250 hours for production
                    cwd='/opt/bist-pattern',
                    env=env,  # Pass environment variables including DRY_RUN_TRIALS
                    preexec_fn=lambda: os.nice(-5) if hasattr(os, 'nice') else None
                )
            finally:
                release_hpo_slot(slot_fd)
                logger.info(f"🔓 HPO slot released for {symbol} {horizon}d")
            
            # ✅ FIX: Log subprocess output for debugging (use INFO level to ensure visibility)
            # ⚡ CRITICAL: Filter out web app logs (pattern_detector, unified_collector, etc.)
            # Only log HPO-specific messages ([hpo], Starting HPO, HPO completed, etc.)
            if result.stdout:
                # Filter stdout: only keep HPO-related messages
                filtered_stdout = '\n'.join([
                    line for line in result.stdout.split('\n')
                    if any(keyword in line for keyword in [
                        '[hpo]', 'Starting HPO', 'HPO completed', 'Trial', 'Best trial',
                        'DirHit', 'Training completed', 'model in memory', 'predict_enhanced'
                    ])
                ])
                if filtered_stdout.strip():
                    logger.info(f"HPO stdout for {symbol} {horizon}d:\n{filtered_stdout[:10000]}")
            
            if result.stderr:
                # Filter stderr: only keep HPO-related messages
                filtered_stderr = '\n'.join([
                    line for line in result.stderr.split('\n')
                    if any(keyword in line for keyword in [
                        '[hpo]', 'Starting HPO', 'HPO completed', 'Trial', 'Best trial',
                        'DirHit', 'Training completed', 'model in memory', 'predict_enhanced',
                        'ERROR', 'WARNING'  # Keep errors and warnings
                    ]) and not any(skip in line for skip in [
                        'pattern_detector', 'unified_collector', 'broadcaster',
                        'No news items', 'TA-Lib detected', 'Pattern Validation'
                    ])
                ])
                if filtered_stderr.strip():
                    logger.info(f"HPO stderr for {symbol} {horizon}d:\n{filtered_stderr[:10000]}")
            
            if result.returncode != 0:
                error_details = []
                if result.stderr:
                    error_details.append(f"stderr: {result.stderr[:500]}")
                if result.stdout:
                    # Look for error messages in stdout
                    error_lines = [line for line in result.stdout.split('\n') if any(keyword in line.lower() for keyword in ['error', 'failed', 'exception', 'traceback'])]
                    if error_lines:
                        error_details.append(f"stdout errors: {'; '.join(error_lines[:5])}")
                
                error_msg = f"HPO script failed (exit code {result.returncode})"
                if error_details:
                    error_msg += f" - {', '.join(error_details)}"
                
                logger.error(f"❌ HPO script failed for {symbol} {horizon}d: {error_msg}")
                return None
            
            # ✅ CRITICAL FIX: Find the generated JSON file with cycle number
            # JSON format: optuna_pilot_features_on_h{horizon}_c{cycle}_{timestamp}.json
            # Also check legacy format (no cycle) for backward compatibility
            json_patterns = [
                f'optuna_pilot_features_on_h{horizon}_c{self.cycle}_*.json',  # Current format with cycle
                f'optuna_pilot_features_on_h{horizon}_*.json'  # Legacy format (fallback)
            ]
            json_files = []
            for pattern in json_patterns:
                found_files = sorted(
                    Path('/opt/bist-pattern/results').glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
                json_files.extend(found_files)
                # If we found files with cycle number, don't use legacy format
                if pattern.startswith(f'optuna_pilot_features_on_h{horizon}_c{self.cycle}_'):
                    if found_files:
                        json_files = found_files  # Use only cycle-specific files
                        break
            # Remove duplicates and sort by mtime
            json_files = sorted(set(json_files), key=lambda p: p.stat().st_mtime, reverse=True)
            
            if not json_files:
                # ✅ CRITICAL FIX: If JSON file not found, try to recover from study file
                # This handles cases where JSON file creation failed but study has 1500+ trials
                logger.warning(f"⚠️ No HPO JSON file found for {symbol} {horizon}d, attempting recovery from study file...")
                
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
                    cycle_file = study_dir / f"hpo_with_features_{symbol}_h{horizon}_c{self.cycle}.db"
                    if cycle_file.exists():
                        study_file = cycle_file
                        break
                    # Check legacy format (only for cycle 1)
                    if self.cycle == 1:
                        legacy_file = study_dir / f"hpo_with_features_{symbol}_h{horizon}.db"
                        if legacy_file.exists():
                            study_file = legacy_file
                            break
                
                if study_file and study_file.exists():
                    try:
                        import optuna
                        study_recovered = optuna.load_study(study_name=None, storage=f"sqlite:///{study_file}")
                        complete_trials = len([t for t in study_recovered.trials if t.state == optuna.trial.TrialState.COMPLETE])
                        
                        if complete_trials >= 1490:
                            logger.info(f"✅ Found completed HPO in study file ({complete_trials} trials), creating JSON file from study...")
                            
                            # Create JSON file from study
                            best_trial = study_recovered.best_trial
                            best_params = study_recovered.best_params
                            best_value = float(study_recovered.best_value) if study_recovered.best_value is not None else 0.0
                            
                            # Get best_dirhit from user_attrs
                            best_dirhit = None
                            try:
                                _val = best_trial.user_attrs.get('avg_dirhit', None)
                                if isinstance(_val, (int, float)) and np.isfinite(_val):
                                    best_dirhit = float(_val)
                            except Exception:
                                pass
                            if best_dirhit is None or not np.isfinite(best_dirhit):
                                best_dirhit = best_value
                            
                            # Extract feature flags, feature parameters, and hyperparameters
                            feature_flags = {k: v for k, v in best_params.items() if k.startswith('enable_')}
                            feature_params_keys = [
                                'ml_loss_mse_weight', 'ml_loss_threshold', 'ml_dir_penalty', 'n_seeds',
                                'meta_stacking_alpha', 'yolo_min_conf', 'smart_consensus_weight',
                                'smart_performance_weight', 'smart_sigma', 'fingpt_confidence_threshold',
                                'external_min_days', 'external_smooth_alpha',
                                'regime_scale_low', 'regime_scale_high',
                            ]
                            feature_params_keys += [k for k in best_params.keys() if k.startswith('ml_adaptive_k_') or k.startswith('ml_pattern_weight_scale_')]
                            feature_params = {k: v for k, v in best_params.items() if k in feature_params_keys}
                            hyperparameters = {k: v for k, v in best_params.items() if not k.startswith('enable_') and k not in feature_params_keys}
                            
                            # Create JSON result
                            result_recovered = {
                                'best_value': best_value,
                                'best_dirhit': best_dirhit,
                                'best_params': best_params,
                                'best_trial': {
                                    'number': best_trial.number,
                                    'value': float(best_trial.value) if best_trial.value is not None else 0.0,
                                    'state': str(best_trial.state),
                                },
                                'n_trials': len(study_recovered.trials),
                                'study_name': study_recovered.study_name,
                                'symbols': [symbol],
                                'horizon': horizon,
                                'best_model_choice': best_params.get('model_choice'),
                                'feature_flags': feature_flags,
                                'feature_params': feature_params,
                                'hyperparameters': hyperparameters,
                                'features_enabled': {
                                    'ENABLE_XGBOOST': '1' if best_params.get('model_choice') in ('xgb', 'all') else '0',
                                    'ENABLE_LIGHTGBM': '1' if best_params.get('model_choice') in ('lgbm', 'all') else '0',
                                    'ENABLE_CATBOOST': '1' if best_params.get('model_choice') in ('cat', 'all') else '0',
                                }
                            }
                            
                            # Save recovered JSON file
                            recovered_json_file = f"/opt/bist-pattern/results/optuna_pilot_features_on_h{horizon}_c{self.cycle}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_recovered.json"
                            with open(recovered_json_file, 'w') as f:
                                json.dump(result_recovered, f, indent=2)
                            
                            logger.info(f"✅ Recovered JSON file created: {recovered_json_file}")
                            
                            # Return recovered result
                            return {
                                'best_value': best_value,
                                'best_dirhit': best_dirhit,
                                'best_params': best_params,
                                'best_trial_number': best_trial.number,
                                'json_file': recovered_json_file,
                                'n_trials': len(study_recovered.trials),
                                'features_enabled': result_recovered['features_enabled'],
                                'feature_params': feature_params,
                                'feature_flags': feature_flags,
                                'hyperparameters': hyperparameters
                            }
                        else:
                            logger.warning(f"⚠️ Study file found but only {complete_trials} trials completed (need 1490+)")
                    except Exception as recover_err:
                        logger.warning(f"⚠️ Failed to recover from study file: {recover_err}")
                
                error_msg = f"No HPO JSON file found for {symbol} {horizon}d"
                logger.warning(f"⚠️ {error_msg}")
                # Store error for retry analysis
                return {'error': error_msg}
            
            # ✅ CRITICAL FIX: Check recent files for our symbol with timestamp validation
            # Also verify that the HPO actually completed with sufficient trials
            # Required keys for validation
            required_keys = ['best_params', 'best_value', 'symbols', 'horizon', 'n_trials']
            
            # ✅ FIX: Collect all valid JSON files, then select the best one (by DirHit or best_value)
            # This handles cases where multiple valid JSON files exist for the same symbol-horizon
            valid_json_candidates = []
            
            for json_file in json_files[:50]:
                try:
                    # ✅ CRITICAL FIX: Verify JSON file was created after HPO started
                    json_mtime = json_file.stat().st_mtime
                    
                    # ✅ FIX: First check if JSON file contains valid HPO result (symbol, horizon, trials)
                    # If valid, we can use it even if timestamp is outside normal range (handles restart cases)
                    json_file_valid = False
                    try:
                        with open(json_file, 'r') as f:
                            hpo_data_precheck = json.load(f)
                            # Quick validation: check if this JSON is for our symbol-horizon
                            # with sufficient trials
                            if (isinstance(hpo_data_precheck, dict) and
                                    symbol in hpo_data_precheck.get('symbols', []) and
                                    hpo_data_precheck.get('horizon') == horizon and
                                    isinstance(hpo_data_precheck.get('n_trials', 0), int) and
                                    hpo_data_precheck.get('n_trials', 0) >= 1490):
                                json_file_valid = True
                    except Exception:
                        pass
                    
                    # ✅ FIX: Timestamp validation - flexible for valid JSON files (handles restart cases)
                    # If JSON file is valid (correct symbol/horizon/trials), use relaxed timestamp validation
                    # This handles cases where HPO completed before restart but state file wasn't updated
                    if json_file_valid:
                        # ✅ SECURITY FIX: Even for valid JSON, check if it's not too old (max 7 days)
                        # This prevents using very old JSON files from previous cycles
                        current_time = time.time()
                        max_age_seconds = 7 * 24 * 3600  # 7 days
                        if json_mtime < current_time - max_age_seconds:
                            logger.debug(f"⚠️ HPO JSON file {json_file.name} is too old (created {(current_time - json_mtime)/86400:.1f} days ago), skipping even though valid")
                            continue
                        # For valid JSON files, allow wider timestamp range (up to 7 days old, or up to 96 hours after HPO start)
                        if json_mtime > hpo_start_time + 345600:  # 96 hours (4 days) after HPO start
                            logger.debug(f"⚠️ HPO JSON file {json_file.name} is too new (created {json_mtime - hpo_start_time:.0f}s after HPO start), skipping")
                            continue
                    else:
                        # Only do strict timestamp validation if JSON file is not clearly valid
                        # Allow 5 minutes tolerance before HPO start (in case of clock skew)
                        if json_mtime < hpo_start_time - 300:
                            logger.debug(f"⚠️ HPO JSON file {json_file.name} is too old (created {time.time() - json_mtime:.0f}s before HPO start), skipping")
                            continue
                        # For invalid JSON files, also check if too new (strict validation)
                        # HPO can take up to 72 hours for 1500 trials, so allow up to 96 hours (4 days) after start
                        if json_mtime > hpo_start_time + 345600:  # 96 hours (4 days) - increased from 18 hours
                            logger.debug(f"⚠️ HPO JSON file {json_file.name} is too new (created {json_mtime - hpo_start_time:.0f}s after HPO start), skipping")
                            continue
                    
                    with open(json_file, 'r') as f:
                        hpo_data = json.load(f)
                    
                    # ✅ CRITICAL FIX: Validate JSON structure
                    if not isinstance(hpo_data, dict):
                        logger.warning(f"⚠️ Invalid HPO JSON structure in {json_file.name}: not a dict")
                        continue
                    
                    # Check required keys
                    missing_keys = [key for key in required_keys if key not in hpo_data]
                    if missing_keys:
                        logger.warning(f"⚠️ Invalid HPO JSON: {json_file.name} missing required keys: {missing_keys}")
                        continue
                    
                    hpo_symbols = hpo_data.get('symbols', [])
                    if not isinstance(hpo_symbols, list):
                        logger.warning(f"⚠️ Invalid HPO JSON: {json_file.name} symbols is not a list")
                        continue
                    
                    if symbol not in hpo_symbols:
                        continue
                    
                    # ✅ FIX: Verify HPO actually completed with sufficient trials
                    n_trials = hpo_data.get('n_trials', 0)
                    if not isinstance(n_trials, int) or n_trials < 10:
                        logger.warning(f"⚠️ HPO for {symbol} {horizon}d completed with only {n_trials} trials (expected {HPO_TRIALS}), rejecting result")
                        continue
                    
                    # ✅ FIX: Validate best_params
                    best_params = hpo_data.get('best_params', {})
                    if not isinstance(best_params, dict) or len(best_params) == 0:
                        logger.warning(f"⚠️ Invalid HPO JSON: {json_file.name} best_params is empty or not a dict")
                        continue
                    
                    # ✅ FIX: Validate horizon matches
                    hpo_horizon = hpo_data.get('horizon', None)
                    if hpo_horizon != horizon:
                        logger.warning(f"⚠️ HPO JSON horizon mismatch: {json_file.name} has horizon {hpo_horizon}, expected {horizon}")
                        continue
                    
                    # ✅ CRITICAL FIX: Use best_dirhit if available, otherwise fallback to best_value
                    # Method 2 returns score (DirHit * mask_after_thr / 100), not DirHit
                    best_dirhit = hpo_data.get('best_dirhit')
                    best_score = hpo_data.get('best_value', 0)
                    
                    # ✅ FIX: Collect this valid JSON as a candidate (don't return immediately)
                    # We'll select the best one among all candidates
                    best_trial_number = None
                    best_trial_info = hpo_data.get('best_trial', {})
                    if isinstance(best_trial_info, dict):
                        best_trial_number = best_trial_info.get('number')
                    
                    valid_json_candidates.append({
                        'json_file': json_file,
                        'json_mtime': json_mtime,
                        'best_value': hpo_data.get('best_value', 0),
                        'best_dirhit': best_dirhit,
                        'best_mask_after_thr': hpo_data.get('best_mask_after_thr'),
                        'best_params': hpo_data.get('best_params', {}),
                        'best_trial_number': best_trial_number,
                        'n_trials': n_trials,
                        'features_enabled': hpo_data.get('features_enabled', {}),
                        'feature_params': hpo_data.get('feature_params', {}),
                        'feature_flags': hpo_data.get('feature_flags', {}),
                        'hyperparameters': hpo_data.get('hyperparameters', {})
                    })
                    
                    if best_dirhit is not None:
                        logger.debug(f"📋 Candidate JSON for {symbol} {horizon}d: {json_file.name} - DirHit = {best_dirhit:.2f}% (score = {best_score:.2f}, {n_trials} trials)")
                    else:
                        logger.debug(f"📋 Candidate JSON for {symbol} {horizon}d: {json_file.name} - Score = {best_score:.2f} (DirHit not available, {n_trials} trials)")
                except Exception as e:
                    logger.warning(f"⚠️ Error reading JSON file {json_file}: {e}")
                    continue
            
            # ✅ FIX: Select the best JSON from all valid candidates
            # Priority: 1) Highest DirHit, 2) Highest best_value, 3) Most recent
            if valid_json_candidates:
                # Sort by: DirHit (desc), then best_value (desc), then mtime (desc)
                valid_json_candidates.sort(
                    key=lambda x: (
                        x['best_dirhit'] if x['best_dirhit'] is not None else -1,  # DirHit (higher is better)
                        x['best_value'],  # best_value (higher is better)
                        x['json_mtime']  # Most recent
                    ),
                    reverse=True
                )
                
                best_candidate = valid_json_candidates[0]
                
                if len(valid_json_candidates) > 1:
                    logger.info(f"⚠️ Found {len(valid_json_candidates)} valid JSON files for {symbol} {horizon}d, selecting best one:")
                    for i, candidate in enumerate(valid_json_candidates[:3]):  # Show top 3
                        dirhit_str = f"DirHit={candidate['best_dirhit']:.2f}%" if candidate['best_dirhit'] is not None else f"Score={candidate['best_value']:.2f}"
                        logger.info(f"   {i+1}. {candidate['json_file'].name}: {dirhit_str}")
                
                best_dirhit = best_candidate['best_dirhit']
                best_score = best_candidate['best_value']
                if best_dirhit is not None:
                    json_name = best_candidate['json_file'].name
                    logger.info(
                        f"✅ Selected best HPO result for {symbol} {horizon}d: "
                        f"DirHit = {best_dirhit:.2f}% (score = {best_score:.2f}, "
                        f"{best_candidate['n_trials']} trials) from {json_name}"
                    )
                else:
                    json_name = best_candidate['json_file'].name
                    logger.info(
                        f"✅ Selected best HPO result for {symbol} {horizon}d: "
                        f"Score = {best_score:.2f} (DirHit not available, "
                        f"{best_candidate['n_trials']} trials) from {json_name}"
                    )
                
                return {
                    'best_value': best_candidate['best_value'],
                    'best_dirhit': best_candidate['best_dirhit'],
                    'best_mask_after_thr': best_candidate['best_mask_after_thr'],
                    'best_params': best_candidate['best_params'],
                    'best_trial_number': best_candidate['best_trial_number'],
                    'json_file': str(best_candidate['json_file']),
                    'n_trials': best_candidate['n_trials'],
                    'features_enabled': best_candidate['features_enabled'],
                    'feature_params': best_candidate['feature_params'],
                    'feature_flags': best_candidate['feature_flags'],
                    'hyperparameters': best_candidate['hyperparameters']
                }
            
            logger.warning(f"⚠️ HPO result not found for {symbol} {horizon}d (or insufficient trials)")
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
    def _dirhit(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 0.005) -> float:
        """Compute directional hit rate (same as HPO).
        
        ✅ FIX: Use same logic as HPO - only significant predictions are evaluated.
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

    def _evaluate_training_dirhits(self, symbol: str, horizon: int, df: pd.DataFrame, best_params: Optional[Dict] = None, hpo_result: Optional[Dict] = None) -> Dict[str, Optional[float]]:
        """Evaluate DirHit after training using two modes:
        - wfv: adaptive OFF (no leakage)
        - online: adaptive OFF (HPO ile tutarlılık - hibrit yaklaşım)
        
        Args:
            symbol: Stock symbol
            horizon: Prediction horizon
            df: Full dataframe
            best_params: Best HPO parameters (must be provided to ensure correct evaluation)
            hpo_result: Full HPO result dict (includes best_trial_number, features_enabled, etc.)
        """
        # ✅ FIX: Import os at function level to avoid scope issues
        import os
        results: Dict[str, Optional[float]] = {'wfv': None, 'online': None}
        total_days = len(df)
        
        # ✅ FIX: Dynamic minimum data requirement based on horizon
        # Minimum: enough for train (80%) + test (20%) with at least (horizon + 10) test days
        # Formula: min_total = (horizon + 10) / 0.2 = (horizon + 10) * 5
        # For horizon=1: min_total = 11 * 5 = 55 days
        # For horizon=30: min_total = 40 * 5 = 200 days
        # Use a reasonable minimum: 100 days (allows evaluation for most horizons)
        min_total_days = max(100, (horizon + 10) * 5)
        
        if total_days < min_total_days:
            logger.warning(f"⚠️ {symbol} {horizon}d: Insufficient data for evaluation ({total_days} days, need {min_total_days})")
            return results

        # ⚡ NEW: Use multiple splits for walk-forward validation (same as HPO)
        from scripts.optuna_hpo_with_feature_flags import generate_walkforward_splits, calculate_dynamic_split
        
        wfv_splits = generate_walkforward_splits(total_days, horizon, n_splits=4)
        
        if not wfv_splits:
            # Fallback to single split if multiple splits not possible
            split_idx = calculate_dynamic_split(total_days, horizon)
            if split_idx < 1:
                logger.warning(f"⚠️ {symbol} {horizon}d: Invalid split, skipping evaluation")
                return results
            wfv_splits = [(split_idx, total_days)]
        
        logger.info(f"📊 {symbol} {horizon}d: Generated {len(wfv_splits)} walk-forward splits for evaluation")
        
        # 🔍 DEBUG: Log data source and statistics (same format as HPO)
        logger.info(f"🔍 [eval-debug] {symbol} {horizon}d: DATA SOURCE = df parameter (from run_training)")
        logger.info(f"🔍 [eval-debug] {symbol} {horizon}d: Total data: {total_days} days")

        # ✅ FIX: Best params'ı set et (eğer verilmişse)
        if best_params is not None:
            from scripts.train_completed_hpo_with_best_params import set_hpo_params_as_env
            set_hpo_params_as_env(best_params, horizon)
            
            # ✅ CRITICAL FIX: Set feature flags from best_params (if available)
            # HPO JSON contains 'features_enabled' dict with feature flags
            # We need to set these to match HPO best trial exactly
            if isinstance(best_params, dict) and 'features_enabled' not in best_params:
                # Try to get from JSON file if available
                # This is a fallback - normally features_enabled should be in best_params
                pass
            elif isinstance(best_params, dict) and 'features_enabled' in best_params:
                features_enabled = best_params['features_enabled']
                for key, value in features_enabled.items():
                    os.environ[key] = str(value)
                logger.info(f"🔧 {symbol} {horizon}d: Feature flags set from best_params: {len(features_enabled)} flags")
            # Also set feature flags from best_params keys (enable_*)
            feature_flag_keys = [k for k in best_params.keys() if k.startswith('enable_')]
            for key in feature_flag_keys:
                value = best_params[key]
                if key == 'enable_external_features':
                    os.environ['ENABLE_EXTERNAL_FEATURES'] = '1' if value else '0'
                elif key == 'enable_fingpt_features':
                    os.environ['ENABLE_FINGPT_FEATURES'] = '1' if value else '0'
                elif key == 'enable_yolo_features':
                    os.environ['ENABLE_YOLO_FEATURES'] = '1' if value else '0'
                elif key == 'enable_directional_loss':
                    os.environ['ML_USE_DIRECTIONAL_LOSS'] = '1' if value else '0'
                elif key == 'enable_seed_bagging':
                    os.environ['ENABLE_SEED_BAGGING'] = '1' if value else '0'
                elif key == 'enable_talib_patterns':
                    os.environ['ENABLE_TALIB_PATTERNS'] = '1' if value else '0'
                elif key == 'enable_smart_ensemble':
                    os.environ['ML_USE_SMART_ENSEMBLE'] = '1' if value else '0'
                elif key == 'enable_stacked_short':
                    os.environ['ML_USE_STACKED_SHORT'] = '1' if value else '0'
                elif key == 'enable_meta_stacking':
                    os.environ['ENABLE_META_STACKING'] = '1' if value else '0'
                elif key == 'enable_regime_detection':
                    os.environ['ML_USE_REGIME_DETECTION'] = '1' if value else '0'
                elif key == 'enable_fingpt':
                    os.environ['ENABLE_FINGPT'] = '1' if value else '0'
            # Log key params for verification
            xgb_params = {k: v for k, v in best_params.items() if k.startswith('xgb_')}
            if xgb_params:
                logger.info(
                    f"⚙️ {symbol} {horizon}d: Best HPO params set for evaluation: "
                    f"n_est={xgb_params.get('xgb_n_estimators', 'N/A')}, "
                    f"max_depth={xgb_params.get('xgb_max_depth', 'N/A')}, "
                    f"lr={xgb_params.get('xgb_learning_rate', 'N/A')}"
                )
                # ✅ FIX: Verify params are actually set in environment
                env_n_est = os.environ.get('OPTUNA_XGB_N_ESTIMATORS', 'NOT_SET')
                env_max_depth = os.environ.get('OPTUNA_XGB_MAX_DEPTH', 'NOT_SET')
                env_lr = os.environ.get('OPTUNA_XGB_LEARNING_RATE', 'NOT_SET')
                logger.info(f"🔍 {symbol} {horizon}d: Environment vars - OPTUNA_XGB_N_ESTIMATORS={env_n_est}, OPTUNA_XGB_MAX_DEPTH={env_max_depth}, OPTUNA_XGB_LEARNING_RATE={env_lr}")
            else:
                logger.info(f"⚙️ {symbol} {horizon}d: Best HPO params set for evaluation: {len(best_params)} params")
        else:
            logger.warning(f"⚠️ {symbol} {horizon}d: Best params not provided! Evaluation may use default params.")
        
        # 1) WFV (adaptive OFF) - Data leakage önleme: Model'i train_df ile yeniden eğit
        original_adaptive = None
        original_seed_bag = None
        original_directional = None
        # ✅ CRITICAL FIX: Save original smart ensemble parameters for restoration (same as online evaluation)
        original_smart_consensus_weight: Optional[str] = None
        original_smart_performance_weight: Optional[str] = None
        original_smart_sigma: Optional[str] = None
        original_smart_weight_xgb: Optional[str] = None
        original_smart_weight_lgb: Optional[str] = None
        original_smart_weight_cat: Optional[str] = None
        # ✅ CRITICAL FIX: Save all dynamically set environment variables from features_enabled and enable_* keys (same as online evaluation)
        original_env_vars: dict = {}
        try:
            # ✅ CRITICAL FIX: Don't use default value - we need to distinguish between "unset" and "set to '0'"
            original_adaptive = os.environ.get('ML_USE_ADAPTIVE_LEARNING')  # None if not set
            os.environ['ML_USE_ADAPTIVE_LEARNING'] = '0'  # Adaptive OFF
            # ✅ CRITICAL FIX: Capture original values BEFORE modifying them (needed for both if and else paths)
            # Don't use default values - we need to distinguish between "unset" and "set to default"
            original_seed_bag = os.environ.get('ENABLE_SEED_BAGGING')  # None if not set
            original_directional = os.environ.get('ML_USE_DIRECTIONAL_LOSS')  # None if not set
            # ✅ CRITICAL FIX: Set feature flags from best_params to match HPO best trial exactly
            # This ensures evaluation uses the same feature flags as HPO
            if best_params and isinstance(best_params, dict):
                features_enabled = best_params.get('features_enabled', {})
                if features_enabled:
                    # ✅ CRITICAL FIX: Save original values before setting new ones (same as online evaluation)
                    # ✅ CRITICAL FIX: Normalize feature flag keys to uppercase environment variable names
                    for key, value in features_enabled.items():
                        env_key = _normalize_feature_flag_key(key)  # Map lowercase to uppercase if needed
                        if env_key not in original_env_vars:
                            original_env_vars[env_key] = os.environ.get(env_key)  # Save original value (None if not set)
                        os.environ[env_key] = str(value)
                    logger.info(f"🔧 {symbol} {horizon}d WFV: Feature flags set from best_params: {len(features_enabled)} flags")
                # Also set smart-ensemble params from feature_params if available (parity with HPO)
                # ✅ CRITICAL FIX: Save original values before setting new ones (same as online evaluation)
                try:
                    fp = best_params.get('feature_params', {}) if isinstance(best_params, dict) else {}
                    if isinstance(fp, dict) and fp:
                        if 'smart_consensus_weight' in fp:
                            original_smart_consensus_weight = os.environ.get('ML_SMART_CONSENSUS_WEIGHT')
                            os.environ['ML_SMART_CONSENSUS_WEIGHT'] = str(fp['smart_consensus_weight'])
                        if 'smart_performance_weight' in fp:
                            original_smart_performance_weight = os.environ.get('ML_SMART_PERFORMANCE_WEIGHT')
                            os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = str(fp['smart_performance_weight'])
                        if 'smart_sigma' in fp:
                            original_smart_sigma = os.environ.get('ML_SMART_SIGMA')
                            os.environ['ML_SMART_SIGMA'] = str(fp['smart_sigma'])
                        if 'smart_weight_xgb' in fp:
                            original_smart_weight_xgb = os.environ.get('ML_SMART_WEIGHT_XGB')
                            os.environ['ML_SMART_WEIGHT_XGB'] = str(fp['smart_weight_xgb'])
                        if 'smart_weight_lgbm' in fp or 'smart_weight_lgb' in fp:
                            original_smart_weight_lgb = os.environ.get('ML_SMART_WEIGHT_LGB')
                            os.environ['ML_SMART_WEIGHT_LGB'] = str(fp.get('smart_weight_lgbm', fp.get('smart_weight_lgb')))
                        if 'smart_weight_cat' in fp:
                            original_smart_weight_cat = os.environ.get('ML_SMART_WEIGHT_CAT')
                            os.environ['ML_SMART_WEIGHT_CAT'] = str(fp['smart_weight_cat'])
                except Exception:
                    pass
                # Also check enable_* keys in best_params
                # ✅ CRITICAL FIX: Save original values before setting new ones (fallback path, same as online evaluation)
                feature_flag_keys = [k for k in best_params.keys() if k.startswith('enable_')]
                for key in feature_flag_keys:
                    value = best_params[key]
                    if key == 'enable_external_features':
                        env_key = 'ENABLE_EXTERNAL_FEATURES'
                        if env_key not in original_env_vars:
                            original_env_vars[env_key] = os.environ.get(env_key)
                        os.environ[env_key] = '1' if value else '0'
                    elif key == 'enable_fingpt_features':
                        env_key = 'ENABLE_FINGPT_FEATURES'
                        if env_key not in original_env_vars:
                            original_env_vars[env_key] = os.environ.get(env_key)
                        os.environ[env_key] = '1' if value else '0'
                    elif key == 'enable_yolo_features':
                        env_key = 'ENABLE_YOLO_FEATURES'
                        if env_key not in original_env_vars:
                            original_env_vars[env_key] = os.environ.get(env_key)
                        os.environ[env_key] = '1' if value else '0'
                    elif key == 'enable_directional_loss':
                        env_key = 'ML_USE_DIRECTIONAL_LOSS'
                        if env_key not in original_env_vars:
                            original_env_vars[env_key] = os.environ.get(env_key)
                        os.environ[env_key] = '1' if value else '0'
                    elif key == 'enable_seed_bagging':
                        env_key = 'ENABLE_SEED_BAGGING'
                        if env_key not in original_env_vars:
                            original_env_vars[env_key] = os.environ.get(env_key)
                        os.environ[env_key] = '1' if value else '0'
                    elif key == 'enable_talib_patterns':
                        env_key = 'ENABLE_TALIB_PATTERNS'
                        if env_key not in original_env_vars:
                            original_env_vars[env_key] = os.environ.get(env_key)
                        os.environ[env_key] = '1' if value else '0'
                    elif key == 'enable_smart_ensemble':
                        env_key = 'ML_USE_SMART_ENSEMBLE'
                        if env_key not in original_env_vars:
                            original_env_vars[env_key] = os.environ.get(env_key)
                        os.environ[env_key] = '1' if value else '0'
                    elif key == 'enable_stacked_short':
                        env_key = 'ML_USE_STACKED_SHORT'
                        if env_key not in original_env_vars:
                            original_env_vars[env_key] = os.environ.get(env_key)
                        os.environ[env_key] = '1' if value else '0'
                    elif key == 'enable_meta_stacking':
                        env_key = 'ENABLE_META_STACKING'
                        if env_key not in original_env_vars:
                            original_env_vars[env_key] = os.environ.get(env_key)
                        os.environ[env_key] = '1' if value else '0'
                    elif key == 'enable_regime_detection':
                        env_key = 'ML_USE_REGIME_DETECTION'
                        if env_key not in original_env_vars:
                            original_env_vars[env_key] = os.environ.get(env_key)
                        os.environ[env_key] = '1' if value else '0'
                    elif key == 'enable_fingpt':
                        env_key = 'ENABLE_FINGPT'
                        if env_key not in original_env_vars:
                            original_env_vars[env_key] = os.environ.get(env_key)
                        os.environ[env_key] = '1' if value else '0'
            else:
                # Fallback: Align with HPO - disable seed bagging and directional loss during evaluation
                # ✅ CRITICAL FIX: Original values already captured above (before if/else), just set to disabled values
                os.environ['ENABLE_SEED_BAGGING'] = '0'
                # ✅ CRITICAL FIX: Align with HPO - disable directional loss (use MSE loss)
                os.environ['ML_USE_DIRECTIONAL_LOSS'] = '0'  # MSE loss (same as HPO)
            # ✅ CRITICAL FIX: Don't use singleton, create new instance (same as HPO)
            # import enhanced_ml_system
            # enhanced_ml_system._enhanced_ml_system = None  # Not needed, we create new instance
            
            # ✅ FIX: Clear ConfigManager cache to ensure best params are read
            try:
                from bist_pattern.core.config_manager import ConfigManager
                ConfigManager.clear_cache()
            except Exception:
                pass
            # ✅ CRITICAL FIX: Use best trial's seed for evaluation (match HPO best trial)
            # This ensures evaluation uses the same random seed as the best HPO trial
            # If best_trial_number is not available, fallback to 42
            # ✅ FIX: Try to get best_trial_number from best_params first, then from hpo_result
            best_trial_number = None
            if isinstance(best_params, dict):
                best_trial_number = best_params.get('best_trial_number')
            if best_trial_number is None and hpo_result:
                best_trial_number = hpo_result.get('best_trial_number')
            eval_seed = best_trial_number if best_trial_number is not None else 42
            try:
                import random
                random.seed(eval_seed)
                np.random.seed(eval_seed)
                os.environ['PYTHONHASHSEED'] = str(eval_seed)
                # Hint for model libraries (read by EnhancedMLSystem if supported)
                os.environ.setdefault('OPTUNA_XGB_RANDOM_STATE', str(eval_seed))
                os.environ.setdefault('OPTUNA_LGB_RANDOM_STATE', str(eval_seed))
                os.environ.setdefault('OPTUNA_CAT_RANDOM_STATE', str(eval_seed))
                os.environ.setdefault('XGB_SEED', str(eval_seed))
                os.environ.setdefault('LIGHTGBM_SEED', str(eval_seed))
                os.environ.setdefault('CATBOOST_RANDOM_SEED', str(eval_seed))
                logger.info(f"🔧 {symbol} {horizon}d WFV: Using seed={eval_seed} (best_trial={best_trial_number}) for evaluation")
            except Exception:
                pass
            
            # ✅ CRITICAL FIX: Create NEW instance (same as HPO) to avoid singleton cache pollution
            # HPO creates new EnhancedMLSystem() for each trial, we should do the same for evaluation
            from enhanced_ml_system import EnhancedMLSystem
            ml_eval = EnhancedMLSystem()  # New instance, not singleton!
            ml_eval.prediction_horizons = [horizon]
            # ✅ CRITICAL FIX: Set base_seeds to match HPO best trial
            # HPO uses: ml.base_seeds = [42 + trial.number]
            # Evaluation should use: ml_eval.base_seeds = [42 + best_trial_number]
            # ✅ FIX: If best_trial_number is None, use fallback seed (42) but log warning
            if best_trial_number is None:
                logger.warning(f"⚠️ {symbol} {horizon}d WFV: best_trial_number not found, using fallback seed=42 (may cause alignment issues)")
            ml_eval.base_seeds = [42 + eval_seed]  # eval_seed = best_trial_number or 42
            # ✅ FIX: Ensure enable_seed_bagging matches HPO best trial (from environment variable)
            # Note: base_seeds is already set to single seed, so seed bagging will effectively be disabled
            # But we ensure the flag matches HPO for consistency
            logger.info(f"🔧 {symbol} {horizon}d WFV: Set base_seeds={ml_eval.base_seeds}, enable_seed_bagging={ml_eval.enable_seed_bagging} (matching HPO trial {best_trial_number})")
            
            reference_entry = _get_best_trial_metrics_entry(hpo_result, symbol, horizon)
            reference_r2_map = _extract_reference_historical_r2(reference_entry)
            reference_key = f"{symbol}_{horizon}d"
            if reference_r2_map:
                try:
                    ml_eval.reference_historical_r2[reference_key] = reference_r2_map
                    ml_eval.use_reference_historical_r2 = True
                    logger.info(
                        f"📊 {symbol} {horizon}d: Using HPO historical R² reference for ensemble weighting "
                        f"(models: {', '.join(reference_r2_map.keys())})"
                    )
                except Exception as ref_err:
                    logger.warning(f"⚠️ {symbol} {horizon}d: Failed to inject reference historical R²: {ref_err}")
                    ml_eval.use_reference_historical_r2 = False
            else:
                ml_eval.use_reference_historical_r2 = False
                ml_eval.reference_historical_r2.pop(reference_key, None)
            
            # ⚡ NEW: Evaluate on all splits and average DirHit, nRMSE, and Score (same as HPO)
            split_dirhits = []
            split_nrmses = []
            split_scores = []
            for split_idx, (train_end_idx, test_end_idx) in enumerate(wfv_splits, 1):
                train_df_split = df.iloc[:train_end_idx].copy()
                test_df_split = df.iloc[train_end_idx:test_end_idx].copy()
                
                logger.info(f"📊 {symbol} {horizon}d WFV Split {split_idx}/{len(wfv_splits)}: train={len(train_df_split)} days, test={len(test_df_split)} days")
                logger.info(f"🔍 [eval-debug] {symbol} {horizon}d Split {split_idx}: Train period: {train_df_split.index.min()} to {train_df_split.index.max()}")
                logger.info(f"🔍 [eval-debug] {symbol} {horizon}d Split {split_idx}: Test period: {test_df_split.index.min()} to {test_df_split.index.max()}")
                
                y_true_split = self._compute_returns(test_df_split, horizon)
                min_test_days = horizon + 10
                if len(test_df_split) < min_test_days:
                    logger.warning(f"⚠️ {symbol} {horizon}d Split {split_idx}: Insufficient test data ({len(test_df_split)} days, need {min_test_days})")
                    continue
                
                # ✅ FIX: Data leakage önleme - Model'i train_df ile yeniden eğit (best params ile)
                # Model zaten tüm df ile eğitilmiş (adaptive ON), bu yüzden test verisini görmüş
                # Evaluation'da model'i train_df ile yeniden eğiterek data leakage'ı önlüyoruz
                # Best params kullanılarak HPO ile aynı koşullar sağlanıyor
                # Log evaluation environment summary (only for first split)
                if split_idx == 1:
                    logger.info(
                        f"🔧 Eval env (WFV): adaptive=0, seed_bagging={os.environ.get('ENABLE_SEED_BAGGING', '0')}, "
                        f"directional_loss={os.environ.get('ML_USE_DIRECTIONAL_LOSS', '0')}, "
                        f"smart={os.environ.get('ML_USE_SMART_ENSEMBLE', '1')}, stacked={os.environ.get('ML_USE_STACKED_SHORT', '1')}, "
                        f"regime={os.environ.get('ML_USE_REGIME_DETECTION', '1')}"
                    )
                    logger.info(f"🔄 {symbol} {horizon}d WFV: Model'i train_df ile yeniden eğitiyoruz (best params, adaptive OFF, data leakage önleme, NEW instance)")
                
                train_result = ml_eval.train_enhanced_models(symbol, train_df_split)
                if not train_result:
                    logger.warning(f"⚠️ {symbol} {horizon}d WFV Split {split_idx}: Model eğitimi başarısız, split atlanıyor")
                    continue
                
                if reference_r2_map:
                    actual_metrics = _extract_model_metrics_from_train_result(train_result, horizon)
                    if actual_metrics:
                        for model_name, ref_val in reference_r2_map.items():
                            actual_val = actual_metrics.get(model_name, {}).get('raw_r2')
                            if actual_val is None or not math.isfinite(actual_val):
                                continue
                            delta = actual_val - ref_val
                            logger.info(
                                f"📏 {symbol} {horizon}d WFV Split {split_idx}: {model_name} raw_r2 "
                                f"HPO={ref_val:.4f}, train={actual_val:.4f}, delta={delta:+.4f}"
                            )
                preds = np.full(len(test_df_split), np.nan, dtype=float)
                valid_predictions = 0
                for t in range(len(test_df_split) - horizon):
                    try:
                        cur = pd.concat([train_df_split, test_df_split.iloc[:t + 1]], axis=0).copy()
                        p = ml_eval.predict_enhanced(symbol, cur)
                        if isinstance(p, dict):
                            key = f"{horizon}d"
                            obj = p.get(key)
                            if isinstance(obj, dict):
                                pred_price = obj.get('ensemble_prediction')
                                if isinstance(pred_price, (int, float)) and not np.isnan(pred_price):
                                    last_close = float(cur['close'].iloc[-1])
                                    if last_close > 0:
                                        preds[t] = float(pred_price) / last_close - 1.0
                                        valid_predictions += 1
                    except Exception:
                        continue
                
                # 🔍 DEBUG: Calculate detailed metrics (same as HPO)
                valid_mask = ~np.isnan(preds) & ~np.isnan(y_true_split)
                valid_count = valid_mask.sum()
                
                if valid_count > 0:
                    # DirHit
                    dh = self._dirhit(y_true_split, preds)
                    
                    # RMSE and MAPE
                    y_true_valid = y_true_split[valid_mask]
                    preds_valid = preds[valid_mask]
                    rmse = np.sqrt(np.mean((y_true_valid - preds_valid) ** 2))
                    mape = np.mean(np.abs((y_true_valid - preds_valid) / (y_true_valid + 1e-8))) * 100
                    
                    # ✅ NEW: Calculate nRMSE (normalized by std of y_true_valid)
                    nrmse_val = float('inf')
                    try:
                        std_y = float(np.std(y_true_valid)) if y_true_valid.size > 1 else 0.0
                        if std_y > 0:
                            nrmse_val = float(rmse / std_y)
                    except Exception:
                        pass
                    
                    # ✅ NEW: Calculate Score (same formula as HPO)
                    from enhanced_ml_system import EnhancedMLSystem
                    score_val = EnhancedMLSystem._calculate_score(dh, nrmse_val, horizon)
                    
                    # Threshold mask statistics
                    thr = 0.005
                    mask_count = ((np.abs(y_true_valid) > thr) & (np.abs(preds_valid) > thr)).sum()
                    mask_pct = (mask_count / valid_count) * 100 if valid_count > 0 else 0
                    
                    # ✅ FIX: Log prediction magnitude statistics for mask_count=0 debugging
                    if mask_count == 0 and valid_count > 0:
                        pred_abs_max = float(np.abs(preds_valid).max()) if len(preds_valid) > 0 else 0.0
                        pred_abs_mean = float(np.abs(preds_valid).mean()) if len(preds_valid) > 0 else 0.0
                        y_true_abs_max = float(np.abs(y_true_valid).max()) if len(y_true_valid) > 0 else 0.0
                        y_true_abs_mean = float(np.abs(y_true_valid).mean()) if len(y_true_valid) > 0 else 0.0
                        logger.warning(
                            f"⚠️ {symbol} {horizon}d WFV Split {split_idx}: mask_count=0 (threshold={thr}) - "
                            f"pred_abs_max={pred_abs_max:.6f}, pred_abs_mean={pred_abs_mean:.6f}, "
                            f"y_true_abs_max={y_true_abs_max:.6f}, y_true_abs_mean={y_true_abs_mean:.6f}"
                        )
                    
                    # Direction statistics
                    direction_matches = (np.sign(y_true_valid) == np.sign(preds_valid)).sum()
                    direction_pct = (direction_matches / valid_count) * 100 if valid_count > 0 else 0
                    
                    # 🔍 DEBUG: Log detailed metrics (only for first split)
                    if split_idx == 1:
                        logger.info(f"🔍 [eval-debug] {symbol} {horizon}d WFV: METRICS:")
                        logger.info(f"🔍 [eval-debug]   Valid predictions: {valid_count}/{len(preds)}")
                        logger.info(f"🔍 [eval-debug]   DirHit (threshold={thr}): {dh:.2f}% (mask_count={mask_count}, mask_pct={mask_pct:.1f}%)")
                        logger.info(f"🔍 [eval-debug]   Direction match (all): {direction_pct:.2f}% ({direction_matches}/{valid_count})")
                        logger.info(f"🔍 [eval-debug]   RMSE: {rmse:.6f}")
                        logger.info(f"🔍 [eval-debug]   MAPE: {mape:.2f}%")
                        logger.info(f"🔍 [eval-debug]   nRMSE: {nrmse_val:.3f}")
                        logger.info(f"🔍 [eval-debug]   Score: {score_val:.2f}")
                        logger.info(f"🔍 [eval-debug]   Seed = {eval_seed} (best_trial={best_trial_number})")
                    
                    if not np.isnan(dh):
                        split_dirhits.append(dh)
                        split_nrmses.append(nrmse_val)
                        split_scores.append(score_val)
                        logger.info(
                            f"✅ {symbol} {horizon}d WFV Split {split_idx}: DirHit={dh:.2f}%, nRMSE={nrmse_val:.3f}, "
                            f"Score={score_val:.2f} (valid={valid_count}/{len(preds)}, RMSE={rmse:.6f}, MAPE={mape:.2f}%)"
                        )
                    else:
                        logger.warning(f"⚠️ {symbol} {horizon}d WFV Split {split_idx}: DirHit calculation returned NaN")
                else:
                    logger.warning(f"⚠️ {symbol} {horizon}d WFV Split {split_idx}: No valid predictions!")
            
            # Average DirHit, nRMSE, and Score across all splits
            if split_dirhits:
                avg_dirhit = float(np.mean(split_dirhits))
                avg_nrmse = float(np.mean(split_nrmses)) if split_nrmses else float('inf')
                avg_score = float(np.mean(split_scores)) if split_scores else 0.0
                logger.info(f"✅ {symbol} {horizon}d WFV: Average across {len(split_dirhits)} splits: DirHit={avg_dirhit:.2f}%, nRMSE={avg_nrmse:.3f}, Score={avg_score:.2f}")
                results['wfv'] = avg_dirhit
                results['wfv_nrmse'] = avg_nrmse
                results['wfv_score'] = avg_score
            else:
                logger.warning(f"⚠️ {symbol} {horizon}d WFV: No valid DirHit from any split")
                results['wfv'] = None
                results['wfv_nrmse'] = None
                results['wfv_score'] = None
        except Exception as e:
            logger.error(f"❌ WFV evaluation error for {symbol} {horizon}d: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # ✅ CRITICAL FIX: Restore adaptive learning - remove if it wasn't set before (same logic as smart ensemble params)
            if original_adaptive is not None:
                os.environ['ML_USE_ADAPTIVE_LEARNING'] = original_adaptive
            elif 'ML_USE_ADAPTIVE_LEARNING' in os.environ and original_adaptive is None:
                # Was not set before, remove it
                os.environ.pop('ML_USE_ADAPTIVE_LEARNING', None)
            # ✅ CRITICAL FIX: Restore seed bagging - remove if it wasn't set before (same logic as smart ensemble params)
            if original_seed_bag is not None:
                os.environ['ENABLE_SEED_BAGGING'] = original_seed_bag
            elif 'ENABLE_SEED_BAGGING' in os.environ and original_seed_bag is None:
                # Was not set before, remove it
                os.environ.pop('ENABLE_SEED_BAGGING', None)
            # ✅ CRITICAL FIX: Restore directional loss - remove if it wasn't set before (same logic as smart ensemble params)
            if original_directional is not None:
                os.environ['ML_USE_DIRECTIONAL_LOSS'] = original_directional
            elif 'ML_USE_DIRECTIONAL_LOSS' in os.environ and original_directional is None:
                # Was not set before, remove it
                os.environ.pop('ML_USE_DIRECTIONAL_LOSS', None)
            # ✅ CRITICAL FIX: Restore smart ensemble parameters (same as online evaluation)
            if original_smart_consensus_weight is not None:
                os.environ['ML_SMART_CONSENSUS_WEIGHT'] = original_smart_consensus_weight
            elif 'ML_SMART_CONSENSUS_WEIGHT' in os.environ and original_smart_consensus_weight is None:
                # Was not set before, remove it
                os.environ.pop('ML_SMART_CONSENSUS_WEIGHT', None)
            if original_smart_performance_weight is not None:
                os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = original_smart_performance_weight
            elif 'ML_SMART_PERFORMANCE_WEIGHT' in os.environ and original_smart_performance_weight is None:
                os.environ.pop('ML_SMART_PERFORMANCE_WEIGHT', None)
            if original_smart_sigma is not None:
                os.environ['ML_SMART_SIGMA'] = original_smart_sigma
            elif 'ML_SMART_SIGMA' in os.environ and original_smart_sigma is None:
                os.environ.pop('ML_SMART_SIGMA', None)
            if original_smart_weight_xgb is not None:
                os.environ['ML_SMART_WEIGHT_XGB'] = original_smart_weight_xgb
            elif 'ML_SMART_WEIGHT_XGB' in os.environ and original_smart_weight_xgb is None:
                os.environ.pop('ML_SMART_WEIGHT_XGB', None)
            if original_smart_weight_lgb is not None:
                os.environ['ML_SMART_WEIGHT_LGB'] = original_smart_weight_lgb
            elif 'ML_SMART_WEIGHT_LGB' in os.environ and original_smart_weight_lgb is None:
                os.environ.pop('ML_SMART_WEIGHT_LGB', None)
            if original_smart_weight_cat is not None:
                os.environ['ML_SMART_WEIGHT_CAT'] = original_smart_weight_cat
            elif 'ML_SMART_WEIGHT_CAT' in os.environ and original_smart_weight_cat is None:
                os.environ.pop('ML_SMART_WEIGHT_CAT', None)
            # ✅ CRITICAL FIX: Restore all dynamically set environment variables from features_enabled and enable_* keys (same as online evaluation)
            for key, original_value in original_env_vars.items():
                if original_value is not None:
                    os.environ[key] = original_value
                elif key in os.environ:
                    # Was not set before, remove it
                    os.environ.pop(key, None)

        # 2) Online (adaptive OFF) - HİBRİT YAKLAŞIM: Model'i train_df ile yeniden eğit (adaptive OFF - HPO ile tutarlılık)
        # Initialize for finally safety
        original_adaptive2: Optional[str] = None
        original_seed_bag2: Optional[str] = None
        original_directional2: Optional[str] = None
        # ✅ CRITICAL FIX: Save original smart ensemble parameters for restoration
        original_smart_consensus_weight2: Optional[str] = None
        original_smart_performance_weight2: Optional[str] = None
        original_smart_sigma2: Optional[str] = None
        original_smart_weight_xgb2: Optional[str] = None
        original_smart_weight_lgb2: Optional[str] = None
        original_smart_weight_cat2: Optional[str] = None
        # ✅ CRITICAL FIX: Save all dynamically set environment variables from features_enabled
        original_env_vars2: dict = {}
        try:
            # ✅ CRITICAL FIX: Don't use default value - we need to distinguish between "unset" and "set to '0'"
            original_adaptive2 = os.environ.get('ML_USE_ADAPTIVE_LEARNING')  # None if not set
            os.environ['ML_USE_ADAPTIVE_LEARNING'] = '0'  # ✅ HİBRİT YAKLAŞIM: Adaptive OFF (HPO ile tutarlılık)
            
            # ✅ CRITICAL FIX: Set feature flags from best_params to match HPO best trial exactly
            # This ensures evaluation uses the same feature flags as HPO (including seed bagging and directional loss)
            # ✅ CRITICAL FIX: Don't use default values - we need to distinguish between "unset" and "set to default"
            original_seed_bag2 = os.environ.get('ENABLE_SEED_BAGGING')  # None if not set
            original_directional2 = os.environ.get('ML_USE_DIRECTIONAL_LOSS')  # None if not set
            
            if best_params and isinstance(best_params, dict):
                # Set feature flags from features_enabled dict (HPO best trial)
                features_enabled = best_params.get('features_enabled', {})
                if features_enabled:
                    # ✅ CRITICAL FIX: Save original values before setting new ones
                    # ✅ CRITICAL FIX: Normalize feature flag keys to uppercase environment variable names
                    for key, value in features_enabled.items():
                        env_key = _normalize_feature_flag_key(key)  # Map lowercase to uppercase if needed
                        if env_key not in original_env_vars2:
                            original_env_vars2[env_key] = os.environ.get(env_key)  # Save original value (None if not set)
                        os.environ[env_key] = str(value)
                    logger.info(f"🔧 {symbol} {horizon}d Online: Feature flags set from best_params: {len(features_enabled)} flags")
                    seed_bag_val = os.environ.get('ENABLE_SEED_BAGGING', 'NOT_SET')
                    dir_loss_val = os.environ.get('ML_USE_DIRECTIONAL_LOSS', 'NOT_SET')
                    logger.info(f"🔧 {symbol} {horizon}d Online: ENABLE_SEED_BAGGING={seed_bag_val}, ML_USE_DIRECTIONAL_LOSS={dir_loss_val}")
                else:
                    # Fallback: Also check enable_* keys in best_params
                    feature_flag_keys = [k for k in best_params.keys() if k.startswith('enable_')]
                    for key in feature_flag_keys:
                        value = best_params[key]
                        # ✅ CRITICAL FIX: Save original values before setting new ones (fallback path)
                        if key == 'enable_external_features':
                            env_key = 'ENABLE_EXTERNAL_FEATURES'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(env_key)
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_fingpt_features':
                            env_key = 'ENABLE_FINGPT_FEATURES'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(env_key)
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_yolo_features':
                            env_key = 'ENABLE_YOLO_FEATURES'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(env_key)
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_directional_loss':
                            env_key = 'ML_USE_DIRECTIONAL_LOSS'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(env_key)
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_seed_bagging':
                            env_key = 'ENABLE_SEED_BAGGING'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(env_key)
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_talib_patterns':
                            env_key = 'ENABLE_TALIB_PATTERNS'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(env_key)
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_smart_ensemble':
                            env_key = 'ML_USE_SMART_ENSEMBLE'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(env_key)
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_stacked_short':
                            env_key = 'ML_USE_STACKED_SHORT'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(env_key)
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_meta_stacking':
                            env_key = 'ENABLE_META_STACKING'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(env_key)
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_regime_detection':
                            env_key = 'ML_USE_REGIME_DETECTION'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(env_key)
                            os.environ[env_key] = '1' if value else '0'
                        elif key == 'enable_fingpt':
                            env_key = 'ENABLE_FINGPT'
                            if env_key not in original_env_vars2:
                                original_env_vars2[env_key] = os.environ.get(env_key)
                            os.environ[env_key] = '1' if value else '0'
                    logger.info(f"🔧 {symbol} {horizon}d Online: Feature flags set from enable_* keys: {len(feature_flag_keys)} flags")
                # Also set smart-ensemble params from feature_params if available (parity with HPO)
                try:
                    fp = best_params.get('feature_params', {}) if isinstance(best_params, dict) else {}
                    if isinstance(fp, dict) and fp:
                        # ✅ CRITICAL FIX: Save original values before setting new ones
                        if 'smart_consensus_weight' in fp:
                            original_smart_consensus_weight2 = os.environ.get('ML_SMART_CONSENSUS_WEIGHT')
                            os.environ['ML_SMART_CONSENSUS_WEIGHT'] = str(fp['smart_consensus_weight'])
                        if 'smart_performance_weight' in fp:
                            original_smart_performance_weight2 = os.environ.get('ML_SMART_PERFORMANCE_WEIGHT')
                            os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = str(fp['smart_performance_weight'])
                        if 'smart_sigma' in fp:
                            original_smart_sigma2 = os.environ.get('ML_SMART_SIGMA')
                            os.environ['ML_SMART_SIGMA'] = str(fp['smart_sigma'])
                        if 'smart_weight_xgb' in fp:
                            original_smart_weight_xgb2 = os.environ.get('ML_SMART_WEIGHT_XGB')
                            os.environ['ML_SMART_WEIGHT_XGB'] = str(fp['smart_weight_xgb'])
                        if 'smart_weight_lgbm' in fp or 'smart_weight_lgb' in fp:
                            original_smart_weight_lgb2 = os.environ.get('ML_SMART_WEIGHT_LGB')
                            os.environ['ML_SMART_WEIGHT_LGB'] = str(fp.get('smart_weight_lgbm', fp.get('smart_weight_lgb')))
                        if 'smart_weight_cat' in fp:
                            original_smart_weight_cat2 = os.environ.get('ML_SMART_WEIGHT_CAT')
                            os.environ['ML_SMART_WEIGHT_CAT'] = str(fp['smart_weight_cat'])
                except Exception:
                    pass
            else:
                # Fallback: Only disable if best_params not available (should not happen in normal flow)
                logger.warning(f"⚠️ {symbol} {horizon}d Online: best_params not available, using fallback (disabling seed bagging and directional loss)")
                os.environ['ENABLE_SEED_BAGGING'] = '0'
                os.environ['ML_USE_DIRECTIONAL_LOSS'] = '0'  # MSE loss (fallback)
            # ✅ CRITICAL FIX: Don't use singleton, create new instance (same as HPO)
            # import enhanced_ml_system
            # enhanced_ml_system._enhanced_ml_system = None  # Not needed, we create new instance
            
            # ✅ FIX: Clear ConfigManager cache to ensure best params are read
            try:
                from bist_pattern.core.config_manager import ConfigManager
                ConfigManager.clear_cache()
            except Exception:
                pass
            # ✅ CRITICAL FIX: Use best trial's seed for evaluation (match HPO best trial)
            # This ensures evaluation uses the same random seed as the best HPO trial
            # If best_trial_number is not available, fallback to 42
            # ✅ FIX: Try to get best_trial_number from best_params first, then from hpo_result
            best_trial_number = None
            if isinstance(best_params, dict):
                best_trial_number = best_params.get('best_trial_number')
            if best_trial_number is None and hpo_result:
                best_trial_number = hpo_result.get('best_trial_number')
            eval_seed = best_trial_number if best_trial_number is not None else 42
            try:
                import random
                random.seed(eval_seed)
                np.random.seed(eval_seed)
                os.environ['PYTHONHASHSEED'] = str(eval_seed)
                os.environ.setdefault('OPTUNA_XGB_RANDOM_STATE', str(eval_seed))
                os.environ.setdefault('OPTUNA_LGB_RANDOM_STATE', str(eval_seed))
                os.environ.setdefault('OPTUNA_CAT_RANDOM_STATE', str(eval_seed))
                os.environ.setdefault('XGB_SEED', str(eval_seed))
                os.environ.setdefault('LIGHTGBM_SEED', str(eval_seed))
                os.environ.setdefault('CATBOOST_RANDOM_SEED', str(eval_seed))
                logger.info(f"🔧 {symbol} {horizon}d Online: Using seed={eval_seed} (best_trial={best_trial_number}) for evaluation")
            except Exception:
                pass
            
            # ✅ CRITICAL FIX: Create NEW instance (same as HPO) to avoid singleton cache pollution
            # HPO creates new EnhancedMLSystem() for each trial, we should do the same for evaluation
            from enhanced_ml_system import EnhancedMLSystem
            ml_online = EnhancedMLSystem()  # New instance, not singleton!
            ml_online.prediction_horizons = [horizon]
            # ✅ CRITICAL FIX: Set base_seeds to match HPO best trial
            # HPO uses: ml.base_seeds = [42 + trial.number]
            # Evaluation should use: ml_online.base_seeds = [42 + best_trial_number]
            # ✅ FIX: If best_trial_number is None, use fallback seed (42) but log warning
            if best_trial_number is None:
                logger.warning(f"⚠️ {symbol} {horizon}d Online: best_trial_number not found, using fallback seed=42 (may cause alignment issues)")
            ml_online.base_seeds = [42 + eval_seed]  # eval_seed = best_trial_number or 42
            # ✅ FIX: Ensure enable_seed_bagging matches HPO best trial (from environment variable)
            # Note: base_seeds is already set to single seed, so seed bagging will effectively be disabled
            # But we ensure the flag matches HPO for consistency
            logger.info(f"🔧 {symbol} {horizon}d Online: Set base_seeds={ml_online.base_seeds}, enable_seed_bagging={ml_online.enable_seed_bagging} (matching HPO trial {best_trial_number})")
            if reference_r2_map:
                try:
                    ml_online.reference_historical_r2[reference_key] = reference_r2_map
                    ml_online.use_reference_historical_r2 = True
                except Exception as ref_err:
                    logger.warning(f"⚠️ {symbol} {horizon}d Online: Failed to inject reference historical R²: {ref_err}")
                    ml_online.use_reference_historical_r2 = False
            else:
                ml_online.use_reference_historical_r2 = False
                ml_online.reference_historical_r2.pop(reference_key, None)
            
            # ✅ CRITICAL FIX: Evaluation mode - skip Phase 2 to match HPO data usage
            # HPO'da tüm train_df kullanılıyor (adaptive OFF)
            # Evaluation'da (Online) adaptive ON olduğunda train_df içinde split yapılıyor (Phase 1: %67, Phase 2: %33)
            # Bu yüzden evaluation'da daha az veri ile eğitim yapılıyor ve DirHit düşük çıkıyor!
            # Çözüm: Evaluation'da ML_SKIP_ADAPTIVE_PHASE2=1 ile Phase 2'yi atla, tüm train_df ile Phase 1 yap
            os.environ['ML_SKIP_ADAPTIVE_PHASE2'] = '1'  # ✅ Skip Phase 2 in evaluation mode
            
            # ⚡ NEW: Evaluate on all splits and average DirHit (same as HPO and WFV)
            split_dirhits_online = []
            for split_idx, (train_end_idx, test_end_idx) in enumerate(wfv_splits, 1):
                train_df_split = df.iloc[:train_end_idx].copy()
                test_df_split = df.iloc[train_end_idx:test_end_idx].copy()
                
                logger.info(f"📊 {symbol} {horizon}d Online Split {split_idx}/{len(wfv_splits)}: train={len(train_df_split)} days, test={len(test_df_split)} days")
                logger.info(f"🔍 [eval-debug] {symbol} {horizon}d Online Split {split_idx}: Train period: {train_df_split.index.min()} to {train_df_split.index.max()}")
                logger.info(f"🔍 [eval-debug] {symbol} {horizon}d Online Split {split_idx}: Test period: {test_df_split.index.min()} to {test_df_split.index.max()}")
                
                y_true_split = self._compute_returns(test_df_split, horizon)
                min_test_days = horizon + 10
                if len(test_df_split) < min_test_days:
                    logger.warning(f"⚠️ {symbol} {horizon}d Online Split {split_idx}: Insufficient test data ({len(test_df_split)} days, need {min_test_days})")
                    continue
                
                # ✅ HİBRİT YAKLAŞIM: Model'i train_df ile yeniden eğit (best params, adaptive OFF)
                # Evaluation'da model'i train_df ile yeniden eğiterek data leakage'ı önlüyoruz
                # Online için adaptive OFF ile eğitiyoruz (HPO ile tutarlılık), Phase 2'yi atlayarak HPO ile aynı veri miktarını kullanıyoruz
                # Best params kullanılarak HPO ile aynı koşullar sağlanıyor
                # Log evaluation environment summary (only for first split)
                if split_idx == 1:
                    logger.info(
                        f"🔧 Eval env (Online): adaptive=0, skip_phase2=1, seed_bagging={os.environ.get('ENABLE_SEED_BAGGING', '0')}, "
                        f"directional_loss={os.environ.get('ML_USE_DIRECTIONAL_LOSS', '0')}, "
                        f"smart={os.environ.get('ML_USE_SMART_ENSEMBLE', '1')}, stacked={os.environ.get('ML_USE_STACKED_SHORT', '1')}, "
                        f"regime={os.environ.get('ML_USE_REGIME_DETECTION', '1')}"
                    )
                    logger.info(f"🔄 {symbol} {horizon}d Online: Model'i train_df ile yeniden eğitiyoruz (best params, adaptive OFF, skip Phase 2, NEW instance)")
                    logger.info("   ✅ HİBRİT YAKLAŞIM: ML_USE_ADAPTIVE_LEARNING=0, ML_SKIP_ADAPTIVE_PHASE2=1 → HPO ile aynı veri miktarı ve ayarlar")
                
                train_result2 = ml_online.train_enhanced_models(symbol, train_df_split)
                if not train_result2:
                    logger.warning(f"⚠️ {symbol} {horizon}d Online Split {split_idx}: Model eğitimi başarısız, split atlanıyor")
                    continue
                
                # ✅ HİBRİT YAKLAŞIM: enhanced_ml_system.py adaptive OFF olduğu için:
                # - Phase 1: Tüm train_df ile eğitim (HPO ile aynı)
                # - Phase 2: SKIP (ML_SKIP_ADAPTIVE_PHASE2=1)
                if split_idx == 1:
                    logger.info(f"✅ {symbol} {horizon}d Online: Model eğitimi tamamlandı (adaptive OFF - HPO ile tutarlılık)")
                
                preds2 = np.full(len(test_df_split), np.nan, dtype=float)
                if reference_r2_map:
                    actual_metrics_online = _extract_model_metrics_from_train_result(train_result2, horizon)
                    if actual_metrics_online:
                        for model_name, ref_val in reference_r2_map.items():
                            actual_val = actual_metrics_online.get(model_name, {}).get('raw_r2')
                            if actual_val is None or not math.isfinite(actual_val):
                                continue
                            delta = actual_val - ref_val
                            logger.info(
                                f"📏 {symbol} {horizon}d Online Split {split_idx}: {model_name} raw_r2 "
                                f"HPO={ref_val:.4f}, train={actual_val:.4f}, delta={delta:+.4f}"
                            )
                valid_predictions2 = 0
                for t in range(len(test_df_split) - horizon):
                    try:
                        cur = pd.concat([train_df_split, test_df_split.iloc[:t + 1]], axis=0).copy()
                        p = ml_online.predict_enhanced(symbol, cur)
                        if isinstance(p, dict):
                            key = f"{horizon}d"
                            obj = p.get(key)
                            if isinstance(obj, dict):
                                pred_price = obj.get('ensemble_prediction')
                                if isinstance(pred_price, (int, float)) and not np.isnan(pred_price):
                                    last_close = float(cur['close'].iloc[-1])
                                    if last_close > 0:
                                        preds2[t] = float(pred_price) / last_close - 1.0
                                        valid_predictions2 += 1
                    except Exception:
                        continue
                
                # 🔍 DEBUG: Calculate detailed metrics (same as HPO)
                valid_mask2 = ~np.isnan(preds2) & ~np.isnan(y_true_split)
                valid_count2 = valid_mask2.sum()
                
                if valid_count2 > 0:
                    # DirHit
                    dh2 = self._dirhit(y_true_split, preds2)
                    
                    # RMSE and MAPE
                    y_true_valid2 = y_true_split[valid_mask2]
                    preds_valid2 = preds2[valid_mask2]
                    rmse2 = np.sqrt(np.mean((y_true_valid2 - preds_valid2) ** 2))
                    mape2 = np.mean(np.abs((y_true_valid2 - preds_valid2) / (y_true_valid2 + 1e-8))) * 100
                    
                    # Threshold mask statistics
                    thr2 = 0.005
                    mask_count2 = ((np.abs(y_true_valid2) > thr2) & (np.abs(preds_valid2) > thr2)).sum()
                    mask_pct2 = (mask_count2 / valid_count2) * 100 if valid_count2 > 0 else 0
                    
                    # ✅ FIX: Log prediction magnitude statistics for mask_count=0 debugging
                    if mask_count2 == 0 and valid_count2 > 0:
                        pred_abs_max2 = float(np.abs(preds_valid2).max()) if len(preds_valid2) > 0 else 0.0
                        pred_abs_mean2 = float(np.abs(preds_valid2).mean()) if len(preds_valid2) > 0 else 0.0
                        y_true_abs_max2 = float(np.abs(y_true_valid2).max()) if len(y_true_valid2) > 0 else 0.0
                        y_true_abs_mean2 = float(np.abs(y_true_valid2).mean()) if len(y_true_valid2) > 0 else 0.0
                        logger.warning(
                            f"⚠️ {symbol} {horizon}d Online Split {split_idx}: mask_count=0 (threshold={thr2}) - "
                            f"pred_abs_max={pred_abs_max2:.6f}, pred_abs_mean={pred_abs_mean2:.6f}, "
                            f"y_true_abs_max={y_true_abs_max2:.6f}, y_true_abs_mean={y_true_abs_mean2:.6f}"
                        )
                    
                    # Direction statistics
                    direction_matches2 = (np.sign(y_true_valid2) == np.sign(preds_valid2)).sum()
                    direction_pct2 = (direction_matches2 / valid_count2) * 100 if valid_count2 > 0 else 0
                    
                    # 🔍 DEBUG: Log detailed metrics (only for first split)
                    if split_idx == 1:
                        logger.info(f"🔍 [eval-debug] {symbol} {horizon}d Online: METRICS:")
                        logger.info(f"🔍 [eval-debug]   Valid predictions: {valid_count2}/{len(preds2)}")
                        logger.info(f"🔍 [eval-debug]   DirHit (threshold={thr2}): {dh2:.2f}% (mask_count={mask_count2}, mask_pct={mask_pct2:.1f}%)")
                        logger.info(f"🔍 [eval-debug]   Direction match (all): {direction_pct2:.2f}% ({direction_matches2}/{valid_count2})")
                        logger.info(f"🔍 [eval-debug]   RMSE: {rmse2:.6f}")
                        logger.info(f"🔍 [eval-debug]   MAPE: {mape2:.2f}%")
                        logger.info(f"🔍 [eval-debug]   Seed = {eval_seed} (best_trial={best_trial_number})")
                        logger.info("🔍 [eval-debug]   Adaptive Learning: OFF (skip_phase2=1, using full train_df - HPO ile tutarlılık)")
                    
                    if not np.isnan(dh2):
                        split_dirhits_online.append(dh2)
                        logger.info(
                            f"✅ {symbol} {horizon}d Online Split {split_idx}: "
                            f"DirHit = {dh2:.2f}% (valid={valid_count2}/{len(preds2)}, "
                            f"mask={mask_count2}, RMSE={rmse2:.6f}, MAPE={mape2:.2f}%)"
                        )
                    else:
                        logger.warning(f"⚠️ {symbol} {horizon}d Online Split {split_idx}: DirHit calculation returned NaN")
                else:
                    logger.warning(f"⚠️ {symbol} {horizon}d Online Split {split_idx}: No valid predictions!")
            
            # Average DirHit across all splits
            if split_dirhits_online:
                avg_dirhit_online = float(np.mean(split_dirhits_online))
                logger.info(f"✅ {symbol} {horizon}d Online: Average DirHit across {len(split_dirhits_online)} splits: {avg_dirhit_online:.2f}% (splits: {split_dirhits_online})")
                results['online'] = avg_dirhit_online
            else:
                logger.warning(f"⚠️ {symbol} {horizon}d Online: No valid DirHit from any split")
                results['online'] = None
        except Exception as e:
            logger.error(f"❌ Online evaluation error for {symbol} {horizon}d: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # ✅ CRITICAL FIX: Restore all modified environment variables
            # ✅ CRITICAL FIX: Restore adaptive learning - remove if it wasn't set before (same logic as smart ensemble params)
            if original_adaptive2 is not None:
                os.environ['ML_USE_ADAPTIVE_LEARNING'] = original_adaptive2
            elif 'ML_USE_ADAPTIVE_LEARNING' in os.environ and original_adaptive2 is None:
                # Was not set before, remove it
                os.environ.pop('ML_USE_ADAPTIVE_LEARNING', None)
            # ✅ CRITICAL FIX: Restore seed bagging - remove if it wasn't set before (same logic as smart ensemble params)
            if original_seed_bag2 is not None:
                os.environ['ENABLE_SEED_BAGGING'] = original_seed_bag2
            elif 'ENABLE_SEED_BAGGING' in os.environ and original_seed_bag2 is None:
                # Was not set before, remove it
                os.environ.pop('ENABLE_SEED_BAGGING', None)
            # ✅ CRITICAL FIX: Restore directional loss - remove if it wasn't set before (same logic as smart ensemble params)
            if original_directional2 is not None:
                os.environ['ML_USE_DIRECTIONAL_LOSS'] = original_directional2
            elif 'ML_USE_DIRECTIONAL_LOSS' in os.environ and original_directional2 is None:
                # Was not set before, remove it
                os.environ.pop('ML_USE_DIRECTIONAL_LOSS', None)
            # ✅ CRITICAL FIX: Restore smart ensemble parameters
            if original_smart_consensus_weight2 is not None:
                os.environ['ML_SMART_CONSENSUS_WEIGHT'] = original_smart_consensus_weight2
            elif 'ML_SMART_CONSENSUS_WEIGHT' in os.environ and original_smart_consensus_weight2 is None:
                # Was not set before, remove it
                os.environ.pop('ML_SMART_CONSENSUS_WEIGHT', None)
            if original_smart_performance_weight2 is not None:
                os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = original_smart_performance_weight2
            elif 'ML_SMART_PERFORMANCE_WEIGHT' in os.environ and original_smart_performance_weight2 is None:
                os.environ.pop('ML_SMART_PERFORMANCE_WEIGHT', None)
            if original_smart_sigma2 is not None:
                os.environ['ML_SMART_SIGMA'] = original_smart_sigma2
            elif 'ML_SMART_SIGMA' in os.environ and original_smart_sigma2 is None:
                os.environ.pop('ML_SMART_SIGMA', None)
            if original_smart_weight_xgb2 is not None:
                os.environ['ML_SMART_WEIGHT_XGB'] = original_smart_weight_xgb2
            elif 'ML_SMART_WEIGHT_XGB' in os.environ and original_smart_weight_xgb2 is None:
                os.environ.pop('ML_SMART_WEIGHT_XGB', None)
            if original_smart_weight_lgb2 is not None:
                os.environ['ML_SMART_WEIGHT_LGB'] = original_smart_weight_lgb2
            elif 'ML_SMART_WEIGHT_LGB' in os.environ and original_smart_weight_lgb2 is None:
                os.environ.pop('ML_SMART_WEIGHT_LGB', None)
            if original_smart_weight_cat2 is not None:
                os.environ['ML_SMART_WEIGHT_CAT'] = original_smart_weight_cat2
            elif 'ML_SMART_WEIGHT_CAT' in os.environ and original_smart_weight_cat2 is None:
                os.environ.pop('ML_SMART_WEIGHT_CAT', None)
            # ✅ CRITICAL FIX: Restore all dynamically set environment variables from features_enabled
            for key, original_value in original_env_vars2.items():
                try:
                    if original_value is not None:
                        os.environ[key] = original_value
                    elif key in os.environ:
                        # Was not set before, remove it
                        os.environ.pop(key, None)
                except Exception:
                    pass
            try:
                os.environ.pop('ML_SKIP_ADAPTIVE_PHASE2', None)  # ✅ Clean up evaluation mode flag
            except Exception:
                pass

        # Mini-CV bloğu kaldırıldı (bakım ve lint sadeleştirme)

        # ✅ DEBUG DUMP: Save evaluation summary for deep analysis (HPO parity checks)
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
            logger.info(f"🧪 Eval debug written: {dbg_path}")
        except Exception as _e:
            logger.debug(f"Eval debug dump failed: {_e}")

        return results

    def run_training(self, symbol: str, horizon: int, best_params: Dict, hpo_result: Optional[Dict] = None) -> Optional[Dict[str, Optional[float]]]:
        """Run training with best params for a symbol-horizon pair
        
        NOTE: This runs with adaptive learning ENABLED (full training with test data)
        
        Args:
            symbol: Stock symbol
            horizon: Prediction horizon
            best_params: Best hyperparameters dict
            hpo_result: Full HPO result dict (includes features_enabled, feature_params, etc.)
        """
        try:
            logger.info(f"🎯 Starting training for {symbol} {horizon}d with best params...")
            
            # Set parameters as environment variables
            from scripts.train_completed_hpo_with_best_params import set_hpo_params_as_env
            set_hpo_params_as_env(best_params, horizon)
            
            # ✅ CRITICAL FIX: Set feature flags from hpo_result (if available)
            # hpo_result contains 'features_enabled' dict with feature flags from best trial
            if hpo_result and 'features_enabled' in hpo_result:
                features_enabled = hpo_result['features_enabled']
                for key, value in features_enabled.items():
                    os.environ[key] = str(value)
                logger.info(f"🔧 {symbol} {horizon}d: Feature flags set from hpo_result: {len(features_enabled)} flags")
                # Smart-ensemble params from feature_params (HPO best trial)
                try:
                    fp = (best_params.get('feature_params') or
                          hpo_result.get('feature_params') or {})
                    if isinstance(fp, dict) and fp:
                        if 'smart_consensus_weight' in fp:
                            os.environ['ML_SMART_CONSENSUS_WEIGHT'] = str(fp['smart_consensus_weight'])
                        if 'smart_performance_weight' in fp:
                            os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = str(fp['smart_performance_weight'])
                        if 'smart_sigma' in fp:
                            os.environ['ML_SMART_SIGMA'] = str(fp['smart_sigma'])
                        if 'smart_weight_xgb' in fp:
                            os.environ['ML_SMART_WEIGHT_XGB'] = str(fp['smart_weight_xgb'])
                        if 'smart_weight_lgbm' in fp or 'smart_weight_lgb' in fp:
                            os.environ['ML_SMART_WEIGHT_LGB'] = str(fp.get('smart_weight_lgbm', fp.get('smart_weight_lgb')))
                        if 'smart_weight_cat' in fp:
                            os.environ['ML_SMART_WEIGHT_CAT'] = str(fp['smart_weight_cat'])
                except Exception:
                    pass
            # Also set feature flags from best_params keys (enable_*)
            elif isinstance(best_params, dict):
                feature_flag_keys = [k for k in best_params.keys() if k.startswith('enable_')]
                for key in feature_flag_keys:
                    value = best_params[key]
                    if key == 'enable_external_features':
                        os.environ['ENABLE_EXTERNAL_FEATURES'] = '1' if value else '0'
                    elif key == 'enable_fingpt_features':
                        os.environ['ENABLE_FINGPT_FEATURES'] = '1' if value else '0'
                    elif key == 'enable_yolo_features':
                        os.environ['ENABLE_YOLO_FEATURES'] = '1' if value else '0'
                    elif key == 'enable_directional_loss':
                        os.environ['ML_USE_DIRECTIONAL_LOSS'] = '1' if value else '0'
                    elif key == 'enable_seed_bagging':
                        os.environ['ENABLE_SEED_BAGGING'] = '1' if value else '0'
                    elif key == 'enable_talib_patterns':
                        os.environ['ENABLE_TALIB_PATTERNS'] = '1' if value else '0'
                    elif key == 'enable_smart_ensemble':
                        os.environ['ML_USE_SMART_ENSEMBLE'] = '1' if value else '0'
                    elif key == 'enable_stacked_short':
                        os.environ['ML_USE_STACKED_SHORT'] = '1' if value else '0'
                    elif key == 'enable_meta_stacking':
                        os.environ['ENABLE_META_STACKING'] = '1' if value else '0'
                    elif key == 'enable_regime_detection':
                        os.environ['ML_USE_REGIME_DETECTION'] = '1' if value else '0'
                    elif key == 'enable_fingpt':
                        os.environ['ENABLE_FINGPT'] = '1' if value else '0'
                if feature_flag_keys:
                    logger.info(f"🔧 {symbol} {horizon}d: Feature flags set from best_params: {len(feature_flag_keys)} flags")
            
            # Set horizon
            os.environ['ML_HORIZONS'] = str(horizon)
            
            # ✅ UPDATED: Set feature flags from HPO best_params (features_enabled dict)
            # Use HPO-optimized feature flags, but always enable adaptive learning for training
            features_enabled = best_params.get('features_enabled', {})
            if features_enabled:
                # Set feature flags from HPO results
                os.environ['ENABLE_EXTERNAL_FEATURES'] = features_enabled.get('ENABLE_EXTERNAL_FEATURES', '1')
                os.environ['ENABLE_FINGPT_FEATURES'] = features_enabled.get('ENABLE_FINGPT_FEATURES', '1')
                os.environ['ENABLE_YOLO_FEATURES'] = features_enabled.get('ENABLE_YOLO_FEATURES', '1')
                os.environ['ML_USE_DIRECTIONAL_LOSS'] = features_enabled.get('ML_USE_DIRECTIONAL_LOSS', '1')
                os.environ['ENABLE_SEED_BAGGING'] = features_enabled.get('ENABLE_SEED_BAGGING', '1')
                os.environ['ENABLE_TALIB_PATTERNS'] = features_enabled.get('ENABLE_TALIB_PATTERNS', '1')
                os.environ['ML_USE_SMART_ENSEMBLE'] = features_enabled.get('ML_USE_SMART_ENSEMBLE', '1')
                os.environ['ML_USE_STACKED_SHORT'] = features_enabled.get('ML_USE_STACKED_SHORT', '1')
                os.environ['ENABLE_META_STACKING'] = features_enabled.get('ENABLE_META_STACKING', '1')
                os.environ['ML_USE_REGIME_DETECTION'] = features_enabled.get('ML_USE_REGIME_DETECTION', '1')
                os.environ['ENABLE_FINGPT'] = features_enabled.get('ENABLE_FINGPT', '1')
                logger.info(f"✅ {symbol} {horizon}d: Feature flags set from HPO results")
                # Smart-ensemble params from feature_params (for training)
                try:
                    fp = best_params.get('feature_params', {})
                    if isinstance(fp, dict) and fp:
                        if 'smart_consensus_weight' in fp:
                            os.environ['ML_SMART_CONSENSUS_WEIGHT'] = str(fp['smart_consensus_weight'])
                        if 'smart_performance_weight' in fp:
                            os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = str(fp['smart_performance_weight'])
                        if 'smart_sigma' in fp:
                            os.environ['ML_SMART_SIGMA'] = str(fp['smart_sigma'])
                        if 'smart_weight_xgb' in fp:
                            os.environ['ML_SMART_WEIGHT_XGB'] = str(fp['smart_weight_xgb'])
                        if 'smart_weight_lgbm' in fp or 'smart_weight_lgb' in fp:
                            os.environ['ML_SMART_WEIGHT_LGB'] = str(fp.get('smart_weight_lgbm', fp.get('smart_weight_lgb')))
                        if 'smart_weight_cat' in fp:
                            os.environ['ML_SMART_WEIGHT_CAT'] = str(fp['smart_weight_cat'])
                except Exception:
                    pass
            else:
                # Fallback: Enable all features if features_enabled not found (backward compatibility)
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
                logger.warning(f"⚠️ {symbol} {horizon}d: features_enabled not found in best_params, using all features (fallback)")
            
            # ✅ HİBRİT YAKLAŞIM: Training'de adaptive learning KAPALI (HPO ile tutarlılık)
            # Plan'a göre: HPO ve Training aynı veri miktarını kullanmalı
            # Cycle zaten incremental learning etkisi yaratıyor (yeni verilerle yeniden HPO)
            os.environ['ML_USE_ADAPTIVE_LEARNING'] = '0'  # Plan'a göre KAPALI (HPO ile tutarlılık)
            os.environ['ML_SKIP_ADAPTIVE_PHASE2'] = '1'    # Phase 2 skip (HPO ile aynı)
            
            # Set feature internal parameters from HPO (if available in best_params)
            # These are already set by set_hpo_params_as_env, but we ensure they're set here too
            if 'feature_params' in best_params:
                feature_params = best_params['feature_params']
                if 'n_seeds' in feature_params:
                    os.environ['N_SEEDS'] = str(feature_params['n_seeds'])
                if 'meta_stacking_alpha' in feature_params:
                    os.environ['ML_META_STACKING_ALPHA'] = str(feature_params['meta_stacking_alpha'])
                if 'yolo_min_conf' in feature_params:
                    os.environ['YOLO_MIN_CONF'] = str(feature_params['yolo_min_conf'])
                if 'smart_consensus_weight' in feature_params:
                    os.environ['ML_SMART_CONSENSUS_WEIGHT'] = str(feature_params['smart_consensus_weight'])
                if 'smart_performance_weight' in feature_params:
                    os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = str(feature_params['smart_performance_weight'])
                if 'smart_sigma' in feature_params:
                    os.environ['ML_SMART_SIGMA'] = str(feature_params['smart_sigma'])
                    # ✅ NEW: Per-model prior weights for smart ensemble
                    if 'smart_weight_xgb' in feature_params:
                        os.environ['ML_SMART_WEIGHT_XGB'] = str(feature_params['smart_weight_xgb'])
                    if 'smart_weight_lgbm' in feature_params:
                        os.environ['ML_SMART_WEIGHT_LGB'] = str(feature_params['smart_weight_lgbm'])
                    if 'smart_weight_cat' in feature_params:
                        os.environ['ML_SMART_WEIGHT_CAT'] = str(feature_params['smart_weight_cat'])
                if 'fingpt_confidence_threshold' in feature_params:
                    os.environ['FINGPT_CONFIDENCE_THRESHOLD'] = str(feature_params['fingpt_confidence_threshold'])
                if 'ml_loss_mse_weight' in feature_params:
                    os.environ['ML_LOSS_MSE_WEIGHT'] = str(feature_params['ml_loss_mse_weight'])
                if 'ml_loss_threshold' in feature_params:
                    os.environ['ML_LOSS_THRESHOLD'] = str(feature_params['ml_loss_threshold'])
                if 'ml_dir_penalty' in feature_params:
                    os.environ['ML_DIR_PENALTY'] = str(feature_params['ml_dir_penalty'])
            
            os.environ['STRICT_HORIZON_FEATURES'] = '1'   # Horizon-specific features
            logger.info(f"✅ {symbol} {horizon}d: Training with HPO-optimized feature flags (adaptive learning OFF - HPO ile tutarlılık)")
            
            # Clear ConfigManager cache
            try:
                from bist_pattern.core.config_manager import ConfigManager
                ConfigManager.clear_cache()
            except Exception:
                pass
            
            # ✅ CRITICAL FIX: Clear singleton using thread-safe function
            from enhanced_ml_system import clear_enhanced_ml_system
            clear_enhanced_ml_system()
            
            with app.app_context():
                # ✅ CRITICAL FIX: Use fetch_prices (cache bypass) instead of get_stock_data
                # This ensures training uses the same data source as HPO (direct DB, no cache)
                # HPO uses fetch_prices which bypasses cache, so training should too for consistency
                from sqlalchemy import create_engine
                from scripts.optuna_hpo_with_feature_flags import fetch_prices
                
                db_url = os.getenv('DATABASE_URL', 'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:6432/bist_pattern_db').strip()
                engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)
                df = fetch_prices(engine, symbol, limit=1200)
                
                if df is None or len(df) < 50:
                    logger.warning(f"❌ {symbol}: Insufficient data ({len(df) if df is not None else 0} days)")
                    # ✅ FIX: Return special value to indicate insufficient data (for skip logic)
                    return None
                
                ml = get_enhanced_ml_system()
                
                # Train with adaptive learning DISABLED (HPO ile tutarlılık - plan'a göre)
                result = ml.train_enhanced_models(symbol, df)
                
                if result:
                    # Save model
                    try:
                        ml.save_enhanced_models(symbol)
                        # ✅ CRITICAL FIX: Verify model was saved successfully
                        if not ml.has_trained_models(symbol):
                            logger.error(f"❌ Model kaydetme başarısız: {symbol} {horizon}d - Model dosyası bulunamadı")
                            return None
                        logger.info(f"✅ Model kaydedildi ve doğrulandı: {symbol} {horizon}d")
                    except Exception as save_err:
                        logger.error(f"❌ Model kaydetme hatası {symbol} {horizon}d: {save_err}")
                        import traceback
                        logger.error(traceback.format_exc())
                        return None
                    
                    logger.info(f"✅ Training completed for {symbol} {horizon}d (adaptive learning OFF - HPO ile tutarlılık)")
                    
                    # ✅ HİBRİT YAKLAŞIM: Evaluate both WFV (adaptive OFF) and Online (adaptive OFF) DirHit
                    # Best params must be passed to ensure evaluation uses same params as HPO
                    # best_params already contains features_enabled from process_task
                    # ✅ FIX: Pass hpo_result to _evaluate_training_dirhits for best_trial_number
                    ev = self._evaluate_training_dirhits(symbol, horizon, df, best_params=best_params, hpo_result=hpo_result)
                    
                    # Return both WFV (adaptive OFF) and online (adaptive OFF) DirHit for comparison
                    wfv_dirhit = ev.get('wfv')
                    adaptive_dirhit = ev.get('online')
                    
                    if wfv_dirhit is not None:
                        logger.info(f"📊 {symbol} {horizon}d: WFV DirHit (adaptive OFF) = {wfv_dirhit:.2f}%")
                    else:
                        logger.warning(f"⚠️ {symbol} {horizon}d: WFV DirHit could not be evaluated")
                    
                    if adaptive_dirhit is not None:
                        logger.info(f"📊 {symbol} {horizon}d: Online DirHit (adaptive OFF) = {adaptive_dirhit:.2f}%")
                    else:
                        logger.warning(f"⚠️ {symbol} {horizon}d: Adaptive DirHit could not be evaluated")
                    
                    # Compare with HPO DirHit
                    if wfv_dirhit is not None:
                        # Get HPO DirHit from task state
                        task_key = f"{symbol}_{horizon}d"
                        task = self.state.get(task_key)
                        if task and task.hpo_dirhit is not None:
                            hpo_dirhit = task.hpo_dirhit
                            diff = wfv_dirhit - hpo_dirhit
                            logger.info(f"🔍 {symbol} {horizon}d: WFV DirHit ({wfv_dirhit:.2f}%) vs HPO DirHit ({hpo_dirhit:.2f}%) = {diff:+.2f}%")
                            if abs(diff) < 1.0:
                                logger.info(f"✅ {symbol} {horizon}d: WFV DirHit matches HPO DirHit (difference < 1%)")
                            else:
                                logger.warning(f"⚠️ {symbol} {horizon}d: WFV DirHit differs from HPO DirHit by {abs(diff):.2f}%")
                    
                    return {
                        'wfv_dirhit': wfv_dirhit,
                        'adaptive_dirhit': adaptive_dirhit
                    }
                else:
                    logger.warning(f"⚠️ Training failed for {symbol} {horizon}d")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Training error for {symbol} {horizon}d: {e}")
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
            # If cycle was explicitly set (e.g., from process_task_standalone), preserve it
            # Otherwise, load from state file
            preserved_cycle = self.cycle if self.cycle > 0 else None
            
            # ✅ CRITICAL FIX: Helper function to load state while preserving cycle
            def load_state_preserve_cycle():
                """Load state but preserve the cycle if it was explicitly set"""
                self.load_state()
                if preserved_cycle is not None and preserved_cycle > 0:
                    self.cycle = preserved_cycle
            
            # Load current state
            load_state_preserve_cycle()
            
            # ✅ RACE CONDITION FIX: Double-check if task is already in progress
            # This prevents multiple processes from processing the same task
            task = self.state.get(key)
            if task and task.status in ('hpo_in_progress', 'training_in_progress'):
                logger.info(f"⏭️ Skipping {symbol} {horizon}d (already in progress by another process)")
                return False
            
            # ✅ CRITICAL FIX: Check if HPO is already completed but state wasn't updated properly
            # This handles cases where HPO finished but JSON file wasn't found or state wasn't updated
            # After restart, we should check study file and JSON file to resume training
            if task and task.cycle == self.cycle:
                # Check if HPO completed but best_params_file is missing
                if task.status in ('failed', 'completed') and (
                    task.best_params_file is None or not Path(task.best_params_file).exists()
                ):
                    # Try to find completed HPO from study file
                    study_dirs = [
                        Path('/opt/bist-pattern/results/optuna_studies'),
                        Path('/opt/bist-pattern/hpo_studies'),
                    ]
                    study_file = None
                    for study_dir in study_dirs:
                        if not study_dir.exists():
                            continue
                        # Check current cycle format first
                        cycle_file = study_dir / f"hpo_with_features_{symbol}_h{horizon}_c{self.cycle}.db"
                        if cycle_file.exists():
                            study_file = cycle_file
                            break
                        # Check legacy format (only for cycle 1)
                        if self.cycle == 1:
                            legacy_file = study_dir / f"hpo_with_features_{symbol}_h{horizon}.db"
                            if legacy_file.exists():
                                study_file = legacy_file
                                break
                    
                    if study_file and study_file.exists():
                        # Check if HPO is completed (1500+ trials)
                        try:
                            import sqlite3
                            conn = sqlite3.connect(str(study_file), timeout=30.0)
                            cursor = conn.cursor()
                            cursor.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'")
                            complete_trials = cursor.fetchone()[0]
                            conn.close()
                            
                            if complete_trials >= 1490:  # HPO completed
                                logger.info(f"✅ Found completed HPO for {symbol} {horizon}d ({complete_trials} trials), resuming training...")
                                # Try to find JSON file
                                json_files = sorted(
                                    Path('/opt/bist-pattern/results').glob(f'optuna_pilot_features_on_h{horizon}_c{self.cycle}_*.json'),
                                    key=lambda p: p.stat().st_mtime,
                                    reverse=True
                                )
                                # Also check legacy format (no cycle in filename)
                                if not json_files:
                                    json_files = sorted(
                                        Path('/opt/bist-pattern/results').glob(f'optuna_pilot_features_on_h{horizon}_*.json'),
                                        key=lambda p: p.stat().st_mtime,
                                        reverse=True
                                    )
                                
                                # Find JSON file for this symbol
                                hpo_result = None
                                for json_file in json_files[:20]:  # Check recent 20 files
                                    try:
                                        with open(json_file, 'r') as f:
                                            hpo_data = json.load(f)
                                        if symbol in hpo_data.get('symbols', []) and hpo_data.get('horizon') == horizon:
                                            # Found valid JSON file
                                            best_dirhit = hpo_data.get('best_dirhit')
                                            best_value = hpo_data.get('best_value', 0)
                                            hpo_result = {
                                                'best_value': best_value,
                                                'best_dirhit': best_dirhit,
                                                'best_params': hpo_data.get('best_params', {}),
                                                'best_trial_number': hpo_data.get('best_trial', {}).get('number'),
                                                'json_file': str(json_file),
                                                'n_trials': hpo_data.get('n_trials', 0),
                                                'features_enabled': hpo_data.get('features_enabled', {}),
                                                'feature_params': hpo_data.get('feature_params', {}),
                                                'feature_flags': hpo_data.get('feature_flags', {}),
                                                'hyperparameters': hpo_data.get('hyperparameters', {})
                                            }
                                            logger.info(f"✅ Found HPO JSON file: {json_file.name}")
                                            break
                                    except Exception:
                                        continue
                                
                                if hpo_result:
                                    # Update state and proceed to training
                                    task.hpo_completed_at = datetime.now().isoformat()
                                    task.hpo_dirhit = hpo_result.get('best_dirhit') or hpo_result.get('best_value', 0)
                                    task.best_params_file = hpo_result['json_file']
                                    task.status = 'training_in_progress'
                                    task.error = None  # Clear previous error
                                    self.state[key] = task
                                    self.save_state()
                                    
                                    # Proceed directly to training
                                    best_params_with_trial = hpo_result['best_params'].copy()
                                    best_params_with_trial['best_trial_number'] = hpo_result.get('best_trial_number')
                                    best_params_with_trial['features_enabled'] = hpo_result.get('features_enabled', {})
                                    best_params_with_trial['feature_params'] = hpo_result.get('feature_params', {})
                                    best_params_with_trial['feature_flags'] = hpo_result.get('feature_flags', {})
                                    best_params_with_trial['hyperparameters'] = hpo_result.get('hyperparameters', {})
                                    
                                    training_result = self.run_training(symbol, horizon, best_params_with_trial)
                                    
                                    if training_result is None:
                                        load_state_preserve_cycle()
                                        task = self.state.get(key)
                                        if task:
                                            task.status = 'failed'
                                            task.error = 'Training failed after HPO recovery'
                                            task.retry_count = task.retry_count + 1
                                            self.state[key] = task
                                            self.save_state()
                                        return False
                                    
                                    # Training completed
                                    load_state_preserve_cycle()
                                    task = self.state.get(key)
                                    if task:
                                        task.status = 'completed'
                                        task.training_completed_at = datetime.now().isoformat()
                                        if isinstance(training_result, dict):
                                            task.adaptive_dirhit = training_result.get('adaptive_dirhit')
                                            task.training_dirhit_online = training_result.get('adaptive_dirhit')
                                            task.training_dirhit = training_result.get('adaptive_dirhit')
                                            task.training_dirhit_wfv = training_result.get('wfv_dirhit')
                                        else:
                                            task.training_dirhit = training_result
                                            task.adaptive_dirhit = training_result
                                        task.cycle = self.cycle
                                        self.state[key] = task
                                        self.save_state()
                                    logger.info(f"✅ Successfully recovered and completed {symbol} {horizon}d")
                                    return True
                                else:
                                    logger.warning(f"⚠️ HPO completed for {symbol} {horizon}d but JSON file not found, will re-run HPO")
                        except Exception as e:
                            logger.warning(f"⚠️ Error checking study file for {symbol} {horizon}d: {e}")
            
            # ✅ RETRY FIX: Analyze failure reason before retrying
            if task and task.status == 'failed' and task.retry_count < 3:
                error_msg = task.error or ''
                error_lower = error_msg.lower()
                
                # Permanent failures - don't retry, mark as skipped
                permanent_failures = [
                    'insufficient data',
                    'no hpo json file found',
                    'symbol not found',
                    'data not available',
                    'delisted',
                    'inactive'
                ]
                
                # Check if this is a permanent failure
                is_permanent = any(permanent in error_lower for permanent in permanent_failures)
                
                if is_permanent:
                    logger.info(f"⏭️ Skipping {symbol} {horizon}d (permanent failure: {error_msg[:50]})")
                    task.status = 'skipped'
                    task.error = f'Permanent failure: {error_msg}'
                    task.cycle = self.cycle
                    self.state[key] = task
                    self.save_state()
                    return False
                
                # Temporary failures - retry
                # These include: timeout, network errors, subprocess errors, general HPO failures
                logger.info(f"🔄 Retrying {symbol} {horizon}d (retry {task.retry_count + 1}/3, error: {error_msg[:100]})")
                task.status = 'pending'
                task.error = None  # Clear previous error for retry
                task.cycle = self.cycle  # Ensure cycle is current
                self.state[key] = task
                self.save_state()
            
            # ✅ CRITICAL FIX: Data quality check BEFORE HPO
            # Check if symbol has sufficient data for this horizon
            try:
                with app.app_context():
                    det = HybridPatternDetector()
                    df = det.get_stock_data(symbol, days=0)
                    
                    # Minimum data requirements per horizon
                    # ✅ USER REQUEST: All horizons require minimum 100 days
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
                        logger.info(f"⏭️ Skipping {symbol} {horizon}d: insufficient data ({actual_days} < {min_days} days required)")
                        task = self.state.get(key, TaskState(symbol=symbol, horizon=horizon, status='pending', cycle=self.cycle))
                        task.status = 'skipped'
                        task.error = f'Insufficient data: {actual_days}/{min_days} days'
                        task.cycle = self.cycle
                        self.state[key] = task
                        self.save_state()
                        return False
            except Exception as e:
                logger.warning(f"⚠️ Data quality check failed for {symbol} {horizon}d: {e}. Proceeding with HPO...")
            
            # Update state: HPO in progress
            task = self.state.get(key, TaskState(symbol=symbol, horizon=horizon, status='pending', cycle=self.cycle))
            task.status = 'hpo_in_progress'
            task.cycle = self.cycle  # ✅ FIX: Ensure cycle is set
            self.state[key] = task
            self.save_state()
            
            # Step 1: Run HPO
            hpo_result = self.run_hpo(symbol, horizon)
            
            if not hpo_result:
                load_state_preserve_cycle()
                task = self.state.get(key)
                if task:
                    # ✅ FIX: Extract error message from hpo_result if it's a dict with 'error' key
                    error_msg = 'HPO failed'
                    if isinstance(hpo_result, dict) and 'error' in hpo_result:
                        error_msg = hpo_result['error']
                    elif isinstance(hpo_result, str):
                        error_msg = hpo_result
                    
                    task.status = 'failed'
                    task.error = error_msg
                    task.retry_count = task.retry_count + 1  # ✅ FIX: Increment retry count
                    self.state[key] = task
                    self.save_state()
                return False
            
            # Update state: HPO completed
            load_state_preserve_cycle()
            task = self.state.get(key)
            if task:
                task.hpo_completed_at = datetime.now().isoformat()
                # ✅ CRITICAL FIX: Use best_dirhit if available, otherwise fallback to best_value
                # Method 2 returns score (DirHit * mask_after_thr / 100), not DirHit
                best_dirhit = hpo_result.get('best_dirhit')
                if best_dirhit is not None:
                    task.hpo_dirhit = best_dirhit
                else:
                    # Fallback: Use best_value (score) but log warning
                    task.hpo_dirhit = hpo_result.get('best_value', 0)
                    logger.warning(f"⚠️ {symbol} {horizon}d: HPO DirHit not available, using score instead: {task.hpo_dirhit:.2f}")
                task.best_params_file = hpo_result['json_file']
                task.status = 'training_in_progress'
                self.state[key] = task
                self.save_state()
            
            # Step 2: Run Training
            # ✅ UPDATED: Include all HPO results in best_params for training
            best_params_with_trial = hpo_result['best_params'].copy()
            best_params_with_trial['best_trial_number'] = hpo_result.get('best_trial_number')
            # Include features_enabled and feature_params for training
            best_params_with_trial['features_enabled'] = hpo_result.get('features_enabled', {})
            best_params_with_trial['feature_params'] = hpo_result.get('feature_params', {})
            best_params_with_trial['feature_flags'] = hpo_result.get('feature_flags', {})
            best_params_with_trial['hyperparameters'] = hpo_result.get('hyperparameters', {})
            training_result = self.run_training(symbol, horizon, best_params_with_trial)
            
            if training_result is None:
                load_state_preserve_cycle()
                task = self.state.get(key)
                if task:
                    # ✅ FIX: Training failed - Check if it's due to insufficient data
                    # If insufficient data, mark as skipped instead of failed
                    error_msg = 'Training failed'
                    if 'Insufficient data' in str(training_result) or training_result is None:
                        # Check if it's really insufficient data
                        try:
                            with app.app_context():
                                det = HybridPatternDetector()
                                df = det.get_stock_data(symbol, days=0)
                                if df is None or len(df) < 50:
                                    task.status = 'skipped'
                                    error_msg = 'Insufficient data - skipped'
                                    logger.info(f"⏭️ Skipping {symbol} {horizon}d (insufficient data: {len(df) if df is not None else 0} days)")
                                else:
                                    task.status = 'failed'
                                    task.retry_count = task.retry_count + 1
                        except Exception:
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
                # ✅ HİBRİT YAKLAŞIM: Save both WFV (adaptive OFF) and online (adaptive OFF) DirHit
                if isinstance(training_result, dict):
                    # New format: both wfv_dirhit and adaptive_dirhit
                    wfv_dirhit = training_result.get('wfv_dirhit')
                    adaptive_dirhit = training_result.get('adaptive_dirhit')
                    task.adaptive_dirhit = adaptive_dirhit
                    # Backward compatibility: also set training_dirhit_online
                    task.training_dirhit_online = adaptive_dirhit
                    task.training_dirhit = adaptive_dirhit  # For backward compatibility
                    # ✅ NEW: Save WFV DirHit for comparison with HPO
                    task.training_dirhit_wfv = wfv_dirhit
                else:
                    # Legacy format (should not happen with new code)
                    task.training_dirhit = training_result
                    task.adaptive_dirhit = training_result
                task.cycle = self.cycle
                self.state[key] = task
                self.save_state()
            
            logger.info(f"✅ Task completed: {symbol} {horizon}d (Cycle {self.cycle})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Task error for {symbol} {horizon}d: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Preserve cycle even in exception handler
            # Use the same preserved_cycle from the try block if available
            # Preserve cycle manually (avoid referencing inner helper in except scope)
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
        """Process all horizons for a symbol sequentially (1d → 3d → 7d → 14d → 30d).
        
        ✅ NEW APPROACH: Symbol-based sequential processing
        - Processes all horizons for one symbol before moving to next
        - Reduces database load (data fetched once per symbol per cycle)
        - Reduces SQLite conflicts (one symbol at a time)
        - More predictable and easier to debug
        
        Returns True if all horizons completed successfully, False otherwise.
        """
        logger.info(f"🔄 Processing symbol {symbol}: all horizons sequentially")
        
        completed = 0
        failed = 0
        skipped = 0
        
        # Process each horizon in order
        for horizon in HORIZON_ORDER:
            key = f"{symbol}_{horizon}d"
            
            # Check if already completed for current cycle
            self.load_state()
            task = self.state.get(key)
            if task and task.status == 'completed' and task.cycle == self.cycle:
                logger.info(f"⏭️ Skipping {symbol} {horizon}d (already completed in cycle {self.cycle})")
                completed += 1
                continue
            
            # Process this horizon
            logger.info(f"📊 Processing {symbol} {horizon}d ({completed + failed + skipped + 1}/{len(HORIZON_ORDER)})")
            success = self.process_task(symbol, horizon)
            
            if success:
                completed += 1
                logger.info(f"✅ {symbol} {horizon}d completed ({completed}/{len(HORIZON_ORDER)} horizons done)")
            else:
                # Check if it was skipped (insufficient data)
                self.load_state()
                task = self.state.get(key)
                if task and task.status == 'skipped':
                    skipped += 1
                    logger.info(f"⏭️ {symbol} {horizon}d skipped (insufficient data)")
                else:
                    failed += 1
                    logger.warning(f"❌ {symbol} {horizon}d failed ({failed} failed, {completed} completed)")
                    
                    # ✅ OPTION: Continue with next horizon even if one fails
                    # This allows partial completion (e.g., 1d fails but 3d succeeds)
                    # Uncomment to stop on first failure:
                    # logger.error(f"❌ Stopping {symbol} processing due to failure at {horizon}d")
                    # return False
        
        # Summary
        total = completed + failed + skipped
        logger.info(f"📊 {symbol} processing complete: {completed} completed, {failed} failed, {skipped} skipped ({total}/{len(HORIZON_ORDER)})")
        
        # Return True if at least some horizons completed (partial success is acceptable)
        return completed > 0
    
    def cleanup_old_cycle_files(self, keep_cycles: int = 1):
        """Clean up SQLite study files from old cycles
        
        Args:
            keep_cycles: Number of recent cycles to keep (default: 1 = keep only current cycle)
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
            log_file = Path('/opt/bist-pattern/logs/continuous_hpo_pipeline.log')
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            if f'Starting cycle {self.cycle}' in line or f'🔄 Starting cycle {self.cycle}' in line:
                                parts = line.split(' - ', 1)
                                if len(parts) >= 1:
                                    timestamp_str = parts[0].strip()
                                    try:
                                        dt_str = timestamp_str.split(',')[0]
                                        cycle_start = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                                        break
                                    except Exception:
                                        pass
                except Exception:
                    pass
            
            # If cycle start not found, use state file mtime
            if cycle_start is None:
                state_file = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
                if state_file.exists():
                    try:
                        cycle_start = datetime.fromtimestamp(state_file.stat().st_mtime)
                    except Exception:
                        cycle_start = datetime.now()
                else:
                    cycle_start = datetime.now()
            
            # ✅ CRITICAL FIX: Clean up by cycle number, not timestamp
            # Delete all study files from cycles < current_cycle
            # Keep only files from current cycle (c{current_cycle}) and legacy files (no cycle number, only for cycle 1)
            deleted_count = 0
            deleted_size = 0
            for study_dir in existing_dirs:
                for db_file in study_dir.glob('*.db'):
                    try:
                        # Extract cycle number from filename
                        name = db_file.name.replace('.db', '')
                        parts = name.split('_')
                        file_cycle = None
                        
                        # Check if cycle number is in filename (format: ..._c{cycle}.db)
                        if len(parts) >= 6 and parts[-1].startswith('c'):
                            try:
                                file_cycle = int(parts[-1][1:])
                            except (ValueError, IndexError):
                                pass
                        
                        # Delete if:
                        # 1. File has cycle number < current_cycle
                        # 2. File has no cycle number AND current_cycle > 1 (legacy files only valid for cycle 1)
                        should_delete = False
                        if file_cycle is not None:
                            if file_cycle < self.cycle:
                                should_delete = True
                        else:
                            # Legacy file (no cycle number) - only keep if current_cycle is 1
                            if self.cycle > 1:
                                should_delete = True
                        
                        if should_delete:
                            file_size = db_file.stat().st_size
                            db_file.unlink()
                            deleted_count += 1
                            deleted_size += file_size
                            logger.debug(f"🧹 Deleted old cycle file: {db_file.name} (cycle {file_cycle if file_cycle else 'legacy'} < {self.cycle})")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not delete old SQLite file {db_file}: {e}")
            
            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} old SQLite study files ({deleted_size / 1024 / 1024:.2f} MB freed)")
        except Exception as e:
            logger.warning(f"⚠️ Error during cleanup of old cycle files: {e}")
    
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
            if task.cycle == current_cycle and task.status not in ('completed', 'skipped'):
                has_incomplete = True
                break
        
        # Only increment cycle if current cycle is complete (or no tasks exist)
        if not has_incomplete and current_cycle > 0:
            # Check if there are any tasks at all
            has_tasks = any(task.cycle == current_cycle for task in self.state.values())
            if has_tasks:
                # Current cycle is complete, increment for new cycle
                self.cycle += 1
                logger.info(f"🔄 Current cycle {current_cycle} complete, starting new cycle {self.cycle}")
            else:
                # No tasks yet, start with cycle 1
                if current_cycle == 0:
                    self.cycle = 1
                    logger.info(f"🔄 Starting first cycle {self.cycle}")
                else:
                    # Keep current cycle
                    logger.info(f"🔄 Resuming cycle {self.cycle}")
        else:
            if has_incomplete:
                logger.info(f"🔄 Resuming cycle {self.cycle} (incomplete tasks found)")
            else:
                # First cycle
                if current_cycle == 0:
                    self.cycle = 1
                    logger.info(f"🔄 Starting first cycle {self.cycle}")
                else:
                    logger.info(f"🔄 Resuming cycle {self.cycle}")
        
        logger.info(f"🔄 Starting cycle {self.cycle}")
        logger.info("   🎯 Processing mode: HORIZON-FIRST (all symbols per horizon)")
        logger.info("   🚀 Incremental delivery: 1d ready for ALL symbols → 3d ready → ...")
        logger.info(f"   🔥 Parallelism: {MAX_WORKERS} symbols in parallel per horizon phase")
        logger.info("   ✅ HİBRİT YAKLAŞIM: Yeni verilerle HPO yapılacak (incremental learning etkisi)")
        logger.info(f"   ✅ Cycle {self.cycle}: Önceki cycle'dan sonra yeni veriler eklendi, split değerleri güncellenecek")
        
        # ✅ CRITICAL FIX: Save cycle number to state file immediately after determining it
        # This ensures cycle number persists across restarts
        self.save_state()
        
        # ✅ FIX: Clean up old cycle files before starting new cycle
        # Keep only current cycle's files, delete older ones
        if self.cycle > 1:  # Don't clean on first cycle
            logger.info("🧹 Cleaning up old cycle files (keeping only current cycle)...")
            self.cleanup_old_cycle_files(keep_cycles=1)
        
        # ✅ CRITICAL FIX: Ensure all pending tasks have current cycle number
        # ✅ CRITICAL FIX: Reset failed tasks to pending for new cycle (recovery)
        # ⚠️ CRITICAL: Don't reload state here - we just saved it, and reloading could overwrite cycle
        # Instead, reload state but preserve the cycle we just determined
        saved_cycle = self.cycle  # Preserve the cycle we just determined
        self.load_state()
        self.cycle = saved_cycle  # Restore cycle (in case load_state() loaded an older value)
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
                logger.info(f"🔄 Reset {reset_failed_count} failed tasks to pending for cycle {self.cycle} (recovery)")
            logger.info(f"🔄 Updated {sum(1 for t in self.state.values() if t.cycle == self.cycle)} tasks to cycle {self.cycle}")
        
        # ✅ NEW: Horizon-First processing
        # Process all symbols for each horizon before moving to next horizon
        total_completed = 0
        total_failed = 0
        
        for horizon in HORIZON_ORDER:
            logger.info("=" * 80)
            logger.info(f"🎯 HORIZON PHASE: {horizon}d - Processing ALL symbols")
            logger.info("=" * 80)
            
            phase_completed = 0
            phase_failed = 0
            phase_skipped = 0
            
            # Get all active symbols
            try:
                symbols_all = self.get_active_symbols()
            except Exception as e:
                logger.error(f"❌ Failed to get active symbols for {horizon}d: {e}")
                continue
            
            # Process symbols in batches (parallel within horizon phase)
            remaining_symbols = symbols_all.copy()
            
            while remaining_symbols:
                # Take batch of symbols
                batch_symbols = remaining_symbols[:MAX_WORKERS]
                remaining_symbols = remaining_symbols[MAX_WORKERS:]
                
                logger.info(f"🌊 Processing {len(batch_symbols)} symbols for {horizon}d in parallel (max_workers={MAX_WORKERS})")
                
                # Process batch in parallel
                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    future_to_symbol = {}
                    for symbol in batch_symbols:
                        key = f"{symbol}_{horizon}d"
                        
                        # Check if already completed
                        self.load_state()
                        task = self.state.get(key)
                        if task and task.status == 'completed' and task.cycle == self.cycle:
                            # ✅ CRITICAL FIX: Check if completed task has valid best_params_file
                            # If best_params_file is missing, recovery logic should run
                            if task.best_params_file and Path(task.best_params_file).exists():
                                logger.info(f"⏭️ Skipping {symbol} {horizon}d (already completed)")
                                phase_completed += 1
                                continue
                            else:
                                # Completed but best_params_file missing - recovery needed
                                logger.warning(f"⚠️ {symbol} {horizon}d marked as completed but best_params_file missing, will attempt recovery")
                                # Continue to process_task() for recovery
                        
                        # ✅ CRITICAL FIX: Use standalone function with cycle parameter
                        # ProcessPoolExecutor pickles the method, but self.cycle might not be correctly passed
                        # Using standalone function ensures cycle is explicitly passed to child process
                        future = executor.submit(process_task_standalone, symbol, horizon, self.cycle)
                        future_to_symbol[future] = symbol
                    
                    # Collect results
                    for future in as_completed(future_to_symbol):
                        symbol = future_to_symbol[future]
                        try:
                            success = future.result()
                            if success:
                                phase_completed += 1
                                logger.info(f"✅ {symbol} {horizon}d completed ({phase_completed}/{len(symbols_all)})")
                            else:
                                # Check if skipped
                                self.load_state()
                                task = self.state.get(f"{symbol}_{horizon}d")
                                if task and task.status == 'skipped':
                                    phase_skipped += 1
                                    logger.info(f"⏭️ {symbol} {horizon}d skipped (insufficient data)")
                                else:
                                    phase_failed += 1
                                    logger.warning(f"❌ {symbol} {horizon}d failed")
                        except Exception as e:
                            logger.error(f"❌ Exception for {symbol} {horizon}d: {e}")
                            phase_failed += 1
            
            # Phase summary
            logger.info("=" * 80)
            logger.info(f"✅ HORIZON PHASE {horizon}d COMPLETE:")
            logger.info(f"   Completed: {phase_completed}/{len(symbols_all)}")
            logger.info(f"   Failed: {phase_failed}")
            logger.info(f"   Skipped: {phase_skipped}")
            logger.info("=" * 80)
            
            total_completed += phase_completed
            total_failed += phase_failed

        logger.info(f"✅ Cycle {self.cycle} complete: {total_completed} tasks completed, {total_failed} tasks failed")
    
    def run_continuous(self):
        """Run pipeline continuously"""
        logger.info("🚀 Starting Continuous HPO and Training Pipeline")
        logger.info(f"   Max workers: {MAX_WORKERS}")
        logger.info(f"   Horizons: {HORIZONS}")
        logger.info(f"   HPO trials: {HPO_TRIALS}")
        
        try:
            while True:
                # Run one cycle
                self.run_cycle()
                
                # Wait before next cycle
                wait_hours = 24  # Wait 24 hours before next cycle
                logger.info(f"⏳ Waiting {wait_hours} hours before next cycle...")
                time.sleep(wait_hours * 3600)
                
        except KeyboardInterrupt:
            logger.info("⛔ Pipeline stopped by user")
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            import traceback
            logger.error(traceback.format_exc())


def process_task_standalone(symbol: str, horizon: int, cycle: int) -> bool:
    """Standalone function for processing a task (runs in separate process)
    
    ✅ CRITICAL: Cycle is explicitly passed to ensure child process uses correct cycle
    """
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
    """Standalone function for processing a symbol (runs in separate process)
    
    ✅ NEW: Processes all horizons for a symbol sequentially
    """
    # Create a new pipeline instance for this process
    pipeline = ContinuousHPOPipeline()
    pipeline.cycle = cycle
    return pipeline.process_symbol(symbol)


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='Continuous HPO and Training Pipeline')
    parser.add_argument('--test', action='store_true', help='Test mode: process only one symbol-horizon pair')
    parser.add_argument('--symbol', type=str, default='ALKA', help='Symbol for test mode')
    parser.add_argument('--horizon', type=int, default=7, help='Horizon for test mode')
    parser.add_argument('--skip-hpo', action='store_true', help='Skip HPO, use existing best params')
    args = parser.parse_args()
    
    pipeline = ContinuousHPOPipeline()
    
    if args.test:
        logger.info(f"🧪 TEST MODE: Processing {args.symbol} {args.horizon}d")
        logger.info("=" * 80)
        
        if args.skip_hpo:
            logger.info("⏭️ Skipping HPO, using existing best params...")
            # Find existing HPO result for this symbol-horizon
            json_files = sorted(
                Path('/opt/bist-pattern/results').glob(f'optuna_pilot_features_on_h{args.horizon}_*.json'),
                key=lambda p: p.stat().st_mtime,
                reverse=True
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
                    logger.warning(f"⚠️ Error reading {json_file}: {e}")
                    continue
            
            if hpo_data and found_json:
                logger.info(f"✅ Found existing HPO result: {found_json}")
                best_params = hpo_data.get('best_params', {}) or {}
                # ✅ Ensure features_enabled is present in best_params for gating parity
                if 'features_enabled' not in best_params and 'features_enabled' in hpo_data:
                    best_params['features_enabled'] = hpo_data['features_enabled']
                # Run training only (pass full hpo_result so ENABLE_* flags are set from features_enabled)
                result = pipeline.run_training(args.symbol, args.horizon, best_params, hpo_result=hpo_data)
                if result is not None:
                    logger.info(f"✅ Test completed: {args.symbol} {args.horizon}d")
                else:
                    logger.error(f"❌ Test failed: {args.symbol} {args.horizon}d")
            else:
                logger.error(f"❌ No HPO result found for {args.symbol} {args.horizon}d")
                logger.info(f"   Searched {len(json_files)} JSON files for horizon {args.horizon}d")
        else:
            # Run full pipeline: HPO + Training
            success = pipeline.process_task(args.symbol, args.horizon)
            if success:
                logger.info(f"✅ Test completed: {args.symbol} {args.horizon}d")
            else:
                logger.error(f"❌ Test failed: {args.symbol} {args.horizon}d")
    else:
        pipeline.run_continuous()


if __name__ == '__main__':
    main()
