#!/usr/bin/env python3
"""
HPO Pipeline İlerleme Göstergesi - Yeni Versiyon

Gösterir:
- Kaçıncı trial'da olduğunu
- O ana kadarki en iyi DirHit oranını
- Güncel trial'daki DirHit'i
- O an eğitim yapılıyorsa onu
- Tamamlanan semboller ve ufuklar için istatistiksel bilgiler
"""
# pyright: reportUnusedVariable=false, reportUnusedImport=false
import sys
import os
import json
import sqlite3
import subprocess
import logging
from pathlib import Path
# datetime not required currently
from collections import defaultdict
from typing import Dict, Optional

sys.path.insert(0, '/opt/bist-pattern')

logger = logging.getLogger(__name__)

# Configuration
STATE_FILE = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
RESULTS_DIR = Path('/opt/bist-pattern/results')
# Canonical study directory (single source of truth)
HPO_STUDIES_DIR = Path('/opt/bist-pattern/hpo_studies')
# ✅ CRITICAL FIX: Read TARGET_TRIALS from environment variable (default: 1500)
try:
    TARGET_TRIALS = int(os.getenv('HPO_TRIALS', '1500'))
except Exception as e:
    logger.debug(f"Failed to get HPO_TRIALS, using 1500: {e}")
    TARGET_TRIALS = 1500  # PRODUCTION default


def load_state() -> Dict:
    """Load pipeline state"""
    if not STATE_FILE.exists():
        return {}

    try:
        with open(STATE_FILE, 'r') as f:
            content = f.read().strip()
            if content.count('{') > 1:
                last_brace = content.rfind('}')
                if last_brace > 0:
                    brace_count = 0
                    start_pos = last_brace
                    for i in range(last_brace, -1, -1):
                        if content[i] == '}':
                            brace_count += 1
                        elif content[i] == '{':
                            brace_count -= 1
                            if brace_count == 0:
                                start_pos = i
                                break
                    content = content[start_pos:last_brace+1]
            return json.loads(content)
    except Exception as e:
        logger.debug(f"Failed to load state file: {e}")
        return {}


def get_active_hpo_processes() -> Dict[str, Dict]:
    """Get active HPO processes and their information"""
    active = {}
    try:
        result = subprocess.run(
            ['ps', 'aux'], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split('\n'):
            if (
                'optuna_hpo_with_feature_flags' in line
                and '--symbols' in line
                and '--horizon' in line
            ):
                # Parse command line
                parts = line.split()
                symbol = None
                horizon = None
                trials = TARGET_TRIALS

                for i, part in enumerate(parts):
                    if part == '--symbols' and i + 1 < len(parts):
                        symbol = parts[i + 1]
                    elif part == '--horizon' and i + 1 < len(parts):
                        try:
                            horizon = int(parts[i + 1])
                        except Exception as e:
                            logger.debug(
                                f"Failed to parse horizon from command "
                                f"line: {e}"
                            )
                    elif part == '--trials' and i + 1 < len(parts):
                        try:
                            trials = int(parts[i + 1])
                        except Exception as e:
                            logger.debug(
                                f"Failed to parse trials from command "
                                f"line: {e}"
                            )

                if symbol and horizon:
                    key = f"{symbol}_{horizon}d"
                    active[key] = {
                        'symbol': symbol,
                        'horizon': horizon,
                        'target_trials': trials,
                        'pid': int(parts[1]) if len(parts) > 1 else None
                    }
    except Exception as e:
        logger.debug(f"Failed to get active HPO processes: {e}")

    return active


def get_trial_info_from_db(
    db_file: Path,
    symbol: Optional[str] = None,
    horizon: Optional[int] = None
) -> Optional[Dict]:
    """Get trial information from Optuna SQLite database

    Args:
        db_file: Path to Optuna study database
        symbol: Stock symbol (optional, extracted from db_file if not
            provided)
        horizon: Prediction horizon (optional, extracted from db_file if not
            provided)
    """
    try:
        if not db_file.exists():
            return None
        # ✅ FIX: Add timeout to handle concurrent access (96 HPO processes
        # writing simultaneously)
        # Timeout allows retry instead of immediate failure when DB is locked
        conn = sqlite3.connect(str(db_file), timeout=30.0)
        cursor = conn.cursor()

        # Get total trials
        cursor.execute("SELECT COUNT(*) FROM trials")
        total_trials = cursor.fetchone()[0]

        # Get running trials
        cursor.execute("SELECT COUNT(*) FROM trials WHERE state='RUNNING'")
        running_trials = cursor.fetchone()[0]

        # Get complete trials
        cursor.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'")
        complete_trials = cursor.fetchone()[0]

        # Get best trial (highest value) - Optuna v3+ uses trial_values table
        # ✅ FIX: Get both trial_id and number to correctly query user_attrs
        cursor.execute("""
            SELECT t.trial_id, t.number, tv.value, t.state
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state='COMPLETE' AND tv.value IS NOT NULL
                AND tv.value_type='FINITE'
            ORDER BY tv.value DESC
            LIMIT 1
        """)
        best_trial_row = cursor.fetchone()
        best_trial_id = None
        best_trial_number = None
        best_value = None
        if best_trial_row:
            best_trial_id, best_trial_number, best_value, _ = best_trial_row

        # Get current trial (last running or last complete)
        # ✅ FIX: Get RUNNING trial first, if none then get last COMPLETE
        # trial
        # ✅ FIX: Get both trial_id and number to correctly query user_attrs
        cursor.execute("""
            SELECT t.trial_id, t.number, COALESCE(tv.value, NULL), t.state
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
                AND tv.value_type='FINITE'
            WHERE t.state = 'RUNNING'
            ORDER BY t.number DESC
            LIMIT 1
        """)
        current_trial_row = cursor.fetchone()

        # If no RUNNING trial, get last COMPLETE trial
        if not current_trial_row:
            cursor.execute("""
                SELECT t.trial_id, t.number, COALESCE(tv.value, NULL), t.state
                FROM trials t
                LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
                    AND tv.value_type='FINITE'
                WHERE t.state = 'COMPLETE'
                ORDER BY t.number DESC
                LIMIT 1
            """)
            current_trial_row = cursor.fetchone()

        current_trial_id = None
        current_trial_number = None
        current_trial_value = None
        current_trial_state = None
        if current_trial_row:
            (
                current_trial_id,
                current_trial_number,
                current_trial_value,
                current_trial_state
            ) = current_trial_row

        # Get DirHit from user_attrs for best trial
        # ✅ CRITICAL FIX: Use symbol_metrics[symbol_key]['avg_dirhit'] for
        # symbol-specific DirHit
        # avg_dirhit in user_attrs is the average across ALL symbols (if
        # multiple symbols in HPO)
        # We need the symbol-specific DirHit, which is in
        # symbol_metrics[symbol_key]['avg_dirhit']
        best_dirhit = None
        if best_trial_id is not None:
            # ✅ PRIORITY 1: Get symbol-specific DirHit from symbol_metrics
            # This is the correct way - always gives the DirHit for THIS
            # specific symbol
            if symbol is not None and horizon is not None:
                try:
                    cursor.execute("""
                        SELECT value_json
                        FROM trial_user_attributes
                            WHERE trial_id = ? AND key = 'symbol_metrics'
                        """, (best_trial_id,))
                    row = cursor.fetchone()
                    if row:
                        try:
                            import json
                            symbol_metrics = json.loads(row[0])
                            symbol_key = f"{symbol}_{horizon}d"
                            if (
                                isinstance(symbol_metrics, dict)
                                and symbol_key in symbol_metrics
                            ):
                                symbol_metric = symbol_metrics[symbol_key]
                                if isinstance(symbol_metric, dict):
                                    best_dirhit = symbol_metric.get(
                                        'avg_dirhit'
                                    )
                                    if best_dirhit is not None:
                                        best_dirhit = float(best_dirhit)
                        except Exception as e:
                            logger.debug(
                                f"Failed to parse best_dirhit from "
                                f"symbol_metrics: {e}"
                            )
                except Exception as e:
                    logger.debug(
                        f"Failed to get best_dirhit from symbol_metrics: {e}"
                    )

            # ✅ PRIORITY 2: Fallback - if symbol/horizon not provided, try
            # single symbol case
            if best_dirhit is None:
                try:
                    cursor.execute("""
                        SELECT value_json
                        FROM trial_user_attributes
                        WHERE trial_id = ? AND key = 'symbol_metrics'
                    """, (best_trial_id,))
                    row = cursor.fetchone()
                    if row:
                        try:
                            import json
                            symbol_metrics = json.loads(row[0])
                            # If single symbol in symbol_metrics, use its
                            # DirHit
                            if (
                                isinstance(symbol_metrics, dict)
                                and len(symbol_metrics) == 1
                            ):
                                symbol_key = list(symbol_metrics.keys())[0]
                                symbol_metric = symbol_metrics[symbol_key]
                                if isinstance(symbol_metric, dict):
                                    best_dirhit = symbol_metric.get(
                                        'avg_dirhit'
                                    )
                                    if best_dirhit is not None:
                                        best_dirhit = float(best_dirhit)
                        except Exception as e:
                            logger.debug(
                                f"Failed to parse best_dirhit from "
                                f"symbol_metrics: {e}"
                            )
                except Exception as e:
                    logger.debug(
                        f"Failed to get best_dirhit from symbol_metrics: {e}"
                    )

            # ✅ PRIORITY 3: Fallback to 'avg_dirhit' (only if symbol_metrics
            # not available)
            # This works for single-symbol HPO, but may be wrong for
            # multi-symbol HPO
            if best_dirhit is None:
                try:
                    cursor.execute("""
                        SELECT value_json
                        FROM trial_user_attributes
                        WHERE trial_id = ? AND key = 'avg_dirhit'
                    """, (best_trial_id,))
                    row = cursor.fetchone()
                    if row:
                        try:
                            import json
                            best_dirhit = float(json.loads(row[0]))
                        except Exception as e:
                            logger.debug(
                                f"Failed to parse best_dirhit from "
                                f"avg_dirhit: {e}"
                            )
                except Exception as e:
                    logger.debug(
                        f"Failed to get best_dirhit from avg_dirhit: {e}"
                    )

            # ✅ PRIORITY 4: Fallback to 'dirhit' (legacy)
            if best_dirhit is None:
                try:
                    cursor.execute("""
                        SELECT value_json
                        FROM trial_user_attributes
                        WHERE trial_id = ? AND key = 'dirhit'
                    """, (best_trial_id,))
                    row = cursor.fetchone()
                    if row:
                        try:
                            import json
                            best_dirhit = float(json.loads(row[0]))
                        except Exception as e:
                            logger.debug(
                                f"Failed to parse best_dirhit from "
                                f"avg_dirhit: {e}"
                            )
                except Exception as e:
                    logger.debug(
                        f"Failed to get best_dirhit from avg_dirhit: {e}"
                    )

        # Get DirHit from user_attrs for current trial
        # ✅ CRITICAL FIX: Use symbol_metrics[symbol_key]['avg_dirhit'] for
        # symbol-specific DirHit
        current_dirhit = None
        if current_trial_id is not None:
            # ✅ PRIORITY 1: Get symbol-specific DirHit from symbol_metrics
            if symbol is not None and horizon is not None:
                try:
                    cursor.execute("""
                        SELECT value_json
                        FROM trial_user_attributes
                            WHERE trial_id = ? AND key = 'symbol_metrics'
                        """, (current_trial_id,))
                    row = cursor.fetchone()
                    if row:
                        try:
                            import json
                            symbol_metrics = json.loads(row[0])
                            symbol_key = f"{symbol}_{horizon}d"
                            if (
                                isinstance(symbol_metrics, dict)
                                and symbol_key in symbol_metrics
                            ):
                                symbol_metric = symbol_metrics[symbol_key]
                                if isinstance(symbol_metric, dict):
                                    current_dirhit = symbol_metric.get(
                                        'avg_dirhit'
                                    )
                                    if current_dirhit is not None:
                                        current_dirhit = float(
                                            current_dirhit
                                        )
                        except Exception as e:
                            logger.debug(
                                f"Failed to parse current_dirhit from "
                                f"symbol_metrics: {e}"
                            )
                except Exception as e:
                    logger.debug(
                        f"Failed to get current_dirhit from symbol_metrics: "
                        f"{e}"
                    )

            # ✅ PRIORITY 2: Fallback - if symbol/horizon not provided, try
            # single symbol case
            if current_dirhit is None:
                try:
                    cursor.execute("""
                        SELECT value_json
                        FROM trial_user_attributes
                        WHERE trial_id = ? AND key = 'symbol_metrics'
                    """, (current_trial_id,))
                    row = cursor.fetchone()
                    if row:
                        try:
                            import json
                            symbol_metrics = json.loads(row[0])
                            # If single symbol in symbol_metrics, use its
                            # DirHit
                            if (
                                isinstance(symbol_metrics, dict)
                                and len(symbol_metrics) == 1
                            ):
                                symbol_key = list(symbol_metrics.keys())[0]
                                symbol_metric = symbol_metrics[symbol_key]
                                if isinstance(symbol_metric, dict):
                                    current_dirhit = symbol_metric.get(
                                        'avg_dirhit'
                                    )
                                    if current_dirhit is not None:
                                        current_dirhit = float(
                                            current_dirhit
                                        )
                        except Exception as e:
                            logger.debug(
                                f"Failed to parse current_dirhit from "
                                f"symbol_metrics: {e}"
                            )
                except Exception as e:
                    logger.debug(
                        f"Failed to get current_dirhit from symbol_metrics: "
                        f"{e}"
                    )

            # ✅ PRIORITY 3: Fallback to 'avg_dirhit' (only if symbol_metrics
            # not available)
            if current_dirhit is None:
                try:
                    cursor.execute("""
                        SELECT value_json
                        FROM trial_user_attributes
                        WHERE trial_id = ? AND key = 'avg_dirhit'
                    """, (current_trial_id,))
                    row = cursor.fetchone()
                    if row:
                        try:
                            import json
                            current_dirhit = float(json.loads(row[0]))
                        except Exception as e:
                            logger.debug(
                                f"Failed to parse current_dirhit from "
                                f"avg_dirhit: {e}"
                            )
                except Exception as e:
                    logger.debug(
                        f"Failed to get current_dirhit from avg_dirhit: {e}"
                    )

            # ✅ PRIORITY 4: Fallback to 'dirhit' (legacy)
            if current_dirhit is None:
                try:
                    cursor.execute("""
                        SELECT value_json
                        FROM trial_user_attributes
                        WHERE trial_id = ? AND key = 'dirhit'
                    """, (current_trial_id,))
                    row = cursor.fetchone()
                    if row:
                        try:
                            import json
                            current_dirhit = float(json.loads(row[0]))
                        except Exception as e:
                            logger.debug(
                                f"Failed to parse current_dirhit from "
                                f"avg_dirhit: {e}"
                            )
                except Exception as e:
                    logger.debug(
                        f"Failed to get current_dirhit from avg_dirhit: {e}"
                    )

        conn.close()

        return {
            'total_trials': total_trials,
            'running_trials': running_trials,
            'complete_trials': complete_trials,
            'best_trial_number': best_trial_number,
            'best_value': best_value,
            'best_dirhit': best_dirhit,
            'current_trial_number': current_trial_number,
            'current_trial_value': current_trial_value,
            'current_trial_state': current_trial_state,
            'current_dirhit': current_dirhit
        }
    except Exception as e:
        import sys
        import traceback
        print(
            f"DEBUG: Error reading DB {db_file}: {e}",
            file=sys.stderr
        )
        print(
            f"DEBUG: Traceback: {traceback.format_exc()}",
            file=sys.stderr
        )
        return None


def find_study_db(
    symbol: str, horizon: int, cycle: Optional[int] = None
) -> Optional[Path]:
    """Find the most recent study database file for symbol-horizon

    Priority:
    1. New format with cycle: hpo_with_features_SYMBOL_hHORIZON_cCYCLE.db
    2. New format (no cycle): hpo_with_features_SYMBOL_hHORIZON.db (legacy)
    3. Old format (with timestamp):
        hpo_with_features_SYMBOL_hHORIZON_TIMESTAMP.db

    Args:
        symbol: Stock symbol
        horizon: Prediction horizon
        cycle: Current cycle number (if None, uses state file cycle)
    """
    study_dirs = [HPO_STUDIES_DIR]

    # Get cycle from state if not provided
    if cycle is None:
        state = load_state()
        cycle = state.get('cycle', 1)

    for study_dir in study_dirs:
        if not study_dir.exists():
            continue

        # ✅ FIX: Priority 1 - New format with cycle number (current
        # format)
        cycle_format_file = (
            study_dir /
            f"hpo_with_features_{symbol}_h{horizon}_c{cycle}.db"
        )
        if cycle_format_file.exists():
            return cycle_format_file

        # ✅ FIX: Priority 2 - Legacy format (no cycle) - only if cycle is
        # 1 (first cycle)
        if cycle == 1:
            legacy_format_file = (
                study_dir /
                f"hpo_with_features_{symbol}_h{horizon}.db"
            )
            if legacy_format_file.exists():
                return legacy_format_file

        # ✅ FIX: Priority 3 - Old format (with timestamp) - fallback for
        # legacy files
        old_format_pattern = f"hpo_with_features_{symbol}_h{horizon}_*.db"
        old_format_files = list(study_dir.glob(old_format_pattern))
        if old_format_files:
            # Filter out cycle format files, keep only timestamp format
            timestamp_files = [
                f for f in old_format_files
                if not f.name.endswith(f'_c{cycle}.db')
            ]
            if timestamp_files:
                # Return most recent old format file
                timestamp_files = sorted(
                    timestamp_files,
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                return timestamp_files[0]

        # ✅ FIX: Legacy patterns (for backward compatibility)
        legacy_patterns = [
            f"*{symbol}*h{horizon}*.db",
            f"hpo_features_on_{symbol}_h{horizon}_*.db"
        ]
        for pattern in legacy_patterns:
            legacy_files = list(study_dir.glob(pattern))
            if legacy_files:
                # Filter out cycle format files
                filtered_files = [
                    f for f in legacy_files
                    if not f.name.endswith(f'_c{cycle}.db')
                ]
                if filtered_files:
                    filtered_files = sorted(
                        filtered_files,
                        key=lambda p: p.stat().st_mtime,
                        reverse=True
                    )
                    return filtered_files[0]

    return None


def get_completed_tasks() -> Dict[str, Dict]:
    """Get completed tasks from state file and also check study files for
    completed HPOs

    ✅ CRITICAL FIX: Only check study files from current cycle to avoid
    false positives from old cycles
    """
    state = load_state()
    completed = {}
    current_cycle = state.get('cycle', 1)

    # ✅ FIX: State dosyasında 'state' key'i var, 'tasks' değil
    # Format: {"cycle": 1, "state": {"SYMBOL_HORIZONd": {...}, ...}}
    tasks = state.get('state', {})

    # tasks bir dictionary (key: "SYMBOL_HORIZONd", value: task dict)
    for key, task in tasks.items():
        if isinstance(task, dict) and task.get('status') == 'completed':
            # ✅ FIX: Only include tasks from current cycle
            task_cycle = task.get('cycle', 0)
            if task_cycle != current_cycle:
                continue

            symbol = task.get('symbol', '')
            horizon = task.get('horizon', 0)
            task_key = f"{symbol}_{horizon}d"

            completed[task_key] = {
                'symbol': symbol,
                'horizon': horizon,
                'hpo_dirhit': task.get('hpo_dirhit'),
                'training_dirhit': task.get('training_dirhit'),
                'adaptive_dirhit': task.get('adaptive_dirhit'),
                'training_dirhit_wfv': task.get('training_dirhit_wfv'),
                'training_dirhit_online': task.get('training_dirhit_online'),
                'training_completed_at': task.get('training_completed_at')
            }

    # ✅ CRITICAL FIX: Also check study files for HPOs that completed but
    # state file wasn't updated
    # BUT: Only check study files from current cycle to avoid false
    # positives from old cycles
    study_dirs = [HPO_STUDIES_DIR]
    for study_dir in study_dirs:
        if not study_dir.exists():
            continue

        # ✅ FIX: Find study files with cycle number (current format)
        cycle_pattern = f'hpo_with_features_*_h*_c{current_cycle}.db'
        db_files = list(study_dir.glob(cycle_pattern))

        # ✅ FIX: Also check legacy format (no cycle) only if cycle is 1
        if current_cycle == 1:
            legacy_pattern = 'hpo_with_features_*_h*.db'
            legacy_files = list(study_dir.glob(legacy_pattern))
            # Filter out cycle format files
            legacy_files = [
                f for f in legacy_files
                if not f.name.endswith(f'_c{current_cycle}.db')
            ]
            db_files.extend(legacy_files)

        for db_file in db_files:
            try:
                # Extract symbol, horizon, and cycle from filename
                name = db_file.name.replace('.db', '')
                parts = name.split('_')
                if (
                    len(parts) >= 5
                    and parts[0] == 'hpo'
                    and parts[1] == 'with'
                    and parts[2] == 'features'
                ):
                    symbol = parts[3]
                    horizon_str = parts[4]  # e.g., "h1"

                    # Check if cycle number is in filename
                    file_cycle = None
                    if len(parts) >= 6 and parts[5].startswith('c'):
                        try:
                            file_cycle = int(parts[5][1:])
                        except (ValueError, IndexError):
                            pass

                    # ✅ CRITICAL FIX: Only process files from current cycle
                    if file_cycle is not None and file_cycle != current_cycle:
                        continue
                    # Legacy files (no cycle) are only valid for cycle 1
                    if file_cycle is None and current_cycle != 1:
                        continue

                    if horizon_str.startswith('h'):
                        try:
                            horizon = int(horizon_str[1:])
                            task_key = f"{symbol}_{horizon}d"

                            # Skip if already in completed from state file
                            if task_key in completed:
                                continue

                            # Check if this study has enough complete trials
                            # (HPO finished)
                            trial_info = get_trial_info_from_db(db_file)
                            # ✅ CRITICAL: Only detect as completed if HPO is
                            # truly completed (TARGET_TRIALS - 10 trials)
                            min_trials_for_recovery = max(
                                1, TARGET_TRIALS - 10
                            )
                            # We need TARGET_TRIALS trials to find best
                            # parameters - partial progress is not enough
                            if (
                                trial_info
                                and trial_info.get('complete_trials', 0)
                                >= min_trials_for_recovery
                            ):  # HPO completed
                                # This HPO completed but state file doesn't
                                # show it as completed
                                # Add it to completed list with info from
                                # study file
                                if task_key not in completed:
                                    completed[task_key] = {
                                        'symbol': symbol,
                                        'horizon': horizon,
                                        'hpo_dirhit': trial_info.get(
                                            'best_dirhit'
                                        ),
                                        'training_dirhit': None,
                                        'adaptive_dirhit': None,
                                        'training_dirhit_wfv': None,
                                        'training_dirhit_online': None,
                                        # Mark as detected from study file
                                        'from_study_file': True
                                    }
                        except (ValueError, IndexError):
                            continue
            except Exception as e:
                logger.debug(
                    f"Failed to process study file {db_file}: {e}"
                )
                continue

    return completed


def get_training_status() -> Dict[str, Dict]:
    """Get training status from state file"""
    state = load_state()
    training = {}

    # ✅ FIX: State dosyasında 'state' key'i var, 'tasks' değil
    tasks = state.get('state', {})
    if not tasks:
        return training

    # tasks bir dictionary (key: "SYMBOL_HORIZONd", value: task dict)
    for key, task in tasks.items():
        if isinstance(task, dict) and task.get('status') == 'training':
            symbol = task.get('symbol', '')
            horizon = task.get('horizon', 0)
            task_key = f"{symbol}_{horizon}d"

            training[task_key] = {
                'symbol': symbol,
                'horizon': horizon
            }

    return training


def calculate_statistics(completed: Dict[str, Dict]) -> Dict:
    """Calculate statistics for completed tasks"""
    stats = {
        'total_completed': len(completed),
        'total_symbols': len(set(v['symbol'] for v in completed.values())),
        'total_horizons': len(set(v['horizon'] for v in completed.values())),
        'avg_hpo_dirhit': None,
        'avg_training_dirhit': None,
        'avg_adaptive_dirhit': None,
        'best_hpo_dirhit': None,
        'best_training_dirhit': None,
        'best_adaptive_dirhit': None,
        'horizon_stats': defaultdict(
            lambda: {
                'count': 0,
                'avg_hpo': None,
                'avg_training': None
            }
        )
    }

    hpo_dirhits = []
    training_dirhits = []
    adaptive_dirhits = []

    for key, task in completed.items():
        horizon = task['horizon']
        stats['horizon_stats'][horizon]['count'] += 1

        if task.get('hpo_dirhit') is not None:
            hpo_dirhits.append(task['hpo_dirhit'])
            if stats['horizon_stats'][horizon]['avg_hpo'] is None:
                stats['horizon_stats'][horizon]['avg_hpo'] = []
            stats['horizon_stats'][horizon]['avg_hpo'].append(
                task['hpo_dirhit']
            )

        if task.get('adaptive_dirhit') is not None:
            adaptive_dirhits.append(task['adaptive_dirhit'])

        training_dirhit = (
            task.get('adaptive_dirhit')
            or task.get('training_dirhit_online')
            or task.get('training_dirhit_wfv')
            or task.get('training_dirhit')
        )
        if training_dirhit is not None:
            training_dirhits.append(training_dirhit)
            if stats['horizon_stats'][horizon]['avg_training'] is None:
                stats['horizon_stats'][horizon]['avg_training'] = []
            stats['horizon_stats'][horizon]['avg_training'].append(
                training_dirhit
            )

    if hpo_dirhits:
        stats['avg_hpo_dirhit'] = sum(hpo_dirhits) / len(hpo_dirhits)
        stats['best_hpo_dirhit'] = max(hpo_dirhits)

    if training_dirhits:
        stats['avg_training_dirhit'] = (
            sum(training_dirhits) / len(training_dirhits)
        )
        stats['best_training_dirhit'] = max(training_dirhits)

    if adaptive_dirhits:
        stats['avg_adaptive_dirhit'] = (
            sum(adaptive_dirhits) / len(adaptive_dirhits)
        )
        stats['best_adaptive_dirhit'] = max(adaptive_dirhits)

    # Calculate averages per horizon
    for horizon, h_stats in stats['horizon_stats'].items():
        if h_stats['avg_hpo']:
            h_stats['avg_hpo'] = (
                sum(h_stats['avg_hpo']) / len(h_stats['avg_hpo'])
            )
        if h_stats['avg_training']:
            h_stats['avg_training'] = (
                sum(h_stats['avg_training'])
                / len(h_stats['avg_training'])
            )

    return stats


def get_all_tasks_from_state() -> Dict[str, Dict]:
    """Get all tasks from state file, grouped by status"""
    state = load_state()
    current_cycle = state.get('cycle', 1)
    tasks = state.get('state', {})

    all_tasks = {
        'pending': {},
        'failed': {},
        'completed': {},
        'training': {},
        'skipped': {}
    }

    for key, task in tasks.items():
        if not isinstance(task, dict):
            continue

        task_cycle = task.get('cycle', 0)
        if task_cycle != current_cycle:
            continue

        status = task.get('status', 'unknown')
        symbol = task.get('symbol', '')
        horizon = task.get('horizon', 0)

        task_info = {
            'symbol': symbol,
            'horizon': horizon,
            'status': status,
            'error': task.get('error'),
            'hpo_completed_at': task.get('hpo_completed_at'),
            'training_completed_at': task.get('training_completed_at'),
            'retry_count': task.get('retry_count', 0)
        }

        if status in all_tasks:
            all_tasks[status][key] = task_info
        else:
            # Unknown status - add to pending
            all_tasks['pending'][key] = task_info

    return all_tasks


def has_trained_models(symbol: str, horizon: int) -> bool:
    """Check if trained model files exist for a symbol/horizon"""
    model_dir = Path(
        os.getenv(
            'ML_MODEL_PATH',
            '/opt/bist-pattern/.cache/enhanced_ml_models'
        )
    )
    candidates = [
        model_dir / f"{symbol}_{horizon}d_xgboost.pkl",
        model_dir / f"{symbol}_{horizon}d_lightgbm.pkl",
        model_dir / f"{symbol}_{horizon}d_catboost.pkl",
    ]
    return any(p.exists() for p in candidates)


def main():
    """Main function"""
    print("=" * 100)
    print("📊 HPO PIPELINE İLERLEME RAPORU")
    print("=" * 100)
    print()

    # Load state
    state = load_state()
    cycle = state.get('cycle', 0)
    print(f"🔄 Cycle: {cycle}")
    print()

    # Get active HPO processes
    active_hpo = get_active_hpo_processes()

    # Get all tasks from state
    all_tasks = get_all_tasks_from_state()

    # Get training status
    training = get_training_status()

    # Get completed tasks
    completed = get_completed_tasks()

    # Calculate statistics
    stats = calculate_statistics(completed)

    # ✅ FIX: Also check study files for running trials (HPO might be
    # running but process not in ps)
    # This handles cases where HPO is running but process name doesn't match
    # or process died
    study_based_active = {}
    state = load_state()
    current_cycle = state.get('cycle', 1)

    # Check all tasks from state file for study files with running trials
    all_tasks = get_all_tasks_from_state()
    for status_group in ['pending', 'failed', 'hpo_in_progress']:
        tasks = all_tasks.get(status_group, {})
        for key, task in tasks.items():
            if key in active_hpo:  # Already in active_hpo, skip
                continue

            symbol = task['symbol']
            horizon = task['horizon']
            db_file = find_study_db(symbol, horizon, cycle=current_cycle)
            if db_file and db_file.exists():
                trial_info = get_trial_info_from_db(
                    db_file, symbol=symbol, horizon=horizon
                )
                if trial_info and trial_info.get('running_trials', 0) > 0:
                    # HPO is running (has running trials in study file)
                    study_based_active[key] = {
                        'symbol': symbol,
                        'horizon': horizon,
                        'target_trials': TARGET_TRIALS,
                        'pid': None,  # Process not found in ps
                        # Mark as detected from study file
                        'from_study_file': True
                    }

    # Merge active_hpo and study_based_active
    all_active_hpo = {**active_hpo, **study_based_active}

    # Display active HPOs
    if all_active_hpo:
        print("🔬 HPO YAPILIYOR (Aktif Process'ler):")
        print("-" * 100)
        for key, info in sorted(all_active_hpo.items()):
            symbol = info['symbol']
            horizon = info['horizon']
            target_trials = info['target_trials']

            # Find study database (for current cycle)
            state = load_state()
            current_cycle = state.get('cycle', 1)
            db_file = find_study_db(symbol, horizon, cycle=current_cycle)
            if db_file:
                trial_info = get_trial_info_from_db(db_file)
                if trial_info:
                    total = trial_info['total_trials']
                    running = trial_info['running_trials']
                    complete = trial_info['complete_trials']
                    best_dirhit = trial_info.get('best_dirhit')
                    current_dirhit = trial_info.get('current_dirhit')
                    current_trial = trial_info.get('current_trial_number')
                    best_trial = trial_info.get('best_trial_number')

                    # Format output
                    status_line = (
                        f"   {symbol}_{horizon}d: Trial {total}/"
                        f"{target_trials}"
                    )
                    if running > 0:
                        status_line += (
                            f" (Running: {running}, Complete: {complete})"
                        )
                    else:
                        status_line += f" (Complete: {complete})"

                    print(status_line)

                    # Show current trial DirHit
                    if current_trial is not None:
                        current_state = trial_info.get(
                            'current_trial_state', 'UNKNOWN'
                        )
                        if current_state == 'RUNNING':
                            # RUNNING trial - show trial number but indicate
                            # it's still running
                            if current_dirhit is not None:
                                print(
                                    f"      📍 Güncel Trial "
                                    f"#{current_trial} "
                                    f"(Running): DirHit = "
                                    f"{current_dirhit:.2f}%"
                                )
                            elif (
                                trial_info.get('current_trial_value')
                                is not None
                            ):
                                print(
                                    f"      📍 Güncel Trial #{current_trial} "
                                    f"(Running): Score = "
                                    f"{trial_info['current_trial_value']:.2f}"
                                )
                            else:
                                # RUNNING trial has no value yet - try to
                                # show last COMPLETE trial's value
                                # Get last COMPLETE trial from DB
                                db_file = find_study_db(symbol, horizon)
                                if db_file and db_file.exists():
                                    try:
                                        # ✅ FIX: Add timeout for concurrent
                                        # access
                                        conn = sqlite3.connect(
                                            str(db_file), timeout=30.0
                                        )
                                        cursor = conn.cursor()
                                        cursor.execute("""
                                            SELECT t.number,
                                                COALESCE(tv.value, NULL)
                                            FROM trials t
                                            LEFT JOIN trial_values tv
                                                ON t.trial_id = tv.trial_id
                                                AND tv.value_type='FINITE'
                                            WHERE t.state = 'COMPLETE'
                                                AND tv.value IS NOT NULL
                                            ORDER BY t.number DESC
                                            LIMIT 1
                                        """)
                                        last_complete = cursor.fetchone()
                                        conn.close()
                                        if (
                                            last_complete
                                            and last_complete[1] is not None
                                        ):
                                            print(
                                                f"      📍 Güncel Trial "
                                                f"#{current_trial} "
                                                f"(Running - "
                                                f"hesaplanıyor...), "
                                                f"Son Tamamlanan: Trial "
                                                f"#{last_complete[0]} "
                                                f"Score = "
                                                f"{last_complete[1]:.2f}"
                                            )
                                        else:
                                            print(
                                                f"      📍 Güncel Trial "
                                                f"#{current_trial} "
                                                f"(Running - hesaplanıyor...)"
                                            )
                                    except Exception as e:
                                        logger.debug(
                                            f"Failed to get last complete "
                                            f"trial info: {e}"
                                        )
                                        print(
                                            f"      📍 Güncel Trial "
                                            f"#{current_trial} "
                                            f"(Running - hesaplanıyor...)"
                                        )
                                else:
                                    print(
                                        f"      📍 Güncel Trial "
                                        f"#{current_trial} "
                                        f"(Running - hesaplanıyor...)"
                                    )
                        else:
                            # COMPLETE trial - show value
                            if current_dirhit is not None:
                                print(
                                    f"      📍 Güncel Trial "
                                    f"#{current_trial}: "
                                    f"DirHit = {current_dirhit:.2f}%"
                                )
                            elif (
                                trial_info.get('current_trial_value')
                                is not None
                            ):
                                print(
                                    f"      📍 Güncel Trial #{current_trial}: "
                                    f"Score = "
                                    f"{trial_info['current_trial_value']:.2f}"
                                )

                    # Show best DirHit
                    if best_trial is not None:
                        if best_dirhit is not None:
                            print(
                                f"      🏆 En İyi Trial #{best_trial}: "
                                f"DirHit = {best_dirhit:.2f}%"
                            )
                        elif trial_info.get('best_value') is not None:
                            print(
                                f"      🏆 En İyi Trial #{best_trial}: "
                                f"Score = {trial_info['best_value']:.2f}"
                            )

                    print()
                else:
                    import sys
                    print(
                        f"   {symbol}_{horizon}d: Veritabanı okunamadı "
                        f"(DB: {db_file})",
                        file=sys.stderr
                    )
                    print(f"   {symbol}_{horizon}d: Veritabanı okunamadı")
                    print()
            else:
                import sys
                print(
                    f"DEBUG: No DB file found for {symbol}_{horizon}d",
                    file=sys.stderr
                )
                print(f"   {symbol}_{horizon}d: Study dosyası bulunamadı")
                print()
    else:
        print("🔬 HPO YAPILIYOR (Aktif Process'ler): Yok")
        print()

    # Display pending tasks (with study file info if available)
    # ✅ FIX: Exclude tasks that are already in all_active_hpo (they're
    # already shown above)
    pending_tasks = all_tasks.get('pending', {})
    # Remove tasks that are already active
    pending_tasks_filtered = {
        k: v for k, v in pending_tasks.items()
        if k not in all_active_hpo
    }

    if pending_tasks_filtered:
        print("⏳ HPO BEKLEYEN (Pending - Process Yok):")
        print("-" * 100)
        state = load_state()
        current_cycle = state.get('cycle', 1)

        for key in sorted(pending_tasks_filtered.keys()):
            task = pending_tasks_filtered[key]
            symbol = task['symbol']
            horizon = task['horizon']

            # Check if study file exists (HPO might have started but process
            # died)
            db_file = find_study_db(symbol, horizon, cycle=current_cycle)
            if db_file and db_file.exists():
                trial_info = get_trial_info_from_db(
                    db_file, symbol=symbol, horizon=horizon
                )
                if trial_info:
                    total = trial_info['total_trials']
                    complete = trial_info['complete_trials']
                    running = trial_info['running_trials']
                    best_dirhit = trial_info.get('best_dirhit')

                    status_line = (
                        f"   {symbol}_{horizon}d: Trial {total}/"
                        f"{TARGET_TRIALS}"
                    )
                    if running > 0:
                        status_line += (
                            f" (Running: {running}, Complete: {complete})"
                        )
                    else:
                        status_line += f" (Complete: {complete})"
                    print(status_line)

                    if best_dirhit is not None:
                        print(
                            f"      🏆 En İyi DirHit: {best_dirhit:.2f}%"
                        )
                    print()
                else:
                    print(
                        f"   {symbol}_{horizon}d: Pending "
                        f"(study file var ama okunamadı)"
                    )
                    print()
            else:
                print(
                    f"   {symbol}_{horizon}d: Pending (henüz başlamadı)"
                )
                print()
    else:
        print("⏳ HPO BEKLEYEN (Pending): Yok")
        print()

    # Display failed tasks
    # ✅ FIX: Exclude tasks that are already in all_active_hpo (they're
    # running, not failed)
    failed_tasks = all_tasks.get('failed', {})
    failed_tasks_filtered = {
        k: v for k, v in failed_tasks.items()
        if k not in all_active_hpo
    }

    if failed_tasks_filtered:
        print("❌ HPO BAŞARISIZ (Failed):")
        print("-" * 100)
        state = load_state()
        current_cycle = state.get('cycle', 1)

        for key in sorted(failed_tasks_filtered.keys()):
            task = failed_tasks_filtered[key]
            symbol = task['symbol']
            horizon = task['horizon']
            error = task.get('error', 'Unknown error')
            retry_count = task.get('retry_count', 0)

            # Check if study file exists (might have partial progress)
            db_file = find_study_db(symbol, horizon, cycle=current_cycle)
            if db_file and db_file.exists():
                trial_info = get_trial_info_from_db(
                    db_file, symbol=symbol, horizon=horizon
                )
                if trial_info:
                    total = trial_info['total_trials']
                    complete = trial_info['complete_trials']
                    best_dirhit = trial_info.get('best_dirhit')

                    print(
                        f"   {symbol}_{horizon}d: Trial {total}/1500 "
                        f"(Complete: {complete})"
                    )
                    if best_dirhit is not None:
                        print(f"      🏆 En İyi DirHit: {best_dirhit:.2f}%")
                    print(f"      ❌ Hata: {error[:80]}")
                    if retry_count > 0:
                        print(f"      🔄 Retry: {retry_count}/10")
                    print()
                else:
                    print(f"   {symbol}_{horizon}d: Failed - {error[:80]}")
                    if retry_count > 0:
                        print(f"      🔄 Retry: {retry_count}/10")
                    print()
            else:
                print(f"   {symbol}_{horizon}d: Failed - {error[:80]}")
                if retry_count > 0:
                    print(f"      🔄 Retry: {retry_count}/3")
                print()
    else:
        print("❌ HPO BAŞARISIZ (Failed): Yok")
        print()

    # Display skipped tasks
    skipped_tasks = all_tasks.get('skipped', {})
    if skipped_tasks:
        print("⏭️  HPO ATLANAN (Skipped):")
        print("-" * 100)
        for key in sorted(skipped_tasks.keys()):
            task = skipped_tasks[key]
            symbol = task['symbol']
            horizon = task['horizon']
            error = task.get('error', 'Unknown reason')
            print(f"   {symbol}_{horizon}d: {error[:80]}")
        print()

    # Display training
    if training:
        print("🎓 EĞİTİM YAPILIYOR:")
        print("-" * 100)
        for key, info in sorted(training.items()):
            print(
                f"   {info['symbol']}_{info['horizon']}d"
            )
        print()
    else:
        print("🎓 EĞİTİM YAPILIYOR: Yok")
        print()

    # Display completed tasks
    if completed:
        print("✅ TAMAMLANAN GÖREVLER:")
        print("-" * 100)
        for key, task in sorted(completed.items()):
            symbol = task['symbol']
            horizon = task['horizon']
            hpo_dirhit = task.get('hpo_dirhit')
            adaptive_dirhit = task.get('adaptive_dirhit')
            training_dirhit = (
                task.get('training_dirhit_online')
                or task.get('training_dirhit_wfv')
                or task.get('training_dirhit')
            )
            from_study_file = task.get('from_study_file', False)
            best_params_file = task.get('best_params_file')
            models_exist = has_trained_models(symbol, horizon)

            line = f"   {symbol}_{horizon}d:"
            if hpo_dirhit is not None:
                line += f" HPO DirHit={hpo_dirhit:.2f}%"
            # ✅ FIX: Use WFV dirhit for comparison with HPO (HPO also uses
            # WFV methodology)
            # Priority: training_dirhit_wfv > training_dirhit_online >
            # adaptive_dirhit > training_dirhit
            training_dirhit_wfv = task.get('training_dirhit_wfv')
            training_dirhit_online = task.get('training_dirhit_online')
            if training_dirhit_wfv is not None:
                line += (
                    f" Training DirHit={training_dirhit_wfv:.2f}%"
                )
            elif training_dirhit_online is not None:
                line += (
                    f" Training DirHit={training_dirhit_online:.2f}%"
                )
            elif adaptive_dirhit is not None:
                line += f" Training DirHit={adaptive_dirhit:.2f}%"
            elif training_dirhit is not None:
                line += f" Training DirHit={training_dirhit:.2f}%"
            else:
                # Check if training was completed but DirHit is None (likely
                # LOW_SUPPORT)
                training_completed = task.get('training_completed_at')
                if training_completed:
                    reason = task.get('error') or 'yetersiz veri'
                    bp_name = (
                        f" best_params={Path(best_params_file).name}"
                        if best_params_file else ""
                    )
                    models_txt = (
                        "models=yes" if models_exist else "models=no"
                    )
                    line += (
                        f" Training DirHit=LOW_SUPPORT "
                        f"({reason}; {models_txt}{bp_name})"
                    )
            if from_study_file:
                line += (
                    " ⚠️ (State dosyasında 'completed' değil, study "
                    "dosyasından tespit edildi)"
                )
            print(line)
        print()
    else:
        print("✅ TAMAMLANAN GÖREVLER: Yok")
        print()

    # Display statistics
    if stats['total_completed'] > 0:
        print("📊 İSTATİSTİKLER:")
        print("-" * 100)
        print(f"   Toplam Tamamlanan: {stats['total_completed']} görev")
        print(f"   Toplam Sembol: {stats['total_symbols']}")
        print(f"   Toplam Ufuk: {stats['total_horizons']}")

        if stats['avg_hpo_dirhit'] is not None:
            print(
                f"   Ortalama HPO DirHit: "
                f"{stats['avg_hpo_dirhit']:.2f}%"
            )
        if stats['best_hpo_dirhit'] is not None:
            print(
                f"   En İyi HPO DirHit: "
                f"{stats['best_hpo_dirhit']:.2f}%"
            )

        if stats['avg_training_dirhit'] is not None:
            print(
                f"   Ortalama Training DirHit: "
                f"{stats['avg_training_dirhit']:.2f}%"
            )
        if stats['best_training_dirhit'] is not None:
            print(
                f"   En İyi Training DirHit: "
                f"{stats['best_training_dirhit']:.2f}%"
            )

        if stats['avg_adaptive_dirhit'] is not None:
            print(
                f"   Ortalama Adaptive DirHit: "
                f"{stats['avg_adaptive_dirhit']:.2f}%"
            )
        if stats['best_adaptive_dirhit'] is not None:
            print(
                f"   En İyi Adaptive DirHit: "
                f"{stats['best_adaptive_dirhit']:.2f}%"
            )

        print()
        print("   Ufuk Bazında İstatistikler:")
        for horizon in sorted(stats['horizon_stats'].keys()):
            h_stats = stats['horizon_stats'][horizon]
            print(f"      {horizon}d: {h_stats['count']} görev", end="")
            if h_stats['avg_hpo'] is not None:
                print(
                    f", Ort. HPO DirHit: {h_stats['avg_hpo']:.2f}%",
                    end=""
                )
            if h_stats['avg_training'] is not None:
                print(
                    f", Ort. Training DirHit: "
                    f"{h_stats['avg_training']:.2f}%",
                    end=""
                )
            print()
        print()
    else:
        print("📊 İSTATİSTİKLER: Henüz tamamlanan görev yok")
        print()


if __name__ == '__main__':
    main()
