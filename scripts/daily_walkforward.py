"""
Daily Walkforward Report Wrapper

Runs the walkforward comparison with stable parameters and writes a compact
JSON report to logs/walkforward_daily.json. Also writes a text log with
progress lines to logs/walkforward_daily.log.

Safe to call from cron/systemd. Exits 0 on success, 1 on failure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict


def ensure_env() -> None:
    os.environ.setdefault('FLASK_SECRET_KEY', 'daily-report')
    # Database URL with password from file if available
    if not os.getenv('DATABASE_URL'):
        pw = ''
        try:
            with open('/opt/bist-pattern/.secrets/db_password', 'r') as f:
                pw = f.read().strip()
        except Exception:
            pw = ''
        if pw:
            # ✅ FIX: Use PgBouncer port 6432 instead of direct Postgres 5432
            os.environ['DATABASE_URL'] = f'postgresql://bist_user:{pw}@127.0.0.1:6432/bist_pattern_db'
    os.environ.setdefault('BIST_LOG_PATH', '/opt/bist-pattern/logs')
    os.environ.setdefault('ML_MIN_DATA_DAYS', '200')


def run_walkforward(limit: int = 50, horizons: str = '1,3,7,14,30', eval_points: int = 60, lookback_days: int = 365) -> Dict[str, Any]:
    ensure_env()
    logs_dir = os.environ.get('BIST_LOG_PATH', '/opt/bist-pattern/logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, 'walkforward_daily.log')

    # Prefer venv python if present
    py = 'python3'
    try:
        if os.path.exists('/opt/bist-pattern/venv/bin/python3'):
            py = '/opt/bist-pattern/venv/bin/python3'
        elif os.path.exists('venv/bin/python3'):
            py = 'venv/bin/python3'
    except Exception:
        pass

    # Resolve project root and script path
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    script_path = os.path.join(root_dir, 'scripts', 'walkforward_compare.py')

    cmd = [
        py,
        script_path,
        '--limit', str(limit),
        '--horizons', horizons,
        '--eval-points', str(eval_points),
        '--lookback-days', str(lookback_days),
    ]

    # Stream output to log (progress visible) and also keep a buffer to parse JSON
    out_buffer = []
    with open(log_path, 'a', buffering=1) as lf:
        lf.write(f"\n=== {datetime.now().isoformat()} DAILY RUN START ===\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=root_dir,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            out_buffer.append(line)
            if len(out_buffer) > 5000:
                out_buffer = out_buffer[-2500:]
        rc = proc.wait()
        lf.write(f"\n=== {datetime.now().isoformat()} DAILY RUN END (rc={rc}) ===\n")

    # Extract trailing JSON from buffered output
    out = ''.join(out_buffer)
    start = out.rfind('{')
    if start == -1:
        raise RuntimeError('No JSON found in walkforward output')
    try:
        payload = json.loads(out[start:])
    except Exception as e:
        raise RuntimeError(f'JSON parse error: {e}')
    return payload


def main() -> int:
    try:
        payload = run_walkforward()
        logs_dir = os.environ.get('BIST_LOG_PATH', '/opt/bist-pattern/logs')
        out_path = os.path.join(logs_dir, 'walkforward_daily.json')
        with open(out_path, 'w') as wf:
            json.dump(payload, wf, ensure_ascii=False, indent=2)
        print(json.dumps({'status': 'ok', 'output': out_path, 'generated_at': datetime.now().isoformat()}))
        return 0
    except Exception as e:
        print(json.dumps({'status': 'error', 'error': str(e)}))
        return 1


if __name__ == '__main__':
    sys.exit(main())
