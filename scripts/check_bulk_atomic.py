from __future__ import annotations

import json
import os
import time
import hashlib
from typing import Any, Dict


def checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
    path = os.path.join(log_dir, 'ml_bulk_predictions.json')
    if not os.path.exists(path):
        print(json.dumps({'status': 'error', 'error': 'not_found', 'path': path}))
        return 1

    # Read file multiple times quickly to detect partial writes
    anomalies = 0
    reads: list[Dict[str, Any]] = []
    chks = set()
    for _ in range(10):
        try:
            with open(path, 'r') as rf:
                json.load(rf)
            reads.append({'ok': True, 'size': os.path.getsize(path)})
            chks.add(checksum(path))
        except Exception as e:
            anomalies += 1
            reads.append({'ok': False, 'error': str(e)})
        time.sleep(0.05)

    result = {
        'status': 'success' if anomalies == 0 else 'suspicious',
        'path': path,
        'reads': reads,
        'unique_checksums': len(chks),
        'anomalies': anomalies,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if anomalies == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
