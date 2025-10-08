from __future__ import annotations
import os
import time
from contextlib import contextmanager
try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


@contextmanager
def file_lock(target_path: str, timeout_seconds: float = 10.0):
    """Advisory exclusive lock using a sidecar .lock file.

    Best-effort: if fcntl is unavailable, proceeds without locking.
    """
    if fcntl is None:
        yield
        return
    lock_path = f"{target_path}.lock"
    os.makedirs(os.path.dirname(lock_path) or '.', exist_ok=True)
    start_ts = time.time()
    with open(lock_path, 'a') as lf:
        acquired = False
        try:
            while True:
                try:
                    fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    if (time.time() - start_ts) > timeout_seconds:
                        raise TimeoutError(f"Timeout acquiring lock: {lock_path}")
                    time.sleep(0.1)
            yield
        finally:
            if acquired:
                try:
                    fcntl.flock(lf, fcntl.LOCK_UN)
                except Exception:
                    pass
