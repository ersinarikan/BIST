
import os, time
from cycle_runner import get_cycle_runner
if __name__ == "__main__":
    interval = int(os.getenv("CYCLE_INTERVAL","900"))
    r = get_cycle_runner()
    r.start(interval_seconds=interval)
    # Yaşat
    while True:
        time.sleep(3600)
