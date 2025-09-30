"""
Core Cache System
In-memory TTL cache for API responses
"""

import time
import threading
from typing import Any, Optional

# Global cache storage
API_CACHE: dict[str, dict] = {}
API_CACHE_LOCK = threading.Lock()


def cache_get(key: str) -> Optional[Any]:
    """Get value from cache with TTL check"""
    try:
        now = time.time()
    except Exception:
        now = 0.0
    
    try:
        with API_CACHE_LOCK:
            entry = API_CACHE.get(key)
            if not entry:
                return None
            if float(entry.get('exp', 0)) < now:
                API_CACHE.pop(key, None)
                return None
            return entry.get('val')
    except Exception:
        return None


def cache_set(key: str, value: Any, ttl_seconds: float = 5.0) -> bool:
    """Set value in cache with TTL"""
    try:
        with API_CACHE_LOCK:
            API_CACHE[key] = {'val': value, 'exp': time.time() + float(ttl_seconds)}
        return True
    except Exception:
        return False


def cache_clear() -> None:
    """Clear all cache entries"""
    try:
        with API_CACHE_LOCK:
            API_CACHE.clear()
    except Exception:
        pass


def cache_stats() -> dict:
    """Get cache statistics"""
    try:
        with API_CACHE_LOCK:
            total_entries = len(API_CACHE)
            expired_entries = 0
            now = time.time()
            
            for entry in API_CACHE.values():
                if float(entry.get('exp', 0)) < now:
                    expired_entries += 1
            
            return {
                'total_entries': total_entries,
                'active_entries': total_entries - expired_entries,
                'expired_entries': expired_entries
            }
    except Exception:
        return {'error': 'Unable to get cache stats'}
