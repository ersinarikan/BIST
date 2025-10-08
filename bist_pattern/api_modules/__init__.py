"""
API modules package
Modularized API blueprints
"""

# Lazy imports to avoid circular dependency issues
__all__ = []  # Lazy loaded, not directly importable


def __getattr__(name):
    """Lazy import to avoid circular dependencies"""
    if name == 'stocks':
        from . import stocks
        return stocks
    elif name == 'automation':
        from . import automation
        return automation
    elif name == 'watchlist':
        # Watchlist moved to blueprints, kept for compatibility
        try:
            from ..blueprints import api_watchlist
            return api_watchlist
        except Exception:
            return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
