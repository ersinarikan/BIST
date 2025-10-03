from __future__ import annotations

from typing import Any


def register_all_blueprints(app: Any, csrf: Any) -> None:
    """Best‑effort blueprint registration with CSRF exemptions.

    - Registers optional API and web blueprints if present
    - Exempts internal and auth blueprints from CSRF where appropriate
    """

    def _try_register(module_path: str, attr: str = 'register') -> None:
        try:
            from importlib import import_module
            mod = import_module(module_path)
            getattr(mod, attr)(app)
        except Exception:  # pragma: no cover
            try:
                app.logger.warning(f"{module_path} blueprint register failed")
            except Exception:
                pass

    # Core/internal first
    _try_register('bist_pattern.blueprints.api_internal')
    # Auth and public APIs
    _try_register('bist_pattern.blueprints.auth')
    _try_register('bist_pattern.blueprints.api_public')
    _try_register('bist_pattern.blueprints.api_metrics')
    _try_register('bist_pattern.blueprints.api_recent')
    _try_register('bist_pattern.blueprints.api_watchlist')
    _try_register('bist_pattern.blueprints.api_simulation')
    _try_register('bist_pattern.blueprints.api_health')
    # Web pages
    _try_register('bist_pattern.blueprints.web')
    _try_register('bist_pattern.blueprints.admin_dashboard')
    
    # Additional API modules (from api_modules)
    _try_register('bist_pattern.api_modules.stocks')
    _try_register('bist_pattern.api_modules.automation')
    _try_register('bist_pattern.api_modules.watchlist')
    _try_register('bist_pattern.api_modules.dashboard')
    
    # High-performance batch API (NEW!)
    _try_register('bist_pattern.blueprints.api_batch')
    
    # CSRF exemptions for internal and auth endpoints
    try:
        if 'api_internal' in app.blueprints:
            csrf.exempt(app.blueprints['api_internal'])
    except Exception:
        pass
    try:
        if 'auth' in app.blueprints:
            csrf.exempt(app.blueprints['auth'])
    except Exception:
        pass
