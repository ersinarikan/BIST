from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_all_blueprints(app: Any, csrf: Any) -> None:
    """Best‑effort blueprint registration with CSRF exemptions.

    - Registers optional API and web blueprints if present
    - Exempts internal and auth blueprints from CSRF where appropriate
    """

    def _try_register(module_path: str, attr: str = 'register') -> None:
        try:
            from importlib import import_module
            mod = import_module(module_path)
            
            # ✅ FIX: Check if blueprint is already registered to avoid duplicate registration
            # Get the blueprint instance from the module
            bp_instance = getattr(mod, 'bp', None)
            if bp_instance is not None:
                # Check if this blueprint is already registered to this app
                bp_name = getattr(bp_instance, 'name', None)
                if bp_name and bp_name in app.blueprints:
                    # Blueprint already registered, skip
                    return
            
            # Try to register
            getattr(mod, attr)(app)
        except Exception as e:  # pragma: no cover
            # ✅ FIX: Only log if it's not an "already registered" error
            error_msg = str(e)
            if "already been registered" in error_msg or "already registered" in error_msg.lower():
                # This is expected if blueprint was registered elsewhere, skip logging
                return
            try:
                import traceback
                app.logger.warning(f"{module_path} blueprint register failed: {e}")
                try:
                    app.logger.debug(traceback.format_exc())
                except Exception as e2:
                    logger.debug(f"Failed to log traceback: {e2}")
            except Exception as e3:
                logger.debug(f"Failed to log blueprint registration error: {e3}")

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
    # REMOVED: stocks (duplicate or unused)
    _try_register('bist_pattern.api_modules.automation')
    # REMOVED: watchlist (duplicate - blueprints.api_watchlist is primary)
    # REMOVED: dashboard (duplicate or unused)
    
    # High-performance batch API (NEW!)
    _try_register('bist_pattern.blueprints.api_batch')
    
    # CSRF exemptions for internal and auth endpoints
    try:
        if 'api_internal' in app.blueprints:
            csrf.exempt(app.blueprints['api_internal'])
    except Exception as e:
        logger.debug(f"Failed to exempt api_internal from CSRF: {e}")
    try:
        if 'auth' in app.blueprints:
            csrf.exempt(app.blueprints['auth'])
    except Exception as e:
        logger.debug(f"Failed to exempt auth from CSRF: {e}")
