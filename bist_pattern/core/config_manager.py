"""
Unified Configuration Manager
Provides consistent config access across the application
"""

import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Unified configuration access layer
    
    Priority order:
    1. Environment variables (highest priority)
    2. Flask current_app.config (if available)
    3. config.py Config class (fallback)
    4. Default value (lowest priority)
    """
    
    _cache: dict[str, Any] = {}
    _cache_enabled = True
    
    @classmethod
    def get(cls, key: str, default: Any = None, cache: bool = True) -> Any:
        """
        Get configuration value with unified priority order
        
        Args:
            key: Configuration key
            default: Default value if not found
            cache: Whether to use cache
            
        Returns:
            Configuration value or default
        """
        # Check cache first
        if cache and cls._cache_enabled and key in cls._cache:
            return cls._cache[key]
        
        value = None
        
        # 1. Check environment variables (highest priority)
        env_value = os.getenv(key)
        if env_value is not None:
            value = cls._parse_value(env_value)
            if value is not None:
                if cache:
                    cls._cache[key] = value
                return value
        
        # 2. Check Flask current_app.config (if available)
        try:
            from flask import current_app
            try:
                flask_value = current_app.config.get(key)
                if flask_value is not None:
                    value = flask_value
                    if cache:
                        cls._cache[key] = value
                    return value
            except RuntimeError:
                # No app context, skip Flask config
                pass
        except Exception as e:
            logger.debug(f"Failed to get Flask config for {key}: {e}")
        
        # 3. Check config.py Config class (fallback)
        try:
            from config import config
            config_value = getattr(config['default'], key, None)
            if config_value is not None:
                value = config_value
                if cache:
                    cls._cache[key] = value
                return value
        except Exception as e:
            logger.debug(f"Failed to get config.py value for {key}: {e}")
        
        # 4. Return default
        return default
    
    @classmethod
    def _parse_value(cls, value: str) -> Any:
        """Parse string value to appropriate type"""
        if value is None:
            return None
        
        value = value.strip()
        
        # Boolean
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        if value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String (default)
        return value
    
    @classmethod
    def set(cls, key: str, value: Any, cache: bool = True) -> None:
        """Set configuration value (for testing/override)"""
        if cache:
            cls._cache[key] = value
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear configuration cache"""
        cls._cache.clear()
    
    @classmethod
    def enable_cache(cls, enabled: bool = True) -> None:
        """Enable/disable configuration cache"""
        cls._cache_enabled = enabled
        if not enabled:
            cls.clear_cache()


# Convenience function for backward compatibility
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value (convenience wrapper)"""
    return ConfigManager.get(key, default)

