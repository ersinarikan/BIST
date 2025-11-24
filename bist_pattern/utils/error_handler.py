"""
Centralized Error Handler
Provides consistent error handling and logging across the application
"""

import logging
import traceback
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Centralized error handling
    
    Provides unified error handling with context-aware logging
    """
    
    @staticmethod
    def handle(
        exception: Exception,
        context: str,
        level: str = 'debug',
        silent: bool = False,
        reraise: bool = False,
        include_traceback: bool = False
    ) -> None:
        """
        Handle exception with unified logging
        
        Args:
            exception: Exception to handle
            context: Context where exception occurred (e.g., 'pattern_detection', 'ml_training')
            level: Log level ('debug', 'info', 'warning', 'error', 'critical')
            silent: If True, don't log (but still handle)
            reraise: If True, re-raise exception after logging
            include_traceback: If True, include full traceback in log
        """
        if silent:
            if reraise:
                raise
            return
        
        # Get logger function
        logger_func = getattr(logger, level.lower(), logger.debug)
        
        # Format message
        exc_type = type(exception).__name__
        exc_msg = str(exception)
        message = f"[{context}] {exc_type}: {exc_msg}"
        
        # Include traceback if requested
        if include_traceback:
            tb_str = traceback.format_exc()
            message = f"{message}\n{tb_str}"
        
        # Log
        logger_func(message)
        
        # Re-raise if requested
        if reraise:
            raise
    
    @staticmethod
    def handle_database_error(
        exception: Exception,
        context: str,
        operation: str = 'unknown'
    ) -> None:
        """
        Handle database-specific errors
        
        Args:
            exception: Exception to handle
            context: Context where exception occurred
            operation: Database operation (e.g., 'commit', 'query', 'add')
        """
        exc_type = type(exception).__name__
        
        # Determine log level based on exception type
        if exc_type in ('IntegrityError', 'OperationalError'):
            level = 'error'
        elif exc_type in ('DataError', 'ProgrammingError'):
            level = 'warning'
        else:
            level = 'debug'
        
        logger_func = getattr(logger, level, logger.debug)
        message = f"[{context}] Database {operation} failed: {exc_type}: {exception}"
        logger_func(message)
    
    @staticmethod
    def handle_config_error(
        exception: Exception,
        context: str,
        key: str
    ) -> None:
        """
        Handle configuration-specific errors
        
        Args:
            exception: Exception to handle
            context: Context where exception occurred
            key: Configuration key that caused error
        """
        logger.warning(f"[{context}] Config access failed for key '{key}': {type(exception).__name__}: {exception}")
    
    @staticmethod
    def handle_ml_error(
        exception: Exception,
        context: str,
        symbol: Optional[str] = None,
        horizon: Optional[str] = None
    ) -> None:
        """
        Handle ML-specific errors
        
        Args:
            exception: Exception to handle
            context: Context where exception occurred
            symbol: Stock symbol (if applicable)
            horizon: Prediction horizon (if applicable)
        """
        symbol_str = f" {symbol}" if symbol else ""
        horizon_str = f" {horizon}d" if horizon else ""
        logger.error(
            f"[{context}] ML operation failed{symbol_str}{horizon_str}: "
            f"{type(exception).__name__}: {exception}"
        )


# Convenience function for backward compatibility
def handle_error(
    exception: Exception,
    context: str,
    level: str = 'debug',
    silent: bool = False
) -> None:
    """Handle error (convenience wrapper)"""
    ErrorHandler.handle(exception, context, level, silent)

