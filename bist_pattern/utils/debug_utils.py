"""
Debug Utilities - Centralized debug functions
"""
import os
import logging

logger = logging.getLogger(__name__)

DEBUG_VERBOSE = str(os.getenv('DEBUG_VERBOSE', '0')).lower() in ('1', 'true', 'yes')


def ddebug(msg: str, logger_instance: logging.Logger = None) -> None:
    """Centralized debug function to avoid duplication"""
    try:
        if DEBUG_VERBOSE:
            if logger_instance:
                logger_instance.debug(msg)
            else:
                logging.getLogger(__name__).debug(msg)
    except Exception as e:
        logger.debug(f"Failed to log debug message: {e}")
