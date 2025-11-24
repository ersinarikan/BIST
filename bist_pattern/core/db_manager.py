"""
Unified Database Session Manager
Provides consistent database session management across the application
"""

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from models import db

logger = logging.getLogger(__name__)


class DBManager:
    """
    Unified database session management
    
    Provides automatic commit/rollback for database operations
    """
    
    @staticmethod
    @contextmanager
    def session_scope() -> Iterator[Any]:
        """
        Automatic session commit/rollback context manager
        
        Usage:
            with DBManager.session_scope() as session:
                stock = session.query(Stock).filter_by(symbol='AKBNK').first()
                session.add(new_stock)
                # Automatic commit on success, rollback on exception
        """
        try:
            yield db.session
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Database transaction failed, rolled back: {e}")
            raise
    
    @staticmethod
    def commit() -> None:
        """Commit current session"""
        try:
            db.session.commit()
        except Exception as e:
            logger.error(f"Database commit failed: {e}")
            raise
    
    @staticmethod
    def rollback() -> None:
        """Rollback current session"""
        try:
            db.session.rollback()
        except Exception as e:
            logger.error(f"Database rollback failed: {e}")
            # Rollback failure is critical, but we can't do much
    
    @staticmethod
    def query(model_class: Any) -> Any:
        """
        Unified query interface
        
        Usage:
            stocks = DBManager.query(Stock).filter_by(is_active=True).all()
        """
        return db.session.query(model_class)
    
    @staticmethod
    def add(obj: Any) -> None:
        """Add object to session"""
        db.session.add(obj)
    
    @staticmethod
    def delete(obj: Any) -> None:
        """Delete object from session"""
        db.session.delete(obj)
    
    @staticmethod
    def flush() -> None:
        """Flush pending changes to database"""
        try:
            db.session.flush()
        except Exception as e:
            logger.error(f"Database flush failed: {e}")
            raise


# Convenience functions for backward compatibility
def session_scope():
    """Session scope context manager (convenience wrapper)"""
    return DBManager.session_scope()

def commit():
    """Commit current session (convenience wrapper)"""
    return DBManager.commit()

def rollback():
    """Rollback current session (convenience wrapper)"""
    return DBManager.rollback()

