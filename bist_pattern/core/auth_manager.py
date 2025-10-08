"""
Authentication Manager
Centralized authentication utilities and user management
"""

import logging
from typing import Dict, Any
from flask import current_app
from flask_login import current_user

logger = logging.getLogger(__name__)


class AuthManager:
    """
    Centralized authentication manager
    """
    
    @staticmethod
    def is_admin(user=None) -> bool:
        """Check if user is admin with multiple criteria"""
        try:
            if user is None:
                user = current_user
            
            if not user or not hasattr(user, 'is_authenticated') or not user.is_authenticated:
                return False
            
            # Check role field
            role = getattr(user, 'role', None)
            if isinstance(role, str) and role.lower() == 'admin':
                return True
            
            # Check is_admin field
            if getattr(user, 'is_admin', False):
                return True
            
            # Check username
            if getattr(user, 'username', '') == 'systemadmin':
                return True
            
            # Check against config admin email
            try:
                admin_email = (current_app.config.get('ADMIN_EMAIL') or '').lower()
                user_email = getattr(user, 'email', '').lower()
                if admin_email and user_email == admin_email:
                    return True
            except Exception:
                pass
            
            return False
            
        except Exception as e:
            logger.error(f"Admin check error: {e}")
            return False
    
    @staticmethod
    def is_test_user(user=None) -> bool:
        """Check if user is a test user"""
        try:
            if user is None:
                user = current_user
            
            if not user or not hasattr(user, 'is_authenticated') or not user.is_authenticated:
                return False
            
            email = getattr(user, 'email', '').lower()
            username = getattr(user, 'username', '').lower()
            
            # Check for test indicators
            test_indicators = ['test', 'demo', 'sample', 'example']
            
            for indicator in test_indicators:
                if indicator in email or indicator in username:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Test user check error: {e}")
            return False
    
    @staticmethod
    def get_user_info(user=None) -> Dict[str, Any]:
        """Get comprehensive user information"""
        try:
            if user is None:
                user = current_user
            
            if not user or not hasattr(user, 'is_authenticated') or not user.is_authenticated:
                return {'authenticated': False}
            
            user_info = {
                'authenticated': True,
                'id': getattr(user, 'id', None),
                'email': getattr(user, 'email', ''),
                'username': getattr(user, 'username', ''),
                'full_name': getattr(user, 'full_name', '') or user.email.split('@')[0],
                'role': getattr(user, 'role', 'user'),
                'provider': getattr(user, 'provider', 'email'),
                'is_admin': AuthManager.is_admin(user),
                'is_test_user': AuthManager.is_test_user(user),
                'is_premium': getattr(user, 'is_premium', False),
                'email_verified': getattr(user, 'email_verified', False),
                'created_at': getattr(user, 'created_at', None),
                'last_login': getattr(user, 'last_login', None)
            }
            
            # Convert datetime objects to ISO format
            for date_field in ['created_at', 'last_login']:
                if user_info[date_field]:
                    try:
                        user_info[date_field] = user_info[date_field].isoformat()
                    except Exception:
                        user_info[date_field] = str(user_info[date_field])
            
            return user_info
            
        except Exception as e:
            logger.error(f"Get user info error: {e}")
            return {'authenticated': False, 'error': str(e)}
    
    @staticmethod
    def create_test_users_if_missing():
        """No longer creates hardcoded test users - use database users instead"""
        logger.info("Test user creation disabled - using existing database users")
        return True
    
    @staticmethod
    def get_user_stats() -> Dict[str, Any]:
        """Get user statistics"""
        try:
            from models import User
            
            total_users = User.query.count()
            active_users = User.query.filter_by(is_active=True).count()
            admin_users = User.query.filter_by(role='admin').count()
            premium_users = User.query.filter_by(is_premium=True).count()
            verified_users = User.query.filter_by(email_verified=True).count()
            
            # Users by provider
            email_users = User.query.filter_by(provider='email').count()
            google_users = User.query.filter_by(provider='google').count()
            apple_users = User.query.filter_by(provider='apple').count()
            
            return {
                'total_users': total_users,
                'active_users': active_users,
                'admin_users': admin_users,
                'premium_users': premium_users,
                'verified_users': verified_users,
                'providers': {
                    'email': email_users,
                    'google': google_users,
                    'apple': apple_users
                }
            }
            
        except Exception as e:
            logger.error(f"Get user stats error: {e}")
            return {'error': str(e)}


# Global helper functions for backward compatibility
def is_admin(user=None) -> bool:
    """Check if user is admin"""
    return AuthManager.is_admin(user)


def get_user_info(user=None) -> Dict[str, Any]:
    """Get user information"""
    return AuthManager.get_user_info(user)
