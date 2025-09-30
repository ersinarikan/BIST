"""
Core Decorators
Authentication and route protection decorators
"""

from functools import wraps
from flask import request, jsonify, current_app
from flask_login import current_user


def admin_required(fn):
    """Require admin role for route access"""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            if not current_user.is_authenticated:
                return jsonify({'error': 'Authentication required'}), 401
            
            # Check admin role
            user_role = getattr(current_user, 'role', 'user')
            if user_role != 'admin':
                return jsonify({'error': 'Admin access required'}), 403
                
            return fn(*args, **kwargs)
        except Exception as e:
            current_app.logger.error(f"Admin check error: {e}")
            return jsonify({'error': 'Access check failed'}), 500
    return wrapper


def internal_route(fn):
    """Internal-only route: require INTERNAL_API_TOKEN or internal header"""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            # Check internal token
            internal_token = current_app.config.get('INTERNAL_API_TOKEN')
            if internal_token:
                provided_token = request.headers.get('X-Internal-Token')
                if provided_token != internal_token:
                    return jsonify({'error': 'Invalid internal token'}), 403
            
            # Alternative: check for internal header
            internal_header = request.headers.get('X-Internal-Request')
            if not internal_header and not internal_token:
                return jsonify({'error': 'Internal access required'}), 403
                
            return fn(*args, **kwargs)
        except Exception as e:
            current_app.logger.error(f"Internal route check error: {e}")
            return jsonify({'error': 'Internal access check failed'}), 500
    
    # Exempt from CSRF and rate limiting
    try:
        from bist_pattern.extensions import csrf
        csrf.exempt(wrapper)
    except Exception:
        pass
    
    try:
        from bist_pattern.extensions import limiter
        limiter.exempt(wrapper)
    except Exception:
        pass
    
    return wrapper


def rate_limit_exempt(fn):
    """Exempt route from rate limiting"""
    try:
        from flask_limiter import current_app as limiter_app
        if hasattr(limiter_app, 'limiter'):
            limiter_app.limiter.exempt(fn)
    except Exception:
        pass
    return fn


def is_admin_user(user) -> bool:
    """Check if user has admin privileges"""
    try:
        role = getattr(user, 'role', None) or getattr(user, 'roles', None)
        if isinstance(role, str):
            return role.lower() == 'admin'
        return bool(getattr(user, 'is_admin', False))
    except Exception:
        return False
