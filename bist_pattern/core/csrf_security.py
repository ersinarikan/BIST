"""
CSRF Security Manager
Selective CSRF protection for different endpoint types
"""

import logging
from flask import request
from typing import Set

logger = logging.getLogger(__name__)


class CSRFSecurityManager:
    """
    CSRF güvenlik yöneticisi - selective protection
    
    Strateji:
    - READ-ONLY API'ler: CSRF exempt (GET)
    - PUBLIC API'ler: CSRF exempt (belirli endpoint'ler)
    - ADMIN/INTERNAL API'ler: CSRF exempt (authentication required)
    - USER ACTIONS: CSRF protected (POST/PUT/DELETE)
    """
    
    def __init__(self):
        # READ-ONLY endpoints (GET requests only)
        self.read_only_endpoints: Set[str] = {
            '/api/stocks',
            '/api/stock-prices/',
            '/api/pattern-analysis/',
            '/api/dashboard-stats',
            '/api/data-collection/status',
            '/api/system-info',
            '/api/health',
            '/api/automation/status',
            '/api/automation/pipeline-history',
            '/api/recent-tasks',
        }
        
        # PUBLIC endpoints (no auth needed, but safe operations)
        self.public_endpoints: Set[str] = {
            '/api/stocks/search',
            '/api',  # API info endpoint
        }
        
        # INTERNAL/ADMIN endpoints (authentication required)
        self.internal_endpoints: Set[str] = {
            '/api/internal/',
            '/api/automation/start',
            '/api/automation/stop',
            '/api/automation/run-task',
            '/api/data-collection/manual',
        }
        
        # USER ACTION endpoints (MUST be CSRF protected)
        self.protected_endpoints: Set[str] = {
            '/api/watchlist',  # POST/DELETE operations
        }
        
        logger.info("🔒 CSRF Security Manager initialized")
    
    def should_exempt_from_csrf(self, path: str, method: str) -> bool:
        """
        Belirli endpoint'in CSRF'ten muaf tutulup tutulmayacağını belirle
        
        Returns:
            True: CSRF exempt
            False: CSRF protected
        """
        path = path.rstrip('/')
        
        # SocketIO her zaman exempt
        if path.startswith('/socket.io'):
            return True
        
        # Static files exempt
        if path.startswith('/static'):
            return True
        
        # GET requests for read-only endpoints
        if method == 'GET':
            for endpoint in self.read_only_endpoints:
                if path.startswith(endpoint):
                    return True
        
        # Public endpoints (all methods)
        for endpoint in self.public_endpoints:
            if path == endpoint or path.startswith(endpoint):
                return True
        
        # Internal/Admin endpoints (authentication required)
        for endpoint in self.internal_endpoints:
            if path.startswith(endpoint):
                return True
        
        # Protected endpoints - CSRF gerekli
        for endpoint in self.protected_endpoints:
            if path.startswith(endpoint):
                logger.debug(f"🔒 CSRF protection required: {method} {path}")
                return False
        
        # Default: API endpoint'leri için selective protection
        if path.startswith('/api/'):
            # Specific pattern matching
            if method == 'GET' and any(path.startswith(ep) for ep in self.read_only_endpoints):
                return True
            
            # POST/PUT/DELETE operations require CSRF protection by default
            if method in ['POST', 'PUT', 'DELETE', 'PATCH']:
                logger.debug(f"🔒 CSRF protection required for API mutation: {method} {path}")
                return False
        
        # Non-API endpoints (forms, etc.) should be protected
        return False
    
    def get_security_stats(self) -> dict:
        """Güvenlik istatistikleri"""
        return {
            'read_only_endpoints': len(self.read_only_endpoints),
            'public_endpoints': len(self.public_endpoints),
            'internal_endpoints': len(self.internal_endpoints),
            'protected_endpoints': len(self.protected_endpoints),
            'csrf_protection': 'selective'
        }


# Global singleton
_csrf_security_manager = None


def get_csrf_security_manager() -> CSRFSecurityManager:
    """CSRF Security Manager singleton"""
    global _csrf_security_manager
    if _csrf_security_manager is None:
        _csrf_security_manager = CSRFSecurityManager()
    return _csrf_security_manager


def should_exempt_request_from_csrf() -> bool:
    """
    Mevcut request'in CSRF'ten muaf tutulup tutulmayacağını belirle
    
    Flask before_request handler'da kullanılır
    """
    try:
        manager = get_csrf_security_manager()
        path = request.path or ''
        method = request.method or 'GET'
        
        return manager.should_exempt_from_csrf(path, method)
        
    except Exception as e:
        logger.error(f"CSRF exemption check error: {e}")
        # Güvenlik hatası durumunda koruma aktif tut
        return False
