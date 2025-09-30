#!/usr/bin/env python3
"""
Configuration Validation Script
Validates all environment variables and configuration consistency
"""

import os
import sys
import json
from typing import Dict, List, Tuple

# Color codes for output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def validate_required_vars() -> Tuple[List[str], List[str]]:
    """Validate required environment variables"""
    print_header("REQUIRED ENVIRONMENT VARIABLES")
    
    required_vars = [
        ('FLASK_SECRET_KEY', 'Flask session security'),
        ('JWT_SECRET_KEY', 'JWT token security'),
        ('INTERNAL_API_TOKEN', 'Internal API security'),
        ('DATABASE_URL', 'Database connection'),
        ('REDIS_URL', 'Redis connection'),
        ('BIST_LOG_PATH', 'Log directory'),
    ]
    
    missing = []
    present = []
    
    for var, description in required_vars:
        value = os.getenv(var)
        if value:
            # Don't print sensitive values
            if 'SECRET' in var or 'PASSWORD' in var or 'TOKEN' in var:
                print_success(f"{var}: {description} (***CONFIGURED***)")
            else:
                print_success(f"{var}: {description} = {value}")
            present.append(var)
        else:
            print_error(f"{var}: {description} - NOT SET!")
            missing.append(var)
    
    return missing, present

def validate_numeric_vars() -> List[str]:
    """Validate numeric environment variables"""
    print_header("NUMERIC CONFIGURATION VALIDATION")
    
    numeric_vars = [
        ('GUNICORN_WORKERS', int, 1, 10),
        ('COLLECTOR_MAX_WORKERS', int, 1, 10),
        ('ML_ASYNC_WORKERS', int, 1, 5),
        ('SOCKETIO_PING_TIMEOUT', int, 10, 300),
        ('SOCKETIO_PING_INTERVAL', int, 5, 120),
        ('API_CACHE_MAX_SIZE', int, 100, 10000),
        ('PATTERN_CACHE_TTL', int, 60, 3600),
        ('ML_MIN_DATA_DAYS', int, 50, 1000),
        ('YF_MAX_RETRIES', int, 1, 10),
        ('AUTOMATION_CYCLE_SLEEP_SECONDS', int, 60, 7200),
    ]
    
    issues = []
    
    for var, var_type, min_val, max_val in numeric_vars:
        value_str = os.getenv(var)
        if value_str:
            try:
                value = var_type(value_str)
                if min_val <= value <= max_val:
                    print_success(f"{var}: {value} (valid range: {min_val}-{max_val})")
                else:
                    print_warning(f"{var}: {value} (outside recommended range: {min_val}-{max_val})")
                    issues.append(f"{var} outside range")
            except ValueError:
                print_error(f"{var}: '{value_str}' is not a valid {var_type.__name__}")
                issues.append(f"{var} invalid type")
        else:
            print_info(f"{var}: Not set (will use default)")
    
    return issues

def validate_paths() -> List[str]:
    """Validate file and directory paths"""
    print_header("PATH VALIDATION")
    
    path_vars = [
        ('BIST_LOG_PATH', 'directory'),
        ('YOLO_MODEL_PATH', 'file'),
        ('ML_MODEL_PATH', 'directory'),
        ('CATBOOST_TRAIN_DIR', 'directory'),
        ('TRANSFORMERS_CACHE', 'directory'),
        ('DB_PASSWORD_FILE', 'file'),
    ]
    
    issues = []
    
    for var, path_type in path_vars:
        path = os.getenv(var)
        if path:
            if path_type == 'directory':
                if os.path.isdir(path):
                    print_success(f"{var}: Directory exists - {path}")
                else:
                    print_warning(f"{var}: Directory does not exist - {path}")
                    issues.append(f"{var} directory missing")
            elif path_type == 'file':
                if os.path.isfile(path):
                    print_success(f"{var}: File exists - {path}")
                else:
                    print_warning(f"{var}: File does not exist - {path}")
                    issues.append(f"{var} file missing")
        else:
            print_info(f"{var}: Not set")
    
    return issues

def validate_security() -> List[str]:
    """Validate security configuration"""
    print_header("SECURITY VALIDATION")
    
    issues = []
    
    # Check secret strength
    secret_vars = ['FLASK_SECRET_KEY', 'JWT_SECRET_KEY', 'INTERNAL_API_TOKEN']
    for var in secret_vars:
        value = os.getenv(var)
        if value:
            if len(value) < 32:
                print_error(f"{var}: Too short (minimum 32 characters)")
                issues.append(f"{var} too short")
            elif value in ['__GENERATE_STRONG_SECRET_KEY__', '__GENERATE_STRONG_JWT_KEY__', '__GENERATE_STRONG_API_TOKEN__']:
                print_error(f"{var}: Placeholder value not replaced!")
                issues.append(f"{var} placeholder")
            else:
                print_success(f"{var}: Strong secret configured")
    
    # Check SSL configuration consistency
    ssl_secure = os.getenv('SESSION_COOKIE_SECURE', 'True').lower() == 'true'
    preferred_scheme = os.getenv('PREFERRED_URL_SCHEME', 'https')
    
    if ssl_secure and preferred_scheme == 'http':
        print_warning("SSL cookies enabled but HTTP scheme preferred - potential conflict")
        issues.append("SSL configuration mismatch")
    elif not ssl_secure and preferred_scheme == 'https':
        print_warning("HTTPS scheme but SSL cookies disabled - potential security issue")
        issues.append("SSL configuration inconsistent")
    else:
        print_success(f"SSL configuration consistent: secure={ssl_secure}, scheme={preferred_scheme}")
    
    # Check CORS configuration
    cors_origins = os.getenv('CORS_ORIGINS', '')
    if cors_origins == '*':
        print_warning("CORS allows all origins - potential security risk")
        issues.append("CORS too permissive")
    elif cors_origins:
        origins = [o.strip() for o in cors_origins.split(',')]
        print_success(f"CORS configured for {len(origins)} specific origins")
    else:
        print_info("CORS not configured - will use default")
    
    return issues

def validate_threading() -> List[str]:
    """Validate threading configuration"""
    print_header("THREADING CONFIGURATION VALIDATION")
    
    issues = []
    
    # Calculate total thread usage
    thread_vars = [
        ('GUNICORN_WORKERS', 1),
        ('COLLECTOR_MAX_WORKERS', 2),
        ('ML_ASYNC_WORKERS', 2),
        ('VISUAL_ASYNC_WORKERS', 1),
        ('PATTERN_COORDINATOR_WORKERS', 1),
        ('VISUAL_THREAD_POOL_WORKERS', 1),
    ]
    
    total_threads = 0
    for var, default in thread_vars:
        try:
            value = int(os.getenv(var, str(default)))
            total_threads += value
            print_info(f"{var}: {value}")
        except ValueError:
            print_error(f"{var}: Invalid value")
            issues.append(f"{var} invalid")
    
    max_threads = int(os.getenv('TOTAL_MAX_THREADS', '10'))
    if total_threads > max_threads:
        print_error(f"Total threads ({total_threads}) exceeds limit ({max_threads})")
        issues.append("Thread limit exceeded")
    else:
        print_success(f"Total threads: {total_threads}/{max_threads}")
    
    return issues

def generate_secure_tokens():
    """Generate secure tokens for configuration"""
    print_header("SECURE TOKEN GENERATION")
    
    import secrets
    
    tokens = {
        'FLASK_SECRET_KEY': secrets.token_urlsafe(64),
        'JWT_SECRET_KEY': secrets.token_urlsafe(64),
        'INTERNAL_API_TOKEN': secrets.token_urlsafe(32),
    }
    
    print_info("Generated secure tokens (save these to your systemd override):")
    for var, token in tokens.items():
        print(f'{Colors.BOLD}Environment="{var}={token}"{Colors.END}')
    
    return tokens

def main():
    """Main validation function"""
    print(f"{Colors.BOLD}{Colors.MAGENTA}")
    print("🔍 BIST Pattern Detection - Configuration Validator")
    print("=" * 60)
    print(f"{Colors.END}")
    
    all_issues = []
    
    # Run all validations
    missing_vars, present_vars = validate_required_vars()
    all_issues.extend(missing_vars)
    
    numeric_issues = validate_numeric_vars()
    all_issues.extend(numeric_issues)
    
    path_issues = validate_paths()
    all_issues.extend(path_issues)
    
    security_issues = validate_security()
    all_issues.extend(security_issues)
    
    threading_issues = validate_threading()
    all_issues.extend(threading_issues)
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    if not all_issues:
        print_success("🎉 All validations passed! Configuration is ready for production.")
        return 0
    else:
        print_error(f"❌ Found {len(all_issues)} configuration issues:")
        for issue in all_issues:
            print(f"   • {issue}")
        
        print("\n" + Colors.BOLD + "🔧 NEXT STEPS:" + Colors.END)
        print("1. Fix the issues listed above")
        print("2. Generate secure tokens if needed:")
        print("3. Update systemd configuration")
        print("4. Run validation again")
        
        if any('SECRET' in issue or 'TOKEN' in issue for issue in all_issues):
            print("\n" + Colors.YELLOW + "🔑 Generate secure tokens:" + Colors.END)
            generate_secure_tokens()
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
