#!/usr/bin/env python3
"""
Check if systemd override environment variables conflict with HPO/test script settings
"""
import os
import subprocess
import sys

sys.path.insert(0, '/opt/bist-pattern')
# Direct import to avoid Flask dependency
import importlib.util
spec = importlib.util.spec_from_file_location("config_manager", "/opt/bist-pattern/bist_pattern/core/config_manager.py")
config_manager_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_manager_module)
ConfigManager = config_manager_module.ConfigManager

def get_systemd_env():
    """Get environment variables from systemd service"""
    try:
        result = subprocess.run(
            ['systemctl', 'show', 'bist-pattern.service', '--property=Environment', '--value'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            env_vars = {}
            env_line = result.stdout.strip().strip('"').strip("'")
            for var in env_line.split():
                if '=' in var:
                    key, value = var.split('=', 1)
                    value = value.strip('"').strip("'")
                    env_vars[key] = value
            return env_vars
    except Exception as e:
        print(f"⚠️ Could not read systemd environment: {e}")
    return {}

def check_conflicts():
    """Check for conflicts between systemd and script environment variables"""
    print("=" * 80)
    print("🔍 ENVIRONMENT VARIABLES CONFLICT CHECK")
    print("=" * 80)
    print()
    
    # Get systemd environment
    systemd_env = get_systemd_env()
    print(f"📋 Systemd override variables: {len(systemd_env)} found")
    print()
    
    # HPO/Test script environment variables
    script_vars = {
        'ENABLE_TALIB_PATTERNS': '1',
        'ML_USE_SMART_ENSEMBLE': '1',
        'ML_USE_STACKED_SHORT': '1',
        'ML_USE_REGIME_DETECTION': '1',
        'ML_USE_ADAPTIVE_LEARNING': '0',
        'ENABLE_SEED_BAGGING': '0',
        'ENABLE_EXTERNAL_FEATURES': '0',
        'ENABLE_FINGPT_FEATURES': '0',
        'ENABLE_YOLO_FEATURES': '0',
        'ML_USE_DIRECTIONAL_LOSS': '0',
        'ENABLE_FINGPT': '0',
    }
    
    # Check for conflicts
    conflicts = []
    for key, script_value in script_vars.items():
        if key in systemd_env:
            systemd_value = systemd_env[key]
            if str(systemd_value).lower() != str(script_value).lower():
                conflicts.append({
                    'key': key,
                    'systemd': systemd_value,
                    'script': script_value
                })
    
    if conflicts:
        print("❌ CONFLICTS FOUND:")
        print()
        for conflict in conflicts:
            print(f"  {conflict['key']}:")
            print(f"    Systemd: {conflict['systemd']}")
            print(f"    Script:  {conflict['script']}")
            print()
    else:
        print("✅ No conflicts found (or variables not in systemd override)")
        print()
    
    # Check ConfigManager cache
    print("📊 ConfigManager Cache Status:")
    print()
    
    # Set script variables
    for key, value in script_vars.items():
        os.environ[key] = value
    
    # Clear cache
    ConfigManager.clear_cache()
    
    # Check values
    print("After ConfigManager.clear_cache() and setting os.environ:")
    for key in script_vars.keys():
        cached = ConfigManager.get(key, 'NOT_FOUND')
        direct = os.getenv(key, 'NOT_FOUND')
        print(f"  {key}:")
        print(f"    ConfigManager.get(): {cached}")
        print(f"    os.getenv():        {direct}")
        if str(cached).lower() != str(direct).lower():
            print(f"    ⚠️ MISMATCH!")
        print()
    
    print("=" * 80)
    return len(conflicts) > 0

if __name__ == '__main__':
    has_conflicts = check_conflicts()
    sys.exit(1 if has_conflicts else 0)

