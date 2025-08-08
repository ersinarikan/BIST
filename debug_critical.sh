#!/bin/bash

# CRITICAL SYSTEM DEBUG SCRIPT
# Flask service crashed - need immediate diagnosis

echo "=== 🚨 CRITICAL SYSTEM DEBUG ===" 
echo ""
echo "Flask service crashed with 502 error - Diagnosing..."
echo ""

cd /opt/bist-pattern
source venv/bin/activate

# === 🔍 SERVICE STATUS DEBUG ===
echo "=== 🔍 SERVICE STATUS ==="
sudo systemctl status bist-pattern

# === 📋 LOG ANALYSIS ===
echo -e "\n=== 📋 RECENT LOGS ==="
echo "Last 30 lines of service logs:"
sudo journalctl -u bist-pattern --no-pager -n 30

# === 🧪 MANUAL FLASK TEST ===
echo -e "\n=== 🧪 MANUAL FLASK TEST ==="
python3 -c "
try:
    print('🔧 Testing Flask app import...')
    from app import app
    print('✅ Flask app import successful')
    
    print('🔧 Testing app context...')
    with app.app_context():
        print('✅ App context working')
    
    print('🔧 Testing scheduler import...')
    from scheduler import get_automated_pipeline
    print('✅ Scheduler import successful')
    
    print('🔧 Testing template directory...')
    import os
    template_dir = os.path.join(os.getcwd(), 'templates')
    if os.path.exists(template_dir):
        print(f'✅ Templates directory exists: {template_dir}')
        templates = os.listdir(template_dir)
        print(f'📁 Templates found: {templates}')
    else:
        print(f'❌ Templates directory missing: {template_dir}')
    
except Exception as e:
    print(f'❌ Flask test error: {e}')
    import traceback
    traceback.print_exc()
"

# === 🔧 CHECK IMPORT ISSUES ===
echo -e "\n=== 🔧 IMPORT DIAGNOSIS ==="
python3 -c "
import sys
sys.path.append('/opt/bist-pattern')

# Test individual imports
modules_to_test = [
    'models',
    'pattern_detector', 
    'ml_prediction_system',
    'scheduler',
    'data_collector',
    'simple_enhanced_ml'
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f'✅ {module}: OK')
    except Exception as e:
        print(f'❌ {module}: {str(e)[:100]}')
"

# === 🔧 TEMPLATE DIRECTORY FIX ===
echo -e "\n=== 🔧 TEMPLATE DIRECTORY CHECK ==="
if [ ! -d "templates" ]; then
    echo "❌ Templates directory missing - creating..."
    mkdir -p templates
    echo "✅ Templates directory created"
else
    echo "✅ Templates directory exists"
    ls -la templates/
fi

# === 🔧 QUICK FIX ATTEMPT ===
echo -e "\n=== 🔧 SERVICE RESTART ATTEMPT ==="
echo "Attempting service restart..."
sudo systemctl stop bist-pattern
sleep 3
sudo systemctl start bist-pattern
sleep 5

echo "Service status after restart:"
sudo systemctl status bist-pattern | head -10

# === ✅ VERIFY FIX ===
echo -e "\n=== ✅ VERIFICATION TEST ==="
echo "Testing API endpoints..."

# Test basic endpoint
curl -s -k "https://172.20.95.50/" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print('✅ Basic API working!')
    print(f'   Version: {data.get(\"version\", \"unknown\")}')
    print(f'   Status: {data.get(\"status\", \"unknown\")}')
except Exception as e:
    print(f'❌ API still failing: {e}')
    print('Raw response:')
    sys.stdin.seek(0)
    print(repr(sys.stdin.read()[:200]))
"

# Test dashboard endpoint
echo -e "\nTesting dashboard endpoint..."
curl -s -k -I "https://172.20.95.50/dashboard" | head -1

# Test system info
echo -e "\nTesting system info..."
curl -s -k "https://172.20.95.50/api/system-info" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print('✅ System Info API working!')
    print(f'   Features: {data.get(\"features\", [])}')
except Exception as e:
    print(f'❌ System Info still failing: {e}')
"

# === 📊 DIAGNOSIS SUMMARY ===
echo -e "\n=== 📊 DIAGNOSIS SUMMARY ==="
echo ""
echo "🔍 Debug Results:"
echo "1. Service Status: See above"
echo "2. Import Tests: See above" 
echo "3. Template Check: See above"
echo "4. API Tests: See above"
echo ""

# === 🚨 EMERGENCY ROLLBACK IF NEEDED ===
echo "=== 🚨 EMERGENCY ROLLBACK OPTION ==="
echo ""
echo "If service still not working, run emergency rollback:"
echo ""
echo "# Remove problematic dashboard template:"
echo "rm -f templates/dashboard.html"
echo ""
echo "# Restart with basic functionality:"
echo "sudo systemctl restart bist-pattern"
echo ""
echo "# Test basic API:"
echo "curl -k https://172.20.95.50/"
echo ""

echo "=== 🎯 DEBUG COMPLETED ==="
echo ""
echo "Check results above to identify the root cause."
echo "Most likely issues:"
echo "1. ❌ Template import error"
echo "2. ❌ Circular import in scheduler"
echo "3. ❌ Flask context issues"
echo "4. ❌ Missing dependencies"
echo ""
