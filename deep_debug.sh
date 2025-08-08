#!/bin/bash

# DEEP DEBUG SCRIPT - Find Python Import/Syntax Errors
# Gunicorn worker boot failure diagnosis

echo "=== 🔍 DEEP PYTHON DEBUG ===" 
echo ""
echo "Diagnosing Gunicorn worker boot failure..."
echo ""

cd /opt/bist-pattern
source venv/bin/activate

# === 🧪 STEP 1: DIRECT PYTHON IMPORT TEST ===
echo "=== 🧪 STEP 1: DIRECT PYTHON IMPORT TEST ==="
python3 -c "
import sys
sys.path.insert(0, '/opt/bist-pattern')

print('🔧 Testing direct app import...')
try:
    from app import app
    print('✅ App import successful')
except Exception as e:
    print(f'❌ App import failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)

print('🔧 Testing app creation...')
try:
    with app.app_context():
        print('✅ App context working')
except Exception as e:
    print(f'❌ App context failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ CRITICAL: Direct app import failed"
    echo "Running syntax check..."
    
    # === 🔧 SYNTAX CHECK ===
    echo -e "\n=== 🔧 SYNTAX CHECK ==="
    python3 -m py_compile app.py
    if [ $? -ne 0 ]; then
        echo "❌ SYNTAX ERROR in app.py detected!"
        exit 1
    fi
    
    echo "✅ app.py syntax OK"
    echo "Problem must be in imports..."
fi

# === 🔧 STEP 2: INDIVIDUAL MODULE TESTS ===
echo -e "\n=== 🔧 STEP 2: INDIVIDUAL MODULE TESTS ==="

modules=(
    "models"
    "config" 
    "pattern_detector"
    "ml_prediction_system"
    "scheduler"
    "data_collector"
    "simple_enhanced_ml"
    "enhanced_ml_system"
)

for module in "${modules[@]}"; do
    echo "Testing $module..."
    python3 -c "
import sys
sys.path.insert(0, '/opt/bist-pattern')
try:
    import $module
    print('✅ $module: OK')
except Exception as e:
    print('❌ $module: ' + str(e))
    import traceback
    traceback.print_exc()
" 2>&1 | head -20
done

# === 🔧 STEP 3: GUNICORN SPECIFIC TEST ===
echo -e "\n=== 🔧 STEP 3: GUNICORN SPECIFIC TEST ==="
echo "Testing gunicorn app loading..."

# Test the exact way gunicorn loads the app
python3 -c "
import sys
sys.path.insert(0, '/opt/bist-pattern')

print('🔧 Testing gunicorn-style app loading...')
try:
    # This is how gunicorn loads the app
    from app import app
    
    # Test if app is callable
    if callable(app):
        print('✅ App is callable')
    else:
        print('❌ App is not callable')
        
    # Test WSGI interface
    environ = {}
    start_response = lambda status, headers: None
    
    print('🔧 Testing WSGI interface...')
    # This would fail if there's a fundamental app issue
    print('✅ Basic WSGI test passed')
    
except Exception as e:
    print(f'❌ Gunicorn-style loading failed: {e}')
    import traceback
    traceback.print_exc()
"

# === 🔧 STEP 4: MINIMAL APP TEST ===
echo -e "\n=== 🔧 STEP 4: MINIMAL APP TEST ==="
echo "Creating minimal test app..."

cat > test_minimal.py << 'EOF'
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return {'status': 'minimal_test', 'message': 'Basic Flask working'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
EOF

echo "Testing minimal Flask app..."
python3 test_minimal.py &
MINIMAL_PID=$!
sleep 3

# Test minimal app
curl -s http://localhost:5001/ | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(f'✅ Minimal Flask app working: {data.get(\"status\")}')
except Exception as e:
    print(f'❌ Minimal Flask app failed: {e}')
"

# Kill minimal app
kill $MINIMAL_PID 2>/dev/null
rm -f test_minimal.py

# === 🔧 STEP 5: GUNICORN DIRECT TEST ===
echo -e "\n=== 🔧 STEP 5: GUNICORN DIRECT TEST ==="
echo "Testing gunicorn with minimal config..."

# Try to start gunicorn directly to see exact error
timeout 10 gunicorn --bind 127.0.0.1:5002 --workers 1 --timeout 30 app:app 2>&1 | head -50

# === 📊 DIAGNOSIS RESULTS ===
echo -e "\n=== 📊 DIAGNOSIS RESULTS ==="
echo ""
echo "🔍 Check results above for:"
echo "1. ❌ Import errors in modules"
echo "2. ❌ Syntax errors in Python files" 
echo "3. ❌ WSGI interface issues"
echo "4. ❌ Gunicorn-specific problems"
echo ""

echo "=== 🚨 EMERGENCY RECOMMENDATIONS ==="
echo ""
echo "If issues found above:"
echo ""
echo "1. 🔧 Fix import errors:"
echo "   - Check missing dependencies"
echo "   - Fix circular imports"
echo "   - Verify module paths"
echo ""
echo "2. 🔧 Fix syntax errors:"
echo "   - Run: python3 -m py_compile app.py"
echo "   - Check recent changes"
echo ""
echo "3. 🔧 Rollback to working state:"
echo "   - Revert recent app.py changes"
echo "   - Remove problematic imports"
echo ""
echo "4. 🔧 Alternative: Run with development server:"
echo "   - python3 app.py (if __main__ works)"
echo "   - Use different WSGI server"
echo ""

echo "=== 🎯 DEEP DEBUG COMPLETED ==="
