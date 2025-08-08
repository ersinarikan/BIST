#!/bin/bash

# EMERGENCY ROLLBACK SCRIPT
# Rollback to last working state

echo "=== 🔄 EMERGENCY ROLLBACK ===" 
echo ""
echo "Rolling back to last known working configuration..."
echo ""

cd /opt/bist-pattern
source venv/bin/activate

# === 🔧 STEP 1: BACKUP CURRENT STATE ===
echo "🔧 Step 1: Backing up current problematic state..."
cp app.py app.py.broken.backup
cp scheduler.py scheduler.py.broken.backup
echo "✅ Current state backed up"

# === 🔄 STEP 2: REMOVE PROBLEMATIC FEATURES ===
echo -e "\n🔄 Step 2: Removing problematic dashboard features..."

# Create minimal working app.py
cat > app.py << 'EOF'
import os
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_login import LoginManager
from flask_mail import Mail
from flask_migrate import Migrate
from config import config
from models import db, User, Stock, StockPrice
import logging

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(config_name=None):
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'production')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Initialize extensions
    db.init_app(app)
    
    # Login Manager
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Mail
    mail = Mail(app)
    
    # Migration
    migrate = Migrate(app, db)
    
    # Basic Routes
    @app.route('/')
    def index():
        return jsonify({
            "message": "BIST Pattern Detection API",
            "status": "running",
            "version": "2.1.0-stable",
            "database": "PostgreSQL",
            "features": ["Real-time Data", "Yahoo Finance", "Basic API"]
        })

    @app.route('/health')
    def health():
        try:
            from sqlalchemy import text
            with db.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return jsonify({
                "status": "healthy",
                "database": "connected",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    @app.route('/api/system-info')
    def system_info():
        """Basic system information"""
        try:
            info = {
                'status': 'operational',
                'version': '2.1.0-stable',
                'database': {
                    'stocks': Stock.query.count(),
                    'price_records': StockPrice.query.count()
                },
                'features': ['Basic API', 'Database', 'Health Check']
            }
            return jsonify(info)
        except Exception as e:
            logger.error(f"System info error: {e}")
            return jsonify({'error': str(e)}), 500

    return app

app = create_app()

# Basic pattern detector (minimal)
_pattern_detector = None

def get_pattern_detector():
    global _pattern_detector
    if _pattern_detector is None:
        try:
            from pattern_detector import HybridPatternDetector
            _pattern_detector = HybridPatternDetector()
        except ImportError:
            logger.warning("Pattern detector not available")
            _pattern_detector = None
    return _pattern_detector

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
EOF

echo "✅ Minimal app.py created"

# === 🔧 STEP 3: RESTART SERVICE ===
echo -e "\n🔧 Step 3: Restarting with minimal configuration..."
sudo systemctl stop bist-pattern
sleep 3
sudo systemctl start bist-pattern
sleep 8

# === ✅ STEP 4: VERIFY ROLLBACK ===
echo -e "\n✅ Step 4: Verifying rollback success..."

# Test basic functionality
echo "Testing basic API..."
curl -s -k "https://172.20.95.50/" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(f'✅ Basic API restored: {data.get(\"status\")} v{data.get(\"version\")}')
    print(f'   Features: {data.get(\"features\", [])}')
except Exception as e:
    print(f'❌ Rollback failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ ROLLBACK SUCCESSFUL!"
    
    # Test system info
    echo -e "\nTesting system info..."
    curl -s -k "https://172.20.95.50/api/system-info" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    db_info = data.get('database', {})
    print(f'✅ System Info: {data.get(\"status\")}')
    print(f'   Database: {db_info.get(\"stocks\", 0)} stocks, {db_info.get(\"price_records\", 0)} records')
except Exception as e:
    print(f'⚠️ System info issue: {e}')
"

    # Test health
    echo -e "\nTesting health endpoint..."
    curl -s -k "https://172.20.95.50/health" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(f'✅ Health: {data.get(\"status\")}')
    print(f'   Database: {data.get(\"database\")}')
except Exception as e:
    print(f'⚠️ Health check issue: {e}')
"

else
    echo "❌ ROLLBACK FAILED - NEED MANUAL INTERVENTION"
    echo ""
    echo "Manual recovery steps:"
    echo "1. Check service status: sudo systemctl status bist-pattern"
    echo "2. Check logs: sudo journalctl -u bist-pattern -n 50"
    echo "3. Try direct Python: python3 app.py"
    echo ""
    exit 1
fi

# === 🎊 ROLLBACK SUCCESS ===
echo -e "\n=== 🎊 ROLLBACK COMPLETED ==="
echo ""
echo "✅ System restored to stable state!"
echo ""
echo "📊 Current Status:"
echo "   🔗 API: https://172.20.95.50/"
echo "   🔍 Health: https://172.20.95.50/health" 
echo "   📋 System Info: https://172.20.95.50/api/system-info"
echo ""
echo "🔧 Next Steps:"
echo "1. ✅ System is now stable and operational"
echo "2. 🧪 Debug the problematic features separately"
echo "3. 📱 Re-implement dashboard step by step"
echo "4. 🤖 Re-add automation features gradually"
echo ""
echo "📁 Backup Files:"
echo "   app.py.broken.backup - The problematic version"
echo "   scheduler.py.broken.backup - For reference"
echo ""
echo "🚀 System is back online and ready for safe development!"
echo ""
