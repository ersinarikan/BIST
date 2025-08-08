#!/bin/bash
# BIST Hybrid AI System - Cron Automation Setup
# Stable alternative to threading scheduler

echo "🚀 Setting up BIST Cron-based Automation..."

# Create logs directory
sudo mkdir -p /opt/bist-pattern/logs
sudo chown -R root:root /opt/bist-pattern/logs
sudo chmod 755 /opt/bist-pattern/logs

# Make cron script executable
sudo chmod +x /opt/bist-pattern/cron_automation.py

echo "⏰ Setting up cron jobs..."

# Create cron jobs file
cat > /tmp/bist_crontab << 'EOF'
# BIST Hybrid AI System - Automated Tasks
# Runs stable automation without threading issues

# Daily data collection at 06:00
0 6 * * * cd /opt/bist-pattern && /opt/bist-pattern/venv/bin/python3 cron_automation.py daily_data_collection >> /opt/bist-pattern/logs/cron_automation.log 2>&1

# Model retraining at 07:00  
0 7 * * * cd /opt/bist-pattern && /opt/bist-pattern/venv/bin/python3 cron_automation.py auto_model_retraining >> /opt/bist-pattern/logs/cron_automation.log 2>&1

# Daily status report at 08:00
0 8 * * * cd /opt/bist-pattern && /opt/bist-pattern/venv/bin/python3 cron_automation.py daily_status_report >> /opt/bist-pattern/logs/cron_automation.log 2>&1

# System health check every 4 hours
0 */4 * * * cd /opt/bist-pattern && /opt/bist-pattern/venv/bin/python3 cron_automation.py system_health_check >> /opt/bist-pattern/logs/cron_automation.log 2>&1

# Weekly full data collection on Monday at 05:00
0 5 * * 1 cd /opt/bist-pattern && /opt/bist-pattern/venv/bin/python3 cron_automation.py daily_data_collection >> /opt/bist-pattern/logs/cron_automation.log 2>&1

EOF

# Install cron jobs
sudo crontab /tmp/bist_crontab
sudo rm /tmp/bist_crontab

echo "✅ Cron jobs installed successfully!"
echo ""
echo "📋 Scheduled Tasks:"
echo "  📊 06:00 - Daily data collection"
echo "  🧠 07:00 - Model retraining"  
echo "  📄 08:00 - Daily status report"
echo "  🔍 Every 4h - Health check"
echo "  📈 Monday 05:00 - Weekly full collection"
echo ""

# Test cron automation
echo "🧪 Testing cron automation..."
echo "Running health check test..."

cd /opt/bist-pattern
/opt/bist-pattern/venv/bin/python3 cron_automation.py system_health_check

if [ $? -eq 0 ]; then
    echo "✅ Cron automation test successful!"
else
    echo "❌ Cron automation test failed!"
    exit 1
fi

echo ""
echo "🎯 BIST Cron Automation Setup Complete!"
echo "📝 View logs: tail -f /opt/bist-pattern/logs/cron_automation.log"
echo "📋 Check cron jobs: sudo crontab -l"
echo "🚀 System will now run automated tasks via cron (no threading issues!)"
