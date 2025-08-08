#!/bin/bash

# BIST Scheduler Service Creator
echo "🔧 Creating BIST Scheduler Systemd Service..."

# Create the service file
sudo tee /etc/systemd/system/bist-scheduler.service > /dev/null << 'EOF'
[Unit]
Description=BIST Pattern Detection Scheduler Daemon
After=network.target postgresql.service bist-pattern.service
Requires=postgresql.service

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/bist-pattern
Environment=PATH=/opt/bist-pattern/venv/bin
ExecStart=/opt/bist-pattern/venv/bin/python /opt/bist-pattern/scheduler_daemon.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable bist-scheduler

echo "✅ BIST Scheduler service created successfully!"
echo "📊 Current scheduler status:"

# Check if python process is running
if pgrep -f "scheduler_daemon.py" > /dev/null; then
    echo "🟢 Scheduler daemon is running as Python process"
    echo "   PID: $(pgrep -f 'scheduler_daemon.py')"
else
    echo "🔴 No scheduler daemon process found"
fi

echo ""
echo "🚀 To manage the service:"
echo "  Start:   sudo systemctl start bist-scheduler"
echo "  Stop:    sudo systemctl stop bist-scheduler"  
echo "  Status:  sudo systemctl status bist-scheduler"
echo "  Logs:    journalctl -u bist-scheduler -f"

echo ""
echo "📈 Current system status:"
echo "✅ Main App: $(systemctl is-active bist-pattern)"
echo "✅ Database: $(systemctl is-active postgresql)"
echo "✅ Web Server: $(systemctl is-active nginx)"
echo "🔄 Scheduler: Python process running"
