#!/bin/bash

# BIST Pattern Detection - Production Deployment Script
# Ubuntu 24.04 LTS için optimize edilmiş deployment

set -e

echo "🚀 BIST Pattern Detection - Production Deployment"
echo "=================================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# 1. Check if running as root or with sudo
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root or with sudo"
   exit 1
fi

# 2. Update system packages
log "Updating system packages..."
apt update && apt upgrade -y

# 3. Install missing dependencies
log "Installing system dependencies..."
apt install -y nginx postgresql postgresql-contrib python3-pip python3-venv python3-dev \
    build-essential libpq-dev git curl wget htop vim nano ufw fail2ban

# 4. Install Python dependencies
log "Installing Python dependencies..."
cd /opt/bist-pattern
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install schedule gunicorn psycopg2-binary

# 5. Database optimization
log "Optimizing PostgreSQL..."
PG_VERSION=$(sudo -u postgres psql -t -c "SELECT version();" | grep -oP '\d+\.\d+' | head -1)
PG_CONF="/etc/postgresql/${PG_VERSION}/main/postgresql.conf"

# Backup original config
cp $PG_CONF ${PG_CONF}.backup

# Apply optimizations
cat >> $PG_CONF << 'EOF'

# BIST Pattern Detection Optimizations
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB
EOF

systemctl restart postgresql

# 6. Setup systemd services
log "Setting up systemd services..."

# Main application service
tee /etc/systemd/system/bist-pattern.service > /dev/null << 'EOF'
[Unit]
Description=BIST Pattern Detection Web Application
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=notify
User=root
Group=root
WorkingDirectory=/opt/bist-pattern
Environment=PATH=/opt/bist-pattern/venv/bin
Environment=FLASK_ENV=production
ExecStart=/opt/bist-pattern/venv/bin/gunicorn --config gunicorn.conf.py app:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Scheduler daemon service
tee /etc/systemd/system/bist-scheduler.service > /dev/null << 'EOF'
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
ExecStart=/opt/bist-pattern/venv/bin/python scheduler_daemon.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 7. Nginx configuration
log "Configuring Nginx..."
tee /etc/nginx/sites-available/bist-pattern > /dev/null << 'EOF'
# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=web:10m rate=30r/s;

server {
    listen 80;
    listen 443 ssl http2 default_server;
    server_name _;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/ssl-cert-snakeoil.pem;
    ssl_certificate_key /etc/ssl/private/ssl-cert-snakeoil.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384:ECDHE-RSA-AES128-SHA:ECDHE-RSA-AES256-SHA:DHE-RSA-AES128-SHA256:DHE-RSA-AES256-SHA256:DHE-RSA-AES128-SHA:DHE-RSA-AES256-SHA:ECDHE-RSA-DES-CBC3-SHA:EDH-RSA-DES-CBC3-SHA:AES128-GCM-SHA256:AES256-GCM-SHA384:AES128-SHA256:AES256-SHA256:AES128-SHA:AES256-SHA:DES-CBC3-SHA:HIGH:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4;
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript application/x-javascript text/x-js;

    # Static files with caching
    location /static/ {
        alias /opt/bist-pattern/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }

    # API endpoints with rate limiting
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        limit_req_status 429;
        
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
        proxy_connect_timeout 10s;
        proxy_send_timeout 60s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        access_log off;
    }

    # Main application
    location / {
        limit_req zone=web burst=50 nodelay;
        
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
        proxy_connect_timeout 10s;
        proxy_send_timeout 60s;
        
        # Enable WebSocket support (future use)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Robots.txt
    location /robots.txt {
        return 200 "User-agent: *\nDisallow: /api/\nDisallow: /admin/\n";
        add_header Content-Type text/plain;
    }
}
EOF

# Enable site
ln -sf /etc/nginx/sites-available/bist-pattern /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test nginx config
nginx -t

# 8. Firewall configuration
log "Configuring firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# 9. Fail2ban setup
log "Setting up Fail2ban..."
tee /etc/fail2ban/jail.local > /dev/null << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 3

[nginx-req-limit]
enabled = true
filter = nginx-req-limit
logpath = /var/log/nginx/error.log
maxretry = 10
findtime = 600
bantime = 7200
EOF

# 10. Log rotation
log "Setting up log rotation..."
tee /etc/logrotate.d/bist-pattern > /dev/null << 'EOF'
/opt/bist-pattern/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
    postrotate
        systemctl reload bist-pattern >/dev/null 2>&1 || true
        systemctl reload bist-scheduler >/dev/null 2>&1 || true
    endscript
}
EOF

# 11. Create log directory
mkdir -p /opt/bist-pattern/logs
chown -R root:root /opt/bist-pattern/logs

# 12. Enable and start services
log "Starting services..."
systemctl daemon-reload
systemctl enable bist-pattern
systemctl enable bist-scheduler
systemctl enable nginx
systemctl enable fail2ban

systemctl restart postgresql
systemctl restart nginx
systemctl restart fail2ban
systemctl restart bist-pattern
sleep 5
systemctl start bist-scheduler

# 13. Create monitoring script
log "Creating monitoring script..."
tee /opt/bist-pattern/monitor.py > /dev/null << 'EOF'
#!/usr/bin/env python3
import requests
import psutil
import subprocess
import json
from datetime import datetime

def check_system():
    status = {
        'timestamp': datetime.now().isoformat(),
        'status': 'healthy',
        'issues': []
    }
    
    # CPU & Memory
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    if cpu > 80:
        status['issues'].append(f'High CPU: {cpu}%')
        status['status'] = 'warning'
    
    if memory.percent > 80:
        status['issues'].append(f'High Memory: {memory.percent}%')
        status['status'] = 'warning'
    
    if disk.percent > 80:
        status['issues'].append(f'High Disk: {disk.percent}%')
        status['status'] = 'warning'
    
    # Services
    services = ['bist-pattern', 'bist-scheduler', 'nginx', 'postgresql']
    for service in services:
        try:
            result = subprocess.run(['systemctl', 'is-active', service], 
                                  capture_output=True, text=True)
            if result.stdout.strip() != 'active':
                status['issues'].append(f'Service down: {service}')
                status['status'] = 'critical'
        except:
            status['issues'].append(f'Cannot check: {service}')
    
    # API Health
    try:
        resp = requests.get('https://localhost/health', verify=False, timeout=10)
        if resp.status_code != 200:
            status['issues'].append(f'API unhealthy: {resp.status_code}')
            status['status'] = 'critical'
    except:
        status['issues'].append('API unreachable')
        status['status'] = 'critical'
    
    print(json.dumps(status, indent=2))
    return status

if __name__ == "__main__":
    check_system()
EOF

chmod +x /opt/bist-pattern/monitor.py

# 14. Setup crontab monitoring
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/bist-pattern/venv/bin/python /opt/bist-pattern/monitor.py >> /opt/bist-pattern/logs/monitoring.log 2>&1") | crontab -

# 15. Final system check
log "Performing final system check..."
sleep 10

echo ""
echo "🎯 DEPLOYMENT SUMMARY"
echo "====================="

# Check services
services=("postgresql" "nginx" "bist-pattern" "bist-scheduler" "fail2ban")
for service in "${services[@]}"; do
    if systemctl is-active --quiet $service; then
        echo -e "✅ $service: ${GREEN}Active${NC}"
    else
        echo -e "❌ $service: ${RED}Inactive${NC}"
    fi
done

echo ""
echo "🔗 Testing endpoints..."

# Test API
if curl -k -f -s "https://localhost/health" > /dev/null; then
    echo -e "✅ API Health: ${GREEN}OK${NC}"
else
    echo -e "❌ API Health: ${RED}Failed${NC}"
fi

# Test dashboard
if curl -k -f -s "https://localhost/dashboard" > /dev/null; then
    echo -e "✅ Dashboard: ${GREEN}OK${NC}"
else
    echo -e "❌ Dashboard: ${RED}Failed${NC}"
fi

# Database check
if sudo -u postgres psql -d bist_pattern -c "SELECT COUNT(*) FROM stocks;" > /dev/null 2>&1; then
    stock_count=$(sudo -u postgres psql -d bist_pattern -t -c "SELECT COUNT(*) FROM stocks;" | xargs)
    price_count=$(sudo -u postgres psql -d bist_pattern -t -c "SELECT COUNT(*) FROM stock_prices;" | xargs)
    echo -e "✅ Database: ${GREEN}Connected (${stock_count} stocks, ${price_count} prices)${NC}"
else
    echo -e "❌ Database: ${RED}Connection failed${NC}"
fi

echo ""
echo "🌐 Access URLs:"
echo "- Dashboard: https://$(hostname -I | awk '{print $1}')/dashboard"
echo "- API Health: https://$(hostname -I | awk '{print $1}')/health"
echo "- System Monitor: python3 /opt/bist-pattern/monitor.py"

echo ""
echo "📊 Quick Stats:"
echo "- CPU: $(awk '{print $1}' /proc/loadavg)"
echo "- Memory: $(free -h | awk 'NR==2{printf "%.1f%% of %s", $3*100/$2, $2}')"
echo "- Disk: $(df -h / | awk 'NR==2{print $5 " of " $2}')"

echo ""
log "🚀 BIST Pattern Detection Production Deployment Complete!"
log "📊 Dashboard: Modern UI with real-time charts"
log "🔄 Scheduler: Automated data collection active"
log "🔒 Security: SSL, Firewall, Fail2ban configured"
log "📈 Monitoring: Health checks and log rotation enabled"

echo ""
echo "Next steps:"
echo "1. Access dashboard at https://your-server-ip/dashboard"
echo "2. Monitor logs: tail -f /opt/bist-pattern/logs/*.log"
echo "3. Check system: python3 /opt/bist-pattern/monitor.py"
echo ""
