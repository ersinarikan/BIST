"""
Gunicorn Configuration for BIST Pattern Detection
"""
import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes (Single worker for automation state consistency and WebSocket support)
workers = 1
worker_class = "eventlet"  # WebSocket support için eventlet kullan
worker_connections = 1000
timeout = 120  # WebSocket bağlantıları için daha uzun timeout
keepalive = 2

# Restart workers after this many requests, to prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "/opt/bist-pattern/logs/gunicorn_access.log"
errorlog = "/opt/bist-pattern/logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "bist-pattern-gunicorn"

# Daemon mode
daemon = False
pidfile = "/opt/bist-pattern/gunicorn.pid"

# User and group
user = "root"
group = "root"

# Environment
raw_env = [
    "FLASK_ENV=production",
]
