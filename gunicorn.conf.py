"""
Gunicorn Configuration for BIST Pattern Detection
"""
import multiprocessing

# Server socket
bind = "127.0.0.1:5000"
backlog = 2048

# Worker processes (Single worker for automation state consistency and WebSocket support)
workers = 1
worker_class = "gevent"  # Eventlet crash'lerini önlemek için gevent kullan
worker_connections = 1000
# Uzun süren toplama/analizlerde worker'ın timeout ile öldürülmesini
# önlemek için süreyi artırıyoruz
timeout = 600
keepalive = 30

# Otomatik worker yeniden başlatmayı kapat (in-process scheduler state korunur)
max_requests = 0
max_requests_jitter = 0

# Logging (stdout/stderr: systemd-journal'a yönlendirilir)
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "bist-pattern-gunicorn"

# Daemon mode
daemon = False
pidfile = "/opt/bist-pattern/gunicorn.pid"

# User and group
user = "www-data"
group = "www-data"

# Environment
raw_env = [
    "FLASK_ENV=production",
    "PIPELINE_MODE=CONTINUOUS_FULL",
]
