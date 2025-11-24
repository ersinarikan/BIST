"""
Gunicorn Configuration for BIST Pattern Detection
"""
# multiprocessing importu kullanılmıyor
import os

# Server socket (env ile özelleştirilebilir)
bind = os.getenv("GUNICORN_BIND", "127.0.0.1:5000")
backlog = int(os.getenv("GUNICORN_BACKLOG", "2048"))

# Worker processes (env ile özelleştirilebilir)
# Single worker for automation state consistency and WebSocket support (default)
workers = int(os.getenv("GUNICORN_WORKERS", "1"))
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker")  # WebSocket support için GeventWebSocketWorker kullan
worker_connections = int(os.getenv("GUNICORN_WORKER_CONNECTIONS", "1000"))
# Uzun süren toplama/analizlerde worker'ın timeout ile öldürülmesini önlemek için süreyi artırıyoruz
# FinBERT model download için timeout artırıldı
timeout = int(os.getenv("GUNICORN_TIMEOUT", "1800"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "30"))

# Otomatik worker yeniden başlatmayı kapat (in-process scheduler state korunur)
max_requests = 0
max_requests_jitter = 0

# Logging (stdout/stderr: systemd-journal'a yönlendirilir)
accesslog = os.getenv("GUNICORN_ACCESSLOG", "-")
errorlog = os.getenv("GUNICORN_ERRORLOG", "-")
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
)

# Process naming
proc_name = "bist-pattern-gunicorn"

# Daemon mode - Always False when running under systemd
daemon = False
# PID file - optional, systemd doesn't need it but we keep for compatibility
pidfile = os.getenv("GUNICORN_PIDFILE", "/opt/bist-pattern/gunicorn.pid")
# Remove stale PID file on startup (handled by systemd ExecStartPre)
if pidfile and os.path.exists(pidfile):
    try:
        # Check if process is actually running
        with open(pidfile, 'r') as f:
            old_pid = int(f.read().strip())
        try:
            # Check if this PID is actually a gunicorn process
            try:
                import psutil  # type: ignore
                proc = psutil.Process(old_pid)
                if 'gunicorn' not in ' '.join(proc.cmdline()).lower():
                    # PID file is stale (not a gunicorn process)
                    os.remove(pidfile)
            except ImportError:
                # psutil yoksa temizlik yapmak güvenli
                os.remove(pidfile)
            except Exception:
                # Her durumda güvenli tarafta kal: PID dosyasını temizle
                os.remove(pidfile)
        except Exception:
            # PID dosyasını okuyamadıysak sessiz geç
            pass
    except Exception:
        pass

# User and group (env ile özelleştirilebilir)
user = os.getenv("GUNICORN_USER", "www-data")
group = os.getenv("GUNICORN_GROUP", "www-data")

# Environment
raw_env = [
    "FLASK_ENV=production",
    "SYMBOL_FLOW=1",
]
