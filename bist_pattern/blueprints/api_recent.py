from flask import Blueprint, jsonify
import os

bp = Blueprint('api_recent', __name__, url_prefix='/api')


def register(app):
    def _read_pipeline_history(max_items: int = 6):
        try:
            import json
            log_path = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            status_file = os.path.join(log_path, 'pipeline_status.json')
            if not os.path.exists(status_file):
                return []
            with open(status_file, 'r') as f:
                data = json.load(f) or {}
            history = data.get('history', [])
            return history[-max_items:]
        except Exception:
            return []

    @bp.route('/recent-tasks')
    def recent_tasks():
        try:
            hist = _read_pipeline_history(50) or []
            # Keep backward compat with dashboard.js expecting 'tasks'
            tasks = []
            for h in hist[::-1]:
                # Transform to lightweight UI-friendly task entries
                tasks.append({
                    'task': h.get('phase', 'pipeline'),
                    'description': h.get('state', ''),
                    'status': h.get('state', 'pending'),
                    'timestamp': h.get('timestamp', ''),
                    'icon': '🧩'
                })
            resp = jsonify({
                'status': 'success',
                'history': hist,
                'tasks': tasks,
            })
            resp.headers['Cache-Control'] = (
                'no-store, no-cache, must-revalidate, max-age=0'
            )
            return resp
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)})

    app.register_blueprint(bp)
