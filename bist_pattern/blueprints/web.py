from flask import Blueprint, render_template, jsonify, redirect, url_for, make_response
from flask_login import login_required, current_user

bp = Blueprint('web', __name__)


def _is_admin_user(user) -> bool:
    try:
        role = getattr(user, 'role', None) or getattr(user, 'roles', None)
        if isinstance(role, str):
            return role.lower() == 'admin'
        return bool(getattr(user, 'is_admin', False))
    except Exception:
        return False


def register(app):
    @bp.route('/')
    def index():
        try:
            if current_user.is_authenticated:
                target = 'web.dashboard' if _is_admin_user(current_user) else 'web.user_dashboard'
                return redirect(url_for(target))
        except Exception:
            pass
        return redirect(url_for('auth.login'))

    @bp.route('/favicon.ico')
    def favicon():
        try:
            return app.send_static_file('favicon.ico')
        except Exception:
            return ('', 404)

    @bp.route('/dashboard')
    @login_required
    def dashboard():
        try:
            return render_template('dashboard.html')
        except Exception:
            return jsonify({'error': 'Dashboard not available'}), 500

    @bp.route('/stocks')
    def stocks_page():
        return render_template('stocks.html')

    @bp.route('/analysis')
    def analysis_page():
        return render_template('analysis.html')

    @bp.route('/user')
    @login_required
    def user_dashboard():
        try:
            resp = make_response(render_template('user_dashboard.html'))
            resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            resp.headers['Pragma'] = 'no-cache'
            resp.headers['Expires'] = '0'
            return resp
        except Exception:
            return jsonify({'error': 'User dashboard not available'}), 500

    app.register_blueprint(bp)
