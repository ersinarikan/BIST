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
        except Exception as e:
            logger.debug(f"Failed to redirect authenticated user: {e}")
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
            # Inject real user info for UI display and websocket room subscription
            try:
                # ✅ FIX: Ensure current_user is authenticated and has an ID
                if not current_user.is_authenticated:
                    app.logger.warning("User dashboard accessed but user not authenticated")
                    return redirect(url_for('auth.login'))
                
                uid = getattr(current_user, 'id', None)
                if uid is None:
                    app.logger.error(f"User {getattr(current_user, 'email', 'unknown')} has no ID attribute")
                    # Try to get ID from database
                    try:
                        from models import User
                        user = User.query.filter_by(email=getattr(current_user, 'email', None)).first()
                        if user:
                            uid = user.id
                            app.logger.info(f"Retrieved user ID {uid} from database for {user.email}")
                        else:
                            app.logger.error(f"User not found in database: {getattr(current_user, 'email', 'unknown')}")
                    except Exception as db_err:
                        app.logger.error(f"Database error retrieving user ID: {db_err}")
                
                email = getattr(current_user, 'email', None)
                username = getattr(current_user, 'username', None)
                full_name = getattr(current_user, 'full_name', None) or username or (email.split('@')[0] if email else 'Kullanıcı')
            except Exception as e:
                app.logger.error(f"Error extracting user info: {e}")
                uid = None
                email = None
                full_name = 'Kullanıcı'
            
            # ✅ FIX: Always inject user_id (even if None) so frontend can handle it
            resp = make_response(render_template('user_dashboard.html', 
                                                  user_id=uid,
                                                  user_email=email,
                                                  user_name=full_name))
            resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            resp.headers['Pragma'] = 'no-cache'
            resp.headers['Expires'] = '0'
            return resp
        except Exception as e:
            app.logger.error(f"User dashboard error: {e}")
            return jsonify({'error': 'User dashboard not available'}), 500

    app.register_blueprint(bp)
