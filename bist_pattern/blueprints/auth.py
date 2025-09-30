from flask import (
    Blueprint,
    render_template,
    redirect,
    url_for,
    request,
    current_app,
)
from flask_login import login_user, logout_user
from datetime import datetime
from ..extensions import csrf

bp = Blueprint('auth', __name__)


def register(app):
    try:
        from models import db, User
    except ImportError:
        import sys
        sys.path.append('/opt/bist-pattern')
        from models import db, User
    try:
        from flask_wtf.csrf import generate_csrf
    except Exception:
        def generate_csrf():
            return ''

    oauth = None
    try:
        from authlib.integrations.flask_client import OAuth
        oauth = OAuth(app)
    except Exception:
        oauth = None

    @bp.route('/login', methods=['GET', 'POST'])
    @csrf.exempt  # Exempt form login from CSRF to avoid proxy/Referer issues
    def login():
        try:
            if request.method == 'GET':
                return render_template(
                    'login.html',
                    google_enabled=bool(oauth and getattr(oauth, 'google', None)),
                    apple_enabled=bool(oauth and getattr(oauth, 'apple', None)),
                    csrf_token=generate_csrf(),
                )
            email = (request.form.get('email') or '').strip().lower()
            password = request.form.get('password') or ''
            if not email or not password:
                return render_template(
                    'login.html',
                    error='E-posta ve şifre gerekli',
                    google_enabled=bool(oauth and getattr(oauth, 'google', None)),
                )
            user = User.query.filter_by(email=email).first()
            if user and user.check_password(password):
                try:
                    user.last_login = datetime.now()
                    user.last_login_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
                    db.session.commit()
                except Exception:
                    try:
                        db.session.rollback()
                    except Exception:
                        pass
                login_user(user)
                # Role-based redirect: admin -> dashboard, others -> user dashboard
                try:
                    def _is_admin(u) -> bool:
                        try:
                            role = getattr(u, 'role', None)
                            if isinstance(role, str) and role.lower() == 'admin':
                                return True
                            if getattr(u, 'is_admin', False):
                                return True
                            if getattr(u, 'username', '') == 'systemadmin':
                                return True
                            admin_email = (current_app.config.get('ADMIN_EMAIL') or '').lower()
                            if admin_email and getattr(u, 'email', '').lower() == admin_email:
                                return True
                        except Exception:
                            return False
                        return False
                    target = 'web.dashboard' if _is_admin(user) else 'web.user_dashboard'
                    return redirect(url_for(target))
                except Exception:
                    return redirect(url_for('web.user_dashboard'))
            return render_template(
                'login.html',
                error='Geçersiz bilgiler',
                google_enabled=bool(oauth and getattr(oauth, 'google', None)),
                csrf_token=generate_csrf(),
            )
        except Exception:
            try:
                return render_template('login.html', error='Sistem hatası', csrf_token=generate_csrf()), 500
            except Exception:
                return render_template('login.html', error='Sistem hatası'), 500

    @bp.route('/logout')
    def logout():
        try:
            logout_user()
        except Exception:
            pass
        return redirect(url_for('auth.login'))

    @bp.route('/auth/google')
    def auth_google():
        if not oauth or not getattr(oauth, 'google', None):
            return redirect(url_for('auth.login'))
        redirect_uri = url_for('auth.auth_google_callback', _external=True)
        google_client = getattr(oauth, 'google', None)
        if not google_client or not hasattr(google_client, 'authorize_redirect'):
            return redirect(url_for('auth.login'))
        return google_client.authorize_redirect(redirect_uri)

    @bp.route('/auth/google/callback')
    def auth_google_callback():
        try:
            if not oauth or not getattr(oauth, 'google', None):
                return redirect(url_for('auth.login'))
            google_client = getattr(oauth, 'google', None)
            if not google_client or not hasattr(google_client, 'authorize_access_token'):
                return redirect(url_for('auth.login'))
            token = google_client.authorize_access_token()
            userinfo = token.get('userinfo') or {}
            if not userinfo:
                resp = google_client.get('userinfo') if hasattr(google_client, 'get') else None
                userinfo = resp.json() if resp else {}
            email = (userinfo.get('email') or '').lower()
            if not email:
                return redirect(url_for('auth.login'))
            user = User.query.filter_by(email=email).first()
            if not user:
                user = User(
                    email=email,
                    provider='google',
                    provider_id=userinfo.get('sub'),
                    first_name=userinfo.get('given_name'),
                    last_name=userinfo.get('family_name'),
                    avatar_url=userinfo.get('picture'),
                    email_verified=True,
                    is_active=True,
                )
                db.session.add(user)
                db.session.commit()
            try:
                user.last_login = datetime.now()
                user.last_login_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
                db.session.commit()
            except Exception:
                try:
                    db.session.rollback()
                except Exception:
                    pass
            login_user(user)
            # Role-based redirect
            try:
                role = getattr(user, 'role', '')
                if isinstance(role, str) and role.lower() == 'admin':
                    return redirect(url_for('web.dashboard'))
                if getattr(user, 'is_admin', False) or getattr(user, 'username', '') == 'systemadmin':
                    return redirect(url_for('web.dashboard'))
                admin_email = (current_app.config.get('ADMIN_EMAIL') or '').lower()
                if admin_email and getattr(user, 'email', '').lower() == admin_email:
                    return redirect(url_for('web.dashboard'))
            except Exception:
                pass
            return redirect(url_for('web.user_dashboard'))
        except Exception:
            return redirect(url_for('auth.login'))

    @bp.route('/auth/apple')
    def auth_apple():
        if not oauth or not getattr(oauth, 'apple', None):
            return redirect(url_for('auth.login'))
        redirect_uri = url_for('auth.auth_apple_callback', _external=True)
        apple_client = getattr(oauth, 'apple', None)
        if not apple_client or not hasattr(apple_client, 'authorize_redirect'):
            return redirect(url_for('auth.login'))
        return apple_client.authorize_redirect(redirect_uri)

    @bp.route('/auth/apple/callback')
    def auth_apple_callback():
        try:
            if not oauth or not getattr(oauth, 'apple', None):
                return redirect(url_for('auth.login'))
            apple_client = getattr(oauth, 'apple', None)
            if not apple_client or not hasattr(apple_client, 'authorize_access_token'):
                return redirect(url_for('auth.login'))
            token = apple_client.authorize_access_token()
            userinfo = token.get('userinfo') or {}
            email = (userinfo.get('email') or '').lower()
            if not email:
                email = (token.get('id_token_claims') or {}).get('email', '').lower()
            if not email:
                return redirect(url_for('auth.login'))
            user = User.query.filter_by(email=email).first()
            if not user:
                user = User(
                    email=email,
                    provider='apple',
                    provider_id=(
                        userinfo.get('sub')
                        or (token.get('id_token_claims') or {}).get('sub')
                    ),
                    first_name=userinfo.get('name'),
                    email_verified=True,
                    is_active=True,
                )
                db.session.add(user)
                db.session.commit()
            try:
                user.last_login = datetime.now()
                user.last_login_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
                db.session.commit()
            except Exception:
                try:
                    db.session.rollback()
                except Exception:
                    pass
            login_user(user)
            # Role-based redirect
            try:
                role = getattr(user, 'role', '')
                if isinstance(role, str) and role.lower() == 'admin':
                    return redirect(url_for('web.dashboard'))
                if getattr(user, 'is_admin', False) or getattr(user, 'username', '') == 'systemadmin':
                    return redirect(url_for('web.dashboard'))
                admin_email = (current_app.config.get('ADMIN_EMAIL') or '').lower()
                if admin_email and getattr(user, 'email', '').lower() == admin_email:
                    return redirect(url_for('web.dashboard'))
            except Exception:
                pass
            return redirect(url_for('web.user_dashboard'))
        except Exception:
            return redirect(url_for('auth.login'))

    app.register_blueprint(bp)
