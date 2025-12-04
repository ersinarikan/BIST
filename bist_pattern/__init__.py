from flask import Flask
import os
from .settings import load_settings
from .extensions import init_extensions
from .websocket.events import register_socketio_events


def create_app(config_name: str | None = None) -> Flask:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    app = Flask(
        __name__,
        static_folder=os.path.join(base_dir, 'static'),
        template_folder=os.path.join(base_dir, 'templates'),
    )
    app.config.from_object(load_settings())
    init_extensions(app)

    # Register blueprints
    try:
        from .blueprints.api_internal import register as register_internal
        register_internal(app)
    except Exception:
        pass
    try:
        from .blueprints.auth import register as register_auth
        register_auth(app)
    except Exception:
        pass
    try:
        from .blueprints.web import register as register_web
        register_web(app)
    except Exception:
        pass
    try:
        from .blueprints.api_public import register as register_api
        register_api(app)
    except Exception:
        pass
    try:
        from .blueprints.api_automation import register as register_auto
        register_auto(app)
    except Exception:
        pass
    try:
        from .blueprints.api_simulation import register as register_sim
        register_sim(app)
    except Exception:
        pass
    try:
        from .blueprints.api_watchlist import register as register_watch
        register_watch(app)
    except Exception:
        pass
    try:
        from .blueprints.api_metrics import register as register_metrics
        register_metrics(app)
    except Exception:
        pass
    try:
        from .blueprints.api_health import register as register_health
        register_health(app)
    except Exception:
        pass
    try:
        from .blueprints.api_recent import register as register_recent
        register_recent(app)
    except Exception:
        pass

    register_socketio_events(app)

    # In-process automation loop (no separate service)
    try:
        auto = str(os.getenv('AUTO_START_PIPELINE', 'True')).lower() in (
            '1', 'true', 'yes'
        )
        if auto and not os.getenv('BIST_PIPELINE_STARTED'):
            try:
                from working_automation import get_working_automation_pipeline  # type: ignore
                pipeline = get_working_automation_pipeline()
            except Exception:
                from scheduler import get_automated_pipeline
                pipeline = get_automated_pipeline()
            if pipeline and not getattr(pipeline, 'is_running', False):
                started = pipeline.start_scheduler()
                if started:
                    os.environ['BIST_PIPELINE_STARTED'] = '1'
    except Exception:
        pass

    # Ensure calibration is disabled by default on service start if requested by env
    try:
        # If BYPASS_ISOTONIC_CALIBRATION is truthy (default '1'), persist a closed state to file
        _bypass_env = str(os.getenv('BYPASS_ISOTONIC_CALIBRATION', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
        print(f"🔧 Calibration startup: BYPASS_ISOTONIC_CALIBRATION={os.getenv('BYPASS_ISOTONIC_CALIBRATION', '1')}, enabled={_bypass_env}")
        if _bypass_env:
            import json as _json
            import os as _os
            from datetime import datetime as _dt
            _log_dir = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            _os.makedirs(_log_dir, exist_ok=True)
            _state_path = _os.path.join(_log_dir, 'calibration_state.json')
            # Merge with existing content if present, but force bypass=true
            cur = {}
            try:
                if _os.path.exists(_state_path):
                    with open(_state_path, 'r') as rf:
                        cur = _json.load(rf) or {}
            except Exception:
                cur = {}
            cur['bypass'] = True
            # Keep previous penalty_factor if any, else provide conservative default
            if 'penalty_factor' not in cur or cur.get('penalty_factor') is None:
                cur['penalty_factor'] = 0.95
            cur['updated_at'] = _dt.now().isoformat()
            _tmp = _state_path + '.tmp'
            try:
                with open(_tmp, 'w') as wf:
                    wf.write(_json.dumps(cur, ensure_ascii=False))
                _os.replace(_tmp, _state_path)
                print(f"✅ Calibration bypass persisted to {_state_path}")
            except Exception as e:
                print(f"⚠️ Atomic write failed, trying fallback: {e}")
                # Non-atomic fallback
                with open(_state_path, 'w') as wf:
                    wf.write(_json.dumps(cur, ensure_ascii=False))
                print(f"✅ Calibration bypass persisted (fallback) to {_state_path}")
    except Exception as e:
        print(f"❌ Calibration startup error: {e}")

    return app
