"""
Admin Dashboard Blueprint
Admin-only dashboard routes and API endpoints
"""

from flask import Blueprint, jsonify, request
from flask_login import login_required
from datetime import datetime
from sqlalchemy import func, desc
import logging

logger = logging.getLogger(__name__)

bp = Blueprint('admin_dashboard', __name__, url_prefix='/api/admin')


def register(app):
    """Register admin dashboard blueprint"""
    
    # Import dependencies inside register to avoid circular imports
    from models import db, User, Stock, StockPrice, MetricsDaily
    from bist_pattern.core.decorators import admin_required
    from bist_pattern.core.auth_manager import AuthManager
    
    @bp.route('/dashboard-stats')
    @login_required
    @admin_required
    def dashboard_stats():
        """Admin dashboard statistics"""
        try:
            # Basic statistics
            total_stocks = Stock.query.count()
            total_prices = StockPrice.query.count()
            
            # Stocks with most data
            stock_with_most_data = db.session.query(
                Stock.symbol,
                func.count(StockPrice.id).label('price_count')
            ).join(StockPrice).group_by(Stock.symbol)\
            .order_by(desc('price_count')).limit(5).all()
            
            # Sector distribution
            sector_stats = db.session.query(
                Stock.sector,
                func.count(Stock.id).label('stock_count')
            ).group_by(Stock.sector)\
            .order_by(desc('stock_count')).limit(10).all()
            
            # Recent activity
            latest_date = db.session.query(func.max(StockPrice.date)).scalar()
            latest_count = 0
            if latest_date:
                latest_count = StockPrice.query.filter_by(date=latest_date).count()
            
            # User statistics
            user_stats = AuthManager.get_user_stats()
            
            top_stocks_data = [{'symbol': s, 'count': c} for s, c in stock_with_most_data]
            sector_data = [{'sector': s or 'Unknown', 'count': c} for s, c in sector_stats]

            return jsonify({
                'database': {
                    'total_stocks': total_stocks,
                    'total_prices': total_prices,
                    'latest_data_date': str(latest_date) if latest_date else None,
                    'latest_day_records': latest_count
                },
                'users': user_stats,
                'top_stocks': top_stocks_data,
                'sectors': sector_data,
                'system': {
                    'timestamp': datetime.now().isoformat(),
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            })
            
        except Exception as e:
            logger.error(f"Admin dashboard stats error: {e}")
            return jsonify({'error': 'Failed to fetch dashboard statistics'}), 500
    
    @bp.route('/data-collection/status')
    @login_required
    @admin_required
    def data_collection_status():
        """Data collection status for admin"""
        try:
            # Latest data information
            latest_date = db.session.query(func.max(StockPrice.date)).scalar()
            
            # Daily data count
            latest_count = 0
            if latest_date:
                latest_count = StockPrice.query.filter_by(date=latest_date).count()
            
            # Data quality metrics
            stocks_with_data = db.session.query(
                func.count(func.distinct(StockPrice.stock_id))
            ).scalar()
            
            # Recent data trends (last 7 days)
            from datetime import timedelta
            
            recent_data = []
            if latest_date:
                for i in range(7):
                    check_date = latest_date - timedelta(days=i)
                    day_count = StockPrice.query.filter_by(date=check_date).count()
                    recent_data.append({
                        'date': str(check_date),
                        'records': day_count
                    })

            return jsonify({
                'status': 'active',
                'latest_data_date': str(latest_date) if latest_date else None,
                'latest_day_records': latest_count,
                'total_records': StockPrice.query.count(),
                'stocks_with_data': stocks_with_data,
                'recent_trends': recent_data,
                'message': 'Data collection system active'
            })
            
        except Exception as e:
            logger.error(f"Data collection status error: {e}")
            return jsonify({'error': 'Failed to fetch data collection status'}), 500
    
    @bp.route('/data-collection/manual', methods=['POST'])
    @login_required
    @admin_required
    def manual_data_collection():
        """Manual data collection endpoint for admin"""
        try:
            from bist_pattern.core.unified_collector import get_unified_collector
            collector = get_unified_collector()
            
            # Get parameters from request (safe)
            try:
                body = request.get_json(silent=True) or {}
            except Exception:
                body = {}
            symbol_limit = body.get('symbol_limit', 10)
            period = body.get('period', '5d')
            
            # Manual data collection
            symbols = collector.get_bist_symbols()
            results = []
            
            for symbol in symbols[:symbol_limit]:
                try:
                    result = collector.collect_single_stock(symbol, period=period)
                    results.append({
                        'symbol': symbol,
                        'status': 'success' if result else 'failed',
                        'result': result
                    })
                except Exception as e:
                    results.append({
                        'symbol': symbol,
                        'status': 'error',
                        'error': str(e)
                    })
            
            successful = len([r for r in results if r['status'] == 'success'])
            
            return jsonify({
                'status': 'completed',
                'collected_symbols': successful,
                'total_attempted': len(results),
                'success_rate': f"{(successful/len(results)*100):.1f}%" if results else "0%",
                'results': results,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Manual data collection error: {e}")
            return jsonify({'error': 'Manual data collection failed'}), 500
    
    @bp.route('/automation/report')
    @login_required
    @admin_required
    def automation_report():
        """Daily system report for admin"""
        try:
            # Get automation pipeline
            try:
                from working_automation import get_working_automation_pipeline  # type: ignore
                pipeline = get_working_automation_pipeline()
            except Exception:
                try:
                    from scheduler import get_automated_pipeline
                    pipeline = get_automated_pipeline()
                except Exception:
                    pipeline = None
            
            if not pipeline:
                return jsonify({'error': 'Automation pipeline not available'}), 503
            
            # Get system stats
            stats = {
                'system_status': 'running',
                'pipeline_active': hasattr(pipeline, 'running') and pipeline.running,
                'last_cycle': getattr(pipeline, 'last_cycle_time', None),
                'total_cycles': getattr(pipeline, 'cycle_count', 0),
                'errors_today': 0,  # Could be implemented with logging analysis
                'data_freshness': 'good',  # Based on latest data date
                'timestamp': datetime.now().isoformat()
            }
            
            # Recent automation history
            try:
                history = getattr(pipeline, 'get_pipeline_history', lambda: [])()
                recent_history = history[-10:] if history else []
            except Exception:
                recent_history = []
            
            # Performance metrics
            performance = {
                'avg_cycle_time': 0,
                'success_rate': 100,
                'last_errors': []
            }
            
            if recent_history:
                try:
                    successful = len([h for h in recent_history if h.get('status') == 'success'])
                    performance['success_rate'] = (successful / len(recent_history)) * 100
                except Exception:
                    pass
            
            return jsonify({
                'report_date': datetime.now().strftime('%Y-%m-%d'),
                'system_stats': stats,
                'recent_history': recent_history,
                'performance': performance,
                'recommendations': _generate_admin_recommendations(stats, performance)
            })
            
        except Exception as e:
            logger.error(f"Automation report error: {e}")
            return jsonify({'error': 'Failed to generate automation report'}), 500

    @bp.route('/calibration/summary')
    @login_required
    @admin_required
    def calibration_summary():
        """Return param_store summary and recent (7d,30d) metrics per horizon."""
        from datetime import date, timedelta
        import os as _os
        import json as _json
        try:
            # Load param_store.json
            base = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            ppath = _os.path.join(base, 'param_store.json')
            pstore = {}
            if _os.path.exists(ppath):
                try:
                    with open(ppath, 'r') as rf:
                        pstore = _json.load(rf) or {}
                except Exception:
                    pstore = {}

            # Helper to aggregate metrics for the last N days
            def _aggregate_last_ndays(n_days: int):
                _today = date.today()
                _start = _today - timedelta(days=n_days - 1)
                _rows = (
                    db.session.query(MetricsDaily)
                    .filter(MetricsDaily.date >= _start)
                    .filter(MetricsDaily.date <= _today)
                    .all()
                )
                _by_h: dict[str, list] = {}
                for _r in _rows:
                    _key = (_r.horizon or '').strip()
                    if not _key:
                        continue
                    _by_h.setdefault(_key, []).append(_r)
                _metrics = {}
                _acc_vals = []
                for _h, _items in _by_h.items():
                    _acc_list = [float(getattr(_it, 'acc')) for _it in _items if getattr(_it, 'acc') is not None]
                    _acc = (sum(_acc_list) / len(_acc_list)) if _acc_list else None
                    _metrics[_h] = {'acc': _acc, 'count': len(_items)}
                    if _acc is not None:
                        _acc_vals.append(_acc)
                _overall = (sum(_acc_vals) / len(_acc_vals)) if _acc_vals else None
                return _metrics, _overall

            metrics_7d, overall_acc_7d = _aggregate_last_ndays(7)
            metrics_30d, overall_acc_30d = _aggregate_last_ndays(30)
            metrics_90d, overall_acc_90d = _aggregate_last_ndays(90)  # ⚡ NEW: 90-day metrics

            # A/B summary for last 7d (based on PredictionsLog.param_version tag and OutcomesLog.dir_hit)
            from models import PredictionsLog, OutcomesLog  # type: ignore
            hkeys = ['1d', '3d', '7d', '14d', '30d']
            ab_7d = {h: {'prod': {'acc': None, 'n': 0}, 'chall': {'acc': None, 'n': 0}} for h in hkeys}
            dt_today = date.today()
            dt_start = dt_today - timedelta(days=6)
            rows = (
                db.session.query(PredictionsLog, OutcomesLog)
                .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)
                .filter(OutcomesLog.ts_eval >= dt_start)
                .all()
            )
            tmp = {h: {'prod': [], 'chall': []} for h in hkeys}
            for p, o in rows:
                try:
                    h = (p.horizon or '').strip()
                    if h not in tmp:
                        continue
                    pv = str(getattr(p, 'param_version', '') or '')
                    grp = 'chall' if 'ab:chall' in pv else 'prod'
                    tmp[h][grp].append(1.0 if bool(getattr(o, 'dir_hit')) else 0.0)
                except Exception:
                    continue
            for h in hkeys:
                for grp in ('prod', 'chall'):
                    vals = tmp[h][grp]
                    n = len(vals)
                    ab_7d[h][grp]['n'] = n
                    ab_7d[h][grp]['acc'] = (sum(vals) / n) if n else None

            # ⚡ NEW: Calculate magnitude-based metrics for A/B test
            ab_7d_magnitude = {}
            try:
                hkeys2 = ['1d', '3d', '7d', '14d', '30d']
                ab_7d_magnitude = {h: {'prod': {'acc': None, 'n': 0}, 'chall': {'acc': None, 'n': 0}} for h in hkeys2}
                dt_today2 = date.today()
                dt_start2 = dt_today2 - timedelta(days=6)
                rows2 = (
                    db.session.query(PredictionsLog, OutcomesLog)
                    .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)
                    .filter(OutcomesLog.ts_eval >= dt_start2)
                    .all()
                )
                tmp2 = {h: {'prod': [], 'chall': []} for h in hkeys2}
                magnitude_tol = float(_os.getenv('MAGNITUDE_HIT_TOLERANCE', '0.05'))
                for p2, o2 in rows2:
                    try:
                        h2 = (p2.horizon or '').strip()
                        if h2 not in tmp2:
                            continue
                        pv2 = str(getattr(p2, 'param_version', '') or '')
                        grp2 = 'chall' if 'ab:chall' in pv2 else 'prod'
                        # Calculate magnitude hit on-the-fly
                        delta_real = float(getattr(o2, 'delta_real') or 0.0)
                        delta_pred = float(getattr(p2, 'delta_pred') or 0.0)
                        dir_hit_val = bool(getattr(o2, 'dir_hit'))
                        abs_err_val = float(getattr(o2, 'abs_err') or 0.0)
                        price_eval_val = float(getattr(o2, 'price_eval') or 0.0)
                        threshold_val = float(_os.getenv('DIRECTION_HIT_THRESHOLD', '0.005'))
                        
                        if dir_hit_val and price_eval_val > 0:
                            if abs(delta_real) < threshold_val and abs(delta_pred) < threshold_val:
                                mag_hit = 1.0
                            elif abs_err_val / price_eval_val <= magnitude_tol:
                                mag_hit = 1.0
                            else:
                                mag_hit = 0.0
                        else:
                            mag_hit = 0.0
                        tmp2[h2][grp2].append(mag_hit)
                    except Exception:
                        continue
                for h2 in hkeys2:
                    for grp2 in ('prod', 'chall'):
                        vals2 = tmp2[h2][grp2]
                        n2 = len(vals2)
                        ab_7d_magnitude[h2][grp2]['n'] = n2
                        ab_7d_magnitude[h2][grp2]['acc'] = (sum(vals2) / n2) if n2 else None
            except Exception:
                ab_7d_magnitude = {}

            # Calibration state from environment
            try:
                _raw = str(_os.getenv('BYPASS_ISOTONIC_CALIBRATION', '1')).strip().lower()
                _bypass = _raw in ('1', 'true', 'yes', 'on')
            except Exception:
                _bypass = True
            try:
                _pf_env = _os.getenv('THRESHOLD_PENALTY_FACTOR', '')
                _penalty = float(_pf_env) if (_pf_env is not None and str(_pf_env).strip() != '') else None
            except Exception:
                _penalty = None
            calibration_state = {
                'bypass': bool(_bypass),
                'status': ('bypass' if _bypass else 'active'),
                'penalty_factor': _penalty,
            }

            return jsonify({
                'param_store': pstore,
                'metrics_7d': metrics_7d,
                'overall_acc_7d': overall_acc_7d,
                'metrics_30d': metrics_30d,
                'overall_acc_30d': overall_acc_30d,
                'metrics_90d': metrics_90d,  # ⚡ NEW: 90-day metrics
                'overall_acc_90d': overall_acc_90d,  # ⚡ NEW: 90-day overall accuracy
                'ab_7d': ab_7d,
                'ab_7d_magnitude': ab_7d_magnitude,  # ⚡ NEW: Magnitude-based A/B test metrics
                'calibration_state': calibration_state,
                'threshold_config': {  # ⚡ NEW: Threshold configuration
                    'direction_hit_threshold': float(_os.getenv('DIRECTION_HIT_THRESHOLD', '0.005')),
                    'magnitude_hit_tolerance': float(_os.getenv('MAGNITUDE_HIT_TOLERANCE', '0.05')),
                },
            })
        except Exception as e:
            logger.error(f"Calibration summary error: {e}")
            return jsonify({'error': 'Failed to load calibration summary'}), 500
    
    @bp.route('/users/management')
    @login_required
    @admin_required
    def user_management():
        """User management data for admin"""
        try:
            # Get all users with statistics
            users = User.query.order_by(User.created_at.desc()).limit(100).all()
            
            user_data = []
            for user in users:
                user_info = AuthManager.get_user_info(user)
                user_data.append({
                    'id': user.id,
                    'email': user.email,
                    'username': user.username,
                    'role': user.role,
                    'provider': user.provider,
                    'is_active': user.is_active,
                    'email_verified': user.email_verified,
                    'created_at': user.created_at.isoformat() if user.created_at else None,
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                    'is_admin': user_info.get('is_admin', False),
                    'is_test_user': user_info.get('is_test_user', False)
                })
            
            # User statistics
            stats = AuthManager.get_user_stats()
            
            return jsonify({
                'users': user_data,
                'statistics': stats,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"User management error: {e}")
            return jsonify({'error': 'Failed to fetch user management data'}), 500
    
    @bp.route('/system_stats')
    def system_stats():
        """System statistics for dashboard Overall Status check"""
        try:
            # Basic system stats for Overall Status
            total_stocks = Stock.query.count()
            total_prices = StockPrice.query.count()
            
            # Check latest data
            latest_date = db.session.query(func.max(StockPrice.date)).scalar()
            latest_count = 0
            if latest_date:
                latest_count = StockPrice.query.filter_by(date=latest_date).count()
            
            # Check data freshness
            days_old = 0
            if latest_date:
                days_old = (datetime.now().date() - latest_date).days
            
            # Overall status calculation
            if total_stocks > 0 and total_prices > 0 and days_old <= 3:
                status = "success"
                overall_status = "healthy"
            elif total_stocks > 0 and total_prices > 0:
                status = "warning" 
                overall_status = "stale_data"
            else:
                status = "error"
                overall_status = "no_data"
            
            # Check automation status
            try:
                from working_automation import get_working_automation_pipeline  # type: ignore
                pipeline = get_working_automation_pipeline()
                if pipeline:
                    automation_status = "running" if pipeline.is_running else "stopped"
                else:
                    automation_status = "error"
            except Exception:
                automation_status = "unknown"
            
            # Database status
            database_status = "healthy" if total_stocks > 0 and total_prices > 0 else "error"
            
            return jsonify({
                'status': status,
                'overall_status': overall_status,
                'total_stocks': total_stocks,
                'total_prices': total_prices,
                'latest_date': latest_date.isoformat() if latest_date else None,
                'latest_count': latest_count,
                'days_old': days_old,
                'automation': automation_status,
                'database': database_status,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"System stats error: {e}")
            return jsonify({
                'status': 'error',
                'overall_status': 'system_error', 
                'error': str(e)
            }), 500

    @bp.route('/system/health')
    @login_required  
    @admin_required
    def system_health():
        """Comprehensive system health check for admin"""
        try:
            health_data = {
                'database': _check_database_health(),
                'automation': _check_automation_health(),
                'data_quality': _check_data_quality(),
                'system_resources': _check_system_resources(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Overall health score
            scores = [v.get('score', 0) for v in health_data.values() if isinstance(v, dict)]
            health_data['overall_score'] = sum(scores) / len(scores) if scores else 0
            health_data['overall_status'] = 'healthy' if health_data['overall_score'] > 80 else 'warning' if health_data['overall_score'] > 60 else 'critical'
            
            return jsonify(health_data)
            
        except Exception as e:
            logger.error(f"System health check error: {e}")
            return jsonify({'error': 'System health check failed'}), 500
    
    def _generate_admin_recommendations(stats, performance):
        """Generate recommendations for admin"""
        recommendations = []
        
        if performance['success_rate'] < 90:
            recommendations.append({
                'type': 'warning',
                'message': 'Automation success rate is below 90%',
                'action': 'Check system logs and error patterns'
            })
        
        if stats.get('pipeline_active') is False:
            recommendations.append({
                'type': 'critical',
                'message': 'Automation pipeline is not active',
                'action': 'Restart automation service'
            })
        
        return recommendations
    
    def _check_database_health():
        """Check database health"""
        try:
            # Simple connection test
            total_stocks = Stock.query.count()
            total_prices = StockPrice.query.count()
            
            return {
                'status': 'healthy',
                'score': 100,
                'stocks': total_stocks,
                'prices': total_prices,
                'connection': 'active'
            }
        except Exception as e:
            return {
                'status': 'error',
                'score': 0,
                'error': str(e)
            }
    
    def _check_automation_health():
        """Check automation system health"""
        try:
            from working_automation import get_working_automation_pipeline  # type: ignore
            pipeline = get_working_automation_pipeline()
            
            if pipeline:
                return {
                    'status': 'healthy',
                    'score': 100,
                    'pipeline_available': True,
                    'running': getattr(pipeline, 'running', False)
                }
            else:
                return {
                    'status': 'warning',
                    'score': 50,
                    'pipeline_available': False
                }
        except Exception:
            return {
                'status': 'error',
                'score': 0,
                'pipeline_available': False
            }
    
    def _check_data_quality():
        """Check data quality metrics"""
        try:
            # Check recent data
            latest_date = db.session.query(func.max(StockPrice.date)).scalar()
            
            if not latest_date:
                return {'status': 'error', 'score': 0, 'message': 'No data available'}
            
            # Check if data is recent (within last 5 days)
            days_old = (datetime.now().date() - latest_date).days
            
            if days_old <= 1:
                score = 100
                status = 'excellent'
            elif days_old <= 3:
                score = 80
                status = 'good'
            elif days_old <= 7:
                score = 60
                status = 'stale'
            else:
                score = 20
                status = 'very_stale'
            
            return {
                'status': status,
                'score': score,
                'latest_date': str(latest_date),
                'days_old': days_old
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'score': 0,
                'error': str(e)
            }
    
    def _check_system_resources():
        """Check basic system resources (CPU/RAM/Disk)."""
        try:
            import psutil
            import os as _os

            # CPU usage
            cpu_percent = float(psutil.cpu_percent(interval=0.1))

            # Memory usage
            vm = psutil.virtual_memory()
            memory_percent = float(getattr(vm, 'percent', 0.0))
            memory_total_mb = float(getattr(vm, 'total', 0.0)) / (1024 * 1024)
            memory_used_mb = float(getattr(vm, 'used', 0.0)) / (1024 * 1024)

            # Disk usage
            du = psutil.disk_usage('/')
            disk_percent = float(getattr(du, 'percent', 0.0))
            disk_free_gb = float(getattr(du, 'free', 0.0)) / (1024 * 1024 * 1024)
            disk_used_gb = float(getattr(du, 'used', 0.0)) / (1024 * 1024 * 1024)

            # Load average (if available)
            try:
                load_avg = None
                if hasattr(_os, 'getloadavg'):
                    la = _os.getloadavg()
                    load_avg = [float(la[0]), float(la[1]), float(la[2])]
            except Exception:
                load_avg = None

            return {
                'status': 'healthy',
                'score': 90,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_total_mb': memory_total_mb,
                'memory_used_mb': memory_used_mb,
                'disk_percent': disk_percent,
                'disk_free_gb': disk_free_gb,
                'disk_used_gb': disk_used_gb,
                'load_avg': load_avg,
            }

        except ImportError:
            return {
                'status': 'info',
                'score': 75,
                'message': 'psutil not available for detailed metrics'
            }
        except Exception as e:
            return {
                'status': 'error',
                'score': 50,
                'error': str(e)
            }
    
    # Register blueprint with app
    app.register_blueprint(bp)
