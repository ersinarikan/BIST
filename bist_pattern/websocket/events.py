from datetime import datetime
from flask import current_app, request
from flask_socketio import emit, join_room, leave_room


def register_socketio_events(app):
    sock = getattr(app, 'socketio', None)
    if sock is None:
        # Nothing to register against; fail softly
        if current_app and current_app.logger:
            current_app.logger.warning('socketio instance not found on app; WS events not registered')
        return

    def _on_connect(auth=None):
        if current_app and current_app.logger:
            current_app.logger.info(
                f"🔗 Client connected: {getattr(request, 'sid', 'n/a')}"
            )
        emit('status', {
            'message': 'Connected to BIST AI System',
            'timestamp': datetime.now().isoformat(),
            'connection_id': getattr(request, 'sid', None)  # type: ignore[attr-defined]
        })

    def _on_disconnect():
        if current_app and current_app.logger:
            current_app.logger.info(
                f"❌ Client disconnected: {getattr(request, 'sid', 'n/a')}"
            )

    def _on_join_admin():
        join_room('admin')
        if current_app and current_app.logger:
            current_app.logger.info(
                f"👤 Client joined admin room: {getattr(request, 'sid', 'n/a')}"
            )
        emit('room_joined', {'room': 'admin', 'message': 'Admin dashboard connected'})

    def _on_join_user(data):
        user_id = data.get('user_id', 'anonymous')
        join_room(f'user_{user_id}')
        if current_app and current_app.logger:
            current_app.logger.info(
                f"👤 Client joined user room: {getattr(request, 'sid', 'n/a')} -> user_{user_id}"
            )
        emit('room_joined', {'room': f'user_{user_id}', 'message': 'User interface connected'})

    def _on_subscribe_stock(data):
        symbol = data.get('symbol', '').upper()
        if symbol:
            join_room(f'stock_{symbol}')
            if current_app and current_app.logger:
                current_app.logger.info(
                    f"📈 Client subscribed to {symbol}: {getattr(request, 'sid', 'n/a')}"
                )
            emit('subscription_confirmed', {'symbol': symbol, 'message': f'Subscribed to {symbol} updates'})

    def _on_unsubscribe_stock(data):
        symbol = data.get('symbol', '').upper()
        if symbol:
            leave_room(f'stock_{symbol}')
            if current_app and current_app.logger:
                current_app.logger.info(
                    f"📉 Client unsubscribed from {symbol}: {getattr(request, 'sid', 'n/a')}"
                )
            emit('subscription_removed', {'symbol': symbol, 'message': f'Unsubscribed from {symbol}'})

    def _on_request_pattern_analysis(data):
        symbol = data.get('symbol', '').upper()
        if not symbol:
            return
        try:
            from app import get_pattern_detector  # delayed import
            # Try cached fast path first to reduce recomputation storms on refresh
            try:
                from bist_pattern.core.cache import cache_get as _cache_get, cache_set as _cache_set
            except Exception:
                _cache_get = _cache_set = None  # type: ignore

            cache_key = f"pattern_analysis:{symbol}"
            result = None
            if callable(_cache_get):
                result = _cache_get(cache_key)
            if not result:
                result = get_pattern_detector().analyze_stock(symbol)
                try:
                    if callable(_cache_set):
                        _cache_set(cache_key, result, ttl_seconds=30.0)
                except Exception:
                    pass
            emit('pattern_analysis', {
                'symbol': symbol,
                'data': result,
                'timestamp': datetime.now().isoformat()
            })
            sock.emit('pattern_analysis', {
                'symbol': symbol,
                'data': result,
                'timestamp': datetime.now().isoformat()
            }, to=f'stock_{symbol}')
            if current_app and current_app.logger:
                current_app.logger.info(
                    f"📊 Pattern analysis sent for {symbol} to {getattr(request, 'sid', 'n/a')} and stock room"
                )
        except Exception as e:  # pragma: no cover
            emit('error', {'message': f'Pattern analysis failed for {symbol}: {str(e)}'})
            if current_app and current_app.logger:
                current_app.logger.error(f"Pattern analysis error for {symbol}: {e}")

    # Register handlers dynamically to the provided socketio instance
    sock.on('connect')(_on_connect)
    sock.on('disconnect')(_on_disconnect)
    sock.on('join_admin')(_on_join_admin)
    sock.on('join_user')(_on_join_user)
    sock.on('subscribe_stock')(_on_subscribe_stock)
    sock.on('unsubscribe_stock')(_on_unsubscribe_stock)
    sock.on('request_pattern_analysis')(_on_request_pattern_analysis)
