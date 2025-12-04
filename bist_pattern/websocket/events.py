import logging
from datetime import datetime
from flask import current_app, request
from flask_socketio import emit, join_room, leave_room

logger = logging.getLogger(__name__)


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
        # ✅ CRITICAL FIX: Sanitize status data before emitting
        try:
            from bist_pattern.core.broadcaster import _sanitize_json_value
            import json
            status_data = {
                'message': 'Connected to BIST AI System',
                'timestamp': datetime.now().isoformat(),
                'connection_id': str(getattr(request, 'sid', None) or '')[:100]  # type: ignore[attr-defined]
            }
            sanitized_status = _sanitize_json_value(status_data)
            json.dumps(sanitized_status)  # Test serialization
            emit('status', sanitized_status)
        except Exception as e:
            if current_app and current_app.logger:
                current_app.logger.debug(f"Status emit sanitization failed: {e}")
            # Fallback: send minimal status
            try:
                emit('status', {'message': 'Connected', 'timestamp': datetime.now().isoformat()})
            except Exception as e2:
                logger.debug(f"Failed to emit fallback status: {e2}")

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
        # ✅ CRITICAL FIX: Sanitize room_joined data before emitting
        try:
            from bist_pattern.core.broadcaster import _sanitize_json_value
            import json
            room_data = {'room': 'admin', 'message': 'Admin dashboard connected'}
            sanitized_room = _sanitize_json_value(room_data)
            json.dumps(sanitized_room)  # Test serialization
            emit('room_joined', sanitized_room)
        except Exception as e:
            if current_app and current_app.logger:
                current_app.logger.debug(f"Room joined emit sanitization failed: {e}")
            # Fallback: send minimal data
            try:
                emit('room_joined', {'room': 'admin'})
            except Exception as e2:
                logger.debug(f"Failed to emit fallback room_joined (admin): {e2}")

    def _on_join_user(data):
        user_id = data.get('user_id', 'anonymous')
        join_room(f'user_{user_id}')
        if current_app and current_app.logger:
            current_app.logger.info(
                f"👤 Client joined user room: {getattr(request, 'sid', 'n/a')} -> user_{user_id}"
            )
        # ✅ CRITICAL FIX: Sanitize room_joined data before emitting
        try:
            from bist_pattern.core.broadcaster import _sanitize_json_value
            import json
            room_data = {'room': f'user_{user_id}', 'message': 'User interface connected'}
            sanitized_room = _sanitize_json_value(room_data)
            json.dumps(sanitized_room)  # Test serialization
            emit('room_joined', sanitized_room)
        except Exception as e:
            if current_app and current_app.logger:
                current_app.logger.debug(f"Room joined emit sanitization failed: {e}")
            # Fallback: send minimal data
            try:
                emit('room_joined', {'room': f'user_{user_id}'})
            except Exception as e2:
                logger.debug(f"Failed to emit fallback room_joined (user): {e2}")

    def _on_subscribe_stock(data):
        symbol = data.get('symbol', '').upper()
        if symbol:
            join_room(f'stock_{symbol}')
            if current_app and current_app.logger:
                current_app.logger.info(
                    f"📈 Client subscribed to {symbol}: {getattr(request, 'sid', 'n/a')}"
                )
            # ✅ CRITICAL FIX: Sanitize subscription_confirmed data before emitting
            try:
                from bist_pattern.core.broadcaster import _sanitize_json_value
                import json
                sub_data = {'symbol': symbol, 'message': f'Subscribed to {symbol} updates'}
                sanitized_sub = _sanitize_json_value(sub_data)
                json.dumps(sanitized_sub)  # Test serialization
                emit('subscription_confirmed', sanitized_sub)
            except Exception as e:
                if current_app and current_app.logger:
                    current_app.logger.debug(f"Subscription confirmed emit sanitization failed: {e}")
                # Fallback: send minimal data
                try:
                    emit('subscription_confirmed', {'symbol': symbol})
                except Exception as e2:
                    logger.debug(f"Failed to emit fallback subscription_confirmed: {e2}")

    def _on_unsubscribe_stock(data):
        symbol = data.get('symbol', '').upper()
        if symbol:
            leave_room(f'stock_{symbol}')
            if current_app and current_app.logger:
                current_app.logger.info(
                    f"📉 Client unsubscribed from {symbol}: {getattr(request, 'sid', 'n/a')}"
                )
            # ✅ CRITICAL FIX: Sanitize subscription_removed data before emitting
            try:
                from bist_pattern.core.broadcaster import _sanitize_json_value
                import json
                sub_data = {'symbol': symbol, 'message': f'Unsubscribed from {symbol}'}
                sanitized_sub = _sanitize_json_value(sub_data)
                json.dumps(sanitized_sub)  # Test serialization
                emit('subscription_removed', sanitized_sub)
            except Exception as e:
                if current_app and current_app.logger:
                    current_app.logger.debug(f"Subscription removed emit sanitization failed: {e}")
                # Fallback: send minimal data
                try:
                    emit('subscription_removed', {'symbol': symbol})
                except Exception as e2:
                    logger.debug(f"Failed to emit fallback subscription_removed: {e2}")

    def _on_request_pattern_analysis(data):
        # ✅ CRITICAL FIX: DISABLED - User dashboard reads from batch API cache
        # This event handler is no longer needed and was causing unnecessary WebSocket traffic
        symbol = data.get('symbol', '').upper()
        if symbol and current_app and current_app.logger:
            current_app.logger.debug(f"📊 Pattern analysis request ignored for {symbol} - use batch API instead")
        # Do nothing - client should use batch API endpoints instead

    # Register handlers dynamically to the provided socketio instance
    sock.on('connect')(_on_connect)
    sock.on('disconnect')(_on_disconnect)
    sock.on('join_admin')(_on_join_admin)
    sock.on('join_user')(_on_join_user)
    sock.on('subscribe_stock')(_on_subscribe_stock)
    sock.on('unsubscribe_stock')(_on_unsubscribe_stock)
    # ✅ CRITICAL FIX: Disabled request_pattern_analysis handler - use batch API instead
    # sock.on('request_pattern_analysis')(_on_request_pattern_analysis)
