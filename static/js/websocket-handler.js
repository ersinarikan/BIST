/**
 * User Dashboard WebSocket Handler
 * WebSocket bağlantı yönetimi ve event handling
 */

import { WS_CONFIG, WS_EVENTS, CACHE } from './constants.js';
import { logDebug } from './utils.js';

/**
 * WebSocket Handler Class
 */
class WebSocketHandler {
  constructor(userId) {
    this.socket = null;
    this.isConnected = false;
    this.userId = userId || this._generateUserId();
    this.wsRequestedOnce = false;
    this.subscribedSymbols = new Set();
    this.eventHandlers = new Map();
    
    // Statistics
    this.lastPatternTimestamp = 0;
    this.completedSymbols = new Set();
  }

  /**
   * Generate random user ID if not provided
   */
  _generateUserId() {
    return Math.random().toString(36).substr(2, 9);
  }

  /**
   * Initialize WebSocket connection
   * ✅ FIX: Prevent duplicate connections and properly reuse socket
   */
  connect() {
    // ✅ FIX: If already connected, don't create a new connection
    if (this.socket && this.socket.connected) {
      logDebug('WebSocket already connected');
      return this;
    }

    // ✅ FIX: If socket exists but disconnected, try to reconnect instead of creating new
    if (this.socket && !this.socket.connected) {
      logDebug('Reconnecting existing socket...');
      try {
        this.socket.connect();
        return this;
      } catch (e) {
        logDebug('Error reconnecting socket, will create new:', e);
      }
    }

    // ✅ Only create new socket if none exists
    if (!this.socket) {
      logDebug('Initializing new WebSocket connection...');

      this.socket = io({
        path: WS_CONFIG.PATH,
        transports: WS_CONFIG.TRANSPORTS,
        upgrade: true,
        withCredentials: true,
        reconnection: true,
        reconnectionDelay: WS_CONFIG.RECONNECTION_DELAY,
        reconnectionDelayMax: WS_CONFIG.RECONNECTION_DELAY_MAX,
        reconnectionAttempts: WS_CONFIG.RECONNECTION_ATTEMPTS,
        timeout: WS_CONFIG.TIMEOUT,
        randomizationFactor: WS_CONFIG.RANDOMIZATION_FACTOR,
        pingTimeout: WS_CONFIG.PING_TIMEOUT,
        pingInterval: WS_CONFIG.PING_INTERVAL,
        forceNew: false,
        autoConnect: true,
        multiplex: true  // ✅ FIX: Enable multiplexing to reuse connections
      });

      // ✅ Attach event listeners only once when creating new socket
      this._attachEventListeners();
      
      // ✅ CRITICAL FIX: Intercept and filter pattern_analysis messages at protocol level
      // This prevents them from being processed even if server sends them
      if (this.socket.io && this.socket.io.engine) {
        const originalOnPacket = this.socket.io.engine.onPacket;
        this.socket.io.engine.onPacket = (packet) => {
          try {
            // Socket.IO packet types: 0=OPEN, 1=CLOSE, 2=PING, 3=PONG, 4=MESSAGE, 5=UPGRADE, 6=NOOP
            if (packet.type === 4 && packet.data) {
              // Parse Socket.IO message format: ["event_name", {...data}]
              let parsed = null;
              try {
                parsed = typeof packet.data === 'string' ? JSON.parse(packet.data) : packet.data;
              } catch (e) {
                // If parse fails, let it through (will be handled by Socket.IO)
                return originalOnPacket.call(this.socket.io.engine, packet);
              }
              
              // Check if this is a pattern_analysis event
              if (Array.isArray(parsed) && parsed[0] === 'pattern_analysis') {
                logDebug('🚫 Blocked pattern_analysis message at protocol level');
                return; // Drop the packet completely
              }
            }
          } catch (e) {
            // If anything fails, let packet through
          }
          // Pass packet to original handler
          return originalOnPacket.call(this.socket.io.engine, packet);
        };
      }
    }
    
    return this;
  }

  /**
   * Attach WebSocket event listeners
   */
  _attachEventListeners() {
    this.socket.on(WS_EVENTS.CONNECT, () => this._handleConnect());
    this.socket.on(WS_EVENTS.DISCONNECT, (reason) => this._handleDisconnect(reason));
    this.socket.on(WS_EVENTS.CONNECT_ERROR, (err) => this._handleConnectError(err));
    this.socket.on(WS_EVENTS.ROOM_JOINED, (data) => this._handleRoomJoined(data));
    // ✅ FIX: Disabled pattern_analysis event listener - client reads from batch API cache
    // this.socket.on(WS_EVENTS.PATTERN_ANALYSIS, (data) => this._handlePatternAnalysis(data));
    this.socket.on(WS_EVENTS.USER_SIGNAL, (data) => this._handleUserSignal(data));
    this.socket.on(WS_EVENTS.SUBSCRIPTION_CONFIRMED, (data) => this._handleSubscriptionConfirmed(data));
    this.socket.on(WS_EVENTS.ERROR, (data) => this._handleError(data));
    
    // ✅ DEBUG: Add ping/pong event listeners for debugging
    // Note: Socket.IO handles ping/pong internally, but we can monitor connection health
    if (this.socket.io && this.socket.io.engine) {
      this.socket.io.engine.on('ping', () => {
        logDebug('📡 WebSocket ping sent to server');
      });
      this.socket.io.engine.on('pong', () => {
        logDebug('📡 WebSocket pong received from server');
      });
    }
    
    // ✅ DEBUG: Monitor all incoming events to catch parse errors
    // Use onAny to catch all events and identify which one causes parse error
    if (typeof this.socket.onAny === 'function') {
      let lastEventBeforeParseError = null;
      let lastEventTime = null;
      
      this.socket.onAny((event, ...args) => {
        // ✅ FIX: Ignore pattern_analysis events completely - they're for admin only
        // Client reads from batch API cache, doesn't need these events
        if (event === 'pattern_analysis') {
          // Silently ignore - these events shouldn't be received by user clients
          // but if they are, we ignore them to prevent parse errors and UI issues
          return;
        }
        
        // Track last event before potential parse error
        lastEventBeforeParseError = event;
        lastEventTime = new Date().toISOString();
        
        try {
          // Try to stringify to catch JSON parse errors
          if (args.length > 0) {
            const stringified = JSON.stringify(args);
            logDebug(`📥 Event received: "${event}" (${stringified.length} bytes)`);
          } else {
            logDebug(`📥 Event received: "${event}" (no args)`);
          }
        } catch (e) {
          console.error(`❌ Parse error detected in event "${event}":`, e);
          logDebug(`Parse error in event "${event}": ${e.message}, args count: ${args.length}, first arg type: ${args[0] ? typeof args[0] : 'null'}`);
          // Try to get more info about the problematic data
          try {
            const partial = JSON.stringify(args[0], (key, value) => {
              if (typeof value === 'number' && (isNaN(value) || !isFinite(value))) {
                return `[${typeof value}: ${value}]`;
              }
              if (typeof value === 'object' && value !== null) {
                // Limit object depth
                if (Object.keys(value).length > 10) {
                  return `[Object with ${Object.keys(value).length} keys]`;
                }
              }
              return value;
            }, 2);
            logDebug(`Partial data (first 1000 chars): ${partial.substring(0, 1000)}`);
          } catch (e2) {
            logDebug(`Could not stringify even partially: ${e2.message}`);
          }
        }
      });
      
      // Track disconnect events to correlate with parse errors
      // Note: Parse errors occur at Socket.IO protocol level, before event handlers
      // So we can't catch them in onAny, but we can track what happened before
      this.socket.on('disconnect', (reason) => {
        if (reason === 'parse error') {
          const timeSinceLastEvent = lastEventTime ? 
            Math.round((Date.now() - new Date(lastEventTime).getTime()) / 1000) : 'unknown';
          console.error(`❌ Parse error disconnect - Last event before error: "${lastEventBeforeParseError}" at ${lastEventTime} (${timeSinceLastEvent}s ago)`);
          logDebug(`Parse error disconnect correlation: lastEvent="${lastEventBeforeParseError}", time="${lastEventTime}", secondsAgo="${timeSinceLastEvent}"`);
          // Parse errors at protocol level usually mean:
          // 1. Server sent malformed JSON
          // 2. Server sent binary data when JSON expected
          // 3. Network corruption
          // 4. Encoding issues
          logDebug('Parse error likely caused by protocol-level issue (not event-level). Check server-side emit calls for malformed data.');
        }
      });
    }
  }

  /**
   * Internal event handlers
   */
  
  _handleConnect() {
    this.isConnected = true;
    logDebug('WebSocket connected:', this.socket.id);
    
    this._emit('connection_status', { connected: true });
    this._emit('notification', { 
      type: 'success', 
      message: '🔗 Sistem bağlantısı kuruldu' 
    });

    // Join user room
    this.joinUserRoom();
  }

  _handleDisconnect(reason) {
    this.isConnected = false;
    logDebug(`WebSocket disconnected: ${reason || 'unknown reason'}`);
    
    this._emit('connection_status', { connected: false });
    
    // ✅ FIX: Only show error notification for unexpected disconnects
    // Socket.IO will automatically reconnect, so don't spam user with notifications
    if (reason === 'io server disconnect') {
      // Server initiated disconnect - show error
      this._throttledNotify('error', '❌ Sunucu bağlantıyı kapattı');
    } else if (reason === 'io client disconnect') {
      // Client initiated disconnect - no notification needed
      logDebug('Client initiated disconnect');
    } else if (reason === 'parse error') {
      // Parse error - usually temporary, Socket.IO will auto-reconnect
      logDebug('Parse error detected - this may be due to malformed server message. Socket.IO will auto-reconnect.');
      // Don't show notification - it's usually temporary and auto-reconnects
    } else if (reason === 'transport close' || reason === 'transport error') {
      // Transport error - network issue, Socket.IO will auto-reconnect
      logDebug('Transport error, Socket.IO will auto-reconnect');
    } else {
      // Network error or other - Socket.IO will auto-reconnect
      logDebug(`Network disconnect (${reason}), Socket.IO will auto-reconnect`);
    }
  }

  _handleConnectError(error) {
    console.error('WebSocket connect error:', error);
    logDebug(`WebSocket connect error details: ${JSON.stringify(error)}`);
    this._throttledNotify('error', '❌ WebSocket bağlanma hatası');
  }
  
  _handleError(data) {
    console.error('WebSocket error event:', data);
    logDebug(`WebSocket error event details: ${JSON.stringify(data)}`);
    // Don't show notification for parse errors - Socket.IO will auto-reconnect
    if (data && data.message && !data.message.includes('parse')) {
      this._throttledNotify('error', `❌ WebSocket hatası: ${data.message}`);
    }
  }

  _handleRoomJoined(data) {
    logDebug('Joined room:', data.room);
    this._emit('room_joined', data);
    
    // First connection - subscribe to all watched stocks
    if (!this.wsRequestedOnce) {
      this._emit('initial_connection', { userId: this.userId });
      this.wsRequestedOnce = true;
    }
  }

  _handlePatternAnalysis(data) {
    if (!data || !data.symbol) return;
    
    try {
      const symbol = (data.symbol || '').replace(/\uFEFF/g, '').toUpperCase().trim();
      // Ignore symbols that do not belong to this user's watchlist
      if (!this.subscribedSymbols.has(symbol)) {
        return;
      }
      
      // ✅ FIX: Only process if we have a valid socket connection
      if (!this.socket || !this.socket.connected) {
        logDebug(`Skipping pattern analysis for ${symbol} (not connected)`);
        return;
      }
      
      this.lastPatternTimestamp = Math.floor(Date.now() / 1000);
      this.completedSymbols.add(symbol);
      
      logDebug(`Pattern analysis received: ${symbol}`);
      this._emit('pattern_analysis', { ...data, symbol });
    } catch (e) {
      console.error('Pattern analysis handler error:', e);
    }
  }

  _handleUserSignal(data) {
    logDebug('Live signal received:', data);
    this._emit('live_signal', data);
  }

  _handleSubscriptionConfirmed(data) {
    if (data.symbol) {
      const symbol = data.symbol.replace(/\uFEFF/g, '').toUpperCase().trim();
      this.subscribedSymbols.add(symbol);
      logDebug(`Subscribed to ${symbol}`);
    }
    this._emit('notification', { 
      type: 'success', 
      message: `📈 ${data.symbol} takibe eklendi` 
    });
  }

  // ✅ FIX: Removed duplicate _handleError - using the one defined earlier (line 148)

  /**
   * Public WebSocket operations
   */
  
  joinUserRoom() {
    if (!this.socket || !this.socket.connected) {
      console.warn('Cannot join room: not connected');
      return;
    }
    this.socket.emit(WS_EVENTS.JOIN_USER, { user_id: this.userId });
  }

  subscribeToStock(symbol) {
    if (!this.socket || !this.socket.connected) {
      console.warn(`Cannot subscribe to ${symbol}: not connected`);
      return;
    }
    
    const sym = String(symbol || '').replace(/\uFEFF/g, '').toUpperCase().trim();
    this.socket.emit(WS_EVENTS.SUBSCRIBE_STOCK, { symbol: sym });
    this.subscribedSymbols.add(sym);
  }

  unsubscribeFromStock(symbol) {
    if (!this.socket || !this.socket.connected) {
      console.warn(`Cannot unsubscribe from ${symbol}: not connected`);
      return;
    }
    
    const sym = symbol.toUpperCase();
    this.socket.emit(WS_EVENTS.UNSUBSCRIBE_STOCK, { symbol: sym });
    this.subscribedSymbols.delete(sym);
  }

  requestPatternAnalysis(symbol) {
    if (!this.socket || !this.socket.connected) {
      console.warn(`Cannot request analysis for ${symbol}: not connected`);
      return;
    }
    
    this.socket.emit(WS_EVENTS.REQUEST_PATTERN_ANALYSIS, { symbol: symbol.toUpperCase() });
  }

  /**
   * Subscribe to multiple stocks at once
   */
  subscribeToMultiple(symbols) {
    if (!Array.isArray(symbols)) return;
    
    // Reset subscribed symbol set to avoid stale subscriptions
    this.subscribedSymbols.clear();
    
    symbols.forEach(symbol => {
      this.subscribeToStock(symbol);
    });
    
    logDebug(`Subscribed to ${symbols.length} stocks`);
  }

  /**
   * Event emitter for custom handlers
   */
  
  on(eventName, handler) {
    if (!this.eventHandlers.has(eventName)) {
      this.eventHandlers.set(eventName, new Set());
    }
    this.eventHandlers.get(eventName).add(handler);
  }

  off(eventName, handler) {
    if (this.eventHandlers.has(eventName)) {
      this.eventHandlers.get(eventName).delete(handler);
    }
  }

  _emit(eventName, data) {
    if (this.eventHandlers.has(eventName)) {
      this.eventHandlers.get(eventName).forEach(handler => {
        try {
          handler(data);
        } catch (e) {
          console.error(`Event handler error [${eventName}]:`, e);
        }
      });
    }
  }

  /**
   * Throttled notification helper
   */
  _throttledNotify(type, message) {
    const now = Date.now();
    if (!this._lastNotifyTime || (now - this._lastNotifyTime) >= CACHE.NOTIFICATION_THROTTLE) {
      this._lastNotifyTime = now;
      this._emit('notification', { type, message });
    }
  }

  /**
   * Get connection status
   */
  getStatus() {
    return {
      connected: this.isConnected,
      socketId: this.socket?.id || null,
      subscribedCount: this.subscribedSymbols.size,
      userId: this.userId
    };
  }

  /**
   * Disconnect
   */
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.isConnected = false;
      this.subscribedSymbols.clear();
    }
  }

  /**
   * Reconnect
   * ✅ FIX: Let Socket.IO handle reconnection automatically, don't force manual reconnect
   */
  reconnect() {
    // Don't manually disconnect - let Socket.IO handle reconnection
    if (this.socket && !this.socket.connected) {
      this.socket.connect();
    } else if (!this.socket) {
      this.connect();
    }
  }
}

/**
 * Singleton instance
 */
let wsInstance = null;

// ✅ FIX: Clean up socket on page unload to prevent zombie connections
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    if (wsInstance && wsInstance.socket) {
      try {
        // Close socket gracefully before page unload
        wsInstance.socket.disconnect();
        wsInstance.socket.close();
      } catch (e) {
        // Ignore errors during cleanup
      }
    }
  });
  
  // ✅ FIX: Also clean up on page visibility change (tab hidden)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
      // When tab becomes hidden, socket will be handled by Socket.IO's reconnection
      // No action needed here
    } else if (document.visibilityState === 'visible') {
      // When tab becomes visible again, check connection
      if (wsInstance && wsInstance.socket && !wsInstance.socket.connected) {
        wsInstance.socket.connect();
      }
    }
  });
}

export function getWebSocket(userId = null) {
  if (!wsInstance) {
    wsInstance = new WebSocketHandler(userId);
  }
  return wsInstance;
}

// Export class for custom instances
export { WebSocketHandler };

// Default export
export default getWebSocket;

