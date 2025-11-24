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
   */
  connect() {
    if (this.socket && this.socket.connected) {
      console.warn('WebSocket already connected');
      return;
    }

    logDebug('Initializing WebSocket connection...');

    this.socket = io({
      path: WS_CONFIG.PATH,
      transports: WS_CONFIG.TRANSPORTS,
      upgrade: true,
      withCredentials: true,
      reconnection: true,
      reconnectionDelay: WS_CONFIG.RECONNECTION_DELAY,
      reconnectionAttempts: WS_CONFIG.RECONNECTION_ATTEMPTS,
      timeout: WS_CONFIG.TIMEOUT,
      forceNew: false  // ✅ FIX: Don't force new connection, reuse existing if available
    });

    this._attachEventListeners();
    
    return this;
  }

  /**
   * Attach WebSocket event listeners
   */
  _attachEventListeners() {
    this.socket.on(WS_EVENTS.CONNECT, () => this._handleConnect());
    this.socket.on(WS_EVENTS.DISCONNECT, () => this._handleDisconnect());
    this.socket.on(WS_EVENTS.CONNECT_ERROR, (err) => this._handleConnectError(err));
    this.socket.on(WS_EVENTS.ROOM_JOINED, (data) => this._handleRoomJoined(data));
    this.socket.on(WS_EVENTS.PATTERN_ANALYSIS, (data) => this._handlePatternAnalysis(data));
    this.socket.on(WS_EVENTS.USER_SIGNAL, (data) => this._handleUserSignal(data));
    this.socket.on(WS_EVENTS.SUBSCRIPTION_CONFIRMED, (data) => this._handleSubscriptionConfirmed(data));
    this.socket.on(WS_EVENTS.ERROR, (data) => this._handleError(data));
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

  _handleDisconnect() {
    this.isConnected = false;
    logDebug('WebSocket disconnected');
    
    this._emit('connection_status', { connected: false });
    this._throttledNotify('error', '❌ Bağlantı kesildi');
  }

  _handleConnectError(error) {
    console.error('WebSocket connect error:', error);
    this._throttledNotify('error', '❌ WebSocket bağlanma hatası');
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
      this.lastPatternTimestamp = Math.floor(Date.now() / 1000);
      this.completedSymbols.add(data.symbol);
      
      logDebug(`Pattern analysis received: ${data.symbol}`);
      this._emit('pattern_analysis', data);
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
      this.subscribedSymbols.add(data.symbol);
      logDebug(`Subscribed to ${data.symbol}`);
    }
    this._emit('notification', { 
      type: 'success', 
      message: `📈 ${data.symbol} takibe eklendi` 
    });
  }

  _handleError(data) {
    this._emit('notification', { 
      type: 'error', 
      message: `❌ ${data.message}` 
    });
  }

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
    
    const sym = symbol.toUpperCase();
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
   */
  reconnect() {
    this.disconnect();
    setTimeout(() => this.connect(), 1000);
  }
}

/**
 * Singleton instance
 */
let wsInstance = null;

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

