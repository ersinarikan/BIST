/**
 * User Dashboard Main Application
 * Tüm UI logic ve state management
 */

import { 
  HORIZONS, 
  DEFAULTS, 
  STORAGE_KEYS,
  UI_LIMITS,
  CACHE,
  BADGE_COLORS
} from './constants.js';

import { 
  formatCurrency, 
  formatPercentage,
  calculatePercentChange,
  getSignalLabel,
  translateModelLabel,
  translateSource,
  getThresholdForHorizon,
  isElementVisible,
  debounce,
  getLocalStorage,
  setLocalStorage,
  highlightSearchTerm,
  getPriceChangeClass,
  normalizePredictionValue,
  buildSignalHTML,
  buildBadgeHTML,
  formatTimestamp,
  logDebug
} from './utils.js';

import { getAPI } from './api-client.js';
import { getWebSocket } from './websocket-handler.js';

/**
 * ============================================
 * APPLICATION STATE
 * ============================================
 */
class DashboardState {
  constructor() {
    // Core data
    this.watchedStocks = [];
    this.watchedSet = new Set();
    
    // Caches
    this.analysisBySymbol = {};
    this.predictionsBySymbol = {};
    
    // UI state
    this.selectedStockData = null;
    this.searchAbortController = null;
    
    // Timestamps
    this.lastPredFetchTs = 0;
    
    // Listeners
    this.listeners = new Map();
  }

  /**
   * Subscribe to state changes
   */
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
  }

  /**
   * Emit state change
   */
  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (e) {
          console.error(`State listener error [${event}]:`, e);
        }
      });
    }
  }

  /**
   * Update watchlist
   */
  updateWatchlist(stocks) {
    this.watchedStocks = stocks;
    this.watchedSet = new Set(stocks.map(s => s.symbol.toUpperCase()));
    this.emit('watchlist_updated', stocks);
  }

  /**
   * Update analysis for symbol
   */
  updateAnalysis(symbol, analysis) {
    this.analysisBySymbol[symbol] = analysis;
    this.emit('analysis_updated', { symbol, analysis });
  }

  /**
   * Update predictions for symbol
   */
  updatePredictions(symbol, predictions) {
    this.predictionsBySymbol[symbol] = predictions;
    this.emit('predictions_updated', { symbol, predictions });
  }

  /**
   * Check if symbol is watched
   */
  isWatched(symbol) {
    return this.watchedSet.has(symbol.toUpperCase());
  }

  /**
   * Get current horizon from UI
   */
  getCurrentHorizon() {
    const el = document.getElementById('pred-sort-horizon');
    return el ? el.value : DEFAULTS.HORIZON;
  }

  /**
   * Get current filters from UI
   */
  getCurrentFilters() {
    return {
      horizon: this.getCurrentHorizon(),
      order: document.getElementById('pred-sort-order')?.value || DEFAULTS.SORT_ORDER,
      direction: document.getElementById('pred-dir')?.value || DEFAULTS.DIRECTION_FILTER,
      minAbsChange: parseFloat(document.getElementById('pred-min-abs')?.value || '0') || DEFAULTS.MIN_ABS_CHANGE
    };
  }
}

/**
 * ============================================
 * UI RENDERING
 * ============================================
 */
class UIRenderer {
  constructor(state) {
    this.state = state;
    // ✅ FIX: DOM cache for performance
    this._domCache = new Map();
  }
  
  /**
   * ✅ FIX: Cached DOM query helper
   */
  _getElement(id) {
    if (!this._domCache.has(id)) {
      const el = document.getElementById(id);
      if (el) {
        this._domCache.set(id, el);
      }
      return el;
    }
    return this._domCache.get(id);
  }

  /**
   * Render watchlist cards
   */
  renderWatchlist() {
    // ✅ FIX: Use cached DOM query
    const container = this._getElement('watchlist');
    if (!container) return;

    const stocks = this.state.watchedStocks;

    if (stocks.length === 0) {
      container.innerHTML = `
        <div class="text-center text-muted py-4">
          <i class="fas fa-chart-bar fa-3x mb-3"></i>
          <p>Henüz takip edilen hisse bulunmuyor.<br>Hisse eklemek için yukarıdaki butonu kullanın.</p>
        </div>
      `;
      return;
    }

    // Preserve existing prediction content
    const existingPreds = {};
    stocks.forEach(s => {
      const el = document.getElementById(`pred-${s.symbol}`);
      if (el && el.innerHTML && !el.innerHTML.includes('yükleniyor')) {
        existingPreds[s.symbol] = el.innerHTML;
      }
    });

    container.innerHTML = stocks.map(stock => {
      const predContent = existingPreds[stock.symbol] || 'Tahminler yükleniyor...';
      const alertTypeLabel = stock.alertType === 'all' ? 'Tümü' : 
                             stock.alertType === 'buy' ? 'Sadece Alım' : 'Sadece Satış';
      
      return `
        <div class="stock-item" id="stock-${stock.symbol}">
          <div class="d-flex justify-content-between align-items-center">
            <div>
              <h6 class="mb-1">${stock.symbol} <span class="text-muted small" id="name-${stock.symbol}"></span></h6>
              <small class="text-muted">Sinyal türü: ${alertTypeLabel}</small>
            </div>
            <div class="text-end">
              <div id="price-${stock.symbol}" class="text-muted">Fiyat yükleniyor...</div>
              <div id="signal-${stock.symbol}" class="text-muted">Sinyal bekleniyor...</div>
              <div id="meta-${stock.symbol}" class="small text-muted"></div>
              <div id="ts-${stock.symbol}" class="small text-muted"></div>
              <button class="btn btn-sm btn-outline-danger mt-1" onclick="window.dashboard.removeStock('${stock.symbol}')">
                <i class="fas fa-trash"></i>
              </button>
              <button class="btn btn-sm btn-outline-primary mt-1" onclick="window.dashboard.openDetailModal('${stock.symbol}')">
                <i class="fas fa-list"></i> Detay
              </button>
            </div>
          </div>
          <div id="pred-${stock.symbol}" class="mt-0 small text-muted">
            ${predContent}
          </div>
          <div id="patt-${stock.symbol}" class="mt-2"></div>
        </div>
      `;
    }).join('');
  }

  /**
   * Update signal display for a symbol
   */
  updateSignal(symbol, signalData) {
    const signalEl = document.getElementById(`signal-${symbol}`);
    if (!signalEl) return;

    // ✅ FIX: Handle null signalData (no prediction available)
    if (!signalData) {
      signalEl.innerHTML = '<span class="text-muted">Sinyal bekleniyor...</span>';
      return;
    }

    const horizon = this.state.getCurrentHorizon();
    const confidence = Math.round((signalData.confidence || 0) * 100);
    const delta = signalData.delta || 0;

    const signal = getSignalLabel(confidence, delta, horizon);
    // ✅ FIX: Açıklayıcı title - Model güveni ve horizon belirtildi
    const horizonLabel = horizon.toUpperCase().replace('D', 'G');  // 7d -> 7G
    const deltaPct = formatPercentage(delta * 100);
    const title = `${horizonLabel} Model Güveni: %${confidence} (Tahmin: ${deltaPct})`;
    
    signalEl.innerHTML = buildSignalHTML(
      signal.label, 
      confidence, 
      signal.icon, 
      signal.cssClass,
      title
    );
  }

  /**
   * Update price display for a symbol
   */
  updatePrice(symbol, price) {
    const priceEl = document.getElementById(`price-${symbol}`);
    if (priceEl && typeof price === 'number') {
      priceEl.textContent = formatCurrency(price);
    }
  }

  /**
   * Update prediction display for a symbol
   */
  updatePredictions(symbol, predData) {
    const el = document.getElementById(`pred-${symbol}`);
    if (!el) return;

    const { predictions, current_price, model, confidences } = predData;
    const horizon = this.state.getCurrentHorizon();

    // Build prediction HTML
    const horizonKeys = ['1d', '3d', '7d', '14d', '30d'];
    const predItems = horizonKeys.map(h => {
      const value = normalizePredictionValue(predictions[h]);
      return `<span class="me-3">${h.toUpperCase().replace('D', 'G')}: <strong>${formatCurrency(value)}</strong></span>`;
    }).join('');

    // Calculate selected horizon change
    const selectedValue = normalizePredictionValue(predictions[horizon]);
    const changePct = calculatePercentChange(current_price, selectedValue);
    const changeClass = getPriceChangeClass(changePct);
    const changeText = formatPercentage(changePct);

    // ✅ FIX: Determine best model for selected horizon - always show "En iyi" if model available
    const analysis = this.state.analysisBySymbol[symbol];
    let bestModelHTML = '';
    
    try {
      const hu = analysis?.ml_unified?.[horizon];
      // ✅ FIX: Always determine best model for consistency
      let bestTag = null;
      
      if (hu) {
        // Use best field if available
        bestTag = hu.best;
        
        // If best not set, determine based on availability and quality
        if (!bestTag) {
          if (hu.enhanced && hu.basic) {
            // Both exist, compare confidence/reliability
            const enhConf = hu.enhanced.confidence || hu.enhanced.reliability || 0;
            const basConf = hu.basic.confidence || hu.basic.reliability || 0;
            bestTag = enhConf >= basConf ? 'enhanced' : 'basic';
          } else {
            // Only one exists
            bestTag = hu.enhanced ? 'enhanced' : (hu.basic ? 'basic' : null);
          }
        }
      }
      
      // ✅ FIX: If no bestTag from ml_unified, check models_by_horizon
      if (!bestTag) {
        const predData = this.state.predictionsBySymbol[symbol];
        const modelsByHorizon = predData?.models_by_horizon || {};
        const horizonModel = modelsByHorizon[horizon];
        
        // Use horizon-specific model if available, otherwise fallback to overall model
        bestTag = horizonModel || model;
      }
      
      // ✅ FIX: Always show "En iyi: ..." if we have a model (consistency)
      if (bestTag) {
        const badgeColor = bestTag === 'enhanced' ? 'warning text-dark' : 'info';
        bestModelHTML = buildBadgeHTML(
          `En iyi: ${translateModelLabel(bestTag)}`,
          badgeColor.split(' ')[0],
          badgeColor.split(' ')[1] || '',
          ''
        );
      }
    } catch (e) {
      logDebug('Best model HTML error:', e);
    }

    // ✅ FIX: İki satıra böl - "En iyi" badge'ı her zaman aynı hizada olsun
    el.innerHTML = `
      <div class="mb-1">${predItems}</div>
      <div class="d-flex align-items-center">
        <span class="${changeClass}">Seçili ufuk ${horizon.toUpperCase()}: ${changeText}</span>
        ${bestModelHTML ? `<span class="ms-2">${bestModelHTML}</span>` : ''}
      </div>
    `;
  }

  /**
   * Update pattern badges for a symbol
   */
  updatePatterns(symbol, analysis) {
    const pattEl = document.getElementById(`patt-${symbol}`);
    if (!pattEl || !analysis) return;

    const horizon = this.state.getCurrentHorizon();
    const maxShow = UI_LIMITS.MAX_PATTERN_BADGES;

    // ✅ FIX: Debug log to check patterns
    const patterns = Array.isArray(analysis.patterns) ? analysis.patterns : [];
    const fingptPatterns = patterns.filter(p => (p.source || '').toUpperCase() === 'FINGPT');
    if (fingptPatterns.length > 0) {
      logDebug(`${symbol}: updatePatterns - FINGPT pattern found:`, fingptPatterns[0]);
    }

    // ML unified badges
    const mlBadges = this._buildMLBadges(analysis, horizon);
    
    // Technical/Visual pattern badges
    const otherBadges = this._buildPatternBadges(analysis, maxShow - mlBadges.length);

    let html = mlBadges.join('') + otherBadges;

    // Fallback: show model from predictions if no ML unified
    if (!mlBadges.length) {
      const predEntry = this.state.predictionsBySymbol[symbol];
      const model = predEntry?.model;
      if (model) {
        const tag = translateModelLabel(model);
        const color = model === 'enhanced' ? 'warning text-dark' : 'info';
        html = buildBadgeHTML(`${tag} ${horizon.toUpperCase()}`, color.split(' ')[0], 'me-1 mb-1') + html;
      }
    }

    // If no badges at all, show "no patterns" message
    if (!html) {
      const staleText = analysis.stale || analysis.stale_seconds > 0 ? ' • (önbellek)' : '';
      html = `<span class="text-muted small">Formasyon yok${staleText}</span>`;
    }

    pattEl.innerHTML = html;
  }

  /**
   * Build ML unified badges for specific horizon
   * ✅ FIX: Renk mantığı - delta_pct'e göre renk (yükseliş=yeşil, düşüş=kırmızı)
   */
  _buildMLBadges(analysis, horizon) {
    const badges = [];
    const uni = analysis.ml_unified || {};
    const hu = uni[horizon];
    
    if (!hu || (!hu.basic && !hu.enhanced)) {
      return badges;
    }

    const best = (hu.best || '').toLowerCase();
    const sources = [];

    // Basic model
    if (hu.basic) {
      const conf = Math.round((hu.basic.confidence || hu.basic.conf || 0) * 100);
      const deltaPct = hu.basic.delta_pct;
      const isBest = best === 'basic';
      
      // ✅ FIX: delta_pct'e göre renk belirle
      let color = 'info';  // Default (mavi)
      if (typeof deltaPct === 'number') {
        if (deltaPct > 0) {
          color = 'success';  // Yeşil - yükseliş
        } else if (deltaPct < 0) {
          color = 'danger';   // Kırmızı - düşüş
        }
      }
      
      const html = buildBadgeHTML(
        `Temel ${horizon.toUpperCase()}`,
        color,
        `me-1 mb-1 ${isBest ? 'fw-bold' : ''}`,
        `Temel • Güven %${conf}${typeof deltaPct === 'number' ? ` • ${deltaPct >= 0 ? '+' : ''}${(deltaPct * 100).toFixed(1)}%` : ''}`
      );
      sources.push({ key: 'basic', html });
    }

    // Enhanced model
    if (hu.enhanced) {
      const conf = Math.round((hu.enhanced.confidence || hu.enhanced.conf || 0) * 100);
      const deltaPct = hu.enhanced.delta_pct;
      const isBest = best === 'enhanced';
      
      // ✅ FIX: delta_pct'e göre renk belirle
      let color = 'warning';  // Default (sarı)
      if (typeof deltaPct === 'number') {
        if (deltaPct > 0) {
          color = 'success';  // Yeşil - yükseliş
        } else if (deltaPct < 0) {
          color = 'danger';   // Kırmızı - düşüş
        }
      }
      
      const html = buildBadgeHTML(
        `Gelişmiş ${horizon.toUpperCase()}`,
        color,
        `text-dark me-1 mb-1 ${isBest ? 'fw-bold' : ''}`,
        `Gelişmiş • Güven %${conf}${typeof deltaPct === 'number' ? ` • ${deltaPct >= 0 ? '+' : ''}${(deltaPct * 100).toFixed(1)}%` : ''}`
      );
      sources.push({ key: 'enhanced', html });
    }

    // Sort: best first
    sources.sort((a, b) => (a.key === best ? -1 : b.key === best ? 1 : 0));
    
    return sources.map(s => s.html);
  }

  /**
   * Build technical/visual pattern badges
   * ✅ FIX: Renk mantığı - Signal'a göre renk (BULLISH=yeşil, BEARISH=kırmızı)
   */
  _buildPatternBadges(analysis, maxCount) {
    const patterns = Array.isArray(analysis.patterns) ? analysis.patterns : [];
    
    // ✅ FIX: Debug log to check FINGPT patterns
    const fingptPatterns = patterns.filter(p => (p.source || '').toUpperCase() === 'FINGPT');
    if (fingptPatterns.length > 0) {
      logDebug('_buildPatternBadges - FINGPT patterns found:', fingptPatterns.length);
    }
    
    return patterns
      .filter(p => {
        const src = (p.source || '').toUpperCase();
        // Exclude ML patterns (they're shown in ML unified section)
        // ✅ FIX: FINGPT patterns should be included
        return src !== 'ML_PREDICTOR' && src !== 'ENHANCED_ML';
      })
      .slice(0, Math.max(0, maxCount))
      .map(p => {
        const src = (p.source || '').toUpperCase();
        const conf = Math.round((p.confidence || 0) * 100);
        
        // ✅ FIX: For FINGPT patterns, show "Sezgisel" instead of pattern name
        let name;
        if (src === 'FINGPT') {
          const signal = (p.signal || '').toUpperCase();
          const signalLabel = signal === 'BULLISH' ? 'Yükseliş' : signal === 'BEARISH' ? 'Düşüş' : 'Nötr';
          name = `Sezgisel (${signalLabel})`;
        } else {
          name = typeof window.translatePattern === 'function' 
            ? window.translatePattern(p.pattern || '') 
            : (p.pattern || '').toString().replace(/_/g, ' ');
        }
        
        // ✅ FIX: Signal'a göre renk belirle (öncelikli)
        const signal = (p.signal || '').toUpperCase();
        let color;
        if (signal === 'BULLISH') {
          color = 'success';  // Yeşil - yükseliş trendi
        } else if (signal === 'BEARISH') {
          color = 'danger';   // Kırmızı - düşüş trendi
        } else if (signal === 'NEUTRAL') {
          color = 'secondary'; // Gri - nötr
        } else {
          // Fallback: Source'a göre renk (eski mantık)
          color = BADGE_COLORS[src] || BADGE_COLORS.default;
        }
        
        return buildBadgeHTML(
          name,
          color,
          'me-1 mb-1',
          `${translateSource(src)} • %${conf} • ${signal || 'N/A'}`
        );
      })
      .join('');
  }

  /**
   * Update connection status UI
   */
  updateConnectionStatus(connected) {
    const elements = ['connection-status', 'bottom-connection-status', 'system-connection'];
    
    elements.forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;

      if (connected) {
        el.innerHTML = '<i class="fas fa-wifi text-success me-2"></i>Bağlı';
        el.className = el.className.replace(/text-(warning|danger)/g, 'text-success');
      } else {
        el.innerHTML = '<i class="fas fa-wifi text-danger me-2"></i>Bağlantı yok';
        el.className = el.className.replace(/text-(warning|success)/g, 'text-danger');
      }
    });

    // Data stream status
    const dataStreamEl = document.getElementById('data-stream');
    if (dataStreamEl) {
      dataStreamEl.innerHTML = connected 
        ? '<span class="text-success">Aktif</span>' 
        : '<span class="text-danger">Durduruldu</span>';
    }
  }

  /**
   * Show notification
   */
  showNotification(type, message) {
    const container = document.getElementById('notification-container');
    if (!container) return;

    const alertClass = type === 'success' ? 'alert-success' : 
                      type === 'error' ? 'alert-danger' : 
                      type === 'warning' ? 'alert-warning' : 'alert-info';

    const notification = document.createElement('div');
    notification.className = `alert ${alertClass} alert-dismissible fade show`;
    notification.innerHTML = `
      ${message}
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    container.appendChild(notification);

    // Auto remove after 5 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.remove();
      }
    }, 5000);
  }

  /**
   * Update last update timestamp
   */
  updateTimestamp() {
    const el = document.getElementById('last-update');
    if (el) {
      el.textContent = new Date().toLocaleTimeString('tr-TR');
    }
  }
}

/**
 * ============================================
 * OPERATION LOCK (Race Condition Prevention)
 * ============================================
 */
class OperationLock {
  constructor() {
    this.locks = new Map();
  }
  
  acquire(key) {
    if (this.locks.has(key)) {
      return false;
    }
    this.locks.set(key, Date.now());
    return true;
  }
  
  release(key) {
    this.locks.delete(key);
  }
  
  isLocked(key) {
    return this.locks.has(key);
  }
}

/**
 * ============================================
 * MAIN DASHBOARD APPLICATION
 * ============================================
 */
class UserDashboard {
  constructor(userId) {
    this.userId = userId;
    this.state = new DashboardState();
    this.ui = new UIRenderer(this.state);
    this.api = getAPI();
    this.ws = getWebSocket(userId);
    
    this.initialized = false;
    
    // ✅ FIX: Operation lock for race condition prevention
    this.operationLock = new OperationLock();
    
    // ✅ FIX: Cleanup tracking
    this._cleanupFunctions = [];
    this._timestampInterval = null;
    this._domCache = new Map();
    
    // ✅ FIX: Track last processed timestamps to prevent duplicate WebSocket updates
    this._lastProcessedTimestamps = new Map(); // symbol -> timestamp
    this._patternUpdateDebounce = new Map(); // symbol -> timeout ID
  }

  /**
   * ✅ FIX: Helper function to build ml_unified from batch predictions
   * Eliminates duplicate code in loadBatchData, openDetailModal, rerenderPredictionsFromCache
   */
  _buildMLUnifiedFromBatchPredictions(predictions, confidences, models_by_horizon, currentPrice, model) {
    const mlUnified = {};
    const horizons = ['1d', '3d', '7d', '14d', '30d'];
    
    horizons.forEach(horizon => {
      const pred = predictions[horizon];
      const conf = confidences && confidences[horizon];
      
      // Determine model for this specific horizon
      let modelToUse = 'basic';
      if (models_by_horizon && models_by_horizon[horizon]) {
        modelToUse = models_by_horizon[horizon];
      } else if (model) {
        modelToUse = model;
      }
      
      if (pred && typeof pred === 'object') {
        const price = pred.price || pred.ensemble_prediction || pred.prediction;
        if (typeof price === 'number' && price > 0 && currentPrice > 0) {
          const deltaPct = (price - currentPrice) / currentPrice;
          mlUnified[horizon] = {
            [modelToUse]: {
              price: price,
              delta_pct: deltaPct,
              confidence: (typeof conf === 'number' ? conf : (pred.confidence || pred.reliability || 0.3))
            },
            best: modelToUse
          };
        }
      } else if (typeof pred === 'number' && pred > 0 && currentPrice > 0) {
        const deltaPct = (pred - currentPrice) / currentPrice;
        mlUnified[horizon] = {
          [modelToUse]: {
            price: pred,
            delta_pct: deltaPct,
            confidence: (typeof conf === 'number' ? conf : 0.3)
          },
          best: modelToUse
        };
      }
    });
    
    return mlUnified;
  }

  /**
   * Initialize application
   */
  async init() {
    if (this.initialized) {
      console.warn('Dashboard already initialized');
      return;
    }

    logDebug('Initializing User Dashboard...');

    // Load watched stocks from server and localStorage
    await this.loadWatchedStocks();

    // Initialize WebSocket
    this._initializeWebSocket();

    // Setup UI event listeners
    this._setupUIListeners();

    // Initialize Bootstrap components
    this._initializeBootstrap();

    // Start timestamp updater
    this._startTimestampUpdater();

    this.initialized = true;
    logDebug('Dashboard initialization complete');
  }

  /**
   * Initialize WebSocket and attach handlers
   */
  _initializeWebSocket() {
    // ✅ FIX: Don't disconnect if already connected - this causes reconnection loops
    if (this.ws && this.ws.socket && this.ws.socket.connected) {
      logDebug('WebSocket already connected, skipping reconnection');
      return;
    }

    // ✅ FIX: Only connect if not already connected
    if (!this.ws || !this.ws.socket || !this.ws.socket.connected) {
      this.ws.connect();
    }

    // Connection status
    this.ws.on('connection_status', (data) => {
      this.ui.updateConnectionStatus(data.connected);
    });

    // Initial connection - subscribe to all stocks
    this.ws.on('initial_connection', async () => {
      await this.subscribeToAllStocks();
    });

    // Pattern analysis updates
    // ✅ FIX: DISABLED - Client already loads data from batch API cache on initial load
    // WebSocket pattern_analysis events are not needed since:
    // 1. Client loads all data from batch API (cache) on page load
    // 2. Automation cycle writes to cache files, client reads from cache
    // 3. WebSocket events cause unnecessary UI updates and race conditions
    // 4. Only live_signal events are needed for real-time updates
    // 
    // If you need real-time pattern updates, use a polling mechanism or
    // only enable this for specific use cases (e.g., admin dashboard)
    /*
    this.ws.on('pattern_analysis', (data) => {
      // Disabled - using batch API cache instead
      logDebug(`Pattern analysis event ignored (using batch API cache): ${data.symbol}`);
    });
    */

    // Live signals
    this.ws.on('live_signal', (data) => {
      this.handleLiveSignal(data);
    });

    // Notifications
    this.ws.on('notification', (data) => {
      this.ui.showNotification(data.type, data.message);
    });

    // ✅ FIX: Cleanup WebSocket on page unload
    if (typeof window !== 'undefined') {
      const cleanup = () => {
        if (this.ws) {
          try {
            this.ws.disconnect();
          } catch (e) {
            logDebug('Error disconnecting WebSocket on unload:', e);
          }
        }
      };
      window.addEventListener('beforeunload', cleanup);
      window.addEventListener('pagehide', cleanup);
    }
  }

  /**
   * Setup UI event listeners
   */
  _setupUIListeners() {
    // Filter controls
    const filterIds = ['pred-sort-horizon', 'pred-sort-order', 'pred-dir', 'pred-min-abs'];
    
    filterIds.forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;

      el.addEventListener('change', () => {
        this.rerenderPredictionsFromCache();
        this.hydratePatterns();
      });

      // Range slider live update
      if (el.type === 'range') {
        el.addEventListener('input', () => {
          document.getElementById('pred-min-abs-val').textContent = el.value;
        });
      }
    });

    logDebug('UI listeners attached');
  }

  /**
   * Initialize Bootstrap tooltips
   */
  _initializeBootstrap() {
    setTimeout(() => {
      const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
      tooltips.forEach(el => {
        try {
          new bootstrap.Tooltip(el);
        } catch (e) {
          // Ignore tooltip errors
        }
      });
      logDebug('Bootstrap tooltips initialized');
    }, 1000);
  }

  /**
   * Start timestamp updater
   * ✅ FIX: Track interval for cleanup
   */
  /**
   * Start timestamp updater
   * ✅ FIX: Track interval for cleanup (memory leak prevention)
   */
  _startTimestampUpdater() {
    // Clear existing interval if any
    if (this._timestampInterval) {
      clearInterval(this._timestampInterval);
    }
    this._timestampInterval = setInterval(() => {
      this.ui.updateTimestamp();
    }, 1000);
  }

  /**
   * Load watched stocks
   */
  async loadWatchedStocks() {
    logDebug('Loading watched stocks...');

    // Try server first
    try {
      const response = await this.api.getWatchlist();
      if (response.status === 'success') {
        const stocks = (response.watchlist || []).map(w => ({
          symbol: w.symbol,
          alertType: 'all', // Could be extracted from w if available
          addedAt: w.created_at
        }));
        this.state.updateWatchlist(stocks);
        setLocalStorage(STORAGE_KEYS.WATCHED_STOCKS, stocks);
        logDebug(`Loaded ${stocks.length} stocks from server`);
      }
    } catch (e) {
      console.warn('Failed to load from server, trying localStorage:', e);
      
      // Fallback to localStorage
      const saved = getLocalStorage(STORAGE_KEYS.WATCHED_STOCKS, []);
      this.state.updateWatchlist(saved);
      logDebug(`Loaded ${saved.length} stocks from localStorage`);
    }

    // Render
    this.ui.renderWatchlist();
  }

  /**
   * Subscribe to all watched stocks
   */
  async subscribeToAllStocks() {
    const symbols = this.state.watchedStocks.map(s => s.symbol);
    
    // Subscribe for WebSocket updates
    this.ws.subscribeToMultiple(symbols);

    // Load batch data
    await this.loadBatchData(symbols);
  }

  /**
   * Load batch data for symbols
   * ✅ FIX: Added race condition prevention and better error handling
   */
  async loadBatchData(symbols) {
    if (symbols.length === 0) return;
    
    // ✅ FIX: Prevent race condition
    const lockKey = 'loadBatchData';
    if (!this.operationLock.acquire(lockKey)) {
      logDebug('loadBatchData already in progress, skipping...');
      return;
    }
    
    try {
      logDebug(`Loading batch data for ${symbols.length} symbols...`);
      // Load patterns and predictions in parallel
      const [patternsResp, predictionsResp] = await Promise.all([
        this.api.getBatchPatternAnalysis(symbols).catch(e => {
          console.warn('Batch patterns failed:', e);
          return { status: 'error', results: {} };
        }),
        this.api.getBatchPredictions(symbols).catch(e => {
          console.warn('Batch predictions failed:', e);
          return { status: 'error', results: {} };
        })
      ]);

      // Process pattern results
      if (patternsResp.status === 'success') {
        const results = patternsResp.results || {};
        Object.entries(results).forEach(([symbol, analysis]) => {
          if (analysis && analysis.status === 'success') {
            // ✅ FIX: Debug log to check if FINGPT patterns are present
            const patterns = analysis.patterns || [];
            const fingptPatterns = patterns.filter(p => (p.source || '').toUpperCase() === 'FINGPT');
            if (fingptPatterns.length > 0) {
              logDebug(`${symbol}: FINGPT pattern found:`, fingptPatterns[0]);
            }
            
            // ✅ FIX: Track timestamp from batch API to prevent stale WebSocket updates
            const timestamp = analysis.timestamp || patternsResp.timestamp;
            if (timestamp) {
              this._lastProcessedTimestamps.set(symbol, timestamp);
            }
            
            this.state.updateAnalysis(symbol, analysis);
            this.handlePatternUpdate({ symbol, data: analysis });
          }
        });
      }

      // Process prediction results
      if (predictionsResp.status === 'success') {
        const results = predictionsResp.results || {};
        Object.entries(results).forEach(([symbol, result]) => {
          if (result && result.predictions) {
            const predData = {
              predictions: result.predictions || {},
              confidences: result.confidences || {},
              current_price: result.current_price,
              model: result.model,
              models_by_horizon: result.models_by_horizon || {},  // ✅ FIX: Include per-horizon model info
              fetched_at: result.source_timestamp
            };
            this.state.updatePredictions(symbol, predData);
            this.ui.updatePredictions(symbol, predData);
            this.ui.updatePrice(symbol, result.current_price);
            
            // ✅ FIX: Use helper function to build ml_unified (eliminates duplicate code)
            let analysis = this.state.analysisBySymbol[symbol] || { symbol };
            const mlUnified = this._buildMLUnifiedFromBatchPredictions(
              result.predictions,
              result.confidences,
              result.models_by_horizon,
              result.current_price || 0,
              result.model
            );
            
            if (Object.keys(mlUnified).length > 0) {
              if (!analysis.ml_unified) {
                analysis.ml_unified = {};
              }
              Object.assign(analysis.ml_unified, mlUnified);
            }
            
            // Update analysis with current_price
            analysis.current_price = result.current_price;
            
            // Update state and trigger signal update
            this.state.updateAnalysis(symbol, analysis);
            this.handlePatternUpdate({ symbol, data: analysis });
          }
        });
      }

      logDebug('Batch data loaded successfully');
      
      // Hydrate patterns from cache
      this.hydratePatterns();
      
    } catch (e) {
      console.error('Batch data load error:', e);
      this.ui.showNotification('error', 'Veri yükleme hatası');
    } finally {
      // ✅ FIX: Always release lock
      this.operationLock.release(lockKey);
    }
  }

  /**
   * Handle pattern analysis update
   */
  handlePatternUpdate(data) {
    // ✅ FIX: Handle both formats - direct data or nested data.data
    const symbol = data.symbol || (data.data && data.data.symbol) || null;
    const incoming = data.data || data;  // Support both { data: {...} } and direct {...}
    
    if (!symbol || !this.state.isWatched(symbol)) return;

    // Merge with existing analysis to preserve ML predictions/confidence
    const existing = this.state.analysisBySymbol[symbol] || {};
    const merged = {
      ...existing,
      ...incoming,
      ml_unified: incoming.ml_unified || existing.ml_unified,
      patterns: Array.isArray(incoming.patterns) ? incoming.patterns : existing.patterns,
      indicators: incoming.indicators || existing.indicators
    };

    // Update price
    const currentPrice = merged.current_price || merged.price || 0;
    this.ui.updatePrice(symbol, currentPrice);

    // Update signal
    const horizon = this.state.getCurrentHorizon();
    const signalData = this._computeHorizonSignal(merged, horizon);
    if (signalData) {
      this.ui.updateSignal(symbol, signalData);
    } else {
      this.ui.updateSignal(symbol, null);
    }

    // Update patterns
    this.ui.updatePatterns(symbol, merged);
    
    // Update state
    this.state.updateAnalysis(symbol, merged);

    // Update timestamp
    this.ui.updateTimestamp();
  }

  /**
   * Compute signal for specific horizon
   */
  _computeHorizonSignal(analysis, horizon) {
    try {
      const current = analysis.current_price || analysis.price || 0;
      if (!current) return null;

      const uni = analysis.ml_unified || {};
      const hu = uni[horizon];
      
      let delta = null;
      let confidence = null;

      // Try ml_unified first - use best model if available
      if (hu) {
        // ✅ FIX: Use best model if specified, otherwise prefer enhanced over basic
        const bestModel = hu.best || (hu.enhanced ? 'enhanced' : (hu.basic ? 'basic' : null));
        const pick = bestModel && hu[bestModel] ? hu[bestModel] : (hu.enhanced || hu.basic);
        
        if (pick && typeof pick.delta_pct === 'number') {
          delta = pick.delta_pct;
          // ✅ FIX: Use confidence or reliability, with appropriate defaults
          confidence = pick.confidence;
          if (confidence === undefined || confidence === null) {
            confidence = pick.reliability;
          }
          if (confidence === undefined || confidence === null) {
            // ✅ FIX: Default confidence based on model type
            // Basic model: 0.4 (40%) - lower than enhanced (0.6-0.7)
            // Enhanced model or unknown: 0.3 (30%) - indicates uncertainty
            if (bestModel === 'basic') {
              confidence = 0.4;
            } else {
              confidence = 0.3;
            }
          }
        }
      }

      // Fallback: enhanced_predictions
      if (delta === null && analysis.enhanced_predictions) {
        const ep = analysis.enhanced_predictions[horizon];
        if (ep && typeof ep.ensemble_prediction === 'number') {
          delta = (ep.ensemble_prediction - current) / current;
          // ✅ FIX: Lower default (0.3) instead of 0.5
          confidence = ep.confidence || 0.3;
        }
      }

      // Fallback: ml_predictions
      if (delta === null && analysis.ml_predictions) {
        const mp = analysis.ml_predictions[horizon];
        if (mp && typeof mp.price === 'number') {
          delta = (mp.price - current) / current;
          // ✅ FIX: Lower default (0.3) for basic model
          confidence = 0.3; // Default for basic (lower confidence)
        }
      }

      if (delta === null || !isFinite(delta)) return null;

      // ✅ FIX: Lower default (0.3) instead of 0.5 if confidence is still missing
      return {
        delta,
        confidence: typeof confidence === 'number' && confidence > 0 ? confidence : 0.3
      };
    } catch (e) {
      logDebug('Compute horizon signal error:', e);
      return null;
    }
  }

  /**
   * Hydrate patterns from cache
   */
  hydratePatterns() {
    this.state.watchedStocks.forEach(stock => {
      const analysis = this.state.analysisBySymbol[stock.symbol];
      if (analysis) {
        this.ui.updatePatterns(stock.symbol, analysis);
      }
    });
  }

  /**
   * Rerender predictions from cache (no network call)
   */
  rerenderPredictionsFromCache() {
    this.state.watchedStocks.forEach(stock => {
      const predData = this.state.predictionsBySymbol[stock.symbol];
      if (predData) {
        this.ui.updatePredictions(stock.symbol, predData);
        
        // Also update signal based on new horizon
        let analysis = this.state.analysisBySymbol[stock.symbol];
        if (analysis) {
          const horizon = this.state.getCurrentHorizon();
          
          // ✅ FIX: Use helper function to rebuild ml_unified if missing (eliminates duplicate code)
          if (!analysis.ml_unified || !analysis.ml_unified[horizon]) {
            const currentPrice = predData.current_price || analysis.current_price || 0;
            if (currentPrice > 0 && predData.predictions) {
              const mlUnified = this._buildMLUnifiedFromBatchPredictions(
                predData.predictions,
                predData.confidences,
                predData.models_by_horizon,
                currentPrice,
                predData.model
              );
              
              if (Object.keys(mlUnified).length > 0) {
                if (!analysis.ml_unified) {
                  analysis.ml_unified = {};
                }
                Object.assign(analysis.ml_unified, mlUnified);
                // Update state with rebuilt ml_unified
                this.state.updateAnalysis(stock.symbol, analysis);
              }
            }
          }
          
          const signalData = this._computeHorizonSignal(analysis, horizon);
          if (signalData) {
            this.ui.updateSignal(stock.symbol, signalData);
          } else {
            // ✅ FIX: If signal is null, clear the signal display
            this.ui.updateSignal(stock.symbol, null);
          }
        }
      }
    });
  }

  /**
   * Handle live signal from WebSocket
   */
  handleLiveSignal(data) {
    const liveSignalsEl = document.getElementById('live-signals');
    if (!liveSignalsEl) return;

    try {
      const signal = data.signal || {};
      const symbol = signal.symbol || 'UNKNOWN';
      const signalType = (signal.overall_signal?.signal || '').toUpperCase();
      const confidence = Math.round((signal.overall_signal?.confidence || 0) * 100);
      const timestamp = formatTimestamp(data.timestamp || Date.now());

      const signalInfo = getSignalLabel(confidence, 
        signalType === 'BULLISH' ? 0.05 : (signalType === 'BEARISH' ? -0.05 : 0));

      const signalHtml = `
        <div class="border-start border-3 border-primary bg-light p-3 mb-3 rounded-end" style="animation: fadeIn 0.5s;">
          <div class="d-flex justify-content-between align-items-start">
            <div>
              <strong class="text-primary">${symbol}</strong>
              <br>
              <span class="${signalInfo.cssClass}">
                <i class="${signalInfo.icon} me-1"></i>${signalInfo.label}
              </span>
              <br>
              <small class="text-muted">Güven: %${confidence}</small>
            </div>
            <small class="text-muted">${timestamp}</small>
          </div>
        </div>
      `;

      // Add to top
      if (liveSignalsEl.innerHTML.includes('Sinyal bekleniyor')) {
        liveSignalsEl.innerHTML = signalHtml;
      } else {
        liveSignalsEl.innerHTML = signalHtml + liveSignalsEl.innerHTML;
        
        // Keep max signals
        const signals = liveSignalsEl.querySelectorAll('.border-start');
        if (signals.length > CACHE.MAX_LIVE_SIGNALS) {
          for (let i = CACHE.MAX_LIVE_SIGNALS; i < signals.length; i++) {
            signals[i].remove();
          }
        }
      }

      this.ui.showNotification('info', `🔔 ${symbol}: ${signalInfo.label}`);
    } catch (e) {
      console.error('Live signal handler error:', e);
    }
  }

  /**
   * Stock search with debounce
   */
  searchStocks = debounce(async (query) => {
    const resultsContainer = document.getElementById('stockSearchResults');
    const resultsList = document.getElementById('searchResultsList');

    if (query.length < UI_LIMITS.SEARCH_MIN_LENGTH) {
      resultsContainer.style.display = 'none';
      return;
    }

    try {
      // Abort previous request
      if (this.state.searchAbortController) {
        this.state.searchAbortController.abort();
      }
      this.state.searchAbortController = new AbortController();

      const data = await this.api.searchStocks(query, UI_LIMITS.MAX_SEARCH_RESULTS);

      if (data.status === 'success' && data.stocks.length > 0) {
        resultsList.innerHTML = data.stocks.map(stock => {
          const escapedSymbol = stock.symbol.replace(/'/g, "\\'");
          const escapedName = (stock.name || '').replace(/'/g, "\\'");
          const escapedSector = (stock.sector || '').replace(/'/g, "\\'");
          
          return `
            <div class="stock-search-item p-2 border-bottom" 
                 style="cursor: pointer;" 
                 onclick="window.dashboard.selectStock('${escapedSymbol}', '${escapedName}', '${escapedSector}')">
              <div class="d-flex justify-content-between">
                <div>
                  <strong>${highlightSearchTerm(stock.symbol, query)}</strong>
                  <div class="small text-muted">${highlightSearchTerm(stock.name, query)}</div>
                </div>
                <div class="text-end">
                  <small class="badge bg-light text-dark">${stock.sector}</small>
                </div>
              </div>
            </div>
          `;
        }).join('');
        
        resultsContainer.style.display = 'block';
      } else {
        resultsList.innerHTML = '<div class="text-muted p-2">Sonuç bulunamadı</div>';
        resultsContainer.style.display = 'block';
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.error('Search error:', error);
        resultsList.innerHTML = '<div class="text-danger p-2">Arama hatası</div>';
        resultsContainer.style.display = 'block';
      }
    }
  }, UI_LIMITS.DEBOUNCE_DELAY);

  /**
   * Select stock from search results
   */
  selectStock(symbol, name, sector) {
    this.state.selectedStockData = { symbol, name, sector };
    document.getElementById('selectedStock').value = `${symbol} - ${name}`;
    document.getElementById('stockSearchResults').style.display = 'none';
    document.getElementById('stockSearch').value = '';
  }

  /**
   * Clear selected stock
   */
  clearSelectedStock() {
    this.state.selectedStockData = null;
    document.getElementById('selectedStock').value = '';
  }

  /**
   * Add stock to watchlist
   */
  async addStock() {
    const symbol = this.state.selectedStockData ? this.state.selectedStockData.symbol :
                   document.getElementById('selectedStock').value.split(' - ')[0] ||
                   document.getElementById('stockSearch').value.toUpperCase().trim();
    
    const alertType = document.getElementById('alertType').value;

    if (!symbol) {
      this.ui.showNotification('error', 'Lütfen hisse kodu girin');
      return;
    }

    if (this.state.watchedStocks.find(s => s.symbol === symbol)) {
      this.ui.showNotification('error', 'Bu hisse zaten takip ediliyor');
      return;
    }

    // Add to server
    try {
      const response = await this.api.addToWatchlist(symbol, {
        alert_enabled: alertType !== 'sell'
      });
      
      if (response.status !== 'success') {
        this.ui.showNotification('error', 'Watchlist kaydı başarısız: ' + (response.error || ''));
        return;
      }
    } catch (e) {
      console.error('Add to watchlist error:', e);
      this.ui.showNotification('error', 'Hisse eklenemedi');
      return;
    }

    // Update local state
    const newStock = { symbol, alertType, addedAt: new Date().toISOString() };
    this.state.watchedStocks.push(newStock);
    this.state.watchedSet.add(symbol);
    setLocalStorage(STORAGE_KEYS.WATCHED_STOCKS, this.state.watchedStocks);

    // Subscribe to WebSocket updates
    this.ws.subscribeToStock(symbol);

    // Render and refresh
    this.ui.renderWatchlist();
    setTimeout(() => this.loadBatchData([symbol]), 100);

    // Close modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('addStockModal'));
    if (modal) modal.hide();

    // Clear form
    document.getElementById('stockSearch').value = '';
    document.getElementById('selectedStock').value = '';
    document.getElementById('alertType').value = 'all';
    document.getElementById('stockSearchResults').style.display = 'none';
    this.state.selectedStockData = null;

    this.ui.showNotification('success', `📈 ${symbol} takibe eklendi`);
  }

  /**
   * Remove stock from watchlist
   */
  async removeStock(symbol) {
    // Remove from server
    try {
      await this.api.removeFromWatchlist(symbol);
    } catch (e) {
      console.warn('Remove from watchlist error:', e);
    }

    // Update local state
    this.state.watchedStocks = this.state.watchedStocks.filter(s => s.symbol !== symbol);
    this.state.watchedSet.delete(symbol);
    setLocalStorage(STORAGE_KEYS.WATCHED_STOCKS, this.state.watchedStocks);

    // Unsubscribe from WebSocket
    this.ws.unsubscribeFromStock(symbol);

    // Re-render
    this.ui.renderWatchlist();
    
    this.ui.showNotification('info', `📉 ${symbol} takipten çıkarıldı`);
  }

  /**
   * Open detail modal for a symbol
   */
  async openDetailModal(symbol) {
    try {
      const title = document.getElementById('detailTitle');
      if (title) title.textContent = `${symbol} Detay`;

      // Load price history, analysis, and ML predictions in parallel
      const [priceData, analysisData, predictionsData] = await Promise.all([
        this.api.getStockPrices(symbol, { days: 60 }).catch(e => {
          console.warn('Price data load failed:', e);
          return { data: [] };
        }),
        // Load analysis from cache or API
        (async () => {
          let analysis = this.state.analysisBySymbol[symbol];
          if (!analysis) {
            try {
              analysis = await this.api.getPatternAnalysis(symbol, { fast: true });
              this.state.updateAnalysis(symbol, analysis);
            } catch (e) {
              console.warn('Pattern analysis load failed:', e);
              analysis = { symbol, status: 'error' };
            }
          }
          return analysis;
        })(),
        // ✅ FIX: Load ML predictions separately from batch API
        this.api.getBatchPredictions([symbol]).catch(e => {
          console.warn('ML predictions load failed:', e);
          return { status: 'error', results: {} };
        })
      ]);

      const prices = priceData.data || [];
      let analysis = analysisData;

      // ✅ FIX: Use helper function to merge ML predictions (eliminates duplicate code)
      if (predictionsData.status === 'success' && predictionsData.results && predictionsData.results[symbol]) {
        const predResult = predictionsData.results[symbol];
        if (predResult && predResult.predictions) {
          const currentPrice = predResult.current_price || (prices.length > 0 ? prices[prices.length - 1]?.close : 0) || 0;
          const mlUnified = this._buildMLUnifiedFromBatchPredictions(
            predResult.predictions,
            predResult.confidences,
            predResult.models_by_horizon,
            currentPrice,
            predResult.model
          );

          // Merge ml_unified into analysis
          if (Object.keys(mlUnified).length > 0) {
            if (!analysis.ml_unified) {
              analysis.ml_unified = {};
            }
            Object.assign(analysis.ml_unified, mlUnified);
            // Update state with enriched analysis
            this.state.updateAnalysis(symbol, analysis);
          }
        }
      }

      // Render chart, patterns, ML summary, etc.
      this._renderDetailModal(symbol, prices, analysis);

      // Show modal
      const modal = new bootstrap.Modal(document.getElementById('detailModal'));
      modal.show();
    } catch (e) {
      console.error('Detail modal error:', e);
      this.ui.showNotification('error', 'Detay yüklenemedi: ' + (e.message || 'Bilinmeyen hata'));
    }
  }

  /**
   * Render detail modal content
   */
  async _renderDetailModal(symbol, prices, analysis) {
    try {
      logDebug('Rendering detail modal for', symbol);
      
      // Chart rendering
      this._renderDetailChart(prices, analysis);
      
      // Pattern list
      this._renderDetailPatterns(analysis);
      
      // ML summary
      this._renderDetailMLSummary(analysis);
      
      // Volume tier (async)
      await this._renderVolumeTier(symbol);
    } catch (e) {
      console.error('Detail modal rendering error:', e);
    }
  }
  
  /**
   * Render volume tier information
   */
  async _renderVolumeTier(symbol) {
    const volTierEl = document.getElementById('detailVolumeTier');
    const volAvgEl = document.getElementById('detailAvgVolume');
    
    if (!volTierEl || !volAvgEl) return;

    try {
      // Call internal API for volume tiers
      const response = await fetch(
        `/api/internal/automation/volume/tiers?symbol=${encodeURIComponent(symbol)}`,
        {
          headers: {
            'X-Internal-Token': 'IBx_gsmQUL9oxymAgr67PxES7ACfKlk1Ex5F9jCCOFw'
          }
        }
      );

      if (!response.ok) {
        volTierEl.textContent = '-';
        volAvgEl.textContent = '-';
        return;
      }

      const data = await response.json();
      
      // Tier label mapping
      const tierLabels = {
        'very_high': 'Çok Yüksek',
        'high': 'Yüksek',
        'medium': 'Orta',
        'low': 'Düşük',
        'very_low': 'Çok Düşük'
      };

      const tier = data.tier || '-';
      const tierLabel = tierLabels[tier] || tier.replace('_', ' ');
      const avgVolume = data.avg_volume;

      volTierEl.textContent = tierLabel;
      
      if (typeof avgVolume === 'number') {
        volAvgEl.textContent = `30g ort. hacim: ${avgVolume.toLocaleString('tr-TR')}`;
      } else {
        volAvgEl.textContent = '-';
      }
    } catch (e) {
      console.warn('Volume tier load error:', e);
      volTierEl.textContent = '-';
      volAvgEl.textContent = '-';
    }
  }

  _renderDetailChart(prices, analysis) {
    try {
      const labels = prices.map(p => p.date);
      const values = prices.map(p => p.close);
      
      if (values.length === 0) {
        logDebug('No price data for chart');
        return;
      }

      const ctx = document.getElementById('detailSpark');
      if (!ctx) return;
      
      const chartCtx = ctx.getContext('2d');

      // Destroy old chart if exists
      if (window._detailChart) {
        try {
          window._detailChart.destroy();
        } catch (e) {
          // Ignore
        }
      }

      // Normalize pattern ranges for chart overlay (red highlighting)
      const patterns = Array.isArray(analysis?.patterns) ? analysis.patterns : [];
      const normalizedPatterns = this._normalizePatternRanges(patterns, values.length, analysis?.data_points);

      // Create datasets with pattern overlays
      const datasets = [{
        data: values,
        borderColor: '#0d6efd',
        backgroundColor: 'rgba(13,110,253,0.08)',
        fill: true,
        tension: 0.25,
        pointRadius: 0,
        // Segment coloring based on patterns
        segment: {
          borderColor: (ctx) => {
            try {
              const idx = ctx.p1DataIndex;
              for (const pattern of normalizedPatterns) {
                const range = pattern.range;
                if (idx >= range.start_index && idx <= range.end_index) {
                  return '#dc3545'; // Red for pattern areas
                }
              }
            } catch (e) {
              // Ignore
            }
            return '#0d6efd'; // Blue for normal areas
          }
        }
      }];

      // Add red overlay datasets for each pattern range
      normalizedPatterns.forEach((pattern, idx) => {
        const range = pattern.range;
        const overlayData = values.map((v, i) => 
          (i >= range.start_index && i <= range.end_index) ? v : null
        );
        
        datasets.push({
          data: overlayData,
          borderColor: '#dc3545',
          backgroundColor: 'rgba(220,53,69,0.08)',
          borderWidth: 3,
          borderDash: [5, 2],
          fill: false,
          tension: 0.25,
          pointRadius: 0,
          spanGaps: false
        });
      });

      // Create new chart with fixed sizing
      window._detailChart = new Chart(chartCtx, {
        type: 'line',
        data: { labels, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          layout: {
            padding: {
              top: 10,
              right: 10,
              bottom: 10,
              left: 10
            }
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              mode: 'index',
              intersect: false,
              callbacks: {
                label: function(context) {
                  return '₺' + context.parsed.y.toFixed(2);
                }
              }
            }
          },
          scales: {
            x: { 
              display: true,
              grid: { display: false },
              ticks: { display: false }
            },
            y: { 
              display: true,
              position: 'right',
              grid: { 
                color: 'rgba(0, 0, 0, 0.05)',
                drawBorder: false
              },
              ticks: {
                padding: 5,
                font: { size: 10 },
                callback: function(value) {
                  return '₺' + value.toFixed(2);
                }
              }
            }
          },
          elements: {
            line: {
              borderWidth: 2
            },
            point: {
              radius: 0,
              hitRadius: 4,
              hoverRadius: 4
            }
          }
        }
      });

      // Update chart stats (including pattern stats)
      this._updateChartStats(values, normalizedPatterns);

      logDebug('Chart rendered successfully with pattern overlays');
    } catch (e) {
      console.error('Chart rendering error:', e);
    }
  }

  /**
   * Normalize pattern ranges to chart data indices
   */
  _normalizePatternRanges(patterns, displayCount, totalDataPoints) {
    try {
      const normalized = [];
      
      // Calculate offset (if data_points > displayCount, we're showing tail)
      let totalPoints = totalDataPoints || displayCount;
      
      // If no data_points, estimate from max end_index
      if (!totalPoints || totalPoints < displayCount) {
        let maxIdx = 0;
        patterns.forEach(p => {
          const endIdx = p.range?.end_index;
          if (typeof endIdx === 'number') {
            maxIdx = Math.max(maxIdx, endIdx);
          }
        });
        totalPoints = Math.max(totalPoints || 0, maxIdx + 1);
      }

      const offset = Math.max(0, totalPoints - displayCount);

      // Filter only technical/visual patterns (not ML)
      patterns.forEach(p => {
        const src = (p.source || '').toUpperCase();
        if (!src || ['ML_PREDICTOR', 'ENHANCED_ML', 'FINGPT'].includes(src)) {
          return; // Skip ML patterns
        }

        const range = p.range;
        if (!range || typeof range.start_index !== 'number' || typeof range.end_index !== 'number') {
          return;
        }

        // Adjust indices based on offset
        let start = Math.round(range.start_index - offset);
        let end = Math.round(range.end_index - offset);

        // Skip if completely outside visible range
        if (end < 0 || start >= displayCount) {
          return;
        }

        // Clamp to visible range
        start = Math.max(0, start);
        end = Math.min(displayCount - 1, end);

        normalized.push({
          range: { start_index: start, end_index: end },
          pattern: p.pattern,
          source: p.source
        });
      });

      logDebug(`Normalized ${normalized.length} patterns for chart overlay`);
      return normalized;
    } catch (e) {
      console.error('Pattern normalization error:', e);
      return [];
    }
  }

  /**
   * Update chart statistics
   */
  _updateChartStats(values, normalizedPatterns) {
    const statsEl = document.getElementById('detailSparkStats');
    if (statsEl && values.length > 0) {
      const min = Math.min(...values);
      const max = Math.max(...values);
      statsEl.innerHTML = `Bar: <strong>${values.length}</strong> • En düşük: <strong>${formatCurrency(min)}</strong> • En yüksek: <strong>${formatCurrency(max)}</strong>`;
    }

    // Pattern-specific stats (red areas)
    const patternStatsEl = document.getElementById('detailPatternStats');
    if (patternStatsEl && normalizedPatterns.length > 0) {
      const patternIndices = new Set();
      normalizedPatterns.forEach(p => {
        for (let i = p.range.start_index; i <= p.range.end_index; i++) {
          patternIndices.add(i);
        }
      });

      const patternValues = Array.from(patternIndices).map(i => values[i]).filter(v => v != null);
      if (patternValues.length > 0) {
        const minP = Math.min(...patternValues);
        const maxP = Math.max(...patternValues);
        patternStatsEl.innerHTML = `Kırmızı aralık: <strong>${patternValues.length}</strong> bar • En düşük: <strong>${formatCurrency(minP)}</strong> • En yüksek: <strong>${formatCurrency(maxP)}</strong>`;
      }
    }
  }

  /**
   * Render detail modal patterns
   * ✅ FIX: Renk mantığı - Signal'a göre renk (BULLISH=yeşil, BEARISH=kırmızı)
   */
  _renderDetailPatterns(analysis) {
    const patt = document.getElementById('detailPatterns');
    if (!patt) return;

    const patterns = Array.isArray(analysis?.patterns) ? analysis.patterns : [];
    
    if (patterns.length === 0) {
      patt.innerHTML = '<span class="text-muted">Formasyon bulunamadı</span>';
      return;
    }

    const html = patterns.slice(0, 20).map(p => {
      const src = (p.source || '').toUpperCase();
      const conf = Math.round((p.confidence || 0) * 100);
      const name = typeof window.translatePattern === 'function' 
        ? window.translatePattern(p.pattern || '') 
        : (p.pattern || '').replace(/_/g, ' ');
      
      // ✅ FIX: Signal'a göre renk belirle (öncelikli)
      const signal = (p.signal || '').toUpperCase();
      let color;
      if (signal === 'BULLISH') {
        color = 'success';  // Yeşil - yükseliş trendi
      } else if (signal === 'BEARISH') {
        color = 'danger';   // Kırmızı - düşüş trendi
      } else if (signal === 'NEUTRAL') {
        color = 'secondary'; // Gri - nötr
      } else {
        // Fallback: Source'a göre renk (eski mantık)
        color = BADGE_COLORS[src] || BADGE_COLORS.default;
      }
      
      return `
        <div class="mb-1">
          ${buildBadgeHTML(translateSource(src), color, 'me-2')}
          ${name} <span class="text-muted">%${conf}</span>
        </div>
      `;
    }).join('');

    patt.innerHTML = html;
  }

  /**
   * ✅ Helper: Escape HTML to prevent XSS
   */
  _escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  _renderDetailMLSummary(analysis) {
    const mlBox = document.getElementById('detailMlUnified');
    if (!mlBox) return;

    try {
      const uni = analysis?.ml_unified || {};
      const horizons = ['1d', '3d', '7d', '14d', '30d'];
      
      // Extract FinGPT patterns for badge (YOLO already shown in Formasyonlar)
      const patterns = Array.isArray(analysis?.patterns) ? analysis.patterns : [];
      const fingptPatterns = patterns.filter(p => p.source === 'FINGPT');

      if (Object.keys(uni).length === 0) {
        mlBox.innerHTML = '<span class="text-muted">ML tahmin bilgisi yok</span>';
        return;
      }

      const rows = horizons.map(h => {
        const item = uni[h];
        if (!item || (!item.basic && !item.enhanced)) {
          return `<div class="mb-1"><strong>${h.toUpperCase()}</strong>: -</div>`;
        }

        // ✅ FIX: Determine best model - use best field if available, otherwise compare quality
        let best = item.best;
        if (!best) {
          // No best field, determine best based on availability and quality
          if (item.enhanced && item.basic) {
            // Both exist, compare confidence/reliability
            const enhConf = item.enhanced.confidence || item.enhanced.reliability || 0;
            const basConf = item.basic.confidence || item.basic.reliability || 0;
            best = enhConf >= basConf ? 'enhanced' : 'basic';
          } else {
            // Only one exists
            best = item.enhanced ? 'enhanced' : (item.basic ? 'basic' : null);
          }
        }
        const segments = [];

        // Process both basic and enhanced
        ['basic', 'enhanced'].forEach(src => {
          if (!item[src]) return;

          const srcData = item[src];
          const price = srcData.price;
          const deltaPct = srcData.delta_pct;
          const conf = srcData.confidence || srcData.reliability;
          const evidence = srcData.evidence || {};

          let segment = `${translateModelLabel(src)}: <strong>${formatCurrency(price)}</strong>`;
          
          if (typeof deltaPct === 'number') {
            const changeClass = getPriceChangeClass(deltaPct * 100);
            segment += ` (<span class="${changeClass}">${formatPercentage(deltaPct * 100)}</span>)`;
          }
          
          if (typeof conf === 'number') {
            segment += ` • Güven %${Math.round(conf * 100)}`;
          }

          // Evidence details (comprehensive)
          const evidenceParts = [];
          
          // Pattern ve Sentiment skorları
          if (typeof evidence.pattern_score === 'number') {
            evidenceParts.push(`Pat ${evidence.pattern_score >= 0 ? '+' : ''}${evidence.pattern_score.toFixed(2)}`);
          }
          if (typeof evidence.sentiment_score === 'number') {
            evidenceParts.push(`Sent ${evidence.sentiment_score >= 0 ? '+' : ''}${evidence.sentiment_score.toFixed(2)}`);
          }
          
          // Confidence contribution (kalibrasyon etkisi)
          if (typeof evidence.contrib_conf === 'number' && Math.abs(evidence.contrib_conf) > 0.001) {
            const sign = evidence.contrib_conf >= 0 ? '+' : '';
            evidenceParts.push(`Δgüv ${sign}${(evidence.contrib_conf * 100).toFixed(0)}`);
          }
          
          // Booster probability (1D için)
          if (typeof evidence.booster_prob === 'number') {
            evidenceParts.push(`Boost P${Math.round(evidence.booster_prob * 100)}`);
          }
          
          // Booster contribution
          if (typeof evidence.contrib_booster === 'number' && Math.abs(evidence.contrib_booster) > 0.001) {
            const sign = evidence.contrib_booster >= 0 ? '+' : '';
            evidenceParts.push(`Δboost ${sign}${(evidence.contrib_booster * 100).toFixed(0)}`);
          }
          
          // Delta contribution (tilt etkisi)
          if (typeof evidence.contrib_delta === 'number' && Math.abs(evidence.contrib_delta) > 0.001) {
            const sign = evidence.contrib_delta >= 0 ? '+' : '';
            evidenceParts.push(`Δ% ${sign}${(evidence.contrib_delta * 100).toFixed(2)}`);
          }
          
          // Weight bilgileri
          if (typeof evidence.w_pat === 'number' && typeof evidence.w_sent === 'number') {
            evidenceParts.push(`w_pat=${evidence.w_pat.toFixed(2)}, w_sent=${evidence.w_sent.toFixed(2)}`);
          }

          if (evidenceParts.length > 0) {
            segment += `<br><span class="text-muted small ms-3">→ Kanıt: ${evidenceParts.join(' | ')}</span>`;
          }

          segments.push(`<span class="d-block mb-1">${segment}</span>`);
        });

        const bestBadge = best ? buildBadgeHTML(`En iyi: ${translateModelLabel(best)}`, 'light', 'text-dark ms-1') : '';
        
        return `<div class="mb-2"><strong>${h.toUpperCase()}</strong>: ${segments.join('')}${bestBadge}</div>`;
      }).join('');

      // Add FinGPT badge at the end (YOLO already shown in Formasyonlar section)
      const additionalBadges = [];
      
      if (fingptPatterns.length > 0) {
        const fgTop = fingptPatterns[0];
        const fgConf = Math.round((fgTop.confidence || 0) * 100);
        const fgSignal = fgTop.signal || 'NEUTRAL';
        const fgNewsCount = fgTop.news_count || 0;
        const fgIcon = fgSignal === 'BULLISH' ? '📈' : fgSignal === 'BEARISH' ? '📉' : '📊';
        const fgNewsItems = fgTop.news_items || [];
        
        // ✅ DEBUG: Log to check if news_items is present
        logDebug('FinGPT pattern data:', {
          news_count: fgNewsCount,
          news_items: fgNewsItems,
          news_items_length: fgNewsItems.length,
          full_pattern: fgTop
        });
        
        // Build tooltip content with news items
        let tooltipContent = `<strong>Sezgisel Analiz</strong><br>`;
        tooltipContent += `<small>${fgSignal} (%${fgConf}) • ${fgNewsCount} haber</small><br><br>`;
        if (fgNewsItems.length > 0) {
          tooltipContent += `<strong>İlgili Haberler:</strong><br>`;
          fgNewsItems.forEach((news, idx) => {
            const escapedNews = this._escapeHtml(news);
            tooltipContent += `<small>${idx + 1}. ${escapedNews}</small><br>`;
          });
        } else {
          tooltipContent += `<small>Haber detayları mevcut değil</small>`;
        }
        
        // Generate unique ID for tooltip
        const tooltipId = `fingpt-tooltip-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        
        additionalBadges.push(`
          <div class="mb-1">
            <span class="badge bg-warning text-dark me-1 fingpt-badge" 
                  id="${tooltipId}" 
                  data-bs-toggle="tooltip" 
                  data-bs-html="true" 
                  data-bs-placement="top"
                  data-tooltip-content="${this._escapeHtml(tooltipContent)}"
                  style="cursor: help;">💡 Sezgisel</span>
            <span class="text-muted small">${fgIcon} ${fgSignal} (%${fgConf}) • ${fgNewsCount} haber</span>
          </div>
        `);
        
        // Initialize Bootstrap tooltip after DOM update
        setTimeout(() => {
          const tooltipElement = document.getElementById(tooltipId);
          if (tooltipElement) {
            const tooltipContentAttr = tooltipElement.getAttribute('data-tooltip-content');
            if (tooltipContentAttr) {
              tooltipElement.setAttribute('title', tooltipContentAttr);
              if (typeof bootstrap !== 'undefined') {
                new bootstrap.Tooltip(tooltipElement, {
                  html: true,
                  placement: 'top',
                  trigger: 'hover',
                  container: 'body'
                });
              }
            }
          }
        }, 100);
      }

      mlBox.innerHTML = rows + (additionalBadges.length > 0 ? '<hr class="my-2">' + additionalBadges.join('') : '');
      logDebug('ML summary rendered');
    } catch (e) {
      console.error('ML summary render error:', e);
      mlBox.innerHTML = '<span class="text-danger">ML özet yüklenemedi</span>';
    }
  }

  /**
   * ✅ FIX: Cleanup method to prevent memory leaks
   */
  cleanup() {
    // Clear debounce timeouts
    this._patternUpdateDebounce.forEach((timeoutId) => {
      clearTimeout(timeoutId);
    });
    this._patternUpdateDebounce.clear();
    
    // Clear timestamp tracking
    this._lastProcessedTimestamps.clear();
    
    // Clear intervals
    if (this._timestampInterval) {
      clearInterval(this._timestampInterval);
      this._timestampInterval = null;
    }
    
    // Execute cleanup functions
    this._cleanupFunctions.forEach(fn => {
      try {
        fn();
      } catch (e) {
        console.warn('Cleanup function error:', e);
      }
    });
    this._cleanupFunctions = [];
    
    // Disconnect WebSocket
    if (this.ws) {
      try {
        this.ws.disconnect();
      } catch (e) {
        console.warn('WebSocket disconnect error:', e);
      }
    }
    
    // Clear DOM cache
    this._domCache.clear();
  }
}

/**
 * ============================================
 * GLOBAL EXPORTS
 * ============================================
 */

// Initialize on page load
let dashboardInstance = null;

export function initDashboard(userId) {
  if (dashboardInstance) {
    console.warn('Dashboard already initialized');
    return dashboardInstance;
  }

  dashboardInstance = new UserDashboard(userId);
  return dashboardInstance;
}

export function getDashboard() {
  return dashboardInstance;
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
  console.log('🚀 DOMContentLoaded - initializing dashboard...');
  
  // Get user ID from server injection
  const userId = window.APP_USER_ID || null;
  
  // Initialize dashboard
  const dashboard = initDashboard(userId);
  await dashboard.init();
  
  // Make globally accessible for onclick handlers
  window.dashboard = dashboard;
  
  // ✅ FIX: Cleanup on page unload (memory leak prevention)
  window.addEventListener('beforeunload', () => {
    if (dashboard) {
      dashboard.cleanup();
    }
  });
  
  console.log('✅ Dashboard ready');
});

// Export for external use
export { UserDashboard, DashboardState, UIRenderer };

