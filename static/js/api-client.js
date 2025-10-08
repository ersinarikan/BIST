/**
 * User Dashboard API Client
 * Tüm API çağrıları için merkezi client
 */

import { API_ENDPOINTS } from './constants.js';
import { logDebug } from './utils.js';

/**
 * Base API Client Class
 */
class APIClient {
  constructor(baseURL = '/api') {
    this.baseURL = baseURL;
    this.defaultOptions = {
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json'
      }
    };
  }

  /**
   * Generic fetch wrapper with error handling
   */
  async _fetch(endpoint, options = {}) {
    const url = endpoint.startsWith('http') ? endpoint : `${this.baseURL}${endpoint}`;
    
    try {
      logDebug(`API Request: ${options.method || 'GET'} ${url}`);
      
      const response = await fetch(url, {
        ...this.defaultOptions,
        ...options,
        headers: {
          ...this.defaultOptions.headers,
          ...(options.headers || {})
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      logDebug(`API Response: ${url}`, data);
      
      return data;
    } catch (error) {
      console.error(`API Error [${url}]:`, error);
      throw error;
    }
  }

  /**
   * GET request
   */
  async get(endpoint, params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const url = queryString ? `${endpoint}?${queryString}` : endpoint;
    return this._fetch(url, { method: 'GET' });
  }

  /**
   * POST request
   */
  async post(endpoint, data = {}) {
    return this._fetch(endpoint, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  /**
   * PUT request
   */
  async put(endpoint, data = {}) {
    return this._fetch(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }

  /**
   * DELETE request
   */
  async delete(endpoint) {
    return this._fetch(endpoint, { method: 'DELETE' });
  }
}

/**
 * Dashboard-specific API methods
 */
class DashboardAPI extends APIClient {
  
  /**
   * Watchlist Operations
   */
  
  async getWatchlist() {
    return this.get(API_ENDPOINTS.WATCHLIST);
  }

  async addToWatchlist(symbol, options = {}) {
    const data = {
      symbol: symbol.toUpperCase().trim(),
      alert_enabled: options.alert_enabled !== false,
      notes: options.notes || '',
      alert_threshold_buy: options.alert_threshold_buy || null,
      alert_threshold_sell: options.alert_threshold_sell || null
    };
    return this.post(API_ENDPOINTS.WATCHLIST, data);
  }

  async removeFromWatchlist(symbol) {
    return this.delete(`${API_ENDPOINTS.WATCHLIST}/${encodeURIComponent(symbol)}`);
  }

  async updateWatchlistItem(symbol, updates = {}) {
    return this.put(`${API_ENDPOINTS.WATCHLIST}/${encodeURIComponent(symbol)}`, updates);
  }

  /**
   * Predictions
   */
  
  async getWatchlistPredictions() {
    return this.get(API_ENDPOINTS.WATCHLIST_PREDICTIONS);
  }

  async getUserPredictions(symbol) {
    return this.get(`${API_ENDPOINTS.USER_PREDICTIONS}/${encodeURIComponent(symbol)}`);
  }

  async getBatchPredictions(symbols) {
    return this.post(API_ENDPOINTS.BATCH_PREDICTIONS, { symbols });
  }

  /**
   * Pattern Analysis
   */
  
  async getPatternAnalysis(symbol, options = {}) {
    const params = {
      fast: options.fast ? '1' : '0',
      v: options.nocache ? Date.now() : undefined
    };
    return this.get(
      `${API_ENDPOINTS.PATTERN_ANALYSIS}/${encodeURIComponent(symbol)}`,
      params
    );
  }

  async getBatchPatternAnalysis(symbols) {
    return this.post(API_ENDPOINTS.BATCH_PATTERN_ANALYSIS, { symbols });
  }

  /**
   * Stock Operations
   */
  
  async searchStocks(query, limit = 50) {
    return this.get(API_ENDPOINTS.STOCK_SEARCH, { q: query, limit });
  }

  async getStockPrices(symbol, options = {}) {
    const params = {
      days: options.days || 60,
      v: options.nocache ? Date.now() : undefined
    };
    return this.get(
      `${API_ENDPOINTS.STOCK_PRICES}/${encodeURIComponent(symbol)}`,
      params
    );
  }

  /**
   * Cache Report
   */
  
  async getWatchlistCacheReport() {
    return this.get('/watchlist/cache-report');
  }

  /**
   * Batch Operations Helper
   */
  
  async fetchAllWatchlistData(symbols) {
    try {
      const [predictions, patterns] = await Promise.all([
        this.getBatchPredictions(symbols).catch(err => {
          console.warn('Batch predictions failed:', err);
          return { status: 'error', results: {} };
        }),
        this.getBatchPatternAnalysis(symbols).catch(err => {
          console.warn('Batch pattern analysis failed:', err);
          return { status: 'error', results: {} };
        })
      ]);

      return {
        predictions: predictions.results || {},
        patterns: patterns.results || {}
      };
    } catch (error) {
      console.error('Batch fetch error:', error);
      return {
        predictions: {},
        patterns: {}
      };
    }
  }
}

/**
 * Singleton instance
 */
let apiInstance = null;

export function getAPI() {
  if (!apiInstance) {
    apiInstance = new DashboardAPI();
  }
  return apiInstance;
}

// Export class for custom instances
export { DashboardAPI };

// Default export
export default getAPI();

