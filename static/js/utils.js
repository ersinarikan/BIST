/**
 * User Dashboard Utility Functions
 * Format, calculation ve helper fonksiyonlar
 */

import {
  SIGNAL_CONFIDENCE,
  MOVEMENT_THRESHOLDS,
  SIGNALS,
  SIGNAL_LABELS,
  SIGNAL_CLASSES,
  SIGNAL_ICONS,
  MODEL_LABELS,
  SOURCE_LABELS
} from './constants.js';

/**
 * Currency formatting
 */
export function formatCurrency(value) {
  if (typeof value !== 'number' || !isFinite(value)) {
    return '-';
  }
  return `₺${value.toFixed(2)}`;
}

/**
 * Percentage formatting
 */
export function formatPercentage(value, decimals = 1) {
  if (typeof value !== 'number' || !isFinite(value)) {
    return '-';
  }
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}%`;
}

/**
 * Calculate percentage change
 */
export function calculatePercentChange(current, target) {
  if (typeof current !== 'number' || typeof target !== 'number' || current === 0) {
    return NaN;
  }
  return ((target - current) / current) * 100;
}

/**
 * Translate model label to Turkish
 */
export function translateModelLabel(model) {
  try {
    const m = String(model || '').toLowerCase();
    return MODEL_LABELS[m] || model || '';
  } catch (e) {
    return model || '';
  }
}

/**
 * Translate source to Turkish
 */
export function translateSource(source) {
  try {
    const key = String(source || '').toUpperCase();
    return SOURCE_LABELS[key] || source || '';
  } catch (e) {
    return source || '';
  }
}

/**
 * Get signal label based on confidence and delta
 */
export function getSignalLabel(confidence, delta, horizon = '7d') {
  try {
    const abs = Math.abs(delta || 0);
    
    // Determine direction
    let signalType = SIGNALS.HOLD;
    if (delta > 0) {
      signalType = SIGNALS.BULLISH;
    } else if (delta < 0) {
      signalType = SIGNALS.BEARISH;
    }
    
    // Determine strength based on confidence and movement
    let label = SIGNAL_LABELS[SIGNALS.HOLD];
    
    if (signalType === SIGNALS.BULLISH || signalType === SIGNALS.BEARISH) {
      const labels = SIGNAL_LABELS[signalType];
      
      // ✅ FIX: Delta'ya göre esnek ve mantıklı sinyal belirleme
      // Temel mantık: Confidence ve delta kombinasyonu
      // Yüksek güven → küçük delta yeterli
      // Düşük güven → büyük delta gerekli
      // Çok küçük delta (< %0.1) → Bekleme (gürültü olabilir)
      
      // Minimum delta threshold (gürültü filtreleme)
      const MIN_DELTA = 0.001;  // %0.1 - çok küçük hareketler gürültü olabilir
      
      if (abs < MIN_DELTA) {
        // Delta çok küçük, gürültü olabilir → Bekleme
        label = labels.low;
      }
      // 1. Çok yüksek güven (%85+) → Yüksek alım/satış (delta %0.5+)
      else if (confidence >= SIGNAL_CONFIDENCE.VERY_HIGH && abs >= MOVEMENT_THRESHOLDS.SMALL) {
        label = labels.high;
      }
      // 2. Yüksek güven (%70-85) → Alım/Satış sinyali (delta %0.5+)
      else if (confidence >= SIGNAL_CONFIDENCE.HIGH && abs >= MOVEMENT_THRESHOLDS.SMALL) {
        label = labels.medium;
      }
      // 3. Orta-yüksek güven (%60-70) → Delta'ya göre esnek
      else if (confidence >= SIGNAL_CONFIDENCE.MEDIUM) {
        // Orta güven + büyük hareket (%1.5+) → Alım/Satış
        if (abs >= MOVEMENT_THRESHOLDS.MEDIUM) {
          label = labels.medium;
        }
        // Orta güven + küçük ama anlamlı hareket (%0.2+) → Alım/Satış
        else if (abs >= 0.002) {
          label = labels.medium;
        }
        // Orta güven + çok küçük hareket → Bekleme
        else {
          label = labels.low;
        }
      }
      // 4. Düşük güven (%55-60) → Sadece büyük hareketlerde sinyal
      else if (confidence >= SIGNAL_CONFIDENCE.LOW) {
        // Düşük güven + büyük hareket (%2+) → Alım/Satış
        if (abs >= 0.02) {
          label = labels.medium;
        }
        // Düşük güven + orta hareket (%1+) → Alım/Satış (daha riskli)
        else if (abs >= 0.01) {
          label = labels.medium;
        }
        else {
          label = labels.low;
        }
      }
      // 5. Çok düşük güven (< %55) → Bekleme (çok riskli)
      else {
        label = labels.low;
      }
    }
    
    return {
      label,
      signalType,
      cssClass: SIGNAL_CLASSES[signalType],
      icon: SIGNAL_ICONS[signalType]
    };
  } catch (e) {
    return {
      label: 'Bekleme',
      signalType: SIGNALS.HOLD,
      cssClass: SIGNAL_CLASSES[SIGNALS.HOLD],
      icon: SIGNAL_ICONS[SIGNALS.HOLD]
    };
  }
}

/**
 * Get threshold for specific horizon
 */
export function getThresholdForHorizon(horizon) {
  try {
    const key = String(horizon || '').toLowerCase();
    return MOVEMENT_THRESHOLDS.CALIBRATED[key] || 0.03;
  } catch (e) {
    return 0.03;
  }
}

/**
 * Check if element is visible
 */
export function isElementVisible(elementId) {
  try {
    const el = document.getElementById(elementId);
    if (!el) return false;
    
    const style = window.getComputedStyle ? window.getComputedStyle(el) : null;
    return el.style.display !== 'none' && (!style || style.display !== 'none');
  } catch (e) {
    return false;
  }
}

/**
 * Debounce function
 */
export function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

/**
 * Throttle function
 */
export function throttle(func, limit) {
  let inThrottle;
  return function(...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

/**
 * Safe JSON parse
 */
export function safeJSONParse(str, defaultValue = null) {
  try {
    return JSON.parse(str);
  } catch (e) {
    return defaultValue;
  }
}

/**
 * Safe localStorage get
 */
export function getLocalStorage(key, defaultValue = null) {
  try {
    const item = localStorage.getItem(key);
    if (item === null) return defaultValue;
    return safeJSONParse(item, defaultValue);
  } catch (e) {
    return defaultValue;
  }
}

/**
 * Safe localStorage set
 */
export function setLocalStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (e) {
    console.error('LocalStorage write error:', e);
    return false;
  }
}

/**
 * Format timestamp to Turkish locale
 */
export function formatTimestamp(timestamp) {
  try {
    const date = new Date(timestamp);
    return date.toLocaleString('tr-TR');
  } catch (e) {
    return '-';
  }
}

/**
 * Format date to Turkish locale (date only)
 */
export function formatDate(date) {
  try {
    const d = new Date(date);
    return d.toLocaleDateString('tr-TR');
  } catch (e) {
    return '-';
  }
}

/**
 * Format time to Turkish locale (time only)
 */
export function formatTime(time) {
  try {
    const t = new Date(time);
    return t.toLocaleTimeString('tr-TR');
  } catch (e) {
    return '-';
  }
}

/**
 * Normalize prediction value (handle different formats)
 */
export function normalizePredictionValue(value) {
  if (typeof value === 'number') {
    return value;
  }
  
  if (typeof value === 'object' && value !== null) {
    // Try different field names
    const candidates = ['ensemble_prediction', 'price', 'prediction', 'target', 'value'];
    for (const field of candidates) {
      if (typeof value[field] === 'number') {
        return value[field];
      }
    }
  }
  
  return null;
}

/**
 * Build HTML for signal display
 */
export function buildSignalHTML(label, confidence, icon, cssClass, title = '') {
  const titleAttr = title ? `title="${title}"` : '';
  return `<span class="${cssClass}" ${titleAttr}>
    <i class="${icon} me-1"></i>${label} (%${confidence})
  </span>`;
}

/**
 * Build HTML for badge
 */
export function buildBadgeHTML(text, color = 'secondary', additionalClass = '', title = '') {
  const titleAttr = title ? `title="${title}"` : '';
  return `<span class="badge bg-${color} ${additionalClass}" ${titleAttr}>${text}</span>`;
}

/**
 * Validate symbol format (BIST stocks)
 */
export function isValidSymbol(symbol) {
  if (!symbol || typeof symbol !== 'string') return false;
  // BIST symbols are usually 3-6 uppercase letters
  return /^[A-Z]{3,6}$/.test(symbol.trim());
}

/**
 * Highlight search term in text
 */
export function highlightSearchTerm(text, searchTerm) {
  if (!searchTerm || !text) return text;
  
  try {
    const term = searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(term, 'gi');
    return String(text).replace(regex, match => `<mark>${match}</mark>`);
  } catch (e) {
    return text;
  }
}

/**
 * Get CSS class for price change
 */
export function getPriceChangeClass(change) {
  if (typeof change !== 'number' || !isFinite(change)) {
    return 'text-muted';
  }
  return change >= 0 ? 'text-success' : 'text-danger';
}

/**
 * Round to decimal places
 */
export function round(value, decimals = 2) {
  if (typeof value !== 'number' || !isFinite(value)) return null;
  return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals);
}

/**
 * Clamp value between min and max
 */
export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

/**
 * Check if value is within range
 */
export function isInRange(value, min, max) {
  return typeof value === 'number' && value >= min && value <= max;
}

/**
 * Generate unique ID
 */
export function generateId() {
  return Math.random().toString(36).substr(2, 9);
}

/**
 * Sleep/delay function
 */
export function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Log with timestamp (production-safe)
 */
export function logDebug(message, data = null) {
  if (localStorage.getItem('DEBUG_UI') === '1') {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}]`, message, data || '');
  }
}

