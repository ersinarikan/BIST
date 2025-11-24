#!/usr/bin/env python3
"""
Ensemble Utility Functions
Smart weighting strategies for combining model predictions
"""

import numpy as np
from typing import Dict, Tuple, Optional


def consensus_weighted_ensemble(
    predictions: np.ndarray,
    sigma: float = 0.005
) -> Tuple[float, np.ndarray]:
    """
    Consensus-based weighting: Birbirine yakın tahminlerin ağırlığını artır
    
    Mantık:
    - 3 model benzer tahmin yapıyorsa → güvenilir, ağırlık yüksek
    - 1 model çok farklı → outlier, ağırlık düşük
    
    Args:
        predictions: Model tahminleri [pred1, pred2, pred3, ...]
        sigma: Uzaklık hassasiyeti (default 0.005 = 0.5%)
    
    Returns:
        final_prediction: Ağırlıklı ortalama
        weights: Her modelin ağırlığı
    
    Example:
        preds = np.array([0.025, 0.023, 0.024, 0.050])
        # Model 0,1,2 benzer (2.3-2.5%), Model 3 outlier (5.0%)
        final, weights = consensus_weighted_ensemble(preds)
        # weights ≈ [0.35, 0.34, 0.35, 0.10]  # Outlier düşük!
    """
    preds = np.array(predictions)
    n = len(preds)
    
    # Her modelin diğerlerine olan ortalama uzaklığı
    distances = np.zeros(n)
    for i in range(n):
        others = np.delete(preds, i)
        distances[i] = np.mean(np.abs(preds[i] - others))
    
    # Uzaklık düşük → ağırlık yüksek (Gaussian kernel)
    weights = np.exp(-distances / sigma)
    
    # Normalize
    weights = weights / weights.sum()
    
    # Ağırlıklı ortalama
    final_pred = np.sum(preds * weights)
    
    return final_pred, weights


def performance_weighted_ensemble(
    predictions: np.ndarray,
    historical_r2: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Performance-based weighting: Geçmiş performansa göre ağırlıklandır
    
    Args:
        predictions: Model tahminleri
        historical_r2: Geçmiş R² skorları [r2_1, r2_2, ...]
    
    Returns:
        final_prediction: Ağırlıklı ortalama
        weights: Her modelin ağırlığı
    """
    preds = np.array(predictions)
    r2_scores = np.array(historical_r2)
    
    # Negatif R² varsa 0'a çek (kötü modelleri cezalandır)
    r2_scores = np.maximum(r2_scores, 0.0)
    
    # R² bazlı ağırlık
    if r2_scores.sum() > 0:
        weights = r2_scores / r2_scores.sum()
    else:
        # Tüm modeller kötü, eşit ağırlık
        weights = np.ones(len(preds)) / len(preds)
    
    final_pred = np.sum(preds * weights)
    
    return final_pred, weights


def hybrid_ensemble(
    predictions: np.ndarray,
    historical_r2: np.ndarray,
    current_confidence: Optional[np.ndarray] = None,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
    sigma: float = 0.005
) -> Tuple[float, np.ndarray]:
    """
    Hybrid weighting: Konsensüs + Performans + Güven
    
    3 faktör:
    1. Geçmiş performans (R²)
    2. Konsensüs (birbirine yakınlık)
    3. Model güven skoru (opsiyonel)
    
    Args:
        predictions: Model tahminleri
        historical_r2: Geçmiş R² skorları
        current_confidence: Model güven skorları (opsiyonel)
        alpha: Performans ağırlığı (default 0.4)
        beta: Konsensüs ağırlığı (default 0.4)
        gamma: Güven ağırlığı (default 0.2)
        sigma: Konsensüs hassasiyeti
    
    Returns:
        final_prediction: Ağırlıklı ortalama
        weights: Her modelin ağırlığı
    """
    preds = np.array(predictions)
    n = len(preds)
    
    # 1. Performans ağırlığı
    r2_scores = np.maximum(np.array(historical_r2), 0.0)
    if r2_scores.sum() > 0:
        perf_weights = r2_scores / r2_scores.sum()
    else:
        perf_weights = np.ones(n) / n
    
    # 2. Konsensüs ağırlığı
    distances = np.array([
        np.mean(np.abs(preds[i] - np.delete(preds, i)))
        for i in range(n)
    ])
    consensus_weights = np.exp(-distances / sigma)
    consensus_weights /= consensus_weights.sum()
    
    # 3. Güven ağırlığı
    if current_confidence is not None:
        conf = np.array(current_confidence)
        if conf.sum() > 0:
            conf_weights = conf / conf.sum()
        else:
            conf_weights = np.ones(n) / n
    else:
        conf_weights = np.ones(n) / n
    
    # Kombine ağırlık
    final_weights = (
        alpha * perf_weights +
        beta * consensus_weights +
        gamma * conf_weights
    )
    final_weights /= final_weights.sum()
    
    final_pred = np.sum(preds * final_weights)
    
    return final_pred, final_weights


def adaptive_ensemble(
    predictions: Dict[str, float],
    regime: str,
    regime_weights: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Adaptive weighting: Rejime göre model ağırlıkları değişir
    
    Args:
        predictions: Model tahminleri {'xgboost': 0.025, 'lightgbm': 0.023, ...}
        regime: Mevcut rejim ('LOW_VOL', 'HIGH_VOL', 'MEDIUM_VOL')
        regime_weights: Rejim bazlı ağırlık matrisi (opsiyonel)
    
    Returns:
        final_prediction: Ağırlıklı ortalama
        weights: Her modelin ağırlığı
    """
    # Default rejim ağırlıkları
    if regime_weights is None:
        regime_weights = {
            'LOW_VOL': {
                'xgboost': 0.45,   # Karmaşık model, trend yakalar
                'lightgbm': 0.35,
                'catboost': 0.15,
                'ridge': 0.05      # Linear zayıf kalır
            },
            'HIGH_VOL': {
                'xgboost': 0.15,   # Karmaşık model overfitting riski
                'lightgbm': 0.20,
                'catboost': 0.25,
                'ridge': 0.40      # Linear daha stabil
            },
            'MEDIUM_VOL': {
                'xgboost': 0.35,
                'lightgbm': 0.30,
                'catboost': 0.25,
                'ridge': 0.10
            }
        }
    
    # Rejim ağırlıklarını al
    weights = regime_weights.get(regime, regime_weights.get('MEDIUM_VOL', {}))
    
    # Sadece mevcut modelleri kullan
    available_models = set(predictions.keys())
    weights_filtered = {
        model: weights.get(model, 0.0)
        for model in available_models
    }
    
    # Normalize
    total = sum(weights_filtered.values())
    if total > 0:
        weights_filtered = {k: v / total for k, v in weights_filtered.items()}
    else:
        # Eşit ağırlık
        n = len(available_models)
        weights_filtered = {k: 1.0 / n for k in available_models}
    
    # Ağırlıklı ortalama
    final_pred = sum(
        predictions[model] * weights_filtered[model]
        for model in available_models
    )
    
    return final_pred, weights_filtered


def median_ensemble(predictions: np.ndarray) -> float:
    """
    Median ensemble: Outlier'lara karşı en dirençli
    
    Args:
        predictions: Model tahminleri
    
    Returns:
        median_prediction: Medyan değer
    """
    return float(np.median(predictions))


def trimmed_mean_ensemble(
    predictions: np.ndarray,
    trim_fraction: float = 0.25
) -> float:
    """
    Trimmed mean: En yüksek ve en düşük tahminleri at, ortalamasını al
    
    Args:
        predictions: Model tahminleri
        trim_fraction: Atılacak oran (her iki uçtan, default 0.25 = %25)
    
    Returns:
        trimmed_mean: Kırpılmış ortalama
    """
    preds = np.array(predictions)
    n = len(preds)
    
    # Sırala
    sorted_preds = np.sort(preds)
    
    # Kaç tane atılacak (her iki uçtan)
    trim_count = int(n * trim_fraction)
    
    if trim_count == 0 or n - 2 * trim_count < 1:
        # Atılacak eleman yok veya çok az eleman kaldı
        return float(np.mean(preds))
    
    # Uçları at, ortalamasını al
    trimmed = sorted_preds[trim_count: n - trim_count]
    
    return float(np.mean(trimmed))


def smart_ensemble(
    predictions: np.ndarray,
    historical_r2: np.ndarray,
    consensus_weight: float = 0.8,
    performance_weight: float = 0.2,
    sigma: float = 0.005
) -> Tuple[float, np.ndarray]:
    """
    Smart ensemble: Konsensüs + Performans dengeli kombinasyonu
    
    ÖNERİLEN YÖNTEM: En iyi denge
    
    Args:
        predictions: Model tahminleri
        historical_r2: Geçmiş R² skorları
        consensus_weight: Konsensüs ağırlığı (default 0.6)
        performance_weight: Performans ağırlığı (default 0.4)
        sigma: Konsensüs hassasiyeti
    
    Returns:
        final_prediction: Ağırlıklı ortalama
        weights: Her modelin ağırlığı
    """
    preds = np.array(predictions)
    n = len(preds)
    
    # Konsensüs ağırlığı (outlier'ları cezalandır)
    distances = np.array([
        np.mean(np.abs(preds[i] - np.delete(preds, i)))
        for i in range(n)
    ])
    consensus_w = np.exp(-distances / sigma)
    consensus_w /= consensus_w.sum()
    
    # Performans ağırlığı
    r2_scores = np.maximum(np.array(historical_r2), 0.0)
    if r2_scores.sum() > 0:
        perf_w = r2_scores / r2_scores.sum()
    else:
        perf_w = np.ones(n) / n
    
    # Kombine
    final_w = consensus_weight * consensus_w + performance_weight * perf_w
    final_w /= final_w.sum()
    
    final_pred = np.sum(preds * final_w)
    
    return final_pred, final_w


def robust_ensemble_with_fallback(
    predictions: np.ndarray,
    historical_r2: np.ndarray,
    sigma: float = 0.005
) -> Tuple[float, np.ndarray, str]:
    """
    Robust ensemble: Önce smart_ensemble; model sayısı <3 veya ayrışma yüksekse medyan/trimmed fallback.
    Returns: (final_pred, weights, method)
    """
    preds = np.array(predictions)
    n = len(preds)
    if n == 0:
        return 0.0, np.array([]), 'none'
    if n < 3:
        return median_ensemble(preds), np.ones(n)/n, 'median_fallback'
    # Smart ensemble
    final_pred, weights = smart_ensemble(preds, np.array(historical_r2), consensus_weight=0.8, performance_weight=0.2, sigma=sigma)
    # Ayrışma oranı: std / |mean|
    mean = float(np.mean(preds))
    std = float(np.std(preds))
    disagreement = (std / max(abs(mean), 1e-8)) if mean != 0 else 1.0
    if disagreement > 0.5:
        # Çok ayrışma → trimmed mean
        return trimmed_mean_ensemble(preds, trim_fraction=0.25), np.ones(n)/n, 'trimmed_fallback'
    return final_pred, weights, 'smart'


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def detect_outliers(
    predictions: np.ndarray,
    threshold: float = 2.0
) -> np.ndarray:
    """
    Outlier tespiti (Z-score bazlı)
    
    Args:
        predictions: Model tahminleri
        threshold: Z-score eşiği (default 2.0)
    
    Returns:
        is_outlier: Boolean array [False, False, True, False]
    """
    preds = np.array(predictions)
    mean = np.mean(preds)
    std = np.std(preds)
    
    if std == 0:
        return np.zeros(len(preds), dtype=bool)
    
    z_scores = np.abs((preds - mean) / std)
    
    return z_scores > threshold


def calculate_prediction_diversity(predictions: np.ndarray) -> float:
    """
    Tahmin çeşitliliği (diversity) hesapla
    
    Yüksek diversity → Modeller farklı şeyler öğrenmiş → Ensemble faydalı
    Düşük diversity → Modeller benzer → Ensemble az fayda
    
    Args:
        predictions: Model tahminleri
    
    Returns:
        diversity: Çeşitlilik skoru (0-1)
    """
    preds = np.array(predictions)
    
    # Coefficient of variation (CV)
    mean = np.mean(preds)
    std = np.std(preds)
    
    if mean == 0:
        return 0.0
    
    cv = std / abs(mean)
    
    # Normalize to 0-1 (CV > 0.5 → diversity = 1.0)
    diversity = min(cv / 0.5, 1.0)
    
    return float(diversity)


def ensemble_confidence(
    predictions: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Ensemble güven skoru hesapla
    
    Düşük variance + yüksek ağırlık konsensüsü → yüksek güven
    
    Args:
        predictions: Model tahminleri
        weights: Model ağırlıkları
    
    Returns:
        confidence: Güven skoru (0-1)
    """
    preds = np.array(predictions)
    w = np.array(weights)
    
    # Weighted variance
    weighted_mean = np.sum(preds * w)
    weighted_var = np.sum(w * (preds - weighted_mean) ** 2)
    
    # Normalize (variance < 0.01 → confidence = 1.0)
    confidence = 1.0 / (1.0 + weighted_var / 0.01)
    
    return float(confidence)
