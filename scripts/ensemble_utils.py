#!/usr/bin/env python3
"""
Smart ensemble utility

Combines base model predictions using a hybrid of:
- Consensus weighting (penalizes disagreement via Gaussian kernel)
- Performance weighting (uses historical R^2 as a proxy for model quality)

Inputs:
- predictions: np.ndarray shape (n_models,), numeric predictions on the same scale
- historical_r2: np.ndarray shape (n_models,), R^2 per model (can be negative)
- consensus_weight: float in [0,1]
- performance_weight: float in [0,1]
- sigma: float > 0, controls consensus sharpness (smaller → penalize disagreement more)

Returns:
- ensemble_pred: float, weighted average prediction
- final_weights: np.ndarray shape (n_models,), normalized weights used
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _safe_array(x, fallback_len: int, fill: float = 0.0) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float).flatten()
        if arr.size == 0:
            return np.full((fallback_len,), fill, dtype=float)
        return arr
    except Exception:
        return np.full((fallback_len,), fill, dtype=float)


def _normalize(weights: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    s = float(w.sum())
    if s <= 0.0 or not np.isfinite(s):
        n = max(1, w.size)
        return np.full((n,), 1.0 / n, dtype=float)
    return w / s


def smart_ensemble(
    predictions: np.ndarray,
    historical_r2: np.ndarray | list,
    consensus_weight: float = 0.6,
    performance_weight: float = 0.4,
    sigma: float = 0.005,
    prior_weights: np.ndarray | list | None = None,
) -> Tuple[float, np.ndarray]:
    preds = _safe_array(predictions, fallback_len=1, fill=0.0)
    n = preds.size
    if n == 1:
        return float(preds[0]), np.array([1.0], dtype=float)

    r2 = _safe_array(historical_r2, fallback_len=n, fill=0.0)
    if r2.size != n:
        # Align length
        if r2.size < n:
            r2 = np.pad(r2, (0, n - r2.size), mode='constant', constant_values=0.0)
        else:
            r2 = r2[:n]

    # Consensus weights: penalize deviation from mean prediction
    mu = float(np.mean(preds))
    var = float(np.var(preds)) if np.isfinite(np.var(preds)) else 0.0
    # Adaptive sigma: if provided sigma is too small/zero, use a fraction of empirical spread
    adaptive_sigma = max(1e-6, sigma if sigma and sigma > 0 else max(1e-6, math.sqrt(var) * 0.5))
    # Gaussian kernel around mean
    consensus_w = np.exp(-0.5 * ((preds - mu) / adaptive_sigma) ** 2)
    consensus_w = _normalize(consensus_w)

    # Performance weights: softmax over clipped R^2 (allow negatives, but compress extremes)
    r2_clipped = np.clip(r2, -0.5, 1.0)
    # Temperature to avoid overly peaky softmax
    perf_temp = 2.0
    perf_logits = r2_clipped / perf_temp
    # Stable softmax
    perf_logits -= np.max(perf_logits)
    perf_w = np.exp(perf_logits)
    perf_w = _normalize(perf_w)

    # Combine weights (convex combination)
    alpha = float(np.clip(consensus_weight, 0.0, 1.0))
    beta = float(np.clip(performance_weight, 0.0, 1.0))
    if alpha + beta <= 0:
        alpha, beta = 0.5, 0.5
    # Normalize coefficients to sum to 1
    s_ab = alpha + beta
    alpha /= s_ab
    beta /= s_ab

    combined = alpha * consensus_w + beta * perf_w
    final_w = _normalize(combined)

    # Apply optional per-model prior weights (e.g., from HPO)
    if prior_weights is not None:
        try:
            pw = _safe_array(prior_weights, fallback_len=n, fill=1.0)
            if pw.size != n:
                if pw.size < n:
                    pw = np.pad(pw, (0, n - pw.size), mode='constant', constant_values=1.0)
                else:
                    pw = pw[:n]
            # elementwise multiply and renormalize
            final_w = _normalize(final_w * np.maximum(pw, 0.0))
        except Exception:
            # ignore prior if anything goes wrong
            pass

    ensemble_pred = float(np.dot(final_w, preds))
    return ensemble_pred, final_w


__all__ = ["smart_ensemble"]
