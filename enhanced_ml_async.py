"""
Enhanced ML Async Loading System
Background model loading with memory cache and queue-based predictions
"""

import threading
import queue
import time
import logging
import os
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import joblib
from collections import OrderedDict

logger = logging.getLogger(__name__)


class AsyncEnhancedMLSystem:
    """
    Async Enhanced ML System with background loading and memory cache

    Features:
    - Background model pre-loading
    - LRU memory cache for models
    - Queue-based prediction system
    - Non-blocking API calls
    - Intelligent model prioritization
    """

    def __init__(self, max_cache_size: int = 100, worker_threads: int = 4):
        self.max_cache_size = max_cache_size
        self.worker_threads = worker_threads

        # Model memory cache (LRU)
        self._model_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.RLock()

        # Background loading
        self._loader_pool = ThreadPoolExecutor(
            max_workers=worker_threads, thread_name_prefix="MLLoader"
        )
        self._loading_queue = queue.Queue()
        self._is_loading: Dict[str, bool] = {}

        # Prediction queue
        self._prediction_queue = queue.Queue()
        self._prediction_results: Dict[str, Any] = {}
        self._prediction_lock = threading.RLock()

        # Model directory (Enhanced ML models)
        self.model_directory = "/opt/bist-pattern/.cache/enhanced_ml_models"

        # Symbol prioritization (most frequently requested symbols load first)
        self._symbol_frequency: Dict[str, int] = {}

        # Start background workers
        self._start_background_workers()

        logger.info(
            f"🚀 Async Enhanced ML System initialized "
            f"(cache: {max_cache_size}, workers: {worker_threads})"
        )

    def _start_background_workers(self):
        """Start background model loading workers"""
        for i in range(self.worker_threads):
            worker = threading.Thread(
                target=self._model_loader_worker,
                daemon=True,
                name=f"MLLoader-{i}"
            )
            worker.start()

    def _model_loader_worker(self):
        """Background worker that loads models from queue"""
        while True:
            try:
                symbol = self._loading_queue.get(timeout=1.0)
                if symbol is None:  # Shutdown signal
                    break

                self._load_symbol_models_sync(symbol)
                self._loading_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Model loader worker error: {e}")
                continue

    def _load_symbol_models_sync(self, symbol: str):
        """Synchronously load all models for a symbol (runs in background)"""
        try:
            with self._cache_lock:
                # Mark as loading
                self._is_loading[symbol] = True

            models_loaded = {}
            prediction_horizons = ['1d', '3d', '7d', '14d', '30d']
            model_types = ['xgboost', 'lightgbm', 'catboost']

            for horizon in prediction_horizons:
                horizon_models = {}
                for model_type in model_types:
                    model_path = (
                        f"{self.model_directory}/{symbol}_{horizon}_"
                        f"{model_type}.pkl"
                    )

                    if os.path.exists(model_path):
                        try:
                            # This is the blocking operation, but it's in
                            # background thread
                            model_obj = joblib.load(model_path)
                            horizon_models[model_type] = {
                                'model': model_obj,
                                'loaded_at': time.time(),
                                'score': 0.0  # TODO: Load from metadata
                            }
                            logger.debug(
                                f"✅ Loaded {symbol}_{horizon}_"
                                f"{model_type}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to load {symbol}_{horizon}_"
                                f"{model_type}: {e}"
                            )

                if horizon_models:
                    models_loaded[f"{symbol}_{horizon}"] = horizon_models

            # Add to cache
            with self._cache_lock:
                self._add_to_cache(symbol, models_loaded)
                self._is_loading[symbol] = False

            logger.info(
                f"🎯 Background loaded models for {symbol} "
                f"({len(models_loaded)} horizons)"
            )

        except Exception as e:
            logger.error(f"Background model loading error for {symbol}: {e}")
            with self._cache_lock:
                self._is_loading[symbol] = False

    def _add_to_cache(self, symbol: str, models: Dict[str, Any]):
        """Add models to LRU cache"""
        # Remove if already exists (move to end)
        if symbol in self._model_cache:
            del self._model_cache[symbol]

        # Add to end
        self._model_cache[symbol] = models

        # Evict oldest if cache is full
        while len(self._model_cache) > self.max_cache_size:
            oldest_symbol = next(iter(self._model_cache))
            del self._model_cache[oldest_symbol]
            logger.debug(f"🗑️ Evicted {oldest_symbol} from model cache")

    def request_predictions_async(self, symbol: str, data: Any) -> str:
        """
        Request predictions asynchronously
        Returns a request_id for later retrieval
        """
        request_id = f"{symbol}_{int(time.time() * 1000)}"

        # Update symbol frequency
        self._symbol_frequency[symbol] = (
            self._symbol_frequency.get(symbol, 0) + 1
        )

        # Check if models are in cache
        with self._cache_lock:
            if symbol in self._model_cache:
                # Models available - immediate prediction
                try:
                    predictions = self._predict_with_cached_models(
                        symbol, data
                    )
                    with self._prediction_lock:
                        self._prediction_results[request_id] = {
                            'status': 'completed',
                            'predictions': predictions,
                            'timestamp': time.time()
                        }
                except Exception as e:
                    with self._prediction_lock:
                        self._prediction_results[request_id] = {
                            'status': 'error',
                            'error': str(e),
                            'timestamp': time.time()
                        }
            else:
                # Models not cached - queue for background loading
                if not self._is_loading.get(symbol, False):
                    try:
                        self._loading_queue.put_nowait(
                            symbol
                        )
                    except queue.Full:
                        logger.warning(
                            f"Model loading queue full, skipping {symbol}"
                        )

                # Return pending status
                with self._prediction_lock:
                    self._prediction_results[request_id] = {
                        'status': 'pending',
                        'message': f'Loading models for {symbol}...',
                        'timestamp': time.time()
                    }

        return request_id

    def get_predictions_result(
        self, request_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get prediction results by request_id"""
        with self._prediction_lock:
            return self._prediction_results.get(request_id)

    def _predict_with_cached_models(
        self, symbol: str, data: Any
    ) -> Dict[str, Any]:
        """Make predictions using cached models"""
        if symbol not in self._model_cache:
            return {}

        models = self._model_cache[symbol]
        predictions = {}

        # Move to end (LRU)
        self._model_cache.move_to_end(symbol)

        for model_key, horizon_models in models.items():
            # Extract horizon (1d, 3d, etc.)
            horizon = model_key.split('_')[-1]

            if not horizon_models:
                continue

            # Ensemble prediction from available models
            model_predictions = []
            for model_type, model_info in horizon_models.items():
                try:
                    # TODO: Implement actual feature engineering and prediction
                    # For now, return placeholder
                    model_predictions.append({
                        'type': model_type,
                        'prediction': 100.0,  # Placeholder
                        'confidence': model_info.get('score', 0.5)
                    })
                except Exception as e:
                    logger.warning(
                        f"Prediction error {symbol} {model_type}: {e}"
                    )

            if model_predictions:
                # Simple ensemble average
                avg_prediction = (
                    sum(p['prediction'] for p in model_predictions) /
                    len(model_predictions)
                )
                avg_confidence = (
                    sum(p['confidence'] for p in model_predictions) /
                    len(model_predictions)
                )

                predictions[horizon] = {
                    'prediction': avg_prediction,
                    'confidence': avg_confidence,
                    'models_used': [
                        p['type'] for p in model_predictions
                    ]
                }

        return predictions

    def preload_popular_symbols(self, symbols: list, max_preload: int = 50):
        """Preload models for popular symbols"""
        # Sort by frequency
        sorted_symbols = sorted(
            symbols,
            key=lambda s: self._symbol_frequency.get(s, 0),
            reverse=True
        )

        for symbol in sorted_symbols[:max_preload]:
            if (
                symbol not in self._model_cache and
                not self._is_loading.get(symbol, False)
            ):
                try:
                    self._loading_queue.put_nowait(symbol)
                    logger.info(
                        f"🎯 Queued preload for popular symbol: {symbol}"
                    )
                except queue.Full:
                    break

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._cache_lock:
            return {
                'cache_size': len(self._model_cache),
                'max_cache_size': self.max_cache_size,
                'cached_symbols': list(self._model_cache.keys()),
                'loading_symbols': [
                    s for s, loading in self._is_loading.items() if loading
                ],
                'symbol_frequency': dict(self._symbol_frequency),
                'queue_size': (
                    self._loading_queue.qsize()
                )
            }

    def cleanup_old_results(self, max_age_seconds: int = 300):
        """Clean up old prediction results"""
        cutoff_time = time.time() - max_age_seconds

        with self._prediction_lock:
            to_remove = [
                req_id
                for req_id, result in self._prediction_results.items()
                if result.get('timestamp', 0) < cutoff_time
            ]

            for req_id in to_remove:
                del self._prediction_results[req_id]

        if to_remove:
            logger.debug(
                f"🧹 Cleaned up {len(to_remove)} old prediction results"
            )


# Singleton instance
_async_enhanced_ml_instance = None


def get_async_enhanced_ml_system() -> AsyncEnhancedMLSystem:
    """Get singleton async enhanced ML system"""
    global _async_enhanced_ml_instance
    if _async_enhanced_ml_instance is None:
        _async_enhanced_ml_instance = AsyncEnhancedMLSystem()
    return _async_enhanced_ml_instance
