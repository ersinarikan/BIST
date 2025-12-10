"""
Async Visual Pattern Detection System
Background YOLO model loading with queue-based predictions
"""

import threading
import queue
import time
import logging
import os
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AsyncVisualPatternSystem:
    """
    Async YOLO Visual Pattern System with background loading

    Features:
    - Background YOLO model loading
    - Queue-based prediction system
    - Non-blocking API calls
    - Memory-efficient model caching
    - Graceful fallback when models unavailable
    """

    def __init__(self, worker_threads: int = 2):
        # Worker thread count is env-driven, falls back to provided/default
        try:
            self.worker_threads = int(
                os.getenv('YOLO_WORKER_THREADS', str(worker_threads))
            )
        except Exception:
            self.worker_threads = worker_threads

        # Model state
        self._model = None
        self._model_path = os.getenv(
            'YOLO_MODEL_PATH',
            '/opt/bist-pattern/yolo/patterns_all_v7_rectblend.pt'
        )
        self._model_loaded = False
        self._model_loading = False
        self._model_load_lock = threading.RLock()

        # YOLO configuration
        self._enable_yolo = str(
            os.getenv('ENABLE_YOLO', 'true')
        ).lower() in ('1', 'true', 'yes')
        try:
            self._min_conf = float(os.getenv('YOLO_MIN_CONF', '0.12'))
        except (ValueError, TypeError):
            self._min_conf = 0.12

        # YOLO backend
        self.yolo_available = False
        self._YOLO = None
        self._try_import_backend()

        # Prediction queue and workers
        self._prediction_queue = queue.Queue()
        self._prediction_results: Dict[str, Any] = {}
        self._prediction_lock = threading.RLock()

        # Background workers
        self._loader_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="YOLOLoader"
        )
        self._prediction_pool = ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix="YOLOPredict"
        )

        # Statistics
        self._stats = {
            'models_loaded': 0,
            'predictions_requested': 0,
            'predictions_completed': 0,
            'cache_hits': 0,
            'errors': 0,
            'detections_found': 0,   # number of requests with >=1 detection
            'detections_total': 0    # total detections across requests
        }

        # Start background model loading
        if self._enable_yolo and self.yolo_available:
            self._start_background_loading()

        logger.info(
            f"🎯 Async Visual Pattern System initialized "
            f"(enabled: {self._enable_yolo}, "
            f"available: {self.yolo_available})"
        )

    def _try_import_backend(self):
        """Try to import YOLO backend"""
        try:
            from ultralytics import YOLO
            self._YOLO = YOLO
            self.yolo_available = True

            # Set YOLO config directory to avoid permission issues
            os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/ultralytics')

            logger.debug("✅ YOLO backend available")
        except ImportError:
            logger.info(
                "📊 YOLO backend not available "
                "(ultralytics not installed)"
            )
        except Exception as e:
            logger.warning(f"📊 YOLO backend import failed: {e}")

    def _start_background_loading(self):
        """Start background model loading"""
        def load_model_async():
            try:
                with self._model_load_lock:
                    if self._model_loaded or self._model_loading:
                        return

                    self._model_loading = True
                    logger.info(
                        f"🔄 Background loading YOLO model: "
                        f"{self._model_path}"
                    )

                # This is the blocking operation, but in background thread
                if not os.path.exists(self._model_path):
                    logger.warning(
                        f"YOLO model file not found: {self._model_path}"
                    )
                    with self._model_load_lock:
                        self._model_loading = False
                    return

                # Load YOLO model (blocking)
                if not self._YOLO:
                    logger.warning("YOLO not available")
                    with self._model_load_lock:
                        self._model_loading = False
                    return

                model = self._YOLO(self._model_path)

                with self._model_load_lock:
                    self._model = model
                    self._model_loaded = True
                    self._model_loading = False
                    self._stats['models_loaded'] += 1

                logger.info(
                    f"✅ YOLO model loaded successfully: "
                    f"{self._model_path}"
                )

            except Exception as e:
                logger.error(f"❌ YOLO model loading failed: {e}")
                with self._model_load_lock:
                    self._model_loading = False
                    self._stats['errors'] += 1

        # Submit loading task to background thread
        self._loader_pool.submit(load_model_async)

    def _canonicalize_class_name(self, raw_name: str) -> str:
        """Map YOLO class name variants to canonical pattern names.
        Returns UPPER_SNAKE_CASE canonical name when possible; otherwise
        sanitized upper name.
        """
        try:
            n = (
                (raw_name or "").strip().lower()
                .replace('-', ' ').replace('_', ' ')
            )

            def has(*tokens):
                return all(tok in n for tok in tokens)
            if has('inverse', 'shoulder') or has('inverse', 'head'):
                return 'INVERSE_HEAD_SHOULDERS'
            if has('head', 'shoulder'):
                return 'HEAD_SHOULDERS'
            if has('double', 'top'):
                return 'DOUBLE_TOP'
            if has('double', 'bottom'):
                return 'DOUBLE_BOTTOM'
            if has('ascending', 'triangle'):
                return 'ASCENDING_TRIANGLE'
            if has('descending', 'triangle'):
                return 'DESCENDING_TRIANGLE'
            if has('rising', 'wedge'):
                return 'RISING_WEDGE'
            if has('falling', 'wedge'):
                return 'FALLING_WEDGE'
            if ('bull' in n and 'flag' in n) or has('bullish', 'flag'):
                return 'BULLISH_FLAG'
            if ('bear' in n and 'flag' in n) or has('bearish', 'flag'):
                return 'BEARISH_FLAG'
            if has('cup', 'handle'):
                return 'CUP_AND_HANDLE'
            # Fallback: sanitize
            import re
            up = (
                re.sub(r'[^a-z0-9]+', '_', n).strip('_').upper()
            )
            return up or 'UNKNOWN'
        except Exception:
            return 'UNKNOWN'

    def request_visual_analysis_async(
        self, symbol: str, stock_data: Any
    ) -> str:
        """
        Request visual pattern analysis asynchronously
        Returns a request_id for later retrieval
        """
        request_id = f"visual_{symbol}_{int(time.time() * 1000)}"
        self._stats['predictions_requested'] += 1

        if not self._enable_yolo:
            with self._prediction_lock:
                self._prediction_results[request_id] = {
                    'status': 'disabled',
                    'message': 'YOLO visual detection disabled',
                    'timestamp': time.time(),
                    'visual_analysis': {'patterns': []}
                }
            return request_id

        if not self.yolo_available:
            with self._prediction_lock:
                self._prediction_results[request_id] = {
                    'status': 'unavailable',
                    'message': 'YOLO backend not available',
                    'timestamp': time.time(),
                    'visual_analysis': {'patterns': []}
                }
            return request_id

        # Check if model is loaded
        with self._model_load_lock:
            if self._model_loaded and self._model is not None:
                # Model ready - queue for immediate processing
                self._prediction_pool.submit(
                    self._process_visual_prediction,
                    request_id, symbol, stock_data
                )
            elif self._model_loading:
                # Model loading - return pending status
                with self._prediction_lock:
                    self._prediction_results[request_id] = {
                        'status': 'pending',
                        'message': 'YOLO model loading...',
                        'timestamp': time.time(),
                        'visual_analysis': {'patterns': []}
                    }
            else:
                # Model not loaded and not loading - try to start loading
                self._start_background_loading()
                with self._prediction_lock:
                    self._prediction_results[request_id] = {
                        'status': 'pending',
                        'message': 'YOLO model loading...',
                        'timestamp': time.time(),
                        'visual_analysis': {'patterns': []}
                    }

        return request_id

    def _process_visual_prediction(
        self, request_id: str, symbol: str, stock_data: Any
    ):
        """Process visual prediction in background thread"""
        try:
            with self._model_load_lock:
                if not self._model_loaded or self._model is None:
                    with self._prediction_lock:
                        self._prediction_results[request_id] = {
                            'status': 'error',
                            'message': 'YOLO model not available',
                            'timestamp': time.time(),
                            'visual_analysis': {'patterns': []}
                        }
                    return

                model = self._model

            # Render chart image (simplified version)
            img = self._render_chart_image(stock_data)
            if img is None:
                with self._prediction_lock:
                    self._prediction_results[request_id] = {
                        'status': 'error',
                        'message': 'Chart rendering failed',
                        'timestamp': time.time(),
                        'visual_analysis': {'patterns': []}
                    }
                return

            # Run YOLO inference
            results = model(img, conf=self._min_conf, verbose=False)

            # Parse results
            patterns = []
            if results and len(results) > 0:
                result = results[0]
                # Try to get class name mapping from result or model
                try:
                    names = getattr(result, 'names', None)
                    if not names:
                        names = getattr(model, 'names', None)
                    if isinstance(names, list):
                        names = {
                            i: n for i, n in enumerate(names)
                        }
                    if not isinstance(names, dict):
                        names = {}
                except (AttributeError, TypeError, ValueError):
                    names = {}
                # Extract boxes
                boxes = getattr(result, 'boxes', None)
                if (
                    boxes is not None and hasattr(boxes, 'cls') and
                    hasattr(boxes, 'conf')
                ):
                    # Determine number of detections robustly across
                    # ultralytics versions
                    try:
                        if hasattr(boxes, 'cls'):
                            num = int(len(boxes.cls))  # torch tensor length
                        elif hasattr(boxes, 'xyxy'):
                            num = int(len(boxes.xyxy))
                        elif hasattr(boxes, 'data'):
                            num = int(len(boxes.data))
                        else:
                            num = 0
                    except (AttributeError, TypeError, ValueError):
                        num = 0
                    for i in range(num):
                        try:
                            conf = (
                                float(boxes.conf[i])
                                if hasattr(boxes, 'conf') else 0.0
                            )
                            cls_idx = (
                                int(boxes.cls[i])
                                if hasattr(boxes, 'cls') else 0
                            )
                            if conf < self._min_conf:
                                continue
                            raw_name = (
                                str(names.get(cls_idx, f'class_{cls_idx}'))
                                if names else f'class_{cls_idx}'
                            )
                            cls_name = self._canonicalize_class_name(raw_name)
                            patterns.append({
                                'pattern': cls_name,
                                'confidence': conf,
                                'source': 'VISUAL_YOLO'
                            })
                        except (
                            KeyError, IndexError, ValueError, TypeError
                        ) as e:
                            logger.debug(f"Pattern parsing error: {e}")

            # Update detection stats
            try:
                if patterns:
                    self._stats['detections_found'] += 1
                    self._stats['detections_total'] += len(patterns)
                    logger.info(
                        f"📸 YOLO visual detections for {symbol}: "
                        f"{len(patterns)}"
                    )
            except Exception:
                pass

            # Store result
            with self._prediction_lock:
                self._prediction_results[request_id] = {
                    'status': 'completed',
                    'message': f'Found {len(patterns)} visual patterns',
                    'timestamp': time.time(),
                    'visual_analysis': {
                        'patterns': patterns,
                        'model_confidence': self._min_conf,
                        'processing_time': (
                            time.time()
                        )
                    }
                }

            self._stats['predictions_completed'] += 1
            logger.debug(
                f"✅ YOLO visual analysis completed for {symbol}: "
                f"{len(patterns)} patterns"
            )

        except Exception as e:
            logger.error(f"❌ YOLO prediction error for {symbol}: {e}")
            with self._prediction_lock:
                self._prediction_results[request_id] = {
                    'status': 'error',
                    'message': f'YOLO prediction failed: {str(e)}',
                    'timestamp': time.time(),
                    'visual_analysis': {'patterns': []}
                }
            self._stats['errors'] += 1

    def _render_chart_image(self, stock_data):
        """Render stock chart to image for YOLO analysis"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import pandas as pd
            import io
            from PIL import Image

            # Convert to DataFrame if needed
            if not isinstance(stock_data, pd.DataFrame):
                return None

            if len(stock_data) < 20:
                return None

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot candlestick-style chart (simplified)
            # Last 100 days
            stock_data = stock_data.tail(100)
            ax.plot(stock_data['close'], linewidth=2, color='blue')

            if 'volume' in stock_data.columns:
                ax2 = ax.twinx()
                ax2.bar(
                    range(len(stock_data)), stock_data['volume'],
                    alpha=0.3, color='gray'
                )
                ax2.set_ylabel('Volume')

            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)

            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)

            # Convert to PIL Image
            img = Image.open(buf)
            plt.close(fig)

            return img

        except Exception as e:
            logger.warning(f"Chart rendering error: {e}")
            return None

    def get_visual_analysis_result(
        self, request_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get visual analysis results by request_id"""
        with self._prediction_lock:
            return self._prediction_results.get(request_id)

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and statistics"""
        with self._model_load_lock:
            return {
                'yolo_available': self.yolo_available,
                'enabled': self._enable_yolo,
                'model_loaded': self._model_loaded,
                'model_loading': self._model_loading,
                'model_path': self._model_path,
                'min_confidence': self._min_conf,
                'statistics': dict(self._stats),
                'queue_size': (
                    self._prediction_queue.qsize()
                    if hasattr(self._prediction_queue, 'qsize') else 0
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
                f"🧹 Cleaned up {len(to_remove)} old visual analysis results"
            )


# Singleton instance
_async_visual_system_instance = None


def get_async_visual_pattern_system() -> AsyncVisualPatternSystem:
    """Get singleton async visual pattern system"""
    global _async_visual_system_instance
    if _async_visual_system_instance is None:
        _async_visual_system_instance = AsyncVisualPatternSystem()
    return _async_visual_system_instance
