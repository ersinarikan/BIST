"""
Visual Pattern Detector
- Provides get_visual_pattern_system()
- Uses ultralytics YOLO if available to perform visual inference
- Loads model path from YOLO_MODEL_PATH (env) or defaults to 'yolov8n.pt'
- Renders a simple price chart image from stock_data for inference
"""

from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)


class _VisualPatternSystem:

    def __init__(self):
        self.yolo_available = False
        self.backend = None
        self._model = None
        self._model_path = os.getenv('YOLO_MODEL_PATH')
        # Feature toggle
        self._enable_yolo = str(
            os.getenv('ENABLE_YOLO', 'true')
        ).lower() in ('1', 'true', 'yes')
        try:
            self._min_conf = float(os.getenv('YOLO_MIN_CONF', '0.33'))
        except Exception as e:
            logger.debug(f"Failed to get YOLO_MIN_CONF, using 0.33: {e}")
            self._min_conf = 0.33
        # Ensure YOLO config dir is writable
        os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/ultralytics')
        # Initial best-effort import
        self._try_import_backend()

    def _try_import_backend(self):
        try:
            from ultralytics import YOLO  # type: ignore
            self._YOLO = YOLO
            self.backend = 'ultralytics'
            self.yolo_available = True
        except Exception as e:
            logger.debug(f"Failed to import ultralytics YOLO: {e}")
            self._YOLO = None
            self.backend = None
            self.yolo_available = False

    def _ensure_model_loaded(self):
        # REDIRECTED to async system: Heavy YOLO model loading moved to
        # background
        # This method now just checks if async system is available
        if not self.yolo_available:
            # Retry import lazily
            self._try_import_backend()
            if not self.yolo_available:
                return False

        # Check if async visual system is available
        try:
            from visual_pattern_async import get_async_visual_pattern_system
            async_system = get_async_visual_pattern_system()
            return async_system.yolo_available and async_system._enable_yolo
        except ImportError:
            return False
        # Model loading is now handled by async system
        return True

    def get_system_info(self):
        # Get info from async system
        try:
            from visual_pattern_async import get_async_visual_pattern_system
            async_system = get_async_visual_pattern_system()
            return async_system.get_system_info()
        except ImportError:
            return {
                'yolo_available': bool(self.yolo_available),
                'backend': self.backend,
                'model_loaded': False,  # Sync system disabled
                'model_path': self._model_path or 'yolov8n.pt',
                'note': 'Using async visual system'
            }

    def _render_chart_image(self, stock_data):
        """Render a simple price chart (close line) to a numpy RGB image
        for YOLO.
        Returns numpy array or None on failure.
        """
        # Defensive: sanitize data (drop NaN/inf, ensure at least some
        # variance)
        def _sanitize_close(series):
            try:
                # local import to avoid top-level hard dep
                import numpy as _np
                vals = series.astype(float).values
                mask = _np.isfinite(vals)
                vals = vals[mask]
                if vals.size == 0:
                    return None
                return vals
            except Exception as e:
                logger.debug(f"Failed to sanitize close with numpy: {e}")
                try:
                    return series.astype(float).dropna().values  # type: ignore
                except Exception as e2:
                    logger.debug(f"Failed to sanitize close with pandas: {e2}")
                    return None
        # Prefer Matplotlib if available
        img = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np

            if stock_data is None or len(stock_data) < 20:
                raise RuntimeError("insufficient_data_matplotlib")

            close = (
                stock_data['close']
                if 'close' in getattr(stock_data, 'columns', [])
                else stock_data.get('Close')
            )
            if close is None:
                raise RuntimeError("no_close_column_matplotlib")
            series = close.tail(180)
            sane = _sanitize_close(series)
            if sane is None or len(sane) < 10:
                raise RuntimeError("sanitized_empty_matplotlib")

            fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(range(len(sane)), sane, color='#1f77b4', linewidth=2)
            ax.set_axis_off()
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            # tostring_rgb may be unavailable on some backends; guard it
            to_rgb = getattr(fig.canvas, 'tostring_rgb', None)
            if to_rgb is None:
                plt.close(fig)
                raise RuntimeError("no_tostring_rgb_matplotlib")
            buf = np.frombuffer(to_rgb(), dtype=np.uint8)
            img = buf.reshape((h, w, 3))
            plt.close(fig)
        except Exception as e:
            logger.debug(f"Failed to render chart with matplotlib: {e}")
            img = None

        # Fallback: NumPy rasterization (no Matplotlib required)
        if img is None:
            try:
                import numpy as np

                if stock_data is None or len(stock_data) < 20:
                    return None
                close = (
                    stock_data['close']
                    if 'close' in getattr(stock_data, 'columns', [])
                    else stock_data.get('Close')
                )
                if close is None:
                    return None
                series = close.tail(180)
                vals = _sanitize_close(series)
                if vals is None or len(vals) < 10:
                    return None
                vmin, vmax = np.min(vals), np.max(vals)
                if vmax - vmin <= 1e-9:
                    vmax = vmin + 1e-9
                norm = (vals - vmin) / (vmax - vmin)

                W = H = 640
                img = np.ones((H, W, 3), dtype=np.uint8) * 255
                xs = np.linspace(10, W - 10, len(norm)).astype(int)
                ys = (H - 10 - norm * (H - 20)).astype(int)

                for i in range(len(xs) - 1):
                    x0, y0 = xs[i], ys[i]
                    x1, y1 = xs[i + 1], ys[i + 1]
                    # simple Bresenham-like interpolation
                    n = max(abs(x1 - x0), abs(y1 - y0)) + 1
                    for t in range(n):
                        x = int(x0 + (x1 - x0) * t / max(1, n - 1))
                        y = int(y0 + (y1 - y0) * t / max(1, n - 1))
                        if 0 <= x < W and 0 <= y < H:
                            img[y, x, :] = [31, 119, 180]  # line color
                            if y + 1 < H:
                                img[y + 1, x, :] = [31, 119, 180]
            except Exception as e:
                logger.debug(
                    f"Failed to render chart with numpy fallback: {e}"
                )
                img = None
        return img

    def analyze_stock_visual(self, symbol, stock_data):
        """
        Visual analysis using async system (non-blocking)
        Now redirects to async system to prevent websocket blocking
        """
        if not self._enable_yolo:
            return {
                'status': 'disabled',
                'message': 'YOLO disabled by environment',
                'timestamp': datetime.now().isoformat(),
                'visual_analysis': {'patterns': []}
            }

        # Use async visual pattern system to prevent blocking
        try:
            from visual_pattern_async import get_async_visual_pattern_system
            async_system = get_async_visual_pattern_system()

            # Request async analysis
            request_id = async_system.request_visual_analysis_async(
                symbol, stock_data
            )

            # Wait briefly for immediate results (non-blocking check)
            import time
            time.sleep(0.1)  # Brief wait for fast completions

            result = async_system.get_visual_analysis_result(request_id)
            if result and result.get('status') in (
                'completed', 'error', 'disabled', 'unavailable'
            ):
                # Add timestamp for compatibility
                result['timestamp'] = datetime.now().isoformat()
                result['symbol'] = symbol
                return result
            else:
                # Return pending status - client can poll later
                return {
                    'status': 'pending',
                    'message': 'Visual analysis in progress...',
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'request_id': request_id,
                    'visual_analysis': {'patterns': []}
                }

        except ImportError:
            return {
                'status': 'error',
                'message': 'Async visual system not available',
                'timestamp': datetime.now().isoformat(),
                'visual_analysis': {'patterns': []}
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'YOLO analysis failed: {str(e)}',
                'timestamp': datetime.now().isoformat(),
                'visual_analysis': {'patterns': []}
            }


_instance = None


def get_visual_pattern_system():
    global _instance
    if _instance is None:
        _instance = _VisualPatternSystem()
    return _instance
