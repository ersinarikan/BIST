"""
Visual Pattern Detector
- Provides get_visual_pattern_system()
- Uses ultralytics YOLO if available to perform visual inference
- Loads model path from YOLO_MODEL_PATH (env) or defaults to 'yolov8n.pt'
- Renders a simple price chart image from stock_data for inference
"""

from datetime import datetime
import os

class _VisualPatternSystem:
    def __init__(self):
        self.yolo_available = False
        self.backend = None
        self._model = None
        self._model_path = os.getenv('YOLO_MODEL_PATH')
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
        except Exception:
            self._YOLO = None
            self.backend = None
            self.yolo_available = False

    def _ensure_model_loaded(self):
        if not self.yolo_available:
            # Retry import lazily
            self._try_import_backend()
            if not self.yolo_available:
                return False
        if self._model is not None:
            return True
        # Resolve model path
        model_path = self._model_path or 'yolov8n.pt'
        try:
            # Allowlist Ultralytics DetectionModel for PyTorch 2.6+ safe load
            try:
                import torch, ultralytics  # type: ignore
                if hasattr(torch, 'serialization'):
                    try:
                        torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
                    except Exception:
                        pass
            except Exception:
                pass
            self._model = self._YOLO(model_path)
            return True
        except Exception:
            # Fallback: try default remote model path
            try:
                self._model = self._YOLO('yolov8n.pt')
                return True
            except Exception:
                self._model = None
                return False

    def get_system_info(self):
        # Attempt lazy load so model_loaded reflects reality
        try:
            self._ensure_model_loaded()
        except Exception:
            pass
        return {
            'yolo_available': bool(self.yolo_available),
            'backend': self.backend,
            'model_loaded': bool(self._model is not None),
            'model_path': self._model_path or 'yolov8n.pt'
        }

    def _render_chart_image(self, stock_data):
        """Render a simple price chart (close line) to a numpy RGB image for YOLO.
        Returns numpy array or None on failure.
        """
        # Prefer Matplotlib if available
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np

            if stock_data is None or len(stock_data) < 20:
                return None

            close = stock_data['close'] if 'close' in getattr(stock_data, 'columns', []) else stock_data.get('Close')
            if close is None:
                return None
            series = close.tail(180)

            fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(range(len(series)), series.values, color='#1f77b4', linewidth=2)
            ax.set_axis_off()
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = buf.reshape((h, w, 3))
            plt.close(fig)
            return img
        except Exception:
            pass

        # Fallback: NumPy rasterization (no Matplotlib required)
        try:
            import numpy as np
            try:
                from PIL import Image
            except Exception:
                Image = None

            if stock_data is None or len(stock_data) < 20:
                return None
            close = stock_data['close'] if 'close' in getattr(stock_data, 'columns', []) else stock_data.get('Close')
            if close is None:
                return None
            series = close.tail(180).astype(float)
            vals = series.values
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
            return img
        except Exception:
            return None

    def analyze_stock_visual(self, symbol, stock_data):
        # Ensure backend/model
        if not self._ensure_model_loaded():
            return {
                'status': 'unavailable',
                'message': 'YOLOv8 not available',
                'timestamp': datetime.now().isoformat(),
                'visual_analysis': {'patterns': []}
            }
        # Render chart to image
        img = self._render_chart_image(stock_data)
        if img is None:
            return {
                'status': 'error',
                'message': 'Failed to render chart image from stock data',
                'timestamp': datetime.now().isoformat(),
                'visual_analysis': {'patterns': []}
            }
        try:
            results = self._model.predict(source=img, verbose=False)
            vis_patterns = []
            if results and len(results) > 0:
                res = results[0]
                names = getattr(res, 'names', getattr(self._model, 'names', {})) or {}
                boxes = getattr(res, 'boxes', None)
                if boxes is not None:
                    for b in boxes:
                        try:
                            cls_idx = int(b.cls.item()) if hasattr(b.cls, 'item') else int(b.cls)
                            conf = float(b.conf.item()) if hasattr(b.conf, 'item') else float(b.conf)
                            cls_name = str(names.get(cls_idx, f'class_{cls_idx}')).upper()
                            vis_patterns.append({
                                'pattern': cls_name,
                                'confidence': conf,
                                'source': 'VISUAL_YOLO',
                                'bbox': [
                                    float(b.xyxy[0][0]),
                                    float(b.xyxy[0][1]),
                                    float(b.xyxy[0][2]),
                                    float(b.xyxy[0][3]),
                                ] if hasattr(b, 'xyxy') else None
                            })
                        except Exception:
                            continue
            return {
                'status': 'success',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'visual_analysis': {
                    'patterns': vis_patterns
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat(),
                'visual_analysis': {'patterns': []}
            }


_instance = None

def get_visual_pattern_system():
    global _instance
    if _instance is None:
        _instance = _VisualPatternSystem()
    return _instance
