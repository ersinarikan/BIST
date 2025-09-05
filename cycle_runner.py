
"""
Single-Cycle Runner (sequential per symbol)
Her sembol için sırasıyla: veri güncelle → AI analizleri → (opsiyonel) alert → sonraki.
"""
from __future__ import annotations
import os, threading, time, logging
from typing import Optional

logger = logging.getLogger(__name__)

class SingleCycleRunner:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_flag: bool = False
        self._interval_seconds: int = 900
        self._last_cycle_summary: dict = {}

    def is_running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    def start(self, interval_seconds: int = 900) -> bool:
        if self.is_running():
            return True
        self._interval_seconds = max(60, int(interval_seconds))
        self._stop_flag = False
        self._thread = threading.Thread(target=self._run_forever,
                                        name="single_cycle_runner",
                                        daemon=True)
        self._thread.start()
        logger.info(f"SingleCycleRunner started, interval={self._interval_seconds}s")
        return True

    def stop(self) -> None:
        self._stop_flag = True
        logger.info("SingleCycleRunner requested to stop")

    def status(self) -> dict:
        return {
            "running": self.is_running(),
            "interval_seconds": self._interval_seconds,
            "last_cycle_summary": self._last_cycle_summary,
        }

    def _iter_symbols(self):
        from data_collector import get_data_collector
        collector = get_data_collector()
        env_syms = os.getenv("CYCLE_SYMBOLS")
        if env_syms:
            symbols = [s.strip().upper() for s in env_syms.split(",") if s.strip()]
        else:
            symbols = collector.get_bist_symbols()
        return symbols, collector

    def _per_symbol_analysis(self, symbol: str) -> dict:
        """Tek sembol için update → analiz → (opsiyonel) alert"""
        out = {"symbol": symbol, "updated": False, "analysis": "skipped"}
        # 1) Veri güncelle (son 7 gün). Yahoo yoğunluğu için hatada devam.
        try:
            # Tek istek: dene; yfinance hata verirse geç
            ok = self._collector.update_single_stock(symbol, days=7)
            out["updated"] = bool(ok)
            # Ratelimit (yfinance'ı yormama)
            time.sleep(float(os.getenv("CYCLE_PER_SYMBOL_SLEEP","0.4")))
        except Exception as e:
            out["update_error"] = str(e)

        # 2) Analiz (hybrid + ML; görsel YOLO hybrid tarafından tetikleniyorsa tek yerden)
        try:
            from app import app as flask_app
            from app import get_pattern_detector  # type: ignore
            with flask_app.app_context():
                det = get_pattern_detector()
                # Veri DB’den alınır; update başarısızsa mevcut veri ile analiz yapılır
                result = det.analyze_stock(symbol)
                out["analysis"] = "done" if result else "empty"
        except Exception as e:
            out["analysis_error"] = str(e)

        # 3) (Opsiyonel) Alert – konfig varsa kontrol et
        try:
            if os.getenv("CYCLE_ALERTS","1") in ("1","true","True","yes","YES"):
                from alert_system import get_alert_system
                alerts = get_alert_system()
                # Tüm konfigleri gezmek yerine çok sık tetiklenmesin diye düşük sıklık bırakalım
                # (Gerektiğinde symbol-bazlı kontrol eklenecek)
                if int(time.time()) % 5 == 0:
                    alerts.check_signals()
                out["alerts_checked"] = True
        except Exception as e:
            out["alerts_error"] = str(e)

        return out

    def run_once(self) -> dict:
        from app import app as flask_app
        summary: dict = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "per_symbol": []}
        try:
            with flask_app.app_context():
                symbols, collector = self._iter_symbols()
                self._collector = collector
            processed = 0
            for sym in symbols:
                if self._stop_flag:
                    break
                try:
                    item = self._per_symbol_analysis(sym)
                    summary["per_symbol"].append(item)
                    processed += 1
                except Exception as e:
                    summary["per_symbol"].append({"symbol": sym, "fatal_error": str(e)})
            summary["processed"] = processed
        except Exception as e:
            summary["cycle_error"] = str(e)

        summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._last_cycle_summary = summary
        return summary

    def _run_forever(self) -> None:
        while not self._stop_flag:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"SingleCycleRunner cycle error: {e}")
            for _ in range(self._interval_seconds):
                if self._stop_flag:
                    break
                time.sleep(1)

_runner: Optional[SingleCycleRunner] = None
def get_cycle_runner() -> SingleCycleRunner:
    global _runner
    if _runner is None:
        _runner = SingleCycleRunner()
    return _runner
