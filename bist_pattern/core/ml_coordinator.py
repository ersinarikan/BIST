"""
ML Coordinator - Enhanced + Basic ML sistemlerinin akıllı koordinasyonu
"""

import os
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple

logger = logging.getLogger(__name__)

# NOTE: Global training lock replaced with file-based lock for cross-process coordination
# Legacy threading lock kept for backward compatibility within single process
_global_training_lock = threading.RLock()
_global_training_status = {'active': False, 'started_by': None, 'started_at': None}


class MLCoordinator:
    """
    Enhanced ve Basic ML sistemlerinin koordineli çalışmasını sağlar
    
    Strategi:
    1. Basic ML - Her cycle'da hızlı tahmin
    2. Enhanced ML - Akıllı eğitim schedule'u ile background training
    3. Model yaşına göre retrain logic
    4. Resource-aware coordination
    """
    
    def __init__(self):
        # All configuration from environment variables
        log_path = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
        self.model_status_file = os.path.join(log_path, 'ml_model_status.json')
        self.global_lock_file = os.path.join(log_path, 'global_ml_training.lock')
        self._lock_context = None  # For file-based lock context manager
        
        try:
            self.max_model_age_days = int(os.getenv('ML_MAX_MODEL_AGE_DAYS', '10'))
        except Exception:
            self.max_model_age_days = 10
            
        try:
            # Align with enhanced_ml_system default: prefer ML_MIN_DATA_DAYS (fallback ML_MIN_DAYS), default 180
            self.min_data_days = int(os.getenv('ML_MIN_DATA_DAYS', os.getenv('ML_MIN_DAYS', '180')))
        except Exception:
            self.min_data_days = 180
            
        try:
            self.training_cooldown_hours = int(os.getenv('ML_TRAINING_COOLDOWN_HOURS', '6'))
        except Exception:
            self.training_cooldown_hours = 6
        # Round-robin configuration (environment-driven)
        try:
            self.candidate_cooldown_hours = int(os.getenv('ML_CANDIDATE_COOLDOWN_HOURS', '2'))
        except Exception:
            self.candidate_cooldown_hours = 2
            
        try:
            self.top_pool_size = int(os.getenv('ML_TOP_POOL_SIZE', '20'))
        except Exception:
            self.top_pool_size = 20
        
        # ML system instances
        self.basic_ml = None
        self.enhanced_ml = None
        
        # Model status tracking
        self.model_status = self._load_model_status()
        
        logger.info("🎯 ML Coordinator başlatıldı")
    
    def acquire_global_training_lock(self, requester: str = "unknown", timeout: int = 300) -> bool:
        """
        Global eğitim kilidi al - automation vs crontab çakışmasını önler
        
        CROSS-PROCESS FILE-BASED LOCK for true multi-process coordination
        
        Args:
            requester: Kilit isteyen sistem adı (automation, crontab, manual)
            timeout: Maksimum bekleme süresi (saniye)
        
        Returns:
            bool: Kilit başarıyla alındı mı
        """
        # Declare globals at the very beginning
        global _global_training_lock
        global _global_training_status
        
        try:
            from bist_pattern.utils.param_store_lock import file_lock
            
            # Acquire file-based lock
            self._lock_context = file_lock(self.global_lock_file, timeout_seconds=timeout)
            self._lock_context.__enter__()
            
            # Write lock metadata for debugging
            try:
                with open(self.global_lock_file, 'a') as f:
                    f.write(f"\n{requester}|{os.getpid()}|{time.time()}\n")
            except Exception:
                pass
            
            # Also update in-memory status for single-process queries
            _global_training_status.update({
                'active': True,
                'started_by': requester,
                'started_at': time.time()
            })
            
            logger.info(f"🔒 Global ML training lock acquired by {requester} (pid={os.getpid()})")
            return True
            
        except TimeoutError:
            logger.warning(f"⏰ Global ML training lock timeout after {timeout}s")
            return False
        except Exception as e:
            # Fallback to threading lock if file_lock unavailable
            logger.warning(f"⚠️ File-based lock failed, using threading lock: {e}")
            try:
                acquired = _global_training_lock.acquire(timeout=timeout)
                if acquired:
                    _global_training_status.update({
                        'active': True,
                        'started_by': requester,
                        'started_at': time.time()
                    })
                    logger.info(f"🔒 Global ML training lock acquired by {requester} (fallback)")
                    return True
                return False
            except Exception as e2:
                logger.error(f"❌ Global ML training lock error: {e2}")
                return False
    
    def release_global_training_lock(self):
        """Global eğitim kilidini serbest bırak"""
        # Declare globals at the very beginning
        global _global_training_lock
        global _global_training_status
        
        try:
            # Release file-based lock if active
            if self._lock_context is not None:
                try:
                    self._lock_context.__exit__(None, None, None)
                    self._lock_context = None
                    logger.info("🔓 Global ML training lock released (file-based)")
                except Exception as e:
                    logger.warning(f"⚠️ File-based lock release error: {e}")
            
            # Also update in-memory status
            _global_training_status.update({
                'active': False,
                'started_by': None,
                'started_at': None
            })
            
            # Release threading lock if it was acquired
            try:
                _global_training_lock.release()
            except Exception:
                pass  # May not have been acquired
                
        except Exception as e:
            logger.warning(f"⚠️ Global ML training lock release error: {e}")
    
    def is_global_training_active(self) -> Dict[str, Any]:
        """Global eğitim durumunu kontrol et"""
        # Declare global at the very beginning
        global _global_training_status
        
        status = dict(_global_training_status)
        if status['active'] and status['started_at']:
            status['duration'] = time.time() - status['started_at']
        return status
    
    def _load_model_status(self) -> Dict:
        """Model status cache'ini yükle"""
        try:
            if os.path.exists(self.model_status_file):
                with open(self.model_status_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Model status yüklenemedi: {e}")
        return {}
    
    def _save_model_status(self):
        """Model status cache'ini kaydet"""
        try:
            os.makedirs(os.path.dirname(self.model_status_file), exist_ok=True)
            with open(self.model_status_file, 'w') as f:
                json.dump(self.model_status, f, indent=2)
        except Exception as e:
            logger.error(f"Model status kaydedilemedi: {e}")

    # Meta yardımcıları (global durum için)
    def _get_meta(self, key: str, default=None):
        try:
            return (self.model_status.get('__meta__') or {}).get(key, default)
        except Exception:
            return default

    def _set_meta(self, key: str, value) -> None:
        try:
            if '__meta__' not in self.model_status:
                self.model_status['__meta__'] = {}
            self.model_status['__meta__'][key] = value
            self._save_model_status()
        except Exception:
            pass
    
    def _get_basic_ml(self):
        """Basic ML system lazy loading"""
        if self.basic_ml is None:
            try:
                from ml_prediction_system import get_ml_prediction_system
                self.basic_ml = get_ml_prediction_system()
            except Exception as e:
                logger.error(f"Basic ML yüklenemedi: {e}")
        return self.basic_ml
    
    def _get_enhanced_ml(self):
        """Enhanced ML system lazy loading"""
        if self.enhanced_ml is None:
            try:
                from enhanced_ml_system import get_enhanced_ml_system
                self.enhanced_ml = get_enhanced_ml_system()
            except Exception as e:
                logger.error(f"Enhanced ML yüklenemedi: {e}")
        return self.enhanced_ml
    
    def _has_complete_models(self, enhanced_ml, symbol: str) -> Tuple[bool, List[int]]:
        """Symbol için tüm ufuklarda (1/3/7/14/30g) en az bir model var mı?
        Returns: (complete, missing_horizons)
        """
        try:
            horizons: List[int] = list(getattr(enhanced_ml, 'prediction_horizons', [1, 3, 7, 14, 30]))
            model_dir: str = str(getattr(enhanced_ml, 'model_directory', '/opt/bist-pattern/.cache/enhanced_ml_models'))
            model_types: List[str] = ['xgboost', 'lightgbm', 'catboost']
            missing: List[int] = []
            for h in horizons:
                has_any = False
                for m in model_types:
                    fpath = f"{model_dir}/{symbol}_{h}d_{m}.pkl"
                    if os.path.exists(fpath):
                        has_any = True
                        break
                if not has_any:
                    missing.append(h)
            return (len(missing) == 0, missing)
        except Exception:
            # Güvenli varsayım: eksik kabul et (tüm ufuklar)
            try:
                horizons = list(getattr(enhanced_ml, 'prediction_horizons', [1, 3, 7, 14, 30]))
            except Exception:
                horizons = [1, 3, 7, 14, 30]
            return (False, horizons)
    
    def should_train_enhanced_model(self, symbol: str, data_length: int) -> bool:
        """Enhanced model eğitimi gerekli mi?"""
        if data_length < self.min_data_days:
            return False
        
        enhanced_ml = self._get_enhanced_ml()
        if not enhanced_ml:
            return False
        
        symbol_status = self.model_status.get(symbol, {})
        
        # 1) Hiç eğitim yapılmamışsa eğit
        if not symbol_status.get('enhanced_trained_at'):
            return True
        
        # 2) Ufuk seti eksikse eğit (mevcut tazelikten bağımsız)
        try:
            complete, _missing = self._has_complete_models(enhanced_ml, symbol)
            if not complete:
                return True
        except Exception:
            # Belirsizlikte eğitimi tercih et
            return True
        
        # 3) Model yaşı çok eskiyse eğit
        try:
            last_trained = datetime.fromisoformat(symbol_status['enhanced_trained_at'])
            if (datetime.now() - last_trained).days > self.max_model_age_days:
                return True
        except Exception:
            return True
        
        # 4) Cooldown aktifse bekle
        try:
            last_attempt = datetime.fromisoformat(symbol_status.get('last_training_attempt', '1970-01-01'))
            if (datetime.now() - last_attempt).total_seconds() < self.training_cooldown_hours * 3600:
                return False
        except Exception:
            pass
        
        # 5) Tüm ufuklar mevcut ve taze → eğitime gerek yok
        return False

    def evaluate_training_gate(self, symbol: str, data_length: int) -> Tuple[bool, str]:
        """Eğitim kapısını değerlendirin: (izin, neden).
        Neden kodları: ok, insufficient_data, enhanced_unavailable, cooldown_active, model_fresh_or_exists, unknown
        """
        try:
            if data_length < self.min_data_days:
                return False, 'insufficient_data'
            enhanced_ml = self._get_enhanced_ml()
            if not enhanced_ml:
                return False, 'enhanced_unavailable'
            symbol_status = self.model_status.get(symbol, {})
            # Model yaş kontrolü: taze ise engelle, yoksa (yeni/yaşlı) devam et
            is_recent_model = False
            try:
                ts = symbol_status.get('enhanced_trained_at')
                if ts:
                    last_trained = datetime.fromisoformat(ts)
                    is_recent_model = (datetime.now() - last_trained).days <= self.max_model_age_days
            except Exception:
                is_recent_model = False
            # Ufuk seti eksikse eğitime izin ver (tazelikten bağımsız)
            try:
                complete, _ = self._has_complete_models(enhanced_ml, symbol)
                if not complete:
                    return True, 'ok'
            except Exception:
                pass
            if is_recent_model:
                return False, 'model_fresh_or_exists'
            # Cooldown kontrolü
            try:
                last_attempt = datetime.fromisoformat(symbol_status.get('last_training_attempt', '1970-01-01'))
                if (datetime.now() - last_attempt).total_seconds() < self.training_cooldown_hours * 3600:
                    return False, 'cooldown_active'
            except Exception:
                pass
            return True, 'ok'
        except Exception:
            return False, 'unknown'
    
    def predict_with_coordination(self, symbol: str, data, sentiment_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Koordineli ML tahmin sistemi
        
        1. Basic ML her zaman çalışır (hızlı)
        2. Enhanced ML sadece model varsa çalışır
        3. Sonuçları birleştirir
        """
        result = {}
        data_length = len(data) if data is not None else 0
        
        # 1. Basic ML Tahmin (her zaman)
        basic_ml = self._get_basic_ml()
        if basic_ml and data_length >= 50:
            try:
                basic_pred = basic_ml.predict_prices(symbol, data, sentiment_score)
                if basic_pred:
                    result['basic'] = basic_pred
                    logger.debug(f"✅ Basic ML tahmin: {symbol}")
            except Exception as e:
                logger.error(f"Basic ML tahmin hatası {symbol}: {e}")
        
        # 2. Enhanced ML Tahmin (sadece model varsa)
        # ⭐ RE-ENABLED: Enhanced ML (Enhanced + Basic ML akıllı koordinasyonu)
        enhanced_ml = self._get_enhanced_ml()
        if enhanced_ml and data_length >= self.min_data_days:
            try:
                # Model varlığını kontrol et
                if enhanced_ml.has_trained_models(symbol):
                    enhanced_ml.load_trained_models(symbol)
                    # ⚡ Pass sentiment_score for prediction adjustment
                    enhanced_pred = enhanced_ml.predict_enhanced(symbol, data, sentiment_score=sentiment_score)
                    if enhanced_pred:
                        result['enhanced'] = enhanced_pred
                        logger.debug(f"✅ Enhanced ML tahmin: {symbol}")
                        
                        # Son tahmin zamanını güncelle
                        self._update_model_status(symbol, 'last_prediction_at', datetime.now().isoformat())
                else:
                    logger.debug(f"⚠️ Enhanced ML model yok: {symbol}")
            except Exception as e:
                logger.error(f"Enhanced ML tahmin hatası {symbol}: {e}")
        
        result['timestamp'] = datetime.now().isoformat()
        result['coordinator_version'] = '1.0'
        return result
    
    def train_enhanced_model_if_needed(self, symbol: str, data) -> bool:
        """
        Enhanced model eğitimi (sadece gerektiğinde)
        
        Returns:
            bool: Eğitim yapıldı mı?
        """
        data_length = len(data) if data is not None else 0
        
        if not self.should_train_enhanced_model(symbol, data_length):
            return False
        
        enhanced_ml = self._get_enhanced_ml()
        if not enhanced_ml:
            return False
        
        try:
            logger.info(f"🧠 Enhanced ML eğitimi başlatılıyor: {symbol}")
            # Eğitim denemesi kaydı
            self._update_model_status(symbol, 'last_training_attempt', datetime.now().isoformat())

            # File-based lock to avoid cross-process conflicts
            try:
                from bist_pattern.utils.param_store_lock import file_lock  # reuse helper
            except Exception:
                file_lock = None  # type: ignore

            models_root = getattr(enhanced_ml, 'model_directory', '/opt/bist-pattern/.cache/enhanced_ml_models')
            lock_target = os.path.join(str(models_root), f"{symbol}_train.locktarget")

            def _do_train() -> bool:
                # Eğitim yap
                training_result = enhanced_ml.train_enhanced_models(symbol, data)
                if training_result:
                    try:
                        enhanced_ml.save_enhanced_models(symbol)
                    except Exception:
                        pass
                    # Başarılı eğitim kaydı
                    self._update_model_status(symbol, 'enhanced_trained_at', datetime.now().isoformat())
                    self._update_model_status(symbol, 'data_length_at_training', data_length)
                    logger.info(f"✅ Enhanced ML eğitimi tamamlandı: {symbol}")
                    return True
                logger.warning(f"⚠️ Enhanced ML eğitimi başarısız: {symbol}")
                return False

            if file_lock is not None:
                try:
                    with file_lock(lock_target, timeout_seconds=300):
                        return _do_train()
                except TimeoutError:
                    logger.warning(f"⏰ File lock timeout for training: {symbol}")
                    return False
            # Fallback without lock
            return _do_train()
        except Exception as e:
            logger.error(f"Enhanced ML eğitimi hatası {symbol}: {e}")
            return False
    
    def _update_model_status(self, symbol: str, key: str, value: Any):
        """Model status güncelle"""
        if symbol not in self.model_status:
            self.model_status[symbol] = {}
        self.model_status[symbol][key] = value
        self._save_model_status()
    
    def get_training_candidates(self, symbols: List[str], max_candidates: int = 5) -> List[str]:
        """
        Deterministik round-robin: A→Z sıralı listede sıradaki uygun max_candidates sembol
        
        Uygunluk: eğitim cooldown'u dolmuş, mümkünse eski modeller öncelikli
        """
        if not symbols:
            return []

        # A→Z sırala
        ordered = sorted(set([s.strip().upper() for s in symbols if s]), key=lambda x: x)

        # Başlangıç işaretçisi: son seçilen sembolden sonrakinden başla
        last_sym = self._get_meta('rr_last_symbol', None)
        start_idx = 0
        if last_sym in ordered:
            try:
                start_idx = (ordered.index(last_sym) + 1) % len(ordered)
            except Exception:
                start_idx = 0

        picked: List[str] = []
        now = datetime.now()
        seen = 0
        idx = start_idx
        while seen < len(ordered) and len(picked) < max_candidates:
            sym = ordered[idx]
            symbol_status = self.model_status.get(sym, {})

            # Eğitim cooldown kontrolü
            skip = False
            try:
                last_attempt = datetime.fromisoformat(symbol_status.get('last_training_attempt', '1970-01-01'))
                if (now - last_attempt).total_seconds() < self.training_cooldown_hours * 3600:
                    skip = True
            except Exception:
                pass

            if not skip:
                picked.append(sym)

            idx = (idx + 1) % len(ordered)
            seen += 1

        # İşaretçiyi güncelle
        if picked:
            self._set_meta('rr_last_symbol', picked[-1])
        return picked
    
    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Coordinator istatistikleri"""
        total_symbols = len(self.model_status)
        trained_enhanced = len([s for s in self.model_status.values() if s.get('enhanced_trained_at')])
        
        return {
            'total_symbols_tracked': total_symbols,
            'enhanced_models_trained': trained_enhanced,
            'basic_ml_available': self._get_basic_ml() is not None,
            'enhanced_ml_available': self._get_enhanced_ml() is not None,
            'max_model_age_days': self.max_model_age_days,
            'min_data_days': self.min_data_days,
            'training_cooldown_hours': self.training_cooldown_hours
        }


# Global singleton
_ml_coordinator = None


def get_ml_coordinator() -> MLCoordinator:
    """ML Coordinator singleton"""
    global _ml_coordinator
    if _ml_coordinator is None:
        _ml_coordinator = MLCoordinator()
    return _ml_coordinator
