#!/usr/bin/env python3
"""
Sabit Parametre + Tüm Feature Kombinasyonları Testi

Bu script, HPO'dan bağımsız sabit bir parametre seti kullanarak
tüm feature kombinasyonlarını test eder. Bu yaklaşım:
- HPO bias'ı yok (feature setine göre optimize edilmemiş)
- Daha adil karşılaştırma (tüm kombinasyonlar aynı parametrelerle)
- Feature interaction'ları tespit eder
- Synergy'leri bulur
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from itertools import product

sys.path.insert(0, '/opt/bist-pattern')

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

from enhanced_ml_system import EnhancedMLSystem
from bist_pattern.core.config_manager import ConfigManager

# Import from test_feature_combinations
sys.path.insert(0, '/opt/bist-pattern/scripts')
from test_feature_combinations import (
    FEATURES, fetch_prices, setup_test_environment
)

# ⚡ CRITICAL: HPO ile aynı fonksiyonlar
def compute_returns(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Compute forward returns for given horizon (HPO ile aynı)."""
    return df['close'].shift(-horizon) / df['close'] - 1.0

def dirhit(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 0.005) -> float:
    """Compute directional hit rate (HPO ile aynı)."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    yt = np.sign(y_true)
    yp = np.sign(y_pred)
    m = (np.abs(y_true) > thr) & (np.abs(y_pred) > thr)
    if m.sum() == 0:
        return 0.0
    return float(np.mean(yt[m] == yp[m]) * 100.0)

logger = logging.getLogger(__name__)

def check_feature_readiness(symbol: str, train_size: int, test_size: int, min_external_days: int = 30) -> List[str]:
    """Hazır olan feature'ları kontrol et ve listele"""
    ready_features = []
    
    # External Features (FinGPT + YOLO)
    feature_dir = os.getenv('EXTERNAL_FEATURE_DIR', '/opt/bist-pattern/logs/feature_backfill')
    fingpt_file = os.path.join(feature_dir, 'fingpt', f'{symbol}.csv')
    yolo_file = os.path.join(feature_dir, 'yolo', f'{symbol}.csv')
    
    fingpt_ready = False
    yolo_ready = False
    
    if os.path.exists(fingpt_file):
        try:
            import pandas as pd
            df = pd.read_csv(fingpt_file)
            if 'date' in df.columns and len(df) >= min_external_days:
                fingpt_ready = True
        except:
            pass
    
    if os.path.exists(yolo_file):
        try:
            import pandas as pd
            df = pd.read_csv(yolo_file)
            if 'date' in df.columns and len(df) >= min_external_days:
                yolo_ready = True
        except:
            pass
    
    # Feature'ları ekle
    # ⚡ FIX: ENABLE_EXTERNAL_FEATURES bir master switch
    # enhanced_ml_system.py içinde FinGPT ve YOLO ayrı ayrı kontrol ediliyor
    # Bu yüzden FinGPT VEYA YOLO hazırsa ENABLE_EXTERNAL_FEATURES eklenebilir
    if fingpt_ready or yolo_ready:
        ready_features.append('ENABLE_EXTERNAL_FEATURES')
    if fingpt_ready:
        ready_features.append('ENABLE_FINGPT_FEATURES')
    if yolo_ready:
        ready_features.append('ENABLE_YOLO_FEATURES')
    
    # Her zaman hazır olan feature'lar
    always_ready = [
        'ML_USE_DIRECTIONAL_LOSS',
        'ENABLE_SEED_BAGGING',
        'ML_USE_ADAPTIVE_LEARNING',
        'ENABLE_TALIB_PATTERNS',
        'ML_USE_SMART_ENSEMBLE',
        'ML_USE_STACKED_SHORT',
        'ENABLE_META_STACKING',
        'ML_USE_REGIME_DETECTION',
        'ENABLE_FINGPT'
    ]
    ready_features.extend(always_ready)
    
    return ready_features

def get_fixed_params(horizon: int) -> Dict:
    """
    Sabit parametre seti (HPO'dan bağımsız, mantıklı orta değerler)
    
    HPO range'lerinin ortasından alınmış mantıklı değerler.
    Horizon'a göre ayarlanmış.
    """
    if horizon in (1, 3):
        # Kısa horizon için daha konservatif
        return {
            'xgb_n_estimators': 375,
            'xgb_max_depth': 4,
            'xgb_learning_rate': 0.05,
            'xgb_subsample': 0.7,
            'xgb_colsample_bytree': 0.7,
            'xgb_reg_alpha': 0.01,
            'xgb_reg_lambda': 0.1,
            'xgb_min_child_weight': 10,
            'xgb_gamma': 0.01,
            'lgb_n_estimators': 260,
            'lgb_max_depth': 5,
            'lgb_learning_rate': 0.05,
            'lgb_num_leaves': 23,
            'lgb_subsample': 0.7,
            'lgb_colsample_bytree': 0.7,
            'lgb_reg_alpha': 0.01,
            'lgb_reg_lambda': 0.1,
            'cat_iterations': 260,
            'cat_depth': 5,
            'cat_learning_rate': 0.05,
            'cat_l2_leaf_reg': 0.1,
            'cat_subsample': 0.7,
            'cat_rsm': 0.7,
            'adaptive_k': 2.5,
            'pattern_weight': 1.0,
        }
    else:
        # Uzun horizon için daha agresif
        return {
            'xgb_n_estimators': 525,
            'xgb_max_depth': 6,
            'xgb_learning_rate': 0.08,
            'xgb_subsample': 0.8,
            'xgb_colsample_bytree': 0.75,
            'xgb_reg_alpha': 0.01,
            'xgb_reg_lambda': 0.1,
            'xgb_min_child_weight': 8,
            'xgb_gamma': 0.01,
            'lgb_n_estimators': 200,
            'lgb_max_depth': 5,
            'lgb_learning_rate': 0.06,
            'lgb_num_leaves': 23,
            'lgb_subsample': 0.8,
            'lgb_colsample_bytree': 0.8,
            'lgb_reg_alpha': 0.01,
            'lgb_reg_lambda': 0.1,
            'cat_iterations': 200,
            'cat_depth': 5,
            'cat_learning_rate': 0.06,
            'cat_l2_leaf_reg': 0.1,
            'cat_subsample': 0.8,
            'cat_rsm': 0.8,
            'adaptive_k': 2.0,
            'pattern_weight': 1.0,
        }

def convert_to_ml_params(fixed_params: Dict) -> Dict:
    """Sabit parametreleri ML sistem formatına çevir"""
    return {
        'best_trial_number': 42,  # Sabit seed
        'xgb': {
            'n_estimators': int(fixed_params['xgb_n_estimators']),
            'max_depth': int(fixed_params['xgb_max_depth']),
            'learning_rate': fixed_params['xgb_learning_rate'],
            'subsample': fixed_params['xgb_subsample'],
            'colsample_bytree': fixed_params['xgb_colsample_bytree'],
            'reg_alpha': fixed_params['xgb_reg_alpha'],
            'reg_lambda': fixed_params['xgb_reg_lambda'],
            'min_child_weight': int(fixed_params['xgb_min_child_weight']),
            'gamma': fixed_params['xgb_gamma'],
        },
        'lgb': {
            'n_estimators': int(fixed_params['lgb_n_estimators']),
            'max_depth': int(fixed_params['lgb_max_depth']),
            'learning_rate': fixed_params['lgb_learning_rate'],
            'num_leaves': int(fixed_params['lgb_num_leaves']),
            'subsample': fixed_params['lgb_subsample'],
            'colsample_bytree': fixed_params['lgb_colsample_bytree'],
            'reg_alpha': fixed_params['lgb_reg_alpha'],
            'reg_lambda': fixed_params['lgb_reg_lambda'],
        },
        'cat': {
            'iterations': int(fixed_params['cat_iterations']),
            'depth': int(fixed_params['cat_depth']),
            'learning_rate': fixed_params['cat_learning_rate'],
            'l2_leaf_reg': fixed_params['cat_l2_leaf_reg'],
            'subsample': fixed_params['cat_subsample'],
            'rsm': fixed_params['cat_rsm'],
        },
        'adaptive_k': fixed_params['adaptive_k'],
        'pattern_weight': fixed_params['pattern_weight'],
    }

def set_features(feature_config: Dict[str, bool]):
    """Feature'ları ayarla (test_feature_combinations.py'den)"""
    os.environ['ML_USE_SMART_ENSEMBLE'] = '1'
    os.environ['ML_USE_STACKED_SHORT'] = '1'
    os.environ['ML_USE_REGIME_DETECTION'] = '1'
    os.environ['ENABLE_TALIB_PATTERNS'] = '1'
    os.environ['ENABLE_FINGPT'] = '0'  # HPO'da kapalı
    
    for feature, enabled in feature_config.items():
        os.environ[feature] = '1' if enabled else '0'
    
    if not feature_config.get('ENABLE_EXTERNAL_FEATURES', False):
        os.environ['ENABLE_FINGPT_FEATURES'] = '0'
        os.environ['ENABLE_YOLO_FEATURES'] = '0'
    
    if feature_config.get('ENABLE_SEED_BAGGING', False):
        os.environ['N_SEEDS'] = '3'
    else:
        os.environ.pop('N_SEEDS', None)
    
    # ⚡ CRITICAL: Evaluation mode'da Phase 2'yi her zaman skip et
    os.environ['ML_SKIP_ADAPTIVE_PHASE2'] = '1'
    if feature_config.get('ML_USE_ADAPTIVE_LEARNING', False):
        logger.warning("⚠️ Adaptive Learning AÇIK ama Phase 2 SKIP (evaluation mode)")
    else:
        logger.info("⚙️ Adaptive Learning KAPALI: Phase 2 skip edilecek")

def train_and_evaluate(symbol: str, horizon: int, train_df: pd.DataFrame, 
                      test_df: pd.DataFrame, fixed_params: Dict, 
                      feature_config: Dict[str, bool], test_folder: Path) -> Dict:
    """Sabit parametrelerle eğitim ve değerlendirme"""
    # ConfigManager cache'i temizle
    ConfigManager.clear_cache()
    
    # Feature'ları ayarla
    set_features(feature_config)
    
    # ML sistem oluştur
    ml = EnhancedMLSystem()
    ml.save_enhanced_models = lambda s: None  # Model kaydetme
    
    # Seed bagging ayarları
    if feature_config.get('ENABLE_SEED_BAGGING', False):
        n_seeds = int(ConfigManager.get('N_SEEDS', '3'))
        ml.enable_seed_bagging = True
        ml.n_seeds = n_seeds
        ml.base_seeds = [42 + i for i in range(n_seeds)]  # Sabit seed'ler
    else:
        ml.enable_seed_bagging = False
        ml.n_seeds = 1
        ml.base_seeds = [42]
    
    # Parametreleri ayarla
    ml_params = convert_to_ml_params(fixed_params)
    
    # Eğitim
    logger.info(f"🚀 Eğitim başlıyor: {symbol} {horizon}d")
    logger.info(f"   Feature config: {feature_config}")
    logger.info(f"   Sabit parametreler: {fixed_params}")
    
    # ⚡ CRITICAL: Save original adaptive learning setting (before try block)
    original_adaptive = os.environ.get('ML_USE_ADAPTIVE_LEARNING', '0')
    
    try:
        # Parametreleri environment variable'lara set et
        for key, value in fixed_params.items():
            os.environ[f'OPTUNA_{key.upper()}'] = str(value)
        
        train_result = ml.train_enhanced_models(
            symbol=symbol,
            data=train_df
        )
        
        # ⚡ FIX: train_enhanced_models başarılı durumda dict döndürür (results)
        # Hata durumunda False döndürür
        # Dict döndürmüşse başarılı, False ise başarısız
        if train_result is False or train_result is None:
            logger.error(f"❌ Eğitim başarısız: {train_result if train_result else 'Unknown'}")
            return {'dirhit': None, 'error': 'Training returned False'}
        
        # train_result bir dict ise (başarılı), devam et
        if not isinstance(train_result, dict):
            logger.error(f"❌ Eğitim beklenmeyen sonuç döndürdü: {type(train_result)}")
            return {'dirhit': None, 'error': f'Unexpected return type: {type(train_result)}'}
        
        # ⚡ CRITICAL: Disable adaptive learning during validation to prevent data leakage
        # Training sırasında adaptive learning açık olabilir, ama validation'da KAPALI olmalı
        os.environ['ML_USE_ADAPTIVE_LEARNING'] = '0'
        logger.info(f"🔒 Adaptive learning disabled for validation (was: {original_adaptive})")
        
        # ⚡ CRITICAL: HPO ile TAM AYNI format ve mantık
        # HPO'da: y_true = compute_returns(test_df, horizon) → Series (NaN'lar var)
        # HPO'da: preds = np.full(len(test_df), np.nan) → Array
        # HPO'da: for t in range(len(test_df) - horizon)
        # HPO'da: preds[t] = ... → Index alignment garantili
        
        logger.info(f"🔮 Tahmin başlıyor: {symbol} {horizon}d")
        
        # HPO ile aynı: compute_returns kullan
        y_true = compute_returns(test_df, horizon)  # Series (HPO ile aynı)
        preds = np.full(len(test_df), np.nan, dtype=float)  # Array (HPO ile aynı)
        
        valid_predictions = 0
        failed_predictions = 0
        
        # HPO ile aynı loop: for t in range(len(test_df) - horizon)
        for t in range(len(test_df) - horizon):
            # ⚡ CRITICAL: HPO ile aynı - train_df + test_df[:t+1] birleştir
            # Bu sayede model hem train hem de test verisini görüyor (walk-forward)
            # Ama sadece t'ye kadar olan test verisi kullanılıyor (future data leakage yok)
            try:
                cur = pd.concat([train_df, test_df.iloc[: t + 1]], axis=0).copy()
                
                # ⚡ DEFENSIVE: Validate concat result (HPO ile aynı)
                if cur.index.duplicated().any():
                    logger.warning(f"⚠️ Duplicate indices after concat at t={t}, skipping")
                    failed_predictions += 1
                    continue
                
                if not cur.index.is_monotonic_increasing:
                    logger.warning(f"⚠️ Non-monotonic index after concat at t={t}, skipping")
                    failed_predictions += 1
                    continue
                
                # ⚡ DEFENSIVE: Validate cur has enough data for feature engineering
                if len(cur) < 60:  # Minimum for 60-day rolling features
                    logger.warning(f"⚠️ Insufficient data for feature engineering at t={t}: {len(cur)} days, skipping")
                    # Continue anyway (ffill will handle missing features)
                
                pred_result = ml.predict_enhanced(
                    symbol=symbol,
                    current_data=cur
                )
            except Exception as concat_err:
                logger.warning(f"⚠️ Concat/prediction failed at t={t}: {concat_err}, skipping")
                failed_predictions += 1
                continue
            
            # ⚡ CRITICAL: HPO ile aynı prediction parsing
            if not isinstance(pred_result, dict):
                failed_predictions += 1
                continue
            
            horizon_key = f"{horizon}d"
            pred_data = pred_result.get(horizon_key)
            if isinstance(pred_data, dict):
                pred_price = pred_data.get('ensemble_prediction')
                try:
                    if isinstance(pred_price, (int, float)) and not np.isnan(pred_price):
                        last_close = float(cur['close'].iloc[-1])
                        if last_close > 0:
                            # HPO ile aynı: preds[t] = float(pred_price) / last_close - 1.0
                            preds[t] = float(pred_price) / last_close - 1.0
                            valid_predictions += 1
                        else:
                            failed_predictions += 1
                    else:
                        failed_predictions += 1
                except Exception as parse_err:
                    logger.warning(f"⚠️ Prediction parsing failed at t={t}: {parse_err}")
                    failed_predictions += 1
            else:
                failed_predictions += 1
        
        # HPO ile aynı: dirhit(y_true, preds) - direkt Series ve array olarak
        # y_true Series, preds array - pandas otomatik dönüştürüyor
        dirhit_value = dirhit(y_true.values, preds)
        
        # HPO ile aynı: mask count hesapla
        thr = 0.005
        try:
            mask_count = int(((np.abs(y_true.values) > thr) & (np.abs(preds) > thr)).sum())
        except Exception:
            mask_count = 0
        
        logger.info(f"✅ DirHit: {dirhit_value:.2f}% (valid_predictions={valid_predictions}, failed_predictions={failed_predictions}, mask_after_thr={mask_count}, test_days={len(test_df)})")
        
        # ⚡ CRITICAL: Restore original adaptive learning setting
        os.environ['ML_USE_ADAPTIVE_LEARNING'] = original_adaptive
        logger.debug(f"🔓 Adaptive learning restored to: {original_adaptive}")
        
        return {
            'dirhit': dirhit_value if not np.isnan(dirhit_value) else None,
            'n_predictions': valid_predictions,
            'n_failed': failed_predictions,
            'n_masked': mask_count
        }
        
    except Exception as e:
        logger.error(f"❌ Hata: {e}", exc_info=True)
        # ⚡ CRITICAL: Restore original adaptive learning setting even on error
        os.environ['ML_USE_ADAPTIVE_LEARNING'] = original_adaptive
        return {'dirhit': None, 'error': str(e)}

def test_all_combinations(symbol: str, horizon: int, train_df: pd.DataFrame,
                         test_df: pd.DataFrame, test_folder: Path,
                         test_features: Optional[List[str]] = None) -> List[Dict]:
    """
    Tüm feature kombinasyonlarını test et
    
    Args:
        test_features: Test edilecek feature'lar (None ise 12 feature)
    """
    if test_features is None:
        # ⚡ SMART: Sadece hazır olan feature'ları kullan
        # Bu sayede gereksiz kombinasyonlardan kaçınırız
        ready_features = check_feature_readiness(symbol, len(train_df), len(test_df), min_external_days=30)
        
        # Tüm olası feature'lar
        all_possible_features = [
            'ENABLE_EXTERNAL_FEATURES',
            'ENABLE_FINGPT_FEATURES',
            'ENABLE_YOLO_FEATURES',
            'ML_USE_DIRECTIONAL_LOSS',
            'ENABLE_SEED_BAGGING',
            'ML_USE_ADAPTIVE_LEARNING',
            'ENABLE_TALIB_PATTERNS',
            'ML_USE_SMART_ENSEMBLE',
            'ML_USE_STACKED_SHORT',
            'ENABLE_META_STACKING',
            'ML_USE_REGIME_DETECTION',
            'ENABLE_FINGPT'
        ]
        
        # Sadece hazır olanları filtrele
        test_features = [f for f in all_possible_features if f in ready_features]
        
        logger.info(f"📊 Feature readiness check:")
        logger.info(f"   Hazır feature'lar: {len(test_features)}/{len(all_possible_features)}")
        logger.info(f"   Kombinasyon sayısı: 2^{len(test_features)} = {2**len(test_features)}")
        if len(test_features) < len(all_possible_features):
            skipped = [f for f in all_possible_features if f not in ready_features]
            logger.info(f"   Atlanan feature'lar: {skipped}")
    
    fixed_params = get_fixed_params(horizon)
    
    print(f"\n{'='*80}")
    print(f"🔬 Sabit Parametre + Tüm Feature Kombinasyonları Testi")
    print(f"{'='*80}\n")
    print(f"Symbol: {symbol}, Horizon: {horizon}d")
    print(f"Test edilecek feature'lar: {test_features}")
    print(f"Kombinasyon sayısı: 2^{len(test_features)} = {2**len(test_features)}")
    print(f"Sabit parametreler: {fixed_params}\n")
    
    results = []
    total_combinations = 2 ** len(test_features)
    
    # Tüm kombinasyonları oluştur
    for idx, combo in enumerate(product([False, True], repeat=len(test_features))):
        feature_config = {test_features[i]: combo[i] for i in range(len(test_features))}
        
        # Base feature'ları ekle (eğer test_features içindeyse, config'den al)
        # Aksi halde her zaman açık olarak ayarla
        if 'ENABLE_TALIB_PATTERNS' not in feature_config:
            feature_config['ENABLE_TALIB_PATTERNS'] = True
        if 'ML_USE_SMART_ENSEMBLE' not in feature_config:
            feature_config['ML_USE_SMART_ENSEMBLE'] = True
        if 'ML_USE_STACKED_SHORT' not in feature_config:
            feature_config['ML_USE_STACKED_SHORT'] = True
        if 'ML_USE_REGIME_DETECTION' not in feature_config:
            feature_config['ML_USE_REGIME_DETECTION'] = True
        if 'ENABLE_META_STACKING' not in feature_config:
            feature_config['ENABLE_META_STACKING'] = True
        
        combo_name = '+'.join([f for f, enabled in feature_config.items() if enabled and f in test_features])
        if not combo_name:
            combo_name = 'Base'
        
        print(f"[{idx+1}/{total_combinations}]: {combo_name}")
        
        result = train_and_evaluate(symbol, horizon, train_df, test_df, fixed_params, feature_config, test_folder)
        
        results.append({
            'config': feature_config,
            'config_name': combo_name,
            'dirhit': result.get('dirhit'),
            'n_predictions': result.get('n_predictions'),
            'n_masked': result.get('n_masked'),
            'error': result.get('error')
        })
        
        if result.get('dirhit') is not None:
            print(f"   DirHit: {result['dirhit']:.2f}%\n")
        else:
            print(f"   ❌ Hata: {result.get('error', 'Unknown')}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Sabit Parametre + Tüm Feature Kombinasyonları Testi')
    parser.add_argument('--symbol', type=str, default='ASELS', help='Symbol to test')
    parser.add_argument('--horizon', type=int, default=7, help='Horizon in days')
    parser.add_argument('--features', type=str, nargs='+', default=None,
                       help='Test edilecek feature\'lar (varsayılan: 12 feature)')
    args = parser.parse_args()
    
    symbol = args.symbol.upper()
    horizon = args.horizon
    test_features = args.features
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Veri yükle
    db_url = 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db'
    engine = create_engine(db_url, poolclass=None)
    df = fetch_prices(engine, symbol, limit=1200)
    
    if df is None or df.empty:
        print(f"❌ Veri bulunamadı: {symbol}")
        return 1
    
    # ⚡ CRITICAL: Train/test split (HPO ile TAM AYNI)
    total_days = len(df)
    min_test_days = horizon + 10  # HPO ile aynı
    
    if total_days >= 240:
        split_idx = total_days - 120
    elif total_days >= 180:
        split_idx = int(total_days * 2 / 3)
    else:
        split_idx = max(1, int(total_days * 2 / 3))
    
    # HPO ile aynı: Ensure test set is large enough
    if total_days - split_idx < min_test_days:
        split_idx = total_days - min_test_days
        if split_idx < 1:
            print(f"❌ Yetersiz veri: {total_days} gün, {horizon}d horizon için en az {min_test_days} test günü gerekli")
            return 1
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # HPO ile aynı: Check if we have enough test days for predictions
    if len(test_df) < min_test_days:
        print(f"❌ Test seti çok küçük: {len(test_df)} < {min_test_days} (horizon={horizon}d)")
        return 1
    
    print(f"📊 Veri split: Total={total_days}, Train={len(train_df)}, Test={len(test_df)} (min_test_days={min_test_days})")
    
    # Test environment
    test_folder = setup_test_environment(symbol, horizon, 'fixed_params_all_combinations')
    
    # Tüm kombinasyonları test et
    results = test_all_combinations(symbol, horizon, train_df, test_df, test_folder, test_features)
    
    # Sonuçları kaydet
    results_file = test_folder / 'results' / f'fixed_params_all_combinations_{symbol}_{horizon}d.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    fixed_params = get_fixed_params(horizon)
    
    with open(results_file, 'w') as f:
        json.dump({
            'symbol': symbol,
            'horizon': horizon,
            'timestamp': datetime.now().isoformat(),
            'fixed_params': fixed_params,
            'test_features': test_features or list(FEATURES.keys()),
            'total_combinations': len(results),
            'results': results
        }, f, indent=2)
    
    # Özet
    valid_results = [r for r in results if r.get('dirhit') is not None]
    if valid_results:
        dirhits = [r['dirhit'] for r in valid_results]
        best = max(valid_results, key=lambda x: x.get('dirhit', 0))
        worst = min(valid_results, key=lambda x: x.get('dirhit', 0))
        
        print(f"\n{'='*80}")
        print(f"📊 ÖZET")
        print(f"{'='*80}\n")
        print(f"Toplam kombinasyon: {len(results)}")
        print(f"Başarılı test: {len(valid_results)}")
        print(f"Ortalama DirHit: {np.mean(dirhits):.2f}%")
        print(f"En iyi DirHit: {best['dirhit']:.2f}% ({best['config_name']})")
        print(f"En kötü DirHit: {worst['dirhit']:.2f}% ({worst['config_name']})")
        print(f"\n✅ Sonuçlar kaydedildi: {results_file}")
        print(f"{'='*80}\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

