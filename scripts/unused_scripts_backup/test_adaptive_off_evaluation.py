#!/usr/bin/env python3
"""
Test: Adaptive OFF ile evaluation yap ve HPO DirHit ile karşılaştır
"""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'
os.environ.setdefault('PYTHONWARNINGS', 'ignore')

# Ensure DATABASE_URL is set
if 'DATABASE_URL' not in os.environ:
    os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:5432/bist_pattern_db'

from scripts.continuous_hpo_training_pipeline import ContinuousHPOPipeline
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test adaptive OFF evaluation"""
    pipeline = ContinuousHPOPipeline()
    
    # Test için tamamlanmış task'ları bul
    completed_tasks = {
        k: v for k, v in pipeline.state.items() 
        if v.status == 'completed' and v.hpo_dirhit is not None
    }
    
    if not completed_tasks:
        logger.error("❌ Tamamlanmış task bulunamadı!")
        return
    
    logger.info(f"📊 {len(completed_tasks)} tamamlanmış task bulundu")
    logger.info("=" * 80)
    
    # İlk 2 task'ı test et (A1YEN_1d ve A1YEN_3d)
    test_tasks = sorted(completed_tasks.items())[:2]
    
    for key, task in test_tasks:
        symbol = task.symbol
        horizon = task.horizon
        hpo_dirhit = task.hpo_dirhit
        best_params_file = task.best_params_file
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"🔬 TEST: {key}")
        logger.info(f"{'=' * 80}")
        logger.info(f"   HPO DirHit: {hpo_dirhit:.2f}%")
        logger.info(f"   Best Params File: {best_params_file}")
        
        if not best_params_file or not Path(best_params_file).exists():
            logger.error(f"❌ Best params file bulunamadı: {best_params_file}")
            continue
        
        # Load best params
        with open(best_params_file, 'r') as f:
            best_params = json.load(f)
        
        logger.info(f"   ✅ Best params yüklendi: {len(best_params.get('best_params', {}))} params")
        
        # Get data
        try:
            from app import app
            with app.app_context():
                from pattern_detector import HybridPatternDetector
                det = HybridPatternDetector()
                df = det.get_stock_data(symbol, days=0)
                
                if df is None or len(df) < 100:
                    logger.error(f"❌ Yetersiz veri: {len(df) if df is not None else 0} gün")
                    continue
                
                logger.info(f"   ✅ Veri yüklendi: {len(df)} gün")
                
                # Evaluate with adaptive OFF (WFV)
                logger.info(f"\n   🔬 Adaptive OFF (WFV) ile evaluation yapılıyor...")
                ev = pipeline._evaluate_training_dirhits(symbol, horizon, df, best_params=best_params)
                
                wfv_dirhit = ev.get('wfv')
                adaptive_dirhit = ev.get('online')
                
                if wfv_dirhit is not None:
                    logger.info(f"\n   📊 SONUÇLAR:")
                    logger.info(f"      HPO DirHit:        {hpo_dirhit:.2f}%")
                    logger.info(f"      WFV DirHit (OFF):  {wfv_dirhit:.2f}%")
                    if adaptive_dirhit is not None:
                        logger.info(f"      Adaptive DirHit:  {adaptive_dirhit:.2f}%")
                    
                    diff = wfv_dirhit - hpo_dirhit
                    logger.info(f"\n   🔍 KARŞILAŞTIRMA:")
                    logger.info(f"      WFV - HPO = {diff:+.2f}%")
                    
                    if abs(diff) < 1.0:
                        logger.info(f"      ✅ WFV DirHit HPO DirHit ile eşleşiyor (fark < 1%)")
                        logger.info(f"      ✅ Sorun adaptive learning'de DEĞİL!")
                    else:
                        logger.warning(f"      ⚠️ WFV DirHit HPO DirHit'ten farklı (fark: {abs(diff):.2f}%)")
                        logger.warning(f"      ⚠️ Sorun adaptive learning'de olmayabilir, başka bir sorun var!")
                    
                    if adaptive_dirhit is not None:
                        adaptive_diff = adaptive_dirhit - wfv_dirhit
                        logger.info(f"\n   🔍 ADAPTIVE ETKİSİ:")
                        logger.info(f"      Adaptive - WFV = {adaptive_diff:+.2f}%")
                        if adaptive_diff < 0:
                            logger.warning(f"      ⚠️ Adaptive learning performansı düşürüyor!")
                        else:
                            logger.info(f"      ✅ Adaptive learning performansı artırıyor!")
                else:
                    logger.error(f"   ❌ WFV DirHit hesaplanamadı!")
                    
        except Exception as e:
            logger.error(f"❌ Hata: {e}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()

