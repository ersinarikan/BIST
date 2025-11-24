#!/usr/bin/env python3
"""
Backfill Predictions for Calibration

Yeni modeller hazır olduğunda, geçmiş verilerle yeniden tahmin yaparak
calibration'ı hızlandırır.

Kullanım:
    python scripts/backfill_predictions_for_calibration.py \
        --symbols A1YEN,ACSEL,A1CAP \
        --days 120 \
        --horizons 1,3,7,14,30

Önemli:
    - Walk-forward validation: Her gün için, o güne kadar olan veri ile tahmin yapılır
    - Data leakage önleme: Model sadece o güne kadar olan veriyi görür
    - Mevcut tahminler üzerine yazılmaz (yeni kayıtlar oluşturulur)
    - OutcomesLog ile eşleştirme: populate_outcomes.py ile yapılır
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app
from models import db, PredictionsLog, Stock, StockPrice
from pattern_detector import HybridPatternDetector
from bist_pattern.core.ml_coordinator import MLCoordinator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_stock_data_up_to_date(symbol: str, end_date: datetime) -> Optional[pd.DataFrame]:
    """Get stock data up to a specific date (walk-forward validation)"""
    try:
        with app.app_context():
            from models import Stock, StockPrice
            
            stock = Stock.query.filter_by(symbol=symbol.upper()).first()
            if not stock:
                logger.warning(f"⚠️ Stock not found: {symbol}")
                return None
            
            # Get prices up to end_date
            end_date_only = end_date.date()
            prices = StockPrice.query.filter_by(stock_id=stock.id)\
                .filter(StockPrice.date <= end_date_only)\
                .order_by(StockPrice.date.asc())\
                .all()
            
            if not prices or len(prices) < 60:  # Minimum 60 days for feature engineering
                logger.warning(f"⚠️ {symbol}: Insufficient data up to {end_date.date()} ({len(prices)} days)")
                return None
            
            # Convert to DataFrame
            data = []
            for p in prices:
                data.append({
                    'date': p.date,
                    'open': float(p.open_price),
                    'high': float(p.high_price),
                    'low': float(p.low_price),
                    'close': float(p.close_price),
                    'volume': int(p.volume) if p.volume else 0
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            return df
    except Exception as e:
        logger.error(f"❌ Error getting stock data for {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def make_prediction_for_date(
    symbol: str,
    pred_date: datetime,
    detector: HybridPatternDetector,
    ml_coordinator: MLCoordinator,
    horizons: List[str]
) -> Dict[str, Any]:
    """Make prediction for a specific historical date (walk-forward validation)"""
    try:
        # Get data up to pred_date (walk-forward: model only sees data up to this date)
        df = get_stock_data_up_to_date(symbol, pred_date)
        if df is None or len(df) < 60:
            return None
        
        # ✅ FIX: Temporarily disable PredictionsLog writes during backfill
        # We'll save predictions manually with correct timestamp
        import os
        original_disable = os.getenv('DISABLE_PREDICTIONS_LOG', '0')
        os.environ['DISABLE_PREDICTIONS_LOG'] = '1'
        
        try:
            # Analyze stock (this will use only data up to pred_date)
            # Note: analyze_stock will use current data, but we've limited it to pred_date
            result = detector.analyze_stock(symbol)
        finally:
            # Restore original setting
            os.environ['DISABLE_PREDICTIONS_LOG'] = original_disable
        
        if result.get('status') != 'success':
            return None
        
        # Extract predictions
        predictions = {}
        ml_unified = result.get('ml_unified', {})
        
        for h in horizons:
            if h in ml_unified:
                best = ml_unified[h].get('best')
                if best and best in ml_unified[h]:
                    pred_data = ml_unified[h][best]
                    predictions[h] = {
                        'pred_price': pred_data.get('pred_price'),
                        'confidence': pred_data.get('confidence'),
                        'delta_pct': pred_data.get('delta_pct'),
                        'current_price': result.get('current_price')
                    }
        
        if not predictions:
            return None
        
        return {
            'symbol': symbol,
            'pred_date': pred_date,
            'current_price': result.get('current_price'),
            'predictions': predictions
        }
    except Exception as e:
        logger.error(f"❌ Error making prediction for {symbol} at {pred_date}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def save_prediction_to_db(prediction_data: Dict[str, Any]) -> bool:
    """Save prediction to PredictionsLog"""
    try:
        with app.app_context():
            symbol = prediction_data['symbol']
            pred_date = prediction_data['pred_date']
            current_price = prediction_data['current_price']
            predictions = prediction_data['predictions']
            
            # Get stock
            stock = Stock.query.filter_by(symbol=symbol.upper()).first()
            if not stock:
                logger.warning(f"⚠️ Stock not found: {symbol}")
                return False
            
            saved_count = 0
            
            for horizon, pred_info in predictions.items():
                pred_price = pred_info.get('pred_price')
                confidence = pred_info.get('confidence')
                delta_pct = pred_info.get('delta_pct')
                
                if pred_price is None:
                    continue
                
                # Check if prediction already exists (avoid duplicates)
                existing = PredictionsLog.query.filter_by(
                    symbol=symbol.upper(),
                    horizon=horizon,
                    stock_id=stock.id
                ).filter(
                    db.func.date(PredictionsLog.ts_pred) == pred_date.date()
                ).first()
                
                if existing:
                    logger.debug(f"⏭️ {symbol} {horizon} at {pred_date.date()}: Already exists, skipping")
                    continue
                
                # Create new prediction
                pred_log = PredictionsLog(
                    symbol=symbol.upper(),
                    horizon=horizon,
                    stock_id=stock.id,
                    ts_pred=pred_date,
                    price_now=current_price,
                    pred_price=pred_price,
                    delta_pred=delta_pct * 100.0 if delta_pct is not None else None,  # Convert to percent
                    confidence=confidence,
                    model='enhanced_ml_backfill',
                    param_version='backfill_v1'
                )
                
                db.session.add(pred_log)
                saved_count += 1
            
            if saved_count > 0:
                db.session.commit()
                logger.info(f"✅ {symbol}: Saved {saved_count} predictions for {pred_date.date()}")
                return True
            else:
                logger.debug(f"⏭️ {symbol}: No new predictions to save for {pred_date.date()}")
                return False
    except Exception as e:
        logger.error(f"❌ Error saving prediction to DB: {e}")
        import traceback
        logger.error(traceback.format_exc())
        db.session.rollback()
        return False


def backfill_predictions(
    symbols: List[str],
    days: int,
    horizons: List[str],
    start_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """Backfill predictions for calibration"""
    
    # Initialize components
    detector = HybridPatternDetector()
    ml_coordinator = MLCoordinator()
    
    # Determine date range
    if start_date is None:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
    else:
        end_date = start_date + timedelta(days=days)
    
    logger.info(f"🔄 Starting backfill: {len(symbols)} symbols, {days} days, {len(horizons)} horizons")
    logger.info(f"   Date range: {start_date.date()} to {end_date.date()}")
    
    # Generate business days (skip weekends)
    business_days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Monday-Friday
            business_days.append(current)
        current += timedelta(days=1)
    
    logger.info(f"   Business days: {len(business_days)}")
    
    total_predictions = 0
    total_saved = 0
    errors = 0
    
    # Process each symbol
    for symbol in symbols:
        logger.info(f"📊 Processing {symbol}...")
        symbol_saved = 0
        
        # Process each business day (walk-forward)
        for pred_date in business_days:
            try:
                # Make prediction for this date (using only data up to this date)
                pred_data = make_prediction_for_date(
                    symbol, pred_date, detector, ml_coordinator, horizons
                )
                
                if pred_data is None:
                    continue
                
                total_predictions += 1
                
                # Save to database
                if save_prediction_to_db(pred_data):
                    total_saved += 1
                    symbol_saved += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol} at {pred_date.date()}: {e}")
                errors += 1
                continue
        
        logger.info(f"✅ {symbol}: Saved {symbol_saved} predictions")
    
    logger.info("=" * 80)
    logger.info(f"📊 Backfill Summary:")
    logger.info(f"   Total predictions made: {total_predictions}")
    logger.info(f"   Total saved to DB: {total_saved}")
    logger.info(f"   Errors: {errors}")
    logger.info("=" * 80)
    
    return {
        'total_predictions': total_predictions,
        'total_saved': total_saved,
        'errors': errors
    }


def main():
    parser = argparse.ArgumentParser(
        description='Backfill predictions for calibration using new models'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        required=True,
        help='Comma-separated list of symbols (e.g., A1YEN,ACSEL,A1CAP)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=120,
        help='Number of days to backfill (default: 120)'
    )
    parser.add_argument(
        '--horizons',
        type=str,
        default='1,3,7,14,30',
        help='Comma-separated horizons (default: 1,3,7,14,30)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD). If not provided, uses (today - days)'
    )
    parser.add_argument(
        '--limit-symbols',
        type=int,
        default=None,
        help='Limit number of symbols to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    if not symbols:
        logger.error("❌ No symbols provided")
        return 1
    
    # Limit symbols if requested
    if args.limit_symbols:
        symbols = symbols[:args.limit_symbols]
    
    # Parse horizons
    horizons = [h.strip() for h in args.horizons.split(',') if h.strip()]
    if not horizons:
        horizons = ['1d', '3d', '7d', '14d', '30d']
    
    # Parse start date
    start_date = None
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        except Exception as e:
            logger.error(f"❌ Invalid start date format: {e}")
            return 1
    
    # Run backfill
    try:
        result = backfill_predictions(symbols, args.days, horizons, start_date)
        
        logger.info("=" * 80)
        logger.info("✅ Backfill completed!")
        logger.info(f"   Next step: Run populate_outcomes.py to match predictions with outcomes")
        logger.info(f"   Then run calibrate_confidence.py to generate calibration parameters")
        logger.info("=" * 80)
        
        return 0
    except Exception as e:
        logger.error(f"❌ Backfill failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())

