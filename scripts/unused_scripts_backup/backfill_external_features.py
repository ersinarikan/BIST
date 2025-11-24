#!/usr/bin/env python3
"""
External Features Backfill Script
- Collects FinGPT sentiment and YOLO pattern data from pattern detector
- Aggregates daily data and writes to CSV files for ML training
- Runs as part of automation cycle or standalone

Expected CSV format:
- fingpt/{SYMBOL}.csv: date, sentiment_score, news_count
- yolo/{SYMBOL}.csv: date, yolo_density, yolo_bull, yolo_bear, yolo_score
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Tuple

# Activate virtual environment if available
venv_path = '/opt/bist-pattern/venv'
if os.path.exists(os.path.join(venv_path, 'bin', 'activate_this.py')):
    try:
        activate_this = os.path.join(venv_path, 'bin', 'activate_this.py')
        exec(open(activate_this).read(), {'__file__': activate_this})
    except Exception:
        pass  # Fallback: assume venv is already activated

sys.path.insert(0, '/opt/bist-pattern')
os.environ.setdefault('PYTHONWARNINGS', 'ignore')

# Set DATABASE_URL if not set (required for app context)
if 'DATABASE_URL' not in os.environ:
    os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db'

# ✅ FIX: Set TRANSFORMERS_CACHE and HF_HOME for FinGPT model loading
os.environ.setdefault('TRANSFORMERS_CACHE', '/opt/bist-pattern/.cache/huggingface')
os.environ.setdefault('HF_HOME', '/opt/bist-pattern/.cache/huggingface')

import pandas as pd  # noqa: E402

# Note: app and models imported inside functions to avoid circular imports

logger = logging.getLogger(__name__)


def analyze_symbol_for_features(symbol: str, pattern_result: Dict[str, Any], lookback_days: int = 60) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Analyze symbol and extract FinGPT and YOLO features from pattern_result
    
    Args:
        symbol: Stock symbol
        pattern_result: Result dictionary from detector.analyze_stock()
        lookback_days: Lookback period (not used currently, kept for API compatibility)
    
    Returns:
        (fingpt_data, yolo_data) dictionaries with date -> value mappings
    """
    fingpt_data = {}  # {date: {'sentiment_score': float, 'news_count': int}}
    yolo_data = {}    # {date: {'density': float, 'bull': float, 'bear': float, 'score': float}}
    
    try:
        if not pattern_result:
            logger.debug(f"No pattern result for {symbol}")
            return fingpt_data, yolo_data
        
        patterns = pattern_result.get('patterns', [])
        
        # Extract FinGPT sentiment data
        fingpt_patterns = [p for p in patterns if p.get('source') == 'FINGPT']
        if fingpt_patterns:
            # Use most recent analysis result
            # Note: This is current snapshot, not historical
            # For historical backfill, we'd need to analyze each date separately
            for p in fingpt_patterns:
                sentiment_score = 0.0
                news_count = p.get('news_count', 0)
                
                # Convert signal to numeric score (-1 to +1)
                signal = p.get('signal', 'NEUTRAL')
                confidence = p.get('confidence', 0.0)
                
                if signal == 'BULLISH':
                    sentiment_score = confidence  # 0.0 to 1.0
                elif signal == 'BEARISH':
                    sentiment_score = -confidence  # -1.0 to 0.0
                else:
                    sentiment_score = 0.0
                
                # Use today's date for current snapshot
                today = datetime.now().date()
                fingpt_data[today] = {
                    'sentiment_score': sentiment_score,
                    'news_count': news_count
                }
        
        # Extract YOLO pattern data
        yolo_patterns = [p for p in patterns if p.get('source') == 'VISUAL_YOLO']
        if yolo_patterns:
            # Count patterns and calculate metrics
            bull_count = sum(1 for p in yolo_patterns if p.get('signal') == 'BULLISH')
            bear_count = sum(1 for p in yolo_patterns if p.get('signal') == 'BEARISH')
            total_count = len(yolo_patterns)
            
            # Density: normalized pattern count (0.0 to 1.0)
            density = min(1.0, total_count / 5.0)  # Max 5 patterns = 1.0
            
            # Bull/bear: normalized counts (0.0 to 1.0)
            bull_ratio = bull_count / max(1, total_count)
            bear_ratio = bear_count / max(1, total_count)
            
            # Average confidence score
            avg_conf = sum(p.get('confidence', 0.0) for p in yolo_patterns) / max(1, total_count)
            
            today = datetime.now().date()
            yolo_data[today] = {
                'density': density,
                'bull': bull_ratio,
                'bear': bear_ratio,
                'score': avg_conf
            }
    
    except Exception as e:
        logger.error(f"Error analyzing {symbol} for features: {e}")
    
    return fingpt_data, yolo_data


def write_feature_csvs(symbol: str, fingpt_data: Dict[str, Any], yolo_data: Dict[str, Any], 
                       feature_dir: str = '/opt/bist-pattern/logs/feature_backfill'):
    """Write feature data to CSV files"""
    try:
        os.makedirs(os.path.join(feature_dir, 'fingpt'), exist_ok=True)
        os.makedirs(os.path.join(feature_dir, 'yolo'), exist_ok=True)
        
        # Write FinGPT CSV
        if fingpt_data:
            fingpt_df = pd.DataFrame.from_dict(fingpt_data, orient='index')
            fingpt_df.index.name = 'date'
            fingpt_df = fingpt_df.reset_index()
            fingpt_df['date'] = pd.to_datetime(fingpt_df['date'])
            fingpt_csv = os.path.join(feature_dir, 'fingpt', f'{symbol}.csv')
            
            # Append to existing CSV or create new
            if os.path.exists(fingpt_csv):
                try:
                    existing_df = pd.read_csv(fingpt_csv)
                    existing_df['date'] = pd.to_datetime(existing_df['date'])
                    # Merge and deduplicate (keep latest)
                    combined_df = pd.concat([existing_df, fingpt_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                    combined_df = combined_df.sort_values('date')
                    combined_df.to_csv(fingpt_csv, index=False)
                except Exception as e:
                    # If merge fails, overwrite with new data
                    logger.warning(f"FinGPT CSV merge failed for {symbol}, overwriting: {e}")
                    fingpt_df.to_csv(fingpt_csv, index=False)
            else:
                fingpt_df.to_csv(fingpt_csv, index=False)
            logger.debug(f"✅ Wrote FinGPT CSV for {symbol}: {len(fingpt_df)} rows")
        
        # Write YOLO CSV
        if yolo_data:
            yolo_df = pd.DataFrame.from_dict(yolo_data, orient='index')
            yolo_df.index.name = 'date'
            yolo_df = yolo_df.reset_index()
            yolo_df['date'] = pd.to_datetime(yolo_df['date'])
            # Rename columns to match expected format
            yolo_df = yolo_df.rename(columns={
                'density': 'yolo_density',
                'bull': 'yolo_bull',
                'bear': 'yolo_bear',
                'score': 'yolo_score'
            })
            yolo_csv = os.path.join(feature_dir, 'yolo', f'{symbol}.csv')
            
            # Append to existing CSV or create new
            if os.path.exists(yolo_csv):
                try:
                    existing_df = pd.read_csv(yolo_csv)
                    existing_df['date'] = pd.to_datetime(existing_df['date'])
                    # Merge and deduplicate (keep latest)
                    combined_df = pd.concat([existing_df, yolo_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                    combined_df = combined_df.sort_values('date')
                    combined_df.to_csv(yolo_csv, index=False)
                except Exception as e:
                    # If merge fails, overwrite with new data
                    logger.warning(f"YOLO CSV merge failed for {symbol}, overwriting: {e}")
                    yolo_df.to_csv(yolo_csv, index=False)
            else:
                yolo_df.to_csv(yolo_csv, index=False)
            logger.debug(f"✅ Wrote YOLO CSV for {symbol}: {len(yolo_df)} rows")
    
    except Exception as e:
        logger.error(f"Error writing CSVs for {symbol}: {e}")


def backfill_all_symbols(lookback_days: int = 60):
    """Backfill features for all active symbols"""
    from app import app, get_pattern_detector
    from models import Stock
    
    with app.app_context():
        detector = get_pattern_detector()
        symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).all()]
        
        import re
        denylist = re.compile(r"USDTR|USDTRY|^XU|^OPX|^F_|VIOP|INDEX", re.IGNORECASE)
        symbols = [s for s in symbols if s and not denylist.search(s)]
        
        logger.info(f"📊 Backfilling features for {len(symbols)} symbols...")
        
        fingpt_count = 0
        yolo_count = 0
        
        for i, symbol in enumerate(symbols, 1):
            try:
                # Analyze stock to get pattern results
                result = detector.analyze_stock(symbol)
                fingpt_data, yolo_data = analyze_symbol_for_features(symbol, result, lookback_days)
                
                if fingpt_data:
                    write_feature_csvs(symbol, fingpt_data, {}, feature_dir=os.getenv(
                        'EXTERNAL_FEATURE_DIR', '/opt/bist-pattern/logs/feature_backfill'))
                    fingpt_count += 1
                
                if yolo_data:
                    write_feature_csvs(symbol, {}, yolo_data, feature_dir=os.getenv(
                        'EXTERNAL_FEATURE_DIR', '/opt/bist-pattern/logs/feature_backfill'))
                    yolo_count += 1
                
                if i % 10 == 0:
                    logger.info(f"  Processed {i}/{len(symbols)} symbols...")
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        logger.info(f"✅ Backfill complete: FinGPT={fingpt_count}, YOLO={yolo_count}")


def main():
    """Main entry point"""
    import argparse
    from app import app, get_pattern_detector
    
    parser = argparse.ArgumentParser(description='Backfill external features for ML training')
    parser.add_argument('--symbols', help='Comma-separated symbols (default: all active)', default=None)
    parser.add_argument('--lookback-days', type=int, default=60, help='Lookback period in days')
    parser.add_argument('--fingpt-only', action='store_true', help='Only backfill FinGPT features')
    parser.add_argument('--yolo-only', action='store_true', help='Only backfill YOLO features')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    with app.app_context():
        detector = get_pattern_detector()
        
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
            logger.info(f"📊 Processing {len(symbols)} specific symbols...")
            
            fingpt_count = 0
            yolo_count = 0
            
            for i, symbol in enumerate(symbols, 1):
                try:
                    # Analyze stock to get pattern results
                    result = detector.analyze_stock(symbol)
                    fingpt_data, yolo_data = analyze_symbol_for_features(symbol, result, args.lookback_days)
                    
                    if not args.yolo_only and fingpt_data:
                        write_feature_csvs(symbol, fingpt_data, {}, 
                                         feature_dir=os.getenv('EXTERNAL_FEATURE_DIR', '/opt/bist-pattern/logs/feature_backfill'))
                        fingpt_count += 1
                    
                    if not args.fingpt_only and yolo_data:
                        write_feature_csvs(symbol, {}, yolo_data,
                                         feature_dir=os.getenv('EXTERNAL_FEATURE_DIR', '/opt/bist-pattern/logs/feature_backfill'))
                        yolo_count += 1
                    
                    if i % 10 == 0:
                        logger.info(f"  Processed {i}/{len(symbols)} symbols... (FinGPT: {fingpt_count}, YOLO: {yolo_count})")
                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            logger.info(f"✅ Complete: FinGPT={fingpt_count}, YOLO={yolo_count}")
        else:
            # Use backfill_all_symbols for better progress tracking
            backfill_all_symbols(args.lookback_days)


if __name__ == '__main__':
    main()
