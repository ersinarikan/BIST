#!/usr/bin/env python3
"""
Train BIST30 symbols with new Directional Loss
Quick test before full retraining
"""
import sys
import os
from datetime import datetime
import pandas as pd

# Ensure project root on path
sys.path.insert(0, '/opt/bist-pattern')

from app import app  # noqa: E402
from models import db, Stock  # noqa: E402
from enhanced_ml_system import EnhancedMLSystem  # noqa: E402

# BIST30 symbols
BIST30 = [
    'AKBNK', 'ASELS', 'BIMAS', 'EKGYO', 'EREGL', 'FROTO', 
    'GARAN', 'GUBRF', 'HEKTS', 'ISCTR', 'KCHOL', 'KOZAL',
    'KOZAA', 'KRDMD', 'PETKM', 'PGSUS', 'SAHOL', 'SASA',
    'SISE', 'TAVHL', 'TCELL', 'THYAO', 'TKFEN', 'TOASO',
    'TTKOM', 'TUPRS', 'VAKBN', 'VESTL', 'YKBNK', 'ODAS'
]

 
def train_bist30():
    """Train BIST30 symbols with directional loss"""
    
    print("=" * 100)
    print("🎯 BIST30 DIRECTIONAL LOSS TRAINING")
    print("=" * 100)
    print()
    print(f"Symbols to train: {len(BIST30)}")
    print(f"Symbols: {', '.join(BIST30)}")
    print()
    print("Configuration:")
    print(f"  • ML_USE_DIRECTIONAL_LOSS: {os.getenv('ML_USE_DIRECTIONAL_LOSS', '1')}")
    print(f"  • ML_LOSS_MSE_WEIGHT: {os.getenv('ML_LOSS_MSE_WEIGHT', '0.3')}")
    print(f"  • ML_LOSS_THRESHOLD: {os.getenv('ML_LOSS_THRESHOLD', '0.005')}")
    print("  • Caps: 5x increased (1d: ±30%, 3d: ±60%, ...)")
    print()
    print("=" * 100)
    print()
    
    start_time = datetime.now()
    
    with app.app_context():
        enhanced_ml = EnhancedMLSystem()
        
        success_count = 0
        fail_count = 0
        skip_count = 0
        
        for i, symbol in enumerate(BIST30, 1):
            print(f"\n{'='*100}")
            print(f"[{i}/{len(BIST30)}] Training: {symbol}")
            print(f"{'='*100}\n")
            
            try:
                # Get stock from database
                stock = Stock.query.filter_by(symbol=symbol).first()
                
                if not stock:
                    print(f"❌ {symbol} not found in database")
                    skip_count += 1
                    continue
                
                # Get price data
                query = (
                    "SELECT date, open, high, low, close, volume "
                    "FROM stock_prices "
                    "WHERE stock_id = :stock_id "
                    "ORDER BY date DESC "
                    "LIMIT 1000"
                )
                result = db.session.execute(db.text(query), {"stock_id": stock.id})
                rows = result.fetchall()
                
                if not rows or len(rows) < 100:
                    print(f"❌ {symbol} insufficient data ({len(rows) if rows else 0} days)")
                    skip_count += 1
                    continue
                
                # Convert to DataFrame
                data = pd.DataFrame([
                    {
                        'date': row[0],
                        'open': float(row[1]),
                        'high': float(row[2]),
                        'low': float(row[3]),
                        'close': float(row[4]),
                        'volume': float(row[5]) if row[5] else 0.0
                    }
                    for row in rows
                ])
                data = data.sort_values('date').reset_index(drop=True)
                
                print(f"✅ Data loaded: {len(data)} days")
                print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
                
                # Train models
                print("\n🧠 Training models...")
                success = enhanced_ml.train_enhanced_models(symbol, data)
                
                if success:
                    print(f"✅ {symbol} training completed!")
                    success_count += 1
                    
                    # Try to save
                    try:
                        enhanced_ml.save_enhanced_models(symbol)
                        print(f"💾 {symbol} models saved to disk")
                    except Exception as e:
                        print(f"⚠️  {symbol} save warning: {e}")
                else:
                    print(f"❌ {symbol} training failed")
                    fail_count += 1
                    
            except Exception as e:
                print(f"❌ {symbol} error: {e}")
                import traceback
                traceback.print_exc()
                fail_count += 1
            
            # Progress update
            elapsed = datetime.now() - start_time
            avg_time = elapsed.total_seconds() / i
            remaining = (len(BIST30) - i) * avg_time
            
            print(f"\n📊 Progress: {i}/{len(BIST30)} ({i/len(BIST30)*100:.1f}%)")
            print(f"   Success: {success_count}, Failed: {fail_count}, Skipped: {skip_count}")
            print(f"   Elapsed: {elapsed}, Remaining: ~{int(remaining/60)}min")
    
    # Final summary
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print()
    print("=" * 100)
    print("✅ BIST30 TRAINING COMPLETE!")
    print("=" * 100)
    print()
    print(f"Total symbols: {len(BIST30)}")
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Failed: {fail_count}")
    print(f"  ⏭️  Skipped: {skip_count}")
    print()
    print(f"Total time: {total_time}")
    print(f"Avg time per symbol: {total_time.total_seconds()/len(BIST30):.1f}s")
    print()
    print("=" * 100)
    print()
    print("🔍 NEXT STEPS:")
    print("  1. Wait for predictions to accumulate (1-2 days)")
    print("  2. Check accuracy in dashboard")
    print("  3. If improved, train all symbols with: scripts/run_bulk_train_all.sh")
    print()

 
if __name__ == '__main__':
    train_bist30()
