#!/usr/bin/env python3
"""
Recalculate dir_hit for existing outcomes with NEW threshold (0.5%)
"""
import os
from datetime import datetime

from app import app
from models import db, OutcomesLog, PredictionsLog

def recalculate_all_outcomes():
    """Update dir_hit for all existing outcomes with new 0.5% threshold"""
    with app.app_context():
        threshold = 0.005  # NEW: 0.5% (was 1.0%)
        
        print(f"🔄 Recalculating dir_hit with NEW threshold={threshold} (0.5%)")
        print()
        
        # Get all outcomes with predictions
        outcomes = db.session.query(OutcomesLog, PredictionsLog).join(
            PredictionsLog, OutcomesLog.prediction_id == PredictionsLog.id
        ).all()
        
        total = len(outcomes)
        updated = 0
        changed = 0
        
        print(f"📊 Found {total} outcomes to recalculate")
        print()
        
        for i, (outcome, prediction) in enumerate(outcomes):
            if i % 1000 == 0 and i > 0:
                print(f"Progress: {i}/{total} ({i/total*100:.1f}%)")
            
            delta_real = float(outcome.delta_real or 0)
            delta_pred = float(prediction.delta_pred or 0)
            
            # Old dir_hit (current DB value with 1% threshold)
            old_dir_hit = outcome.dir_hit
            
            # New threshold-based logic with 0.5% threshold
            real_dir = 0 if abs(delta_real) < threshold else (1 if delta_real > 0 else -1)
            pred_dir = 0 if abs(delta_pred) < threshold else (1 if delta_pred > 0 else -1)
            
            if real_dir == 0 and pred_dir == 0:
                new_dir_hit = True
            elif real_dir == 0 or pred_dir == 0:
                new_dir_hit = False
            else:
                new_dir_hit = (real_dir == pred_dir)
            
            # Update if changed
            if old_dir_hit != new_dir_hit:
                outcome.dir_hit = new_dir_hit
                changed += 1
            
            updated += 1
        
        # Commit changes
        print()
        print(f"💾 Committing changes...")
        db.session.commit()
        
        print()
        print("=" * 60)
        print(f"✅ TAMAMLANDI!")
        print("=" * 60)
        print(f"Toplam:    {total}")
        print(f"Değişen:   {changed} ({changed/total*100:.1f}%)")
        print(f"Değişmeyen: {total-changed} ({(total-changed)/total*100:.1f}%)")
        print()
        print("📝 Not: Threshold 1.0% → 0.5% değişti")
        print()

if __name__ == '__main__':
    recalculate_all_outcomes()
