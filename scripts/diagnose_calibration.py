#!/usr/bin/env python3
"""
Calibration System Diagnostic Tool

Checks the entire prediction → outcome → metrics → calibration pipeline
and reports detailed status and issues.

Usage:
    python scripts/diagnose_calibration.py
    python scripts/diagnose_calibration.py --window-days 30
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from typing import Dict, List

from app import app
from models import db, PredictionsLog, OutcomesLog, MetricsDaily, Stock


def check_database_health():
    """Check database connectivity and table existence"""
    print("=" * 70)
    print("DATABASE HEALTH CHECK")
    print("=" * 70)
    
    try:
        with app.app_context():
            db.create_all()
            
            # Check tables
            tables = {
                'Stock': Stock.query.count(),
                'PredictionsLog': PredictionsLog.query.count(),
                'OutcomesLog': OutcomesLog.query.count(),
                'MetricsDaily': MetricsDaily.query.count(),
            }
            
            for table, count in tables.items():
                print(f"✅ {table:20s}: {count:>10,} rows")
            print()
            return True
    except Exception as e:
        print(f"❌ Database health check failed: {e}")
        return False


def check_predictions(window_days: int = 30):
    """Check predictions status"""
    print("=" * 70)
    print(f"PREDICTIONS CHECK (last {window_days} days)")
    print("=" * 70)
    
    with app.app_context():
        cutoff = datetime.utcnow() - timedelta(days=window_days)
        
        # Total predictions
        total = PredictionsLog.query.filter(
            PredictionsLog.ts_pred >= cutoff
        ).count()
        print(f"Total predictions: {total}")
        
        # With confidence
        with_conf = PredictionsLog.query.filter(
            PredictionsLog.ts_pred >= cutoff,
            PredictionsLog.confidence.isnot(None)
        ).count()
        print(f"  └─ With confidence: {with_conf}")
        
        # Per horizon
        print("\n  Per horizon:")
        for h in ['1d', '3d', '7d', '14d', '30d']:
            count = PredictionsLog.query.filter(
                PredictionsLog.horizon == h,
                PredictionsLog.ts_pred >= cutoff
            ).count()
            with_conf_h = PredictionsLog.query.filter(
                PredictionsLog.horizon == h,
                PredictionsLog.ts_pred >= cutoff,
                PredictionsLog.confidence.isnot(None)
            ).count()
            print(f"    {h}: {count:>6} total, {with_conf_h:>6} with conf")
        
        # Date distribution
        print("\n  Last 10 days:")
        from sqlalchemy import func, cast, Date
        results = db.session.query(
            cast(PredictionsLog.ts_pred, Date).label('date'),
            func.count().label('count')
        ).filter(
            PredictionsLog.ts_pred >= cutoff
        ).group_by(
            cast(PredictionsLog.ts_pred, Date)
        ).order_by(
            cast(PredictionsLog.ts_pred, Date).desc()
        ).limit(10).all()
        
        for r in results:
            print(f"    {r.date}: {r.count} predictions")
        
        # Latest prediction
        latest = PredictionsLog.query.order_by(
            PredictionsLog.ts_pred.desc()
        ).first()
        
        if latest:
            print(f"\n  Latest prediction:")
            print(f"    Symbol: {latest.symbol} {latest.horizon}")
            print(f"    Time: {latest.ts_pred}")
            print(f"    Confidence: {latest.confidence}")
            print(f"    Model: {latest.model}")
        
        print()
        
        return total


def check_outcomes(window_days: int = 30):
    """Check outcomes status"""
    print("=" * 70)
    print(f"OUTCOMES CHECK (last {window_days} days)")
    print("=" * 70)
    
    with app.app_context():
        cutoff = datetime.utcnow() - timedelta(days=window_days)
        
        # Total outcomes
        total = OutcomesLog.query.filter(
            OutcomesLog.ts_eval >= cutoff
        ).count()
        print(f"Total outcomes: {total}")
        
        # Predictions without outcomes
        waiting = db.session.query(PredictionsLog)\
            .outerjoin(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)\
            .filter(OutcomesLog.id.is_(None))\
            .filter(PredictionsLog.ts_pred >= cutoff)\
            .count()
        print(f"Predictions waiting for outcome: {waiting}")
        
        if waiting > 0:
            # Show oldest waiting
            oldest = db.session.query(PredictionsLog)\
                .outerjoin(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)\
                .filter(OutcomesLog.id.is_(None))\
                .order_by(PredictionsLog.ts_pred.asc())\
                .limit(5)\
                .all()
            
            print("\n  Oldest 5 waiting:")
            for p in oldest:
                days_ago = (datetime.utcnow() - p.ts_pred).days
                print(f"    {p.symbol} {p.horizon} at {p.ts_pred.date()} ({days_ago} days ago)")
        
        print()
        return total


def check_calibration(window_days: int = 30):
    """Check calibration pipeline"""
    print("=" * 70)
    print(f"CALIBRATION PIPELINE CHECK (last {window_days} days)")
    print("=" * 70)
    
    with app.app_context():
        cutoff = datetime.utcnow() - timedelta(days=window_days)
        
        # Joined prediction-outcome pairs
        total_pairs = db.session.query(PredictionsLog, OutcomesLog)\
            .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)\
            .filter(PredictionsLog.ts_pred >= cutoff)\
            .count()
        print(f"Prediction-Outcome pairs: {total_pairs}")
        
        # With confidence
        with_conf = db.session.query(PredictionsLog, OutcomesLog)\
            .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)\
            .filter(PredictionsLog.ts_pred >= cutoff)\
            .filter(PredictionsLog.confidence.isnot(None))\
            .count()
        print(f"  └─ With confidence: {with_conf}")
        
        # Per horizon
        print("\n  Per horizon (with confidence):")
        for h in ['1d', '3d', '7d', '14d', '30d']:
            count = db.session.query(PredictionsLog, OutcomesLog)\
                .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)\
                .filter(PredictionsLog.horizon == h)\
                .filter(PredictionsLog.ts_pred >= cutoff)\
                .filter(PredictionsLog.confidence.isnot(None))\
                .count()
            print(f"    {h}: {count:>6} pairs")
        
        print()
        
        # Calibration readiness
        min_samples = 150
        print(f"Calibration Requirements:")
        print(f"  Minimum samples needed: {min_samples}")
        
        for h in ['1d', '3d', '7d', '14d', '30d']:
            count = db.session.query(PredictionsLog, OutcomesLog)\
                .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)\
                .filter(PredictionsLog.horizon == h)\
                .filter(PredictionsLog.ts_pred >= cutoff)\
                .filter(PredictionsLog.confidence.isnot(None))\
                .count()
            
            status = "✅ READY" if count >= min_samples else f"⚠️  INSUFFICIENT ({count}/{min_samples})"
            print(f"    {h}: {status}")
        
        print()
        return with_conf


def check_calibration_files():
    """Check calibration output files"""
    print("=" * 70)
    print("CALIBRATION FILES CHECK")
    print("=" * 70)
    
    log_path = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
    
    files = {
        'param_store.json': os.path.join(log_path, 'param_store.json'),
        'calibration_state.json': os.path.join(log_path, 'calibration_state.json'),
        'params/active.json': os.path.join(log_path, 'params/active.json'),
    }
    
    for name, path in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            age = (datetime.now() - mtime).total_seconds() / 3600  # hours
            print(f"✅ {name:25s}: {size:>8,} bytes, {age:.1f}h old")
            
            # Validate JSON
            try:
                import json
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                if 'param_store.json' in name:
                    horizons = data.get('horizons', {})
                    print(f"     └─ Horizons: {len(horizons)} configured")
                    for h, hdata in horizons.items():
                        if isinstance(hdata, dict):
                            th = hdata.get('thresholds', {})
                            print(f"        {h}: delta={th.get('delta_thr')}, conf={th.get('conf_thr')}")
                
                if 'calibration_state.json' in name:
                    horizons = data.get('horizons', {})
                    for h, hdata in horizons.items():
                        if isinstance(hdata, dict) and h != 'meta':
                            n_pairs = hdata.get('n_pairs', 0)
                            used_prev = hdata.get('used_prev', False)
                            status = "prev" if used_prev else "calibrated"
                            print(f"     └─ {h}: {n_pairs} samples, using {status}")
                            
            except Exception as e:
                print(f"     ⚠️  JSON validation failed: {e}")
        else:
            print(f"❌ {name:25s}: NOT FOUND")
    
    print()


def check_models():
    """Check ML models status"""
    print("=" * 70)
    print("ML MODELS CHECK")
    print("=" * 70)
    
    # Enhanced models
    enh_path = os.getenv('ML_MODEL_PATH', '/opt/bist-pattern/.cache/enhanced_ml_models')
    if os.path.exists(enh_path):
        enh_models = len([f for f in os.listdir(enh_path) if f.endswith('.pkl')])
        print(f"✅ Enhanced models: {enh_models} files in {enh_path}")
    else:
        print(f"⚠️  Enhanced models directory not found: {enh_path}")
    
    # Basic models
    basic_path = '/opt/bist-pattern/.cache/basic_ml_models'
    if os.path.exists(basic_path):
        basic_models = len([f for f in os.listdir(basic_path) if f.endswith('.pkl')])
        print(f"✅ Basic models: {basic_models} files in {basic_path}")
    else:
        print(f"⚠️  Basic models directory not found: {basic_path}")
    
    print()


def diagnose(window_days: int = 30):
    """Main diagnostic function"""
    print()
    print("🔍 BIST-PATTERN CALIBRATION SYSTEM DIAGNOSTICS")
    print(f"   Window: {window_days} days")
    print(f"   Time: {datetime.now().isoformat()}")
    print()
    
    # Run all checks
    db_ok = check_database_health()
    if not db_ok:
        print("\n❌ Database check failed. Cannot continue.")
        return 1
    
    pred_count = check_predictions(window_days)
    outcome_count = check_outcomes(window_days)
    calib_pairs = check_calibration(window_days)
    check_calibration_files()
    check_models()
    
    # Summary and recommendations
    print("=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    if pred_count == 0:
        print("🔴 CRITICAL: No predictions in last {window_days} days")
        print("   Recommendations:")
        print("   1. Check if automation is running: systemctl status bist-pattern")
        print("   2. Check automation logs: journalctl -u bist-pattern -n 100")
        print("   3. Verify ML models exist (see above)")
        print("   4. Check if analyze_stock() is being called")
        return 1
    
    elif pred_count > 0 and calib_pairs == 0:
        print("🟡 WARNING: Predictions exist but no prediction-outcome pairs")
        print("   Recommendations:")
        print("   1. Check if populate_outcomes is running")
        print("   2. Check if predictions have matured (need time to evaluate)")
        print("   3. Run: /opt/bist-pattern/scripts/run_populate_outcomes.sh")
        print("   4. Check populate_outcomes.log for errors")
        return 2
    
    elif calib_pairs < 150:
        print(f"🟡 WARNING: Only {calib_pairs} calibration pairs (need ≥150)")
        print("   Recommendations:")
        print("   1. Wait for more data to accumulate")
        print("   2. Or reduce --min-samples parameter")
        print("   3. System will use previous/default parameters until enough data")
        return 3
    
    else:
        print(f"✅ System looks healthy!")
        print(f"   - {pred_count} predictions")
        print(f"   - {outcome_count} outcomes")
        print(f"   - {calib_pairs} calibration pairs")
        print("   - Calibration should work properly")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose BIST-Pattern calibration system health'
    )
    parser.add_argument(
        '--window-days',
        type=int,
        default=30,
        help='Time window for analysis (default: 30 days)'
    )
    args = parser.parse_args()
    
    return diagnose(args.window_days)


if __name__ == '__main__':
    import sys
    sys.exit(main())

