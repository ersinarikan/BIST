#!/usr/bin/env python3
"""
Check how many days of data exist in database for given symbols
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from bist_pattern.core.config_manager import ConfigManager


def get_database_url() -> str:
    """Get database URL from config"""
    # Read password from secrets file
    secrets_file = project_root / '.secrets' / 'db_password'
    if not secrets_file.exists():
        raise ValueError(f"Secrets file not found: {secrets_file}")
    
    with open(secrets_file, 'r') as f:
        db_password = f.read().strip()
    
    # Use PgBouncer port 6432
    db_url = f"postgresql://bist_user:{db_password}@127.0.0.1:6432/bist_pattern_db"
    return db_url


def check_symbol_data_days(symbols: List[str]) -> List[Dict[str, any]]:
    """Check how many days of data exist for each symbol"""
    engine = create_engine(get_database_url())
    results = []
    
    # Calculate date 100 days ago
    date_100_days_ago = datetime.now().date() - timedelta(days=100)
    
    query = text("""
        SELECT 
            COUNT(*) as total_days,
            MIN(p.date) as first_date,
            MAX(p.date) as last_date,
            COUNT(CASE WHEN p.date >= :date_threshold THEN 1 END) as days_last_100
        FROM stock_prices p
        JOIN stocks s ON s.id = p.stock_id
        WHERE s.symbol = :sym
    """)
    
    with engine.connect() as conn:
        for symbol in symbols:
            symbol = symbol.strip().upper()
            try:
                result = conn.execute(query, {
                    "sym": symbol,
                    "date_threshold": date_100_days_ago
                }).fetchone()
                
                if result:
                    total_days = result[0] or 0
                    first_date = result[1]
                    last_date = result[2]
                    days_last_100 = result[3] or 0
                    
                    results.append({
                        'symbol': symbol,
                        'total_days': total_days,
                        'first_date': first_date.strftime('%Y-%m-%d') if first_date else None,
                        'last_date': last_date.strftime('%Y-%m-%d') if last_date else None,
                        'days_last_100': days_last_100,
                        'has_data': total_days > 0
                    })
                else:
                    results.append({
                        'symbol': symbol,
                        'total_days': 0,
                        'first_date': None,
                        'last_date': None,
                        'days_last_100': 0,
                        'has_data': False
                    })
            except Exception as e:
                results.append({
                    'symbol': symbol,
                    'total_days': 0,
                    'first_date': None,
                    'last_date': None,
                    'days_last_100': 0,
                    'has_data': False,
                    'error': str(e)
                })
    
    return results


def print_table(results: List[Dict[str, any]]):
    """Print results in a formatted table"""
    # Header
    print(f"\n{'Symbol':<12} {'Total Days':<12} {'Last 100 Days':<15} {'First Date':<12} {'Last Date':<12} {'Status':<20}")
    print("-" * 95)
    
    # Sort by total_days descending
    results_sorted = sorted(results, key=lambda x: x['total_days'], reverse=True)
    
    for r in results_sorted:
        symbol = r['symbol']
        total_days = r['total_days']
        days_last_100 = r['days_last_100']
        first_date = r['first_date'] or 'N/A'
        last_date = r['last_date'] or 'N/A'
        
        if 'error' in r:
            status = f"ERROR: {r['error'][:15]}"
        elif total_days == 0:
            status = "NO DATA"
        elif days_last_100 < 100:
            status = f"INSUFFICIENT ({days_last_100}/100)"
        else:
            status = "OK"
        
        print(f"{symbol:<12} {total_days:<12} {days_last_100:<15} {first_date:<12} {last_date:<12} {status:<20}")
    
    # Summary
    total_symbols = len(results)
    symbols_with_data = sum(1 for r in results if r['total_days'] > 0)
    symbols_sufficient = sum(1 for r in results if r['days_last_100'] >= 100)
    
    print("-" * 95)
    print(f"\nSummary:")
    print(f"  Total symbols checked: {total_symbols}")
    print(f"  Symbols with data: {symbols_with_data}")
    print(f"  Symbols with sufficient data (≥100 days): {symbols_sufficient}")
    print(f"  Symbols with insufficient data: {symbols_with_data - symbols_sufficient}")
    print(f"  Symbols with no data: {total_symbols - symbols_with_data}")


def main():
    # Symbols from user's list
    symbols = [
        'AKDFA', 'AKFK', 'AKMEN', 'AKSFA', 'AKTIF', 'AKTVK',
        'ALFIN', 'ALJF', 'ALNTF', 'ALNUS', 'ALROS', 'APX30',
        'ARSVY', 'ATAVK', 'ATAYM', 'ATLFA', 'BRKT', 'DGRVK',
        'DNFIN', 'DNYVA', 'DOGVY', 'DRPHN', 'DTBMK', 'DTRND',
        'DVRLK', 'DYBNK', 'DZYMK', 'EKER', 'EKOFA', 'EKOVR',
        'EKTVK', 'EMIRV', 'EMVAR', 'ERGLI', 'EXIMB', 'FBBNK',
        'FIBAF', 'FSDAT', 'GARFL', 'GGBVK', 'GSIPD', 'HALKF',
        'HALKI', 'HALKS', 'HAYVK', 'HDFFL', 'HDFVK', 'HDFYB',
        'HEPFN', 'HLVKS', 'HUZFA', 'INALR', 'INVAZ'
    ]
    
    print("Checking data availability for symbols...")
    results = check_symbol_data_days(symbols)
    print_table(results)


if __name__ == '__main__':
    main()

