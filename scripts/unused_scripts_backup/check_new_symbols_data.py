#!/usr/bin/env python3
"""
Check how much data has been collected for newly added symbols
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/opt/bist-pattern')

from sqlalchemy import create_engine, text

# Newly added symbols
new_symbols = [
    'ADLVY', 'AKCVR', 'BIGTK', 'BLUME', 'DKVRL', 'DMD', 'DSYAT', 'DUNYH',
    'ECOGR', 'EFOR', 'FAIRF', 'GLB', 'ISTVY', 'KFILO', 'KSFIN', 'KTEST',
    'MARMR', 'MDASM', 'MDS', 'PAHOL', 'TRALT', 'TRENJ', 'TRMET', 'TVM',
    'YKB', 'YKR'
]

def get_db_engine():
    """Get database engine"""
    project_root = Path('/opt/bist-pattern')
    secrets_file = project_root / '.secrets' / 'db_password'
    with open(secrets_file, 'r') as f:
        db_password = f.read().strip()
    db_url = f'postgresql://bist_user:{db_password}@127.0.0.1:6432/bist_pattern_db'
    return create_engine(db_url)

def check_symbols_data():
    """Check data for newly added symbols"""
    engine = get_db_engine()
    
    print("=" * 100)
    print("YENİ EKLENEN SEMBOLLERİN VERİ DURUMU")
    print("=" * 100)
    print()
    
    results = []
    
    with engine.connect() as conn:
        for symbol in sorted(new_symbols):
            query = text("""
                SELECT 
                    s.symbol,
                    s.name,
                    COUNT(sp.id) as total_records,
                    MIN(sp.date) as first_date,
                    MAX(sp.date) as last_date
                FROM stocks s
                LEFT JOIN stock_prices sp ON sp.stock_id = s.id
                WHERE s.symbol = :sym
                GROUP BY s.id, s.symbol, s.name
            """)
            
            result = conn.execute(query, {"sym": symbol}).fetchone()
            
            if result:
                total = result[2] or 0
                first_date = result[3]
                last_date = result[4]
                
                results.append({
                    'symbol': symbol,
                    'name': result[1][:40] if result[1] else 'N/A',
                    'total': total,
                    'first': first_date,
                    'last': last_date
                })
    
    # Print table
    print(f"{'Symbol':<10} {'Total Days':<12} {'First Date':<12} {'Last Date':<12} {'Status':<20}")
    print("-" * 100)
    
    total_with_data = 0
    total_without_data = 0
    
    for r in results:
        status = "NO DATA" if r['total'] == 0 else f"OK ({r['total']} days)"
        if r['total'] > 0:
            total_with_data += 1
        else:
            total_without_data += 1
        
        first_str = str(r['first']) if r['first'] else 'N/A'
        last_str = str(r['last']) if r['last'] else 'N/A'
        
        print(f"{r['symbol']:<10} {r['total']:<12} {first_str:<12} {last_str:<12} {status:<20}")
    
    print("-" * 100)
    print()
    print("ÖZET:")
    print(f"  Toplam sembol: {len(results)}")
    print(f"  Veri olan: {total_with_data}")
    print(f"  Veri olmayan: {total_without_data}")
    print()
    
    if total_with_data > 0:
        print("✅ working_automation veri çekmeye başlamış!")
        print(f"   {total_with_data} sembol için veri çekilmiş")
    else:
        print("⏳ Henüz veri çekilmemiş, working_automation bir tur daha tamamlamalı...")

if __name__ == '__main__':
    check_symbols_data()

