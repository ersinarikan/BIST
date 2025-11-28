#!/usr/bin/env python3
"""
Add missing symbols to database
"""

import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

from sqlalchemy import create_engine, text

# Missing symbols data (25 symbols + EFOR)
missing_symbols = [
    {'symbol': 'ADLVY', 'name': 'ADİL VARLIK YÖNETİM A.Ş.'},
    {'symbol': 'AKCVR', 'name': 'AKADEMİ ÇEVRE ENTEGRE ATIK YÖNETİMİ ENDÜSTRİ A.Ş.'},
    {'symbol': 'BIGTK', 'name': 'BİG MEDYA TEKNOLOJİ A.Ş.'},
    {'symbol': 'BLUME', 'name': 'BLUME METAL KİMYA A.Ş.'},
    {'symbol': 'DKVRL', 'name': 'DK VARLIK KİRALAMA A.Ş.'},
    {'symbol': 'DMD', 'name': 'DESTEK YATIRIM MENKUL DEĞERLER A.Ş.'},
    {'symbol': 'DSYAT', 'name': 'DESTEK YATIRIM MENKUL DEĞERLER A.Ş.'},
    {'symbol': 'DUNYH', 'name': 'DÜNYA HOLDİNG A.Ş.'},
    {'symbol': 'ECOGR', 'name': 'ECOGREEN ENERJİ HOLDİNG A.Ş.'},
    {'symbol': 'EFOR', 'name': 'EFOR YATIRIM SANAYİ TİCARET A.Ş.'},
    {'symbol': 'FAIRF', 'name': 'FAİR FİNANSMAN A.Ş.'},
    {'symbol': 'GLB', 'name': 'GLOBAL MENKUL DEĞERLER A.Ş.'},
    {'symbol': 'ISTVY', 'name': 'İSTANBUL VARLIK YÖNETİM A.Ş.'},
    {'symbol': 'KFILO', 'name': 'KAYATUR FİLO KİRALAMA A.Ş.'},
    {'symbol': 'KSFIN', 'name': 'KOÇ STELLANTİS FİNANSMAN A.Ş.'},
    {'symbol': 'KTEST', 'name': 'KAP TEST A.Ş.'},
    {'symbol': 'MARMR', 'name': 'MARMARA HOLDİNG A.Ş.'},
    {'symbol': 'MDASM', 'name': 'MİDAS MENKUL DEĞERLER A.Ş.'},
    {'symbol': 'MDS', 'name': 'MİDAS MENKUL DEĞERLER A.Ş.'},
    {'symbol': 'PAHOL', 'name': 'PASİFİK HOLDİNG A.Ş.'},
    {'symbol': 'TRALT', 'name': 'TÜRK ALTIN İŞLETMELERİ A.Ş.'},
    {'symbol': 'TRENJ', 'name': 'TR DOĞAL ENERJİ KAYNAKLARI ARAŞTIRMA VE ÜRETİM A.Ş.'},
    {'symbol': 'TRMET', 'name': 'TR ANADOLU METAL MADENCİLİK İŞLETMELERİ A.Ş.'},
    {'symbol': 'TVM', 'name': 'TRIVE YATIRIM MENKUL DEĞERLER A.Ş.'},
    {'symbol': 'YKB', 'name': 'YAPI VE KREDİ BANKASI A.Ş.'},
    {'symbol': 'YKR', 'name': 'YAPI KREDİ YATIRIM MENKUL DEĞERLER A.Ş.'},
]

def get_db_engine():
    """Get database engine with correct port"""
    project_root = Path('/opt/bist-pattern')
    secrets_file = project_root / '.secrets' / 'db_password'
    if not secrets_file.exists():
        raise ValueError(f"Secrets file not found: {secrets_file}")
    
    with open(secrets_file, 'r') as f:
        db_password = f.read().strip()
    
    db_url = f'postgresql://bist_user:{db_password}@127.0.0.1:6432/bist_pattern_db'
    return create_engine(db_url)

def add_symbols():
    """Add missing symbols to database"""
    engine = get_db_engine()
    
    added_count = 0
    skipped_count = 0
    errors = []
    
    print("=" * 80)
    print("EKSİK SEMBOLLERİ VERİTABANINA EKLEME")
    print("=" * 80)
    print()
    
    with engine.connect() as conn:
        for item in missing_symbols:
            symbol = item['symbol'].upper()
            name = item['name']
            
            try:
                # Check if symbol already exists
                check_query = text("SELECT symbol, name FROM stocks WHERE symbol = :sym")
                existing = conn.execute(check_query, {"sym": symbol}).fetchone()
                
                if existing:
                    print(f"⏭️  {symbol:<8} - Zaten mevcut: {existing[1][:50]}")
                    skipped_count += 1
                    continue
                
                # Insert new stock
                insert_query = text("""
                    INSERT INTO stocks (symbol, name, sector, is_active, created_at, updated_at)
                    VALUES (:sym, :name, 'BIST', true, NOW(), NOW())
                    ON CONFLICT (symbol) DO NOTHING
                """)
                result = conn.execute(insert_query, {"sym": symbol, "name": name})
                conn.commit()
                
                if result.rowcount > 0:
                    print(f"✅ {symbol:<8} - Eklendi: {name[:50]}")
                    added_count += 1
                else:
                    print(f"⏭️  {symbol:<8} - Conflict (zaten var olabilir)")
                    skipped_count += 1
                
            except Exception as e:
                error_msg = f"{symbol}: {str(e)}"
                errors.append(error_msg)
                print(f"❌ {symbol:<8} - Hata: {e}")
                conn.rollback()
    
    print()
    print("=" * 80)
    print("SONUÇ:")
    print("=" * 80)
    print(f"✅ Eklenen: {added_count}")
    print(f"⏭️  Atlanan: {skipped_count}")
    if errors:
        print(f"❌ Hatalar: {len(errors)}")
        for err in errors[:5]:  # Show first 5 errors
            print(f"   - {err}")
        if len(errors) > 5:
            print(f"   ... ve {len(errors) - 5} hata daha")
    print()
    
    if added_count > 0:
        print("✅ Semboller başarıyla eklendi!")
        return True
    elif skipped_count > 0 and len(errors) == 0:
        print("ℹ️  Tüm semboller zaten mevcut.")
        return True
    else:
        print("❌ Hata oluştu!")
        return False

if __name__ == '__main__':
    success = add_symbols()
    sys.exit(0 if success else 1)

