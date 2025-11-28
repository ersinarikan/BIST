#!/usr/bin/env python3
"""
Prepare INSERT statements for missing symbols
"""

# Missing symbols with their company names from the list
missing_symbols_data = {
    'ADLVY': {
        'name': 'ADİL VARLIK YÖNETİM A.Ş.',
        'sector': 'BIST'
    },
    'AKCVR': {
        'name': 'AKADEMİ ÇEVRE ENTEGRE ATIK YÖNETİMİ ENDÜSTRİ A.Ş.',
        'sector': 'BIST'
    },
    'BIGTK': {
        'name': 'BİG MEDYA TEKNOLOJİ A.Ş.',
        'sector': 'BIST'
    },
    'BLUME': {
        'name': 'BLUME METAL KİMYA A.Ş.',
        'sector': 'BIST'
    },
    'DKVRL': {
        'name': 'DK VARLIK KİRALAMA A.Ş.',
        'sector': 'BIST'
    },
    'DMD': {
        'name': 'DESTEK YATIRIM MENKUL DEĞERLER A.Ş.',
        'sector': 'BIST'
    },
    'DSYAT': {
        'name': 'DESTEK YATIRIM MENKUL DEĞERLER A.Ş.',
        'sector': 'BIST'
    },
    'DUNYH': {
        'name': 'DÜNYA HOLDİNG A.Ş.',
        'sector': 'BIST'
    },
    'ECOGR': {
        'name': 'ECOGREEN ENERJİ HOLDİNG A.Ş.',
        'sector': 'BIST'
    },
    'FAIRF': {
        'name': 'FAİR FİNANSMAN A.Ş.',
        'sector': 'BIST'
    },
    'GLB': {
        'name': 'GLOBAL MENKUL DEĞERLER A.Ş.',
        'sector': 'BIST'
    },
    'ISTVY': {
        'name': 'İSTANBUL VARLIK YÖNETİM A.Ş.',
        'sector': 'BIST'
    },
    'KFILO': {
        'name': 'KAYATUR FİLO KİRALAMA A.Ş.',
        'sector': 'BIST'
    },
    'KSFIN': {
        'name': 'KOÇ STELLANTİS FİNANSMAN A.Ş.',
        'sector': 'BIST'
    },
    'KTEST': {
        'name': 'KAP TEST A.Ş.',
        'sector': 'BIST'
    },
    'MARMR': {
        'name': 'MARMARA HOLDİNG A.Ş.',
        'sector': 'BIST'
    },
    'MDASM': {
        'name': 'MİDAS MENKUL DEĞERLER A.Ş.',
        'sector': 'BIST'
    },
    'MDS': {
        'name': 'MİDAS MENKUL DEĞERLER A.Ş.',
        'sector': 'BIST'
    },
    'PAHOL': {
        'name': 'PASİFİK HOLDİNG A.Ş.',
        'sector': 'BIST'
    },
    'TRALT': {
        'name': 'TÜRK ALTIN İŞLETMELERİ A.Ş.',
        'sector': 'BIST'
    },
    'TRENJ': {
        'name': 'TR DOĞAL ENERJİ KAYNAKLARI ARAŞTIRMA VE ÜRETİM A.Ş.',
        'sector': 'BIST'
    },
    'TRMET': {
        'name': 'TR ANADOLU METAL MADENCİLİK İŞLETMELERİ A.Ş.',
        'sector': 'BIST'
    },
    'TVM': {
        'name': 'TRIVE YATIRIM MENKUL DEĞERLER A.Ş.',
        'sector': 'BIST'
    },
    'YKB': {
        'name': 'YAPI VE KREDİ BANKASI A.Ş.',
        'sector': 'BIST'
    },
    'YKR': {
        'name': 'YAPI KREDİ YATIRIM MENKUL DEĞERLER A.Ş.',
        'sector': 'BIST'
    }
}


def print_insert_table():
    """Print formatted table for INSERT statements"""
    print("\n" + "="*120)
    print("EKSİK SEMBOLLER İÇİN EKLEME TABLOSU")
    print("="*120)
    print(f"\n{'Symbol':<10} {'Şirket Adı':<60} {'Sektör':<30}")
    print("-" * 120)
    
    for symbol, data in sorted(missing_symbols_data.items()):
        name = data['name']
        sector = data['sector']
        # Truncate if too long
        if len(name) > 58:
            name = name[:55] + "..."
        if len(sector) > 28:
            sector = sector[:25] + "..."
        print(f"{symbol:<10} {name:<60} {sector:<30}")
    
    print("\n" + "="*120)
    print("SQL INSERT STATEMENTS:")
    print("="*120)
    print()
    
    for symbol, data in sorted(missing_symbols_data.items()):
        name = data['name'].replace("'", "''")  # Escape single quotes
        sector = data['sector']
        print(f"INSERT INTO stocks (symbol, name, sector, is_active, created_at, updated_at)")
        print(f"VALUES ('{symbol}', '{name}', '{sector}', true, NOW(), NOW())")
        print(f"ON CONFLICT (symbol) DO NOTHING;")
        print()


def print_python_insert():
    """Print Python code for inserting"""
    print("\n" + "="*120)
    print("PYTHON INSERT CODE:")
    print("="*120)
    print()
    print("from models import Stock, db")
    print("from datetime import datetime")
    print()
    print("missing_symbols = [")
    for symbol, data in sorted(missing_symbols_data.items()):
        name = data['name'].replace("'", "\\'")
        sector = data['sector']
        print(f"    {{'symbol': '{symbol}', 'name': '{name}', 'sector': '{sector}'}},")
    print("]")
    print()
    print("for item in missing_symbols:")
    print("    stock = Stock(")
    print("        symbol=item['symbol'],")
    print("        name=item['name'],")
    print("        sector=item['sector'],")
    print("        is_active=True")
    print("    )")
    print("    db.session.add(stock)")
    print("try:")
    print("    db.session.commit()")
    print("    print(f'Successfully added {len(missing_symbols)} symbols')")
    print("except Exception as e:")
    print("    db.session.rollback()")
    print("    print(f'Error: {e}')")


if __name__ == '__main__':
    print_insert_table()
    print_python_insert()

