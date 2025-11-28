#!/usr/bin/env python3
"""
Compare symbols from external list with database
"""

import sys
import os
from pathlib import Path
from typing import Set, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text


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


def get_db_symbols() -> Set[str]:
    """Get all symbols from database"""
    engine = create_engine(get_database_url())
    
    query = text("""
        SELECT DISTINCT UPPER(s.symbol) as symbol
        FROM stocks s
        ORDER BY symbol
    """)
    
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
        return {row[0] for row in rows}


def parse_symbols_from_list() -> tuple[Set[str], dict, dict]:
    """Parse symbols from the provided list"""
    # Symbols from the user's list (extracted from the text)
    symbols_text = """
    ACSEL ADEL ADESE ADLVY ADGYO AFYON AGHOL AGESA AGROT AHSGY AHGAZ AKSFA AKFK AKMEN AKCVR AKBNK AKCKM AKCNS AKDFA AKYHO AKENR AKFGY AKFIS AKFYE ATEKS AKSGY AKMGY AKSA AKSEN AKGRT AKSUE AKTVK AKTIF ALCAR ALGYO ALARK ALBRK ALCTL ALFAS ALJF ALKIM ALKA ALNUS ALFIN AYCES ALTNY ALKLC ALVES ANSGR AEFES ANHYT ASUZU ANGEN ANELE ARCLK ARDYZ ARENA ARMGD ARSAN ARSVY ARTMS ARZUM ASGYO ASELS ASTOR ATAGY ATAVK ATAYM ATAKP AGYO ATLFA ATSYH ATLAS ATATP AVOD AVGYO AVTUR AVHOL AVPGY AYDEM AYEN AYES AYGAZ AZTEK A1CAP A1YEN
    BAGFS BAHKM BAKAB BALAT BALSU BNTAS BANVT BARMA BSRFK BASGZ BASCM BEGYO BTCIM BSOKE BYDNR BAYRK BERA BRKT BRKSN BESLR BJKAS BEYAZ BIENY BIGTK BLCYT BLKOM BIMAS BINBN BIOEN BRKVY BRKO BIGEN BRLSM BRMEN BIZIM BLUME BMSTL BMSCH BNPPI BOBET BORSK BORLS BRSAN BRYAT BFREN BOSSA BRISA BULGS BLS BURCE BURVA BRGFK BUCIM BVSAN BIGCH
    CRFSA CASA CEMZY CEOEM CCOLA CONSE COSMO CRDFA CVKMD CWENE
    CGCAM CAGFA CANTE CATES CLEBI CELHA CLKMT CEMAS CEMTS CMBTN CMENT CIMSA CUSAN
    DVRLK DYBNK DAGI DAPGM DARDL DGATE DCTTR DGRVK DMSAS DENGE DENFA DNFIN DZGYO DZYMK DERIM DERHL DESA DESPC DSTKF DMD DSYAT DEVA DNISI DIRIT DITAS DKVRL DMRGD DOCO DOFRB DOFER DOHOL DTRND DGNMO DOGVY ARASE DOGUB DGGYO DOAS DFKTR DOKTA DURDO DURKN DUNYH DNYVA DYOBY
    EBEBK ECOGR ECZYT EDATA EDIP EFOR EGEEN EGGUB EGPRO EGSER EPLAS EGEGY ECILC EKER EKIZ EKOFA EKOS EKOVR EKSUN ELITE EMKEL EMNIS EMIRV EKTVK DMLKT EKGYO EMVAR ENDAE ENJSA ENERY ENKAI ENSRI ERBOS ERCB EREGL ERGLI KIMMR ERSU ESCAR ESCOM ESEN ETILR EUKYO EUYO ETYAT EUHOL TEZOL EUREN EUPWR EYGYO
    FADE FAIRF FSDAT FMIZP FENER FBBNK FLAP FONET FROTO FORMT FORTE FRIGO FZLGY
    GWIND GSRAY GARFA GARFL GRNYO GEDIK GEDZA GLCVY GENIL GENTS GEREL GZNMI GIPTA GMTAS GESAN GLB GLYHO GGBVK GSIPD GOODY GOKNR GOLTS GOZDE GRTHO GSDDE GSDHO GUBRF GLRYH GLRMK GUNDG GRSEL
    SAHOL HALKF HLGYO HLVKS HALKI HRKET HATEK HATSN HAYVK HDFFL HDFGS HEDEF HDFVK HDFYB HEKTS HEPFN HKTM HTTBT HOROZ HUBVC HUNER HUZFA HURGZ
    ENTRA ICB ICUGS INGRM INVEO INVAZ INVES ISKPL IEYHO
    IDGYO IHEVA IHLGM IHGZT IHAAS IHLAS IHYAY IMASM INALR INDES INFO INTEK INTEM ISDMR ISTFK ISTVY ISFAK ISFIN ISGYO ISGSY ISMEN ISYAT ISBIR ISSEN IZINV IZENR IZMDC IZFAS
    JANTS
    KFEIN KLKIM KLSER KLVKS KLYPV KTEST KAPLM KRDMA KAREL KARSN KRTEK KARTN KATVK KTLEV KATMR KFILO KAYSE KNTFA KENT KRVGD KERVN TCKRC KZBGY KLGYO KLRHO KMPUR KLMSN KCAER KOCFN KCHOL KOCMT KSFIN KLSYN KNFRT KONTR KONYA KONKA KGYO KORDS KRPLS KORTS KOTON KOPOL KRGYO KRSTL KRONT KTKVK KTSVK KSTUR KUVVA KUYAS KBORU KZGYO KUTPO KTSKR
    LIDER LIDFA LILAK LMKDC LINK LOGO LKMNH LRSHO LUKSK LYDHO LYDYE
    MACKO MAKIM MAKTK MANAS MRBAS MAGEN MRMAG MARKA MARMR MAALT MRSHL MRGYO MARTI MTRKS MAVI MZHLD MEDTR MEGMT MEGAP MEKAG MEKMD KLMSN KCAER KOCFN KCHOL KOCMT KSFIN KLSYN KNFRT KONTR KONYA KONKA KGYO KORDS KRPLS KORTS KOTON KOPOL KRGYO KRSTL KRONT KTKVK KTSVK KSTUR KUVVA KUYAS KBORU KZGYO KUTPO KTSKR
    EGEPO NATEN NTGAZ NTHOL NETAS NIBAS NUHCM NUGYO NRLIN NURVK NRBNK
    OBAMS OBASE ODAS ODINE OFSYM ONCSM ONRYT OPET ORCAY ORFIN ORGE ORMA OMD OSMEN OSTIM OTKAR OTOKC OTOSR OTTO OYAKC OYA OYYAT OYAYO OYLUM
    OZKGY OZATD OZGYO OZRDN OZSUB OZYSR
    PAMEL PNLSN PAGYO PAPIL PRFFK PRDGS PRKME PARSN PBT PBTR PASEU PSGYO PAHOL PATEK PCILT PGSUS PEKGY PENGD PENTA PSDTC PETKM PKENT PETUN PINSU PNSUT PKART PLTUR POLHO POLTK PRZMA
    QFINF QYATB QYHOL FIN QNBTR QNBFF QNBFK QNBVK QUAGR QUFIN
    RNPOL RALYH RAYSG REEDR RYGYO RYSAS RODRG ROYAL RGYAS RTALB RUBNS RUZYE
    SAFKR SANEL SNICA SANFM SANKO SAMAT SARKY SARTN SASA SAYAS SDTTR SEGMN SEKUR SELEC SELVA SNKRN SERNT SRVGY SEYKM SILVR SNGYO SKYLP SMRTG SMART SODSN SOKE SKTAS SONME SNPAM SUMAS SUNTK SURGY SUWEN SZUKI SMRFA SMRVA
    SEKFK SEGYO SKY SKYMD SEK SKBNK SOKM
    TABGD TAC TCRYT TAMFA TNZTP TARKM TATGD TATEN TAVHL DRPHN TEBFA TEBCE TEKTU TKFEN TKNSA TMPOL TRHOL TAE TRBNK TERA TRA TEHOL TFNVK TGSAS TIMUR TRYKI TOASO TRGYO TRMET TRENJ TLMAN TSPOR TDGYO TRMEN TVM TSGYO TUCLK TUKAS TRCAS TUREX MARBL TRKFN TRILC TCELL TBA TRKSH TRKNT TMSN TUPRS TRALT THYAO PRKAB TTKOM TTRAK TBORG TURGG GARAN TGB HALKB THL EXIMB THR ISATR ISBTR ISCTR ISKUR TIB KLN KLNMA TSK TSKB TURSG SISE TVB VAKBN TV8TV
    UFUK ULAS ULUFA ULUSE ULUUN UMPAS USAK
    ULKER UNLU
    VAKFA VAKFN VKGYO VKFYO VAKVK VAKKO VANGD VBTYZ VDFLO VRGYO VERUS VERTU VESBE VESTL VKING VSNMD VDFAS
    YKFKT YKFIN YKR YKYAT YKB YKBNK YAPRK YATAS YYLGD YAYLA YGGYO YEOTK YGYO YYAPI YESIL YBTAS YIGIT YONGA YKSLN YUNSA
    ZEDUR ZRGYO ZKBVK ZKBVR ZOREN
    BINHO
    """
    
    # Also include alternative codes mentioned in parentheses
    # Extract all symbols (including alternatives)
    symbols = set()
    lines = symbols_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        for part in parts:
            # Remove any trailing commas or special chars
            symbol = part.strip().upper()
            if symbol and len(symbol) >= 2:
                symbols.add(symbol)
    
    # Alternative code mappings from the list
    # Format: alternative -> main code
    alternative_mappings = {
        'AKM': 'AKMEN',
        'AFB': 'AKTIF',
        'ALBRK': 'ALK',
        'ALNUS': 'ANC',
        'A1CAP': 'ACP',
        'BLS': 'BLSMD',
        'MRBAS': 'MRS',
        'ICB': 'ICBCT',
        'IAZ': 'INVAZ',
        'INFO': 'IYF',
        'ISMEN': 'IYM',
        'KRDMA': 'KRDMB',
        'KRDMD': 'KRDMA',
        'MDASM': 'MDS',
        'MSY': 'MSYBN',
        'OMD': 'OSMEN',
        'OYA': 'OYYAT',
        'PBT': 'PBTR',
        'QYATB': 'YBQ',
        'FIN': 'QNBTR',
        'SKY': 'SKYMD',
        'SEK': 'SKBNK',
        'TAC': 'TCRYT',
        'TAE': 'TRBNK',
        'TERA': 'TRA',
        'TBA': 'TRKSH',
        'GARAN': 'TGB',
        'HALKB': 'THL',
        'EXIMB': 'THR',
        'ISATR': 'ISBTR',
        'ISCTR': 'ISKUR',
        'ISKUR': 'TIB',
        'KLN': 'KLNMA',
        'TSK': 'TSKB',
        'TVB': 'VAKBN',
    }
    
    # Reverse mapping: main -> alternatives
    reverse_mappings = {}
    for alt, main in alternative_mappings.items():
        if main not in reverse_mappings:
            reverse_mappings[main] = []
        reverse_mappings[main].append(alt)
    
    # Add all alternative codes
    for alt, main in alternative_mappings.items():
        symbols.add(alt)
        symbols.add(main)
    
    return symbols, alternative_mappings, reverse_mappings
    
    return symbols


def main():
    print("Fetching symbols from database...")
    db_symbols = get_db_symbols()
    print(f"Found {len(db_symbols)} symbols in database")
    
    print("\nParsing symbols from list...")
    list_symbols, alt_mappings, reverse_mappings = parse_symbols_from_list()
    print(f"Found {len(list_symbols)} symbols in list")
    
    # Find symbols in list but not in database
    missing_in_db_raw = sorted(list_symbols - db_symbols)
    
    # Check if missing symbols have alternatives that exist in DB
    missing_in_db = []
    missing_with_alt = []
    
    for sym in missing_in_db_raw:
        # Check if this is an alternative code and main code exists in DB
        if sym in alt_mappings:
            main_code = alt_mappings[sym]
            if main_code in db_symbols:
                missing_with_alt.append((sym, main_code))
                continue
        
        # Check if this is a main code and any alternative exists in DB
        if sym in reverse_mappings:
            for alt in reverse_mappings[sym]:
                if alt in db_symbols:
                    missing_with_alt.append((sym, alt))
                    break
            else:
                missing_in_db.append(sym)
        else:
            missing_in_db.append(sym)
    
    # Find symbols in database but not in list (optional, for completeness)
    extra_in_db = sorted(db_symbols - list_symbols)
    
    print("\n" + "="*80)
    print("SYMBOLS IN LIST BUT NOT IN DATABASE (with alternatives):")
    print("="*80)
    
    if missing_with_alt:
        print("\n⚠️  Symbols that are missing but have alternatives in DB:")
        for missing, exists in sorted(missing_with_alt):
            print(f"  {missing:<8} -> {exists} (alternative exists)")
    
    print("\n" + "="*80)
    print("SYMBOLS TRULY MISSING IN DATABASE:")
    print("="*80)
    
    if missing_in_db:
        # Group by first letter for better readability
        by_letter = {}
        for sym in missing_in_db:
            first_letter = sym[0] if sym else '?'
            if first_letter not in by_letter:
                by_letter[first_letter] = []
            by_letter[first_letter].append(sym)
        
        for letter in sorted(by_letter.keys()):
            print(f"\n{letter}:")
            symbols = sorted(by_letter[letter])
            for i, sym in enumerate(symbols, 1):
                print(f"  {sym:<8}", end="")
                if i % 10 == 0:
                    print()
            if len(symbols) % 10 != 0:
                print()
    else:
        print("  None - All symbols from list are in database!")
    
    print(f"\n\nTotal truly missing: {len(missing_in_db)}")
    print(f"Missing but have alternatives: {len(missing_with_alt)}")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"Symbols in list:        {len(list_symbols)}")
    print(f"Symbols in database:    {len(db_symbols)}")
    print(f"Symbols in both:        {len(list_symbols & db_symbols)}")
    print(f"Missing in database:    {len(missing_in_db)}")
    print(f"Extra in database:      {len(extra_in_db)}")


if __name__ == '__main__':
    main()

