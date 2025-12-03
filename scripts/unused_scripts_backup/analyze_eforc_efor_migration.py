#!/usr/bin/env python3
"""
Analyze EFORC -> EFOR migration impact on HPO process
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, '/opt/bist-pattern')

from sqlalchemy import create_engine, text


def get_db_connection():
    """Get database connection"""
    project_root = Path('/opt/bist-pattern')
    secrets_file = project_root / '.secrets' / 'db_password'
    with open(secrets_file, 'r') as f:
        db_password = f.read().strip()
    db_url = f'postgresql://bist_user:{db_password}@127.0.0.1:6432/bist_pattern_db'
    return create_engine(db_url)


def analyze_eforc_efor_situation():
    """Analyze EFORC/EFOR situation"""
    print("=" * 80)
    print("EFORC -> EFOR MİGRASYON ANALİZİ")
    print("=" * 80)
    print()
    
    # 1. Check database
    print("1. VERİTABANI DURUMU:")
    print("-" * 80)
    engine = get_db_connection()
    with engine.connect() as conn:
        # Check EFORC
        eforc = conn.execute(text("SELECT symbol, name, created_at FROM stocks WHERE symbol = 'EFORC'")).fetchone()
        if eforc:
            print(f"✅ EFORC exists: {eforc[1]} (created: {eforc[2]})")
        else:
            print("❌ EFORC not found in database")
        
        # Check EFOR
        efor = conn.execute(text("SELECT symbol, name, created_at FROM stocks WHERE symbol = 'EFOR'")).fetchone()
        if efor:
            print(f"✅ EFOR exists: {efor[1]} (created: {efor[2]})")
        else:
            print("❌ EFOR not found in database")
        
        # Check price data
        if eforc:
            price_count = conn.execute(text("""
                SELECT COUNT(*) FROM stock_prices sp
                JOIN stocks s ON s.id = sp.stock_id
                WHERE s.symbol = 'EFORC'
            """)).scalar()
            print(f"   EFORC price records: {price_count}")
        
        if efor:
            price_count = conn.execute(text("""
                SELECT COUNT(*) FROM stock_prices sp
                JOIN stocks s ON s.id = sp.stock_id
                WHERE s.symbol = 'EFOR'
            """)).scalar()
            print(f"   EFOR price records: {price_count}")
    
    print()
    
    # 2. Check HPO state
    print("2. HPO SÜREÇ DURUMU:")
    print("-" * 80)
    state_file = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        eforc_tasks = {k: v for k, v in state.items() if 'EFORC' in k.upper()}
        efor_tasks = {k: v for k, v in state.items() if 'EFOR' in k.upper() and 'EFORC' not in k.upper()}
        
        if eforc_tasks:
            print(f"✅ EFORC tasks found: {len(eforc_tasks)}")
            for key, task in eforc_tasks.items():
                print(f"   {key}: Status={task.get('status')}, Cycle={task.get('cycle')}")
        else:
            print("❌ No EFORC tasks found")
        
        if efor_tasks:
            print(f"✅ EFOR tasks found: {len(efor_tasks)}")
            for key, task in efor_tasks.items():
                print(f"   {key}: Status={task.get('status')}, Cycle={task.get('cycle')}")
        else:
            print("❌ No EFOR tasks found")
    else:
        print("❌ State file not found")
    
    print()
    
    # 3. Check HPO PID files
    print("3. HPO PID DOSYALARI:")
    print("-" * 80)
    pid_dir = Path('/opt/bist-pattern/results/hpo_pids')
    if pid_dir.exists():
        eforc_pids = list(pid_dir.glob('EFORC_*.pid'))
        efor_pids = list(pid_dir.glob('EFOR_*.pid'))
        
        if eforc_pids:
            print(f"✅ EFORC PID files: {len(eforc_pids)}")
            for pid_file in eforc_pids:
                try:
                    pid = int(pid_file.read_text().strip())
                    print(f"   {pid_file.name}: PID={pid}")
                except:
                    print(f"   {pid_file.name}: Invalid PID")
        else:
            print("❌ No EFORC PID files")
        
        if efor_pids:
            print(f"✅ EFOR PID files: {len(efor_pids)}")
            for pid_file in efor_pids:
                print(f"   {pid_file.name}")
        else:
            print("❌ No EFOR PID files")
    else:
        print("❌ PID directory not found")
    
    print()
    
    # 4. Recommendations
    print("4. ÖNERİLER:")
    print("-" * 80)
    print("""
    Senaryo: 3 Kasım 2025'te EFORC -> EFOR değişmiş
    
    Seçenekler:
    
    A) EFORC HPO'yu bitir, sonra EFOR'a geç
       - Mevcut EFORC HPO'yu durdurma (veri kaybı olur)
       - EFORC HPO bitene kadar bekle
       - Sonra veritabanında EFORC -> EFOR güncelle
       - Yeni cycle'da EFOR ile devam et
    
    B) EFORC HPO'yu durdur, hemen EFOR'a geç
       - Mevcut EFORC HPO'yu durdur (PID kill)
       - State dosyasını güncelle (EFORC -> EFOR)
       - Veritabanında EFORC -> EFOR güncelle
       - Yeni cycle'da EFOR ile başla
    
    C) İkisini de tut (geçiş dönemi)
       - EFORC HPO bitene kadar devam et
       - EFOR için yeni task oluştur
       - Geçiş döneminde her ikisi de çalışsın
    
    ÖNERİ: Seçenek A (en güvenli)
    - Mevcut HPO'yu bitir
    - Sonra temiz geçiş yap
    """)


if __name__ == '__main__':
    analyze_eforc_efor_situation()

