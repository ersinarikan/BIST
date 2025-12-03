#!/usr/bin/env python3
"""
Track JSON creation progress - real-time monitoring
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import time

def get_progress():
    """Get current progress"""
    results_dir = Path('/opt/bist-pattern/results')
    json_files = list(results_dir.glob("optuna_pilot_features_on_h1_c2_20251203_*.json"))
    
    today_start = datetime(2025, 12, 3, 2, 44, 0).timestamp()
    processed = {}
    
    for jf in json_files:
        mtime = jf.stat().st_mtime
        if mtime >= today_start:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                symbols = data.get('symbols', [])
                if symbols:
                    symbol = symbols[0]
                    if symbol not in processed or mtime > processed[symbol]['mtime']:
                        processed[symbol] = {
                            'file': jf.name,
                            'mtime': mtime,
                            'time': datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
                        }
            except:
                pass
    
    # Tüm study dosyaları
    study_dir = Path('/opt/bist-pattern/hpo_studies')
    study_files = list(study_dir.glob("hpo_with_features_*_h1_c2.db"))
    all_symbols = sorted([sf.stem.split('_')[3] for sf in study_files if len(sf.stem.split('_')) >= 4])
    
    return processed, all_symbols

def main():
    print("=" * 80)
    print("JSON OLUŞTURMA TAKİP")
    print("=" * 80)
    print("\n💡 Çıkmak için Ctrl+C")
    print("🔄 Her 5 saniyede bir güncelleniyor...\n")
    
    last_count = 0
    last_time = time.time()
    
    try:
        while True:
            processed, all_symbols = get_progress()
            
            current_count = len(processed)
            current_time = time.time()
            
            # İlerleme hızı
            if current_count > last_count:
                elapsed = current_time - last_time
                rate = (current_count - last_count) / elapsed if elapsed > 0 else 0
                eta_seconds = (len(all_symbols) - current_count) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
            else:
                rate = 0
                eta_minutes = 0
            
            # Son işlenen sembol
            processed_list = sorted(processed.items(), key=lambda x: x[1]['mtime'], reverse=True)
            last_symbol = processed_list[0][0] if processed_list else "N/A"
            last_time_str = processed_list[0][1]['time'] if processed_list else "N/A"
            
            # Şu anki sembol tahmini
            processed_set = set(processed.keys())
            current_symbol = None
            for sym in all_symbols:
                if sym not in processed_set:
                    current_symbol = sym
                    break
            
            # Clear screen and print
            print("\033[2J\033[H", end="")  # Clear screen
            print("=" * 80)
            print("JSON OLUŞTURMA TAKİP - CANLI")
            print("=" * 80)
            print(f"\n📊 İlerleme: {current_count}/{len(all_symbols)} ({current_count/len(all_symbols)*100:.1f}%)")
            print(f"⏳ Kalan: {len(all_symbols) - current_count}")
            
            if rate > 0:
                print(f"⚡ Hız: {rate:.2f} sembol/saniye")
                print(f"⏱️  Tahmini kalan süre: {eta_minutes:.1f} dakika")
            
            print(f"\n📅 Son işlenen: {last_symbol} ({last_time_str})")
            if current_symbol:
                idx = all_symbols.index(current_symbol)
                print(f"🔄 Şu an muhtemelen: {current_symbol} (sıra {idx+1}/{len(all_symbols)})")
            
            # Son 10 işlenen
            print(f"\n📋 Son 10 işlenen sembol:")
            for sym, info in processed_list[:10]:
                print(f"   ✅ {sym} ({info['time']})")
            
            # İşlenmeyen ilk 10
            missing = [s for s in all_symbols if s not in processed_set]
            if missing:
                print(f"\n⏳ İşlenmeyen ilk 10:")
                for sym in missing[:10]:
                    print(f"   ⏳ {sym}")
            
            last_count = current_count
            last_time = current_time
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n✅ Takip durduruldu")

if __name__ == '__main__':
    main()

