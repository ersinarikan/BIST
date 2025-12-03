#!/usr/bin/env python3
"""
Disable AUTO_START_CYCLE to prevent working_automation from auto-starting
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

# Check systemd override file
override_file = Path('/etc/systemd/system/bist-pattern.service.d/99-consolidated.conf')

print("=" * 80)
print("WORKING_AUTOMATION AUTO-START KAPATMA")
print("=" * 80)
print()

if override_file.exists():
    print(f"✅ Override dosyası bulundu: {override_file}")
    
    # Read current content
    with open(override_file, 'r') as f:
        content = f.read()
    
    # Check if AUTO_START_CYCLE is already set
    if 'AUTO_START_CYCLE' in content:
        # Replace existing value
        import re
        content = re.sub(r'AUTO_START_CYCLE=.*', 'AUTO_START_CYCLE=False', content)
        print("✅ AUTO_START_CYCLE=False olarak güncellendi")
    else:
        # Add new line
        if '[Service]' in content:
            content = content.replace('[Service]', '[Service]\nEnvironment="AUTO_START_CYCLE=False"')
        else:
            content = '[Service]\nEnvironment="AUTO_START_CYCLE=False"\n' + content
        print("✅ AUTO_START_CYCLE=False eklendi")
    
    # Write back
    with open(override_file, 'w') as f:
        f.write(content)
    
    print()
    print("=" * 80)
    print("SONUÇ:")
    print("=" * 80)
    print("✅ AUTO_START_CYCLE=False olarak ayarlandı")
    print()
    print("⚠️  Değişikliklerin etkili olması için servisi restart etmelisiniz:")
    print("   sudo systemctl daemon-reload")
    print("   sudo systemctl restart bist-pattern.service")
    print()
    print("ℹ️  Not: working_automation artık otomatik başlamayacak.")
    print("   Manuel başlatmak için admin dashboard'dan veya API'den başlatabilirsiniz.")
else:
    print(f"❌ Override dosyası bulunamadı: {override_file}")
    print()
    print("Manuel olarak şu satırı ekleyin:")
    print('   Environment="AUTO_START_CYCLE=False"')
    print()
    print("Dosya: /etc/systemd/system/bist-pattern.service.d/99-consolidated.conf")

