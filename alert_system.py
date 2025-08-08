"""
BIST Alert System
Otomatik sinyal bildirimleri ve uyarı sistemi
"""

import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import threading
import time
from dataclasses import dataclass, asdict
import schedule

logger = logging.getLogger(__name__)

@dataclass
class AlertConfig:
    """Alert konfigürasyonu"""
    symbol: str
    min_signal_strength: float = 70.0  # Minimum sinyal gücü
    signal_types: List[str] = None  # ['BULLISH', 'BEARISH', 'NEUTRAL']
    email_enabled: bool = False
    webhook_enabled: bool = False
    email_addresses: List[str] = None
    webhook_url: str = ""
    check_interval_minutes: int = 15
    active: bool = True

@dataclass
class Alert:
    """Alert verisi"""
    id: str
    symbol: str
    signal_type: str
    strength: float
    message: str
    timestamp: datetime
    config_id: str
    additional_data: Dict = None

class AlertSystem:
    """Otomatik alert sistemi"""
    
    def __init__(self):
        self.configs: Dict[str, AlertConfig] = {}
        self.alert_history: List[Alert] = []
        self.last_signals: Dict[str, Dict] = {}  # Son sinyalleri cache
        self.running = False
        self.scheduler_thread = None
        
        # Email config
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email_username = ""
        self.email_password = ""
        
        logger.info("🚨 Alert System başlatıldı")
    
    def add_alert_config(self, config: AlertConfig) -> str:
        """Alert konfigürasyonu ekle"""
        try:
            config_id = f"{config.symbol}_{datetime.now().timestamp()}"
            
            # Default değerler
            if config.signal_types is None:
                config.signal_types = ['BULLISH', 'BEARISH']
            if config.email_addresses is None:
                config.email_addresses = []
            
            self.configs[config_id] = config
            
            logger.info(f"📋 Alert config eklendi: {config.symbol} (min: {config.min_signal_strength}%)")
            return config_id
            
        except Exception as e:
            logger.error(f"Alert config ekleme hatası: {e}")
            return ""
    
    def remove_alert_config(self, config_id: str) -> bool:
        """Alert konfigürasyonu sil"""
        try:
            if config_id in self.configs:
                symbol = self.configs[config_id].symbol
                del self.configs[config_id]
                logger.info(f"🗑️ Alert config silindi: {symbol}")
                return True
            return False
        except Exception as e:
            logger.error(f"Alert config silme hatası: {e}")
            return False
    
    def check_signals(self):
        """Tüm konfigürasyonlar için sinyalleri kontrol et"""
        try:
            for config_id, config in self.configs.items():
                if not config.active:
                    continue
                
                try:
                    # API'den son analizi al
                    response = requests.get(
                        f"http://localhost:5000/api/pattern-analysis/{config.symbol}",
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        overall_signal = data.get('overall_signal', {})
                        
                        signal_type = overall_signal.get('signal', '').upper()
                        strength = overall_signal.get('strength', 0)
                        
                        # Alert koşullarını kontrol et
                        if self._should_trigger_alert(config, signal_type, strength):
                            self._trigger_alert(config_id, config, data)
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Signal check hatası ({config.symbol}): {e}")
            
        except Exception as e:
            logger.error(f"Signal check genel hatası: {e}")
    
    def _should_trigger_alert(self, config: AlertConfig, signal_type: str, strength: float) -> bool:
        """Alert tetiklensin mi kontrol et"""
        try:
            # Sinyal gücü yeterli mi?
            if strength < config.min_signal_strength:
                return False
            
            # Sinyal tipi izin verilen listede mi?
            if signal_type not in config.signal_types:
                return False
            
            # Son sinyal ile aynı mı? (Spam önleme)
            cache_key = f"{config.symbol}_{signal_type}"
            last_signal = self.last_signals.get(cache_key, {})
            
            # Son 30 dakika içinde aynı sinyal verildi mi?
            if last_signal:
                last_time = last_signal.get('timestamp')
                if last_time and datetime.now() - last_time < timedelta(minutes=30):
                    if abs(last_signal.get('strength', 0) - strength) < 10:  # ±10% fark yoksa
                        return False
            
            # Yeni sinyal, cache'e ekle
            self.last_signals[cache_key] = {
                'timestamp': datetime.now(),
                'strength': strength
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Alert kontrol hatası: {e}")
            return False
    
    def _trigger_alert(self, config_id: str, config: AlertConfig, analysis_data: Dict):
        """Alert tetikle"""
        try:
            overall_signal = analysis_data.get('overall_signal', {})
            signal_type = overall_signal.get('signal', '').upper()
            strength = overall_signal.get('strength', 0)
            
            # Alert mesajı oluştur
            message = self._create_alert_message(config.symbol, analysis_data)
            
            # Alert objesi oluştur
            alert = Alert(
                id=f"alert_{datetime.now().timestamp()}",
                symbol=config.symbol,
                signal_type=signal_type,
                strength=strength,
                message=message,
                timestamp=datetime.now(),
                config_id=config_id,
                additional_data={
                    'patterns': analysis_data.get('patterns', []),
                    'signals': overall_signal.get('signals', [])
                }
            )
            
            # Alert'i kaydet
            self.alert_history.append(alert)
            
            # Bildirimleri gönder
            if config.email_enabled and config.email_addresses:
                self._send_email_alert(alert, config)
            
            if config.webhook_enabled and config.webhook_url:
                self._send_webhook_alert(alert, config)
            
            logger.info(f"🚨 ALERT: {config.symbol} - {signal_type} ({strength}%)")
            
        except Exception as e:
            logger.error(f"Alert tetikleme hatası: {e}")
    
    def _create_alert_message(self, symbol: str, analysis_data: Dict) -> str:
        """Alert mesajı oluştur"""
        try:
            overall_signal = analysis_data.get('overall_signal', {})
            patterns = analysis_data.get('patterns', [])
            
            signal_type = overall_signal.get('signal', '').upper()
            strength = overall_signal.get('strength', 0)
            confidence = overall_signal.get('confidence', 0)
            
            message = f"""
🚨 BIST ALERT: {symbol}

📊 Sinyal: {signal_type} ({strength}%)
🎯 Güven: {confidence}%
⏰ Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔍 Tespit Edilen Patternler:
"""
            
            for pattern in patterns[:3]:  # İlk 3 pattern
                pattern_type = pattern.get('type', 'Unknown')
                pattern_confidence = pattern.get('confidence', 0)
                message += f"  • {pattern_type} ({pattern_confidence}%)\n"
            
            # Sinyal kaynaklarını ekle
            signals = overall_signal.get('signals', [])
            if signals:
                message += f"\n📈 Analiz Kaynakları: {len(signals)} aktif sistem\n"
                for signal in signals[:3]:
                    signal_name = signal.get('source', 'Unknown')
                    signal_value = signal.get('signal', 'N/A')
                    message += f"  • {signal_name}: {signal_value}\n"
            
            message += f"\n🔗 Detaylı Analiz: https://172.20.95.50/api/pattern-analysis/{symbol}"
            
            return message.strip()
            
        except Exception as e:
            logger.error(f"Mesaj oluşturma hatası: {e}")
            return f"🚨 {symbol} için {analysis_data.get('overall_signal', {}).get('signal', 'UNKNOWN')} sinyali"
    
    def _send_email_alert(self, alert: Alert, config: AlertConfig):
        """Email alert gönder"""
        try:
            if not self.email_username or not self.email_password:
                logger.warning("Email credentials ayarlanmamış")
                return
            
            # Email içeriği
            subject = f"🚨 BIST Alert: {alert.symbol} - {alert.signal_type}"
            
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['Subject'] = subject
            
            # HTML formatında mesaj
            html_message = f"""
            <html>
            <body>
                <h2 style="color: {'#e74c3c' if alert.signal_type == 'BEARISH' else '#27ae60'};">
                    {alert.signal_type} SİNYALİ
                </h2>
                <p><strong>Hisse:</strong> {alert.symbol}</p>
                <p><strong>Sinyal Gücü:</strong> {alert.strength}%</p>
                <p><strong>Zaman:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
                <pre>{alert.message}</pre>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_message, 'html', 'utf-8'))
            
            # Email gönder
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            
            for email_address in config.email_addresses:
                msg['To'] = email_address
                server.send_message(msg)
                logger.info(f"📧 Email alert gönderildi: {email_address}")
            
            server.quit()
            
        except Exception as e:
            logger.error(f"Email gönderme hatası: {e}")
    
    def _send_webhook_alert(self, alert: Alert, config: AlertConfig):
        """Webhook alert gönder"""
        try:
            payload = {
                'type': 'bist_alert',
                'symbol': alert.symbol,
                'signal': alert.signal_type,
                'strength': alert.strength,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'config_id': alert.config_id,
                'additional_data': alert.additional_data
            }
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'BIST-Alert-System/1.0'
            }
            
            response = requests.post(
                config.webhook_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"🔗 Webhook alert gönderildi: {config.webhook_url}")
            else:
                logger.warning(f"Webhook hatası: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Webhook gönderme hatası: {e}")
    
    def start_monitoring(self):
        """Alert monitoring'i başlat"""
        try:
            if self.running:
                logger.warning("Alert system zaten çalışıyor")
                return
            
            self.running = True
            
            # Schedule setup
            schedule.every(5).minutes.do(self.check_signals)  # Her 5 dakikada kontrol
            
            def run_scheduler():
                while self.running:
                    schedule.run_pending()
                    time.sleep(60)  # 1 dakika bekleme
            
            self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            logger.info("🚀 Alert monitoring başlatıldı")
            
        except Exception as e:
            logger.error(f"Alert monitoring başlatma hatası: {e}")
    
    def stop_monitoring(self):
        """Alert monitoring'i durdur"""
        try:
            self.running = False
            schedule.clear()
            
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
            
            logger.info("⏹️ Alert monitoring durduruldu")
            
        except Exception as e:
            logger.error(f"Alert monitoring durdurma hatası: {e}")
    
    def get_alert_configs(self) -> List[Dict]:
        """Alert konfigürasyonlarını döndür"""
        return [
            {
                'id': config_id,
                **asdict(config)
            }
            for config_id, config in self.configs.items()
        ]
    
    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Alert geçmişini döndür"""
        sorted_alerts = sorted(
            self.alert_history,
            key=lambda x: x.timestamp,
            reverse=True
        )
        
        return [
            {
                **asdict(alert),
                'timestamp': alert.timestamp.isoformat()
            }
            for alert in sorted_alerts[:limit]
        ]
    
    def configure_email(self, username: str, password: str, smtp_server: str = None, smtp_port: int = None):
        """Email ayarlarını yapılandır"""
        self.email_username = username
        self.email_password = password
        
        if smtp_server:
            self.smtp_server = smtp_server
        if smtp_port:
            self.smtp_port = smtp_port
        
        logger.info("📧 Email yapılandırması güncellendi")
    
    def test_alert(self, symbol: str) -> bool:
        """Test alert gönder"""
        try:
            # Test config oluştur
            test_config = AlertConfig(
                symbol=symbol,
                min_signal_strength=0,  # Test için minimum
                signal_types=['BULLISH', 'BEARISH', 'NEUTRAL'],
                email_enabled=False,
                webhook_enabled=False
            )
            
            # Fake analiz verisi
            test_data = {
                'overall_signal': {
                    'signal': 'BULLISH',
                    'strength': 85,
                    'confidence': 75,
                    'signals': [
                        {'source': 'TEST', 'signal': 'BULLISH', 'weight': 1.0}
                    ]
                },
                'patterns': [
                    {'type': 'TEST_PATTERN', 'confidence': 80}
                ]
            }
            
            # Test alert tetikle
            self._trigger_alert('test_config', test_config, test_data)
            
            logger.info(f"✅ Test alert başarılı: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Test alert hatası: {e}")
            return False

# Global singleton instance
_alert_system = None

def get_alert_system():
    """Alert System singleton'ını döndür"""
    global _alert_system
    if _alert_system is None:
        _alert_system = AlertSystem()
    return _alert_system

if __name__ == "__main__":
    # Test
    alert_system = get_alert_system()
    
    # Test config
    config = AlertConfig(
        symbol="THYAO",
        min_signal_strength=70,
        signal_types=['BULLISH', 'BEARISH'],
        email_enabled=False,
        webhook_enabled=False,
        check_interval_minutes=5
    )
    
    config_id = alert_system.add_alert_config(config)
    print(f"✅ Alert config eklendi: {config_id}")
    
    # Test alert
    if alert_system.test_alert("THYAO"):
        print("✅ Test alert başarılı")
    
    # Monitoring başlat
    alert_system.start_monitoring()
    print("🚀 Alert monitoring başlatıldı")
    
    try:
        time.sleep(10)  # 10 saniye bekle
    finally:
        alert_system.stop_monitoring()
        print("⏹️ Alert monitoring durduruldu")
