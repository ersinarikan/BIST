"""
FinGPT Sentiment Analysis System
FinBERT ile finansal sentiment analizi
"""

import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# FinBERT için gerekli kütüphaneler
AutoTokenizer = None
AutoModelForSequenceClassification = None
torch = None
FINBERT_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    logger.warning("FinBERT dependencies not available. Install with: pip install transformers torch")


class FinGPTAnalyzer:
    """FinBERT tabanlı sentiment analysis sistemi"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.model_name = None
        
        if FINBERT_AVAILABLE:
            try:
                # Türkçe sentiment model tercih et
                import os
                # ✅ FIX: Set TRANSFORMERS_CACHE and HF_HOME before loading model
                cache_dir = os.getenv('TRANSFORMERS_CACHE', '/opt/bist-pattern/.cache/huggingface')
                os.environ.setdefault('TRANSFORMERS_CACHE', cache_dir)
                os.environ.setdefault('HF_HOME', cache_dir)
                
                use_turkish_model = os.getenv('USE_TURKISH_SENTIMENT', 'True').lower() == 'true'
                
                if use_turkish_model:
                    # Türkçe sentiment modeli (öncelikli)
                    model_name = "savasy/bert-base-turkish-sentiment-cased"
                    logger.info(f"🇹🇷 Türkçe sentiment modeli yükleniyor (cache: {cache_dir})...")
                    self.model_name = model_name
                    try:
                        # Try local first - cache'den yükle
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, cache_dir=cache_dir)  # type: ignore
                        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True, cache_dir=cache_dir)  # type: ignore
                        logger.info("✅ Türkçe model yüklendi (local cache)")
                    except Exception as cache_err:
                        # Fallback to network if not in cache
                        logger.info(f"📥 Türkçe model cache'de bulunamadı ({cache_err}), network'ten indiriliyor...")
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)  # type: ignore
                        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)  # type: ignore
                        logger.info("✅ Türkçe model indirildi ve cache'lendi")
                else:
                    # İngilizce FinBERT (fallback)
                    model_name = "ProsusAI/finbert"
                    logger.info("🇺🇸 FinBERT modeli yükleniyor...")
                    self.model_name = model_name
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)  # type: ignore
                
                # Evaluation mode
                self.model.eval()
                self.model_loaded = True
                logger.info(f"✅ Sentiment modeli başarıyla yüklendi: {model_name}")
                
            except Exception as e:
                logger.warning(f"⚠️ Türkçe model yükleme hatası, FinBERT'e geçiliyor: {e}")
                # Fallback to FinBERT
                try:
                    model_name = "ProsusAI/finbert"
                    self.model_name = model_name
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)  # type: ignore
                    self.model.eval()
                    self.model_loaded = True
                    logger.info("✅ FinBERT fallback modeli yüklendi")
                except Exception as e2:
                    logger.error(f"❌ Tüm modeller yükleme hatası: {e2}")
                    self.model_loaded = False
    
    def analyze_sentiment(self, text):
        """Metindeki sentiment'i analiz et"""
        try:
            if not self.model_loaded:
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                    'status': 'model_unavailable'
                }
            
            # Tokenize
            if self.tokenizer is None:
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                    'status': 'model_unavailable'
                }
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            
            # Prediction (guard torch usage)
            if (not FINBERT_AVAILABLE) or (self.model is None):
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                    'status': 'model_unavailable'
                }
            with torch.no_grad():  # type: ignore
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)  # type: ignore
            
            # Sonuçları parse et
            scores = predictions[0].tolist()
            
            # Model type'a göre labels belirle ve normalize et
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
                raw_labels = list(self.model.config.id2label.values())
                labels = [str(label_value).lower() for label_value in raw_labels]
            else:
                labels = ['positive', 'negative', 'neutral']

            # Pozitif/negatif indekslerini bul
            def _idx(name: str, default: int) -> int:
                try:
                    return labels.index(name)
                except ValueError:
                    return default

            if len(scores) >= 3:
                pos_idx = _idx('positive', 0)
                neg_idx = _idx('negative', 1)
                neu_idx = _idx('neutral', 2)
                pos_score = scores[pos_idx]
                neg_score = scores[neg_idx]
                neu_score = scores[neu_idx]
            else:
                # 2 sınıflı Türkçe model: neutral yok, 0.0 say
                # Etiketler 'positive'/'negative' olmayabilir, bu yüzden en yüksek iki skoru pozitif/negatif olarak eşleştir
                pos_idx = _idx('positive', 1 if len(scores) > 1 else 0)
                neg_idx = _idx('negative', 0)
                pos_score = scores[pos_idx]
                neg_score = scores[neg_idx]
                neu_score = max(0.0, 1.0 - (pos_score + neg_score))

            # Normalize sentiment etiketi
            if pos_score >= neg_score and pos_score >= neu_score:
                sentiment = 'positive'
                confidence = float(pos_score)
            elif neg_score >= pos_score and neg_score >= neu_score:
                sentiment = 'negative'
                confidence = float(neg_score)
            else:
                sentiment = 'neutral'
                confidence = float(neu_score)

            # Skor sözlüğünü her zaman aynı anahtarlarla döndür
            scores_dict = {
                'positive': float(pos_score),
                'negative': float(neg_score),
                'neutral': float(neu_score),
            }
                
            result = {
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': scores_dict,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis hatası: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'status': 'error',
                'message': str(e)
            }
    
    def analyze_stock_news(self, symbol, news_texts):
        """Hisse için haber sentiment'lerini analiz et"""
        try:
            if not news_texts:
                try:
                    self._broadcast('INFO', f"FinGPT {symbol}: news=0 overall=neutral conf=0.00 (no_news)", 'news')
                    try:
                        logger.info(f"FinGPT {symbol}: news=0 overall=neutral conf=0.00 (no_news)")
                    except Exception as e:
                        logger.debug(f"Failed to log no_news message: {e}")
                except Exception as e:
                    logger.debug(f"Failed to process no_news case: {e}")
                return {
                    'symbol': symbol,
                    'overall_sentiment': 'neutral',
                    'confidence': 0.0,
                    'news_count': 0,
                    'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                    'status': 'no_news'
                }
            
            sentiments = []
            for text in news_texts:
                sentiment_result = self.analyze_sentiment(text)
                sentiments.append(sentiment_result)
            
            # Genel sentiment hesapla
            positive_count = sum(1 for s in sentiments if str(s.get('sentiment', '')).lower() == 'positive')
            negative_count = sum(1 for s in sentiments if str(s.get('sentiment', '')).lower() == 'negative')
            neutral_count = sum(1 for s in sentiments if str(s.get('sentiment', '')).lower() == 'neutral')
            
            total_positive_score = sum(float(s.get('scores', {}).get('positive', 0.0)) for s in sentiments)
            total_negative_score = sum(float(s.get('scores', {}).get('negative', 0.0)) for s in sentiments)
            total_neutral_score = sum(float(s.get('scores', {}).get('neutral', 0.0)) for s in sentiments)
            
            avg_positive = total_positive_score / len(sentiments)
            avg_negative = total_negative_score / len(sentiments)
            avg_neutral = total_neutral_score / len(sentiments)
            
            # Dominant sentiment
            if (avg_positive > avg_negative) and (avg_positive > avg_neutral):
                overall_sentiment = 'positive'
                overall_confidence = avg_positive
            elif (avg_negative > avg_positive) and (avg_negative > avg_neutral):
                overall_sentiment = 'negative'
                overall_confidence = avg_negative
            else:
                overall_sentiment = 'neutral'
                overall_confidence = avg_neutral
            
            result = {
                'symbol': symbol,
                'overall_sentiment': overall_sentiment,
                'confidence': overall_confidence,
                'news_count': len(news_texts),
                'sentiment_distribution': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count
                },
                'average_scores': {
                    'positive': avg_positive,
                    'negative': avg_negative,
                    'neutral': avg_neutral
                },
                'individual_sentiments': sentiments,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }

            # Broadcast compact log to admin dashboard (toggle with FINGPT_BROADCAST_LOGS)
            try:
                import os as _os
                if str(_os.getenv('FINGPT_BROADCAST_LOGS', '1')).lower() in ('1', 'true', 'yes'):
                    _msg = (f"FinGPT {symbol}: news={len(news_texts)} "
                            f"overall={overall_sentiment} conf={overall_confidence:.2f} "
                            f"pos={positive_count} neg={negative_count} neu={neutral_count}")
                    self._broadcast('INFO', _msg, 'news')
                    try:
                        logger.info(_msg)
                    except Exception as e:
                        logger.debug(f"Failed to log FinGPT message: {e}")
            except Exception as e:
                logger.debug(f"Failed to process FinGPT result: {e}")

            return result
            
        except Exception as e:
            logger.error(f"Stock news analysis hatası: {e}")
            return {
                'symbol': symbol,
                'overall_sentiment': 'neutral',
                'confidence': 0.0,
                'status': 'error',
                'message': str(e)
            }

    # Internal lightweight broadcaster (best-effort)
    def _broadcast(self, level: str, message: str, category: str = 'news') -> None:
        try:
            from flask import current_app
            app_obj = current_app._get_current_object()
            if hasattr(app_obj, 'broadcast_log'):
                # ✅ FIX: Add service identifier to distinguish from HPO logs
                app_obj.broadcast_log(level, message, category='working_automation', service='working_automation')
            else:
                sock = getattr(app_obj, 'socketio', None)
                if sock is not None:
                    sock.emit('log_update', {
                        'level': level,
                        'message': message,
                        'category': 'working_automation',
                        'service': 'working_automation',
                        'timestamp': datetime.now().isoformat(),
                    })
        except Exception as e:
            logger.debug(f"Failed to broadcast FinGPT result (no app context): {e}")
    
    def get_sentiment_signal(self, sentiment_result):
        """Sentiment'den trading sinyal türü belirle"""
        # Check if result has status (from analyze_sentiment) or not (from analyze_stock_news)
        if 'status' in sentiment_result and sentiment_result['status'] != 'success':
            return 'NEUTRAL'
        
        # Handle both 'sentiment' and 'overall_sentiment' keys for compatibility
        sentiment = sentiment_result.get('sentiment') or sentiment_result.get('overall_sentiment', 'neutral')
        confidence = sentiment_result.get('confidence', 0.0)
        
        # ⚡ NEW: Confidence threshold (HPO-optimizable, default 0.3)
        try:
            from bist_pattern.core.config_manager import ConfigManager
            threshold = float(ConfigManager.get('FINGPT_CONFIDENCE_THRESHOLD', '0.3'))
        except Exception as e:
            logger.debug(f"Failed to get FINGPT_CONFIDENCE_THRESHOLD, using 0.3: {e}")
            threshold = 0.3  # Default fallback
        
        if confidence < threshold:
            return 'NEUTRAL'
        
        if sentiment == 'positive':
            return 'BULLISH'
        elif sentiment == 'negative':
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def get_system_info(self):
        """Sistem bilgilerini döndür"""
        return {
            'finbert_available': FINBERT_AVAILABLE,
            'model_loaded': self.model_loaded,
            'model_name': "ProsusAI/finbert" if self.model_loaded else None
        }


# Global singleton instance
_fingpt_analyzer_instance = None


def get_fingpt_analyzer():
    """FinGPT analyzer singleton'ını döndür"""
    global _fingpt_analyzer_instance
    if _fingpt_analyzer_instance is None:
        _fingpt_analyzer_instance = FinGPTAnalyzer()
    return _fingpt_analyzer_instance


if __name__ == "__main__":
    # Test
    analyzer = get_fingpt_analyzer()
    
    test_texts = [
        "Company reports strong quarterly earnings, beating expectations",
        "Stock price falls amid market uncertainty",
        "New product launch expected to boost revenue"
    ]
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
        print("---")
