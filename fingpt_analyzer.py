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
        
        if FINBERT_AVAILABLE:
            try:
                # FinBERT modelini yükle
                model_name = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Evaluation mode
                self.model.eval()
                self.model_loaded = True
                logger.info("FinBERT modeli başarıyla yüklendi")
                
            except Exception as e:
                logger.error(f"FinBERT model yükleme hatası: {e}")
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
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                   max_length=512, padding=True)
            
            # Prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Sonuçları parse et
            scores = predictions[0].tolist()
            labels = ['positive', 'negative', 'neutral']
            
            # En yüksek skoru bul
            max_score_idx = scores.index(max(scores))
            sentiment = labels[max_score_idx]
            confidence = max(scores)
            
            result = {
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': {
                    'positive': scores[0],
                    'negative': scores[1], 
                    'neutral': scores[2]
                },
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
            positive_count = sum(1 for s in sentiments if s['sentiment'] == 'positive')
            negative_count = sum(1 for s in sentiments if s['sentiment'] == 'negative')
            neutral_count = sum(1 for s in sentiments if s['sentiment'] == 'neutral')
            
            total_positive_score = sum(s['scores']['positive'] for s in sentiments)
            total_negative_score = sum(s['scores']['negative'] for s in sentiments)
            total_neutral_score = sum(s['scores']['neutral'] for s in sentiments)
            
            avg_positive = total_positive_score / len(sentiments)
            avg_negative = total_negative_score / len(sentiments)
            avg_neutral = total_neutral_score / len(sentiments)
            
            # Dominant sentiment
            if avg_positive > avg_negative and avg_positive > avg_neutral:
                overall_sentiment = 'positive'
                overall_confidence = avg_positive
            elif avg_negative > avg_positive and avg_negative > avg_neutral:
                overall_sentiment = 'negative'
                overall_confidence = avg_negative
            else:
                overall_sentiment = 'neutral'
                overall_confidence = avg_neutral
            
            return {
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
            
        except Exception as e:
            logger.error(f"Stock news analysis hatası: {e}")
            return {
                'symbol': symbol,
                'overall_sentiment': 'neutral',
                'confidence': 0.0,
                'status': 'error',
                'message': str(e)
            }
    
    def get_sentiment_signal(self, sentiment_result):
        """Sentiment'den trading sinyal türü belirle"""
        if sentiment_result['status'] != 'success':
            return 'NEUTRAL'
        
        sentiment = sentiment_result['sentiment']
        confidence = sentiment_result['confidence']
        
        # Yüksek güvenlik threshold'u
        if confidence < 0.6:
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
