"""
News & Sentiment Analysis System
Optimized FinGPT integration with RSS news feeds
"""

import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
import threading

logger = logging.getLogger(__name__)


class NewsProvider:
    """
    Optimized RSS news provider with smart caching
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = int(os.getenv('NEWS_CACHE_TTL', '600'))  # 10 minutes
        self.max_items = int(os.getenv('NEWS_MAX_ITEMS', '15'))
        self.lookback_hours = int(os.getenv('NEWS_LOOKBACK_HOURS', '24'))
        self.lock = threading.Lock()
        
        # RSS sources from environment
        sources_env = os.getenv('NEWS_SOURCES', '')
        self.rss_sources = [s.strip() for s in sources_env.split(',') if s.strip()]
        
        # Check feedparser availability
        self.feedparser_available = False
        try:
            import feedparser
            self.feedparser = feedparser
            self.feedparser_available = True
            logger.info(f"📰 RSS News Provider initialized with {len(self.rss_sources)} sources")
        except ImportError:
            logger.warning("📰 feedparser not available. RSS news disabled.")
    
    def get_recent_news(self, symbol: str) -> List[str]:
        """Get recent news for a stock symbol"""
        if not self.feedparser_available or not self.rss_sources:
            return []
        
        cache_key = f"news_{symbol}_{len(self.rss_sources)}"
        
        # Check cache
        with self.lock:
            if cache_key in self.cache:
                cached_entry = self.cache[cache_key]
                if time.time() - cached_entry['timestamp'] < self.cache_ttl:
                    return cached_entry['data']
        
        try:
            news_items = []
            symbol_upper = symbol.upper()
            
            for rss_url in self.rss_sources:
                try:
                    feed = self.feedparser.parse(rss_url)
                    
                    for entry in feed.entries[:self.max_items]:
                        # Extract text from entry
                        text = self._extract_entry_text(entry)
                        if not text:
                            continue
                        
                        # Check if news is related to symbol or general market
                        if self._is_relevant_news(text, symbol_upper):
                            news_items.append(text)
                        
                        if len(news_items) >= self.max_items:
                            break
                    
                except Exception as e:
                    logger.warning(f"RSS feed error {rss_url}: {e}")
                    continue
            
            # Cache results
            with self.lock:
                self.cache[cache_key] = {
                    'data': news_items,
                    'timestamp': time.time()
                }
            
            return news_items[:self.max_items]
            
        except Exception as e:
            logger.error(f"News retrieval error for {symbol}: {e}")
            return []
    
    def _extract_entry_text(self, entry) -> str:
        """Extract readable text from RSS entry"""
        try:
            title = getattr(entry, 'title', '').strip()
            description = getattr(entry, 'description', '').strip()
            
            # Combine title and description
            text_parts = []
            if title:
                text_parts.append(title)
            if description and description != title:
                # Clean HTML tags from description
                import re
                clean_desc = re.sub(r'<[^>]+>', '', description)
                text_parts.append(clean_desc[:200])  # Limit description length
            
            return ' - '.join(text_parts)
            
        except Exception:
            return ''
    
    def _is_relevant_news(self, text: str, symbol: str) -> bool:
        """Check if news is relevant to symbol or general market"""
        text_upper = text.upper()
        
        # Symbol-specific check
        if symbol in text_upper:
            return True
        
        # General market keywords
        market_keywords = [
            'BIST', 'BORSA', 'HİSSE', 'HISSE', 'EKONOMİ', 'EKONOMI',
            'FİNANS', 'FINANS', 'TAHVIL', 'DÖVİZ', 'DOVIZ', 'ALTIN',
            'PETROL', 'ENFLASYON', 'FAİZ', 'FAIZ'
        ]
        
        return any(keyword in text_upper for keyword in market_keywords)


class SentimentAnalyzer:
    """
    Simple sentiment analyzer with FinBERT fallback
    """
    
    def __init__(self):
        self.finbert_available = False
        self.model_loaded = False
        self.model = None
        self.tokenizer = None
        
        # Try to load FinBERT
        self._try_load_finbert()
        
        # Simple sentiment keywords as fallback
        self.positive_keywords = [
            'yüksel', 'artt', 'kazanç', 'kar', 'büyü', 'rekor', 'başarı',
            'olumlu', 'güçlü', 'iyi', 'pozitif', 'yatırım', 'hedef',
            'yukarı', 'artış', 'çıkış'
        ]
        
        self.negative_keywords = [
            'düş', 'azal', 'kayıp', 'zarar', 'kriz', 'sorun', 'risk',
            'olumsuz', 'zayıf', 'kötü', 'negatif', 'satış', 'aşağı',
            'düşüş', 'çöküş', 'gerileme'
        ]
    
    def _try_load_finbert(self):
        """Try to load FinBERT model"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            
            self.finbert_available = True
            self.model_loaded = True
            logger.info("✅ FinBERT model loaded successfully")
            
        except ImportError:
            logger.info("📊 FinBERT dependencies not available, using simple sentiment")
        except Exception as e:
            logger.warning(f"📊 FinBERT model loading failed: {e}, using simple sentiment")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        if self.model_loaded and self.finbert_available:
            return self._analyze_with_finbert(text)
        else:
            return self._analyze_with_keywords(text)
    
    def _analyze_with_finbert(self, text: str) -> Dict[str, Any]:
        """Analyze using FinBERT model"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            
            import torch
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            scores = predictions[0].tolist()
            labels = ['positive', 'negative', 'neutral']
            
            max_score_idx = scores.index(max(scores))
            sentiment = labels[max_score_idx]
            confidence = max(scores)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': {
                    'positive': scores[0],
                    'negative': scores[1],
                    'neutral': scores[2]
                },
                'method': 'finbert'
            }
            
        except Exception as e:
            logger.error(f"FinBERT analysis error: {e}")
            return self._analyze_with_keywords(text)
    
    def _analyze_with_keywords(self, text: str) -> Dict[str, Any]:
        """Simple keyword-based sentiment analysis"""
        try:
            text_lower = text.lower()
            
            positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
            negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
            
            if positive_count > negative_count:
                sentiment = 'positive'
                confidence = min(0.8, 0.5 + (positive_count - negative_count) * 0.1)
            elif negative_count > positive_count:
                sentiment = 'negative'
                confidence = min(0.8, 0.5 + (negative_count - positive_count) * 0.1)
            else:
                sentiment = 'neutral'
                confidence = 0.5
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': {
                    'positive': positive_count / max(1, positive_count + negative_count + 1),
                    'negative': negative_count / max(1, positive_count + negative_count + 1),
                    'neutral': 1 / max(1, positive_count + negative_count + 1)
                },
                'method': 'keywords',
                'keyword_counts': {
                    'positive': positive_count,
                    'negative': negative_count
                }
            }
            
        except Exception as e:
            logger.error(f"Keyword sentiment analysis error: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'method': 'error'
            }
    
    def analyze_stock_news(self, symbol: str, news_texts: List[str]) -> Dict[str, Any]:
        """Analyze sentiment for multiple news texts"""
        if not news_texts:
            return {
                'symbol': symbol,
                'overall_sentiment': 'neutral',
                'confidence': 0.0,
                'news_count': 0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'method': 'no_news'
            }
        
        sentiments = []
        for text in news_texts:
            sentiment_result = self.analyze_sentiment(text)
            sentiments.append(sentiment_result)
        
        # Calculate overall sentiment
        positive_count = sum(1 for s in sentiments if s['sentiment'] == 'positive')
        negative_count = sum(1 for s in sentiments if s['sentiment'] == 'negative')
        neutral_count = sum(1 for s in sentiments if s['sentiment'] == 'neutral')
        
        total_confidence = sum(s['confidence'] for s in sentiments)
        avg_confidence = total_confidence / len(sentiments) if sentiments else 0.0
        
        # Determine overall sentiment
        if positive_count > negative_count:
            overall_sentiment = 'positive'
        elif negative_count > positive_count:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'symbol': symbol,
            'overall_sentiment': overall_sentiment,
            'confidence': avg_confidence,
            'news_count': len(news_texts),
            'sentiment_distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            },
            'method': sentiments[0]['method'] if sentiments else 'unknown',
            'individual_sentiments': sentiments
        }
    
    def get_sentiment_signal(self, sentiment_result: Dict) -> str:
        """Convert sentiment to trading signal"""
        sentiment = sentiment_result.get('overall_sentiment', 'neutral')
        confidence = sentiment_result.get('confidence', 0.0)
        
        if sentiment == 'positive' and confidence > 0.6:
            return 'BULLISH'
        elif sentiment == 'negative' and confidence > 0.6:
            return 'BEARISH'
        else:
            return 'NEUTRAL'


class NewsSentimentSystem:
    """
    Complete news & sentiment analysis system
    """
    
    def __init__(self):
        self.news_provider = NewsProvider()
        self.sentiment_analyzer = SentimentAnalyzer()
        logger.info("📊 News & Sentiment System initialized")
    
    def analyze_stock_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Complete sentiment analysis for a stock"""
        try:
            # Get recent news
            news_texts = self.news_provider.get_recent_news(symbol)
            
            # Analyze sentiment
            sentiment_result = self.sentiment_analyzer.analyze_stock_news(symbol, news_texts)
            
            # Add timestamp and system info
            sentiment_result.update({
                'timestamp': datetime.now().isoformat(),
                'system_info': {
                    'rss_sources': len(self.news_provider.rss_sources),
                    'finbert_available': self.sentiment_analyzer.finbert_available,
                    'model_loaded': self.sentiment_analyzer.model_loaded
                }
            })
            
            return sentiment_result
            
        except Exception as e:
            logger.error(f"Stock sentiment analysis error for {symbol}: {e}")
            return {
                'symbol': symbol,
                'overall_sentiment': 'neutral',
                'confidence': 0.0,
                'news_count': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and capabilities"""
        return {
            'news_provider': {
                'feedparser_available': self.news_provider.feedparser_available,
                'rss_sources_count': len(self.news_provider.rss_sources),
                'cache_size': len(self.news_provider.cache)
            },
            'sentiment_analyzer': {
                'finbert_available': self.sentiment_analyzer.finbert_available,
                'model_loaded': self.sentiment_analyzer.model_loaded,
                'fallback_method': 'keywords'
            }
        }


# Global singleton
_news_sentiment_system = None


def get_news_sentiment_system() -> NewsSentimentSystem:
    """Get news sentiment system singleton"""
    global _news_sentiment_system
    if _news_sentiment_system is None:
        _news_sentiment_system = NewsSentimentSystem()
    return _news_sentiment_system


# Backward compatibility
def get_fingpt_analyzer():
    """Backward compatibility function"""
    return get_news_sentiment_system().sentiment_analyzer


def get_recent_news(symbol: str, **kwargs) -> List[str]:
    """Backward compatibility function"""
    return get_news_sentiment_system().news_provider.get_recent_news(symbol)
