"""
FinBERT Sentiment Analysis - Async Non-blocking Version

Background loading of FinBERT model to prevent worker blocking.
Uses ThreadPoolExecutor for model loading and LRU cache for loaded models.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
import uuid
import queue

# Suppress warnings during model loading
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AsyncFinBERTSentimentSystem:
    """Async FinBERT sentiment analysis system for non-blocking operation"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="finbert_async")
        self.model_cache = {}  # Simple cache for loaded model
        self.model_loading = threading.Event()
        self.model_loaded = False
        self.model = None
        self.tokenizer = None
        
        # Request queue for sentiment analysis
        self.request_queue = queue.Queue()
        self.result_store = {}
        self.result_lock = threading.Lock()
        
        # Start background model loading
        self.executor.submit(self._load_model_background)
        
        logger.info("📊 Async FinBERT Sentiment System initialized (background loading)")
    
    def _load_model_background(self):
        """Load FinBERT model in background thread"""
        try:
            logger.debug("🤖 Starting FinBERT model loading in background...")
            
            # Import heavy libraries only when needed
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            model_name = "ProsusAI/finbert"
            
            logger.debug(f"📥 Loading FinBERT tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            logger.debug(f"📥 Loading FinBERT model: {model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            self.model_loaded = True
            self.model_loading.set()
            
            logger.info("✅ FinBERT model loaded successfully in background!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FinBERT model in background: {e}")
            self.model_loaded = False
            self.model_loading.set()
    
    def request_sentiment_analysis_async(self, texts: List[str]) -> str:
        """Request sentiment analysis asynchronously (non-blocking)"""
        request_id = str(uuid.uuid4())
        
        if not texts or len(texts) == 0:
            # Store empty result immediately
            with self.result_lock:
                self.result_store[request_id] = {
                    'status': 'completed',
                    'sentiments': [],
                    'timestamp': time.time()
                }
            return request_id
        
        # Store pending result
        with self.result_lock:
            self.result_store[request_id] = {
                'status': 'pending',
                'texts': texts,
                'timestamp': time.time()
            }
        
        # Submit to background processing
        self.executor.submit(self._process_sentiment_request, request_id, texts)
        
        logger.debug(f"📊 Sentiment analysis requested for {len(texts)} texts (ID: {request_id})")
        return request_id
    
    def _process_sentiment_request(self, request_id: str, texts: List[str]):
        """Process sentiment analysis request in background"""
        try:
            # Wait for model to be loaded (non-blocking from main thread perspective)
            if not self.model_loading.wait(timeout=30):
                logger.warning("⏰ FinBERT model loading timeout")
                with self.result_lock:
                    self.result_store[request_id] = {
                        'status': 'error',
                        'error': 'Model loading timeout',
                        'timestamp': time.time()
                    }
                return
            
            if not self.model_loaded:
                logger.warning("❌ FinBERT model failed to load")
                with self.result_lock:
                    self.result_store[request_id] = {
                        'status': 'error',
                        'error': 'Model loading failed',
                        'timestamp': time.time()
                    }
                return
            
            # Perform sentiment analysis
            import torch
            from torch.nn.functional import softmax
            
            sentiments = []
            
            for text in texts:
                try:
                    if not self.tokenizer or not self.model:
                        sentiments.append({
                            'text': text[:100] + "..." if len(text) > 100 else text,
                            'sentiment': 'neutral',
                            'confidence': 0.33
                        })
                        continue
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True, 
                        padding=True, 
                        max_length=512
                    )
                    
                    # Predict
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        predictions = softmax(outputs.logits, dim=-1)
                    
                    # Get sentiment
                    predicted_class = int(torch.argmax(predictions, dim=-1).item())
                    confidence = torch.max(predictions).item()
                    
                    # Map to sentiment labels
                    labels = ['negative', 'neutral', 'positive']
                    sentiment = labels[predicted_class] if predicted_class < len(labels) else 'neutral'
                    
                    sentiments.append({
                        'text': text[:100] + "..." if len(text) > 100 else text,
                        'sentiment': sentiment,
                        'confidence': confidence
                    })
                    
                except Exception as e:
                    logger.warning(f"❌ Sentiment analysis failed for text: {e}")
                    sentiments.append({
                        'text': text[:100] + "..." if len(text) > 100 else text,
                        'sentiment': 'neutral',
                        'confidence': 0.33,
                        'error': str(e)
                    })
            
            # Store result
            with self.result_lock:
                self.result_store[request_id] = {
                    'status': 'completed',
                    'sentiments': sentiments,
                    'timestamp': time.time()
                }
            
            logger.debug(f"✅ Sentiment analysis completed for request {request_id}")
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis processing failed: {e}")
            with self.result_lock:
                self.result_store[request_id] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': time.time()
                }
    
    def get_sentiment_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get sentiment analysis result (non-blocking)"""
        with self.result_lock:
            return self.result_store.get(request_id)
    
    def cleanup_old_results(self, max_age_seconds: int = 300):
        """Clean up old results to prevent memory leaks"""
        current_time = time.time()
        to_remove = []
        
        with self.result_lock:
            for request_id, result in self.result_store.items():
                if current_time - result.get('timestamp', 0) > max_age_seconds:
                    to_remove.append(request_id)
            
            for request_id in to_remove:
                del self.result_store[request_id]
        
        if to_remove:
            logger.debug(f"🧹 Cleaned up {len(to_remove)} old sentiment results")


# Global singleton instance
_async_finbert_system = None
_async_finbert_lock = threading.Lock()


def get_async_finbert_sentiment_system():
    """Get async FinBERT sentiment system singleton"""
    global _async_finbert_system
    with _async_finbert_lock:
        if _async_finbert_system is None:
            _async_finbert_system = AsyncFinBERTSentimentSystem()
    return _async_finbert_system


# Cleanup function for old results
def cleanup_sentiment_results():
    """Cleanup old sentiment analysis results"""
    system = get_async_finbert_sentiment_system()
    system.cleanup_old_results()
