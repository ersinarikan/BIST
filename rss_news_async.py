"""
Async RSS News Provider System
Background RSS fetching with aiohttp and cache-based delivery
"""

import asyncio
import aiohttp
import threading
import time
import logging
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
# from datetime import datetime  # unused
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AsyncRSSNewsProvider:
    """
    Async RSS News Provider with background fetching
    
    Features:
    - Background RSS feed fetching
    - AsyncIO + aiohttp for non-blocking requests
    - Intelligent caching system
    - Timeout & error handling
    - Turkish financial news focus
    """
    
    def __init__(self, worker_threads: int = 2):
        self.worker_threads = worker_threads
        
        # Configuration from environment
        self.cache_ttl = int(os.getenv('NEWS_CACHE_TTL', '600'))  # 10 minutes
        self.max_items = int(os.getenv('NEWS_MAX_ITEMS', '15'))
        self.lookback_hours = int(os.getenv('NEWS_LOOKBACK_HOURS', '24'))
        self.request_timeout = int(os.getenv('RSS_TIMEOUT', '5'))  # 5 seconds
        
        # RSS sources from environment
        sources_env = os.getenv('NEWS_SOURCES', '')
        self.rss_sources = [s.strip() for s in sources_env.split(',') if s.strip()]
        
        # Cache and state
        self._news_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        self._last_fetch_time = 0
        self._fetching = False
        
        # Background worker
        self._background_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="RSSFetcher")
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self._stats = {
            'fetches_completed': 0,
            'feeds_processed': 0,
            'news_items_cached': 0,
            'cache_hits': 0,
            'errors': 0,
            'last_fetch_duration': 0
        }
        
        # Turkish financial keywords
        self.financial_keywords = [
            'BIST', 'borsa', 'hisse', 'ekonomi', 'finans', 'şirket', 'tahvil',
            'döviz', 'altın', 'petrol', 'enflasyon', 'merkez bankası', 'faiz',
            'yatırım', 'kar', 'zarar', 'büyüme', 'resesyon', 'ihracat', 'ithalat'
        ]
        
        # Start background fetching
        if self.rss_sources:
            self._start_background_fetching()
        
        logger.info(f"📰 Async RSS News Provider initialized ({len(self.rss_sources)} sources)")
    
    def _start_background_fetching(self):
        """Start background RSS fetching"""
        def _run_in_new_loop(coro):
            try:
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    try:
                        loop.close()
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"RSS loop run error: {e}")
                return None
        
        def background_fetcher():
            """Background thread that periodically fetches RSS feeds"""
            while True:
                try:
                    # Check if it's time to fetch (every 5 minutes)
                    current_time = time.time()
                    if current_time - self._last_fetch_time > 300 and not self._fetching:  # 5 minutes
                        _run_in_new_loop(self._fetch_all_feeds_async())
                    
                    # Sleep for 30 seconds before next check
                    time.sleep(30)
                    
                except Exception as e:
                    logger.error(f"Background RSS fetcher error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start background thread
        thread = threading.Thread(target=background_fetcher, daemon=True, name="RSSBackgroundFetcher")
        thread.start()
        
        # Immediate first fetch (separate thread/loop)
        self._background_pool.submit(lambda: _run_in_new_loop(self._fetch_all_feeds_async()))
    
    async def _create_session(self) -> aiohttp.ClientSession:
        """Create aiohttp session with proper configuration"""
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; BIST-Pattern RSS Reader; +https://github.com/bist-pattern)'
        }
        
        return aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=3)
        )
    
    async def _fetch_all_feeds_async(self):
        """Fetch all RSS feeds asynchronously"""
        if self._fetching:
            return
        
        try:
            self._fetching = True
            fetch_start_time = time.time()
            
            logger.debug(f"🔄 Background fetching {len(self.rss_sources)} RSS feeds...")
            
            async with await self._create_session() as session:
                # Fetch all feeds concurrently
                tasks = [self._fetch_single_feed(session, url) for url in self.rss_sources]
                feed_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                all_news_items = []
                for i, result in enumerate(feed_results):
                    if isinstance(result, Exception):
                        logger.warning(f"RSS feed error {self.rss_sources[i]}: {result}")
                        self._stats['errors'] += 1
                    elif isinstance(result, list):
                        all_news_items.extend(result)
                        self._stats['feeds_processed'] += 1
                
                # Cache the news globally
                with self._cache_lock:
                    cache_entry = {
                        'data': all_news_items,
                        'timestamp': time.time(),
                        'items_count': len(all_news_items)
                    }
                    self._news_cache['global_news'] = cache_entry
                
                self._last_fetch_time = time.time()
                self._stats['fetches_completed'] += 1
                self._stats['news_items_cached'] = len(all_news_items)
                self._stats['last_fetch_duration'] = int(time.time() - fetch_start_time)
                
                logger.info(f"✅ RSS fetch completed: {len(all_news_items)} news items cached in {self._stats['last_fetch_duration']:.1f}s")
        
        except Exception as e:
            logger.error(f"RSS background fetch error: {e}")
            self._stats['errors'] += 1
        finally:
            self._fetching = False
    
    async def _fetch_single_feed(self, session: aiohttp.ClientSession, url: str) -> List[Dict[str, Any]]:
        """Fetch and parse a single RSS feed"""
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"RSS feed HTTP {response.status}: {url}")
                    return []
                
                content = await response.text()
                return self._parse_rss_content(content)
                
        except asyncio.TimeoutError:
            logger.warning(f"RSS feed timeout: {url}")
            return []
        except Exception as e:
            logger.warning(f"RSS feed error {url}: {e}")
            return []
    
    def _parse_rss_content(self, content: str) -> List[Dict[str, Any]]:
        """Parse RSS XML content"""
        try:
            root = ET.fromstring(content)
            news_items = []
            
            # Handle different RSS formats
            items = root.findall('.//item') or root.findall('.//{http://purl.org/rss/1.0/}item')
            
            for item in items[:self.max_items]:
                try:
                    title = ''
                    description = ''
                    pub_date = ''
                    
                    # Extract title
                    title_elem = item.find('title')
                    if title_elem is not None and title_elem.text:
                        title = title_elem.text.strip()
                    
                    # Extract description
                    desc_elem = item.find('description') or item.find('summary')
                    if desc_elem is not None and desc_elem.text:
                        description = desc_elem.text.strip()
                    
                    # Extract publication date
                    date_elem = item.find('pubDate') or item.find('published')
                    if date_elem is not None and date_elem.text:
                        pub_date = date_elem.text.strip()
                    
                    # Combine title and description
                    text = f"{title}. {description}".strip()
                    
                    if text and len(text) > 20:  # Minimum text length
                        news_items.append({
                            'text': text,
                            'title': title,
                            'description': description,
                            'pub_date': pub_date,
                            'timestamp': time.time()
                        })
                
                except Exception as e:
                    logger.debug(f"RSS item parsing error: {e}")
                    continue
            
            return news_items
            
        except ET.ParseError as e:
            logger.warning(f"RSS XML parsing error: {e}")
            return []
        except Exception as e:
            logger.warning(f"RSS content parsing error: {e}")
            return []
    
    def get_recent_news_async(self, symbol: str) -> List[str]:
        """
        Get recent news for a symbol (non-blocking)
        Returns cached news immediately
        """
        if not self.rss_sources:
            return []
        
        try:
            with self._cache_lock:
                # Check global news cache
                if 'global_news' in self._news_cache:
                    cached_entry = self._news_cache['global_news']
                    
                    # Check cache freshness
                    if time.time() - cached_entry['timestamp'] < self.cache_ttl:
                        all_news = cached_entry['data']
                        
                        # Filter news relevant to symbol
                        symbol_upper = symbol.upper()
                        relevant_news = []
                        
                        for news_item in all_news:
                            text = news_item.get('text', '')
                            if self._is_relevant_news(text, symbol_upper):
                                relevant_news.append(text)
                                
                                if len(relevant_news) >= self.max_items:
                                    break
                        
                        self._stats['cache_hits'] += 1
                        return relevant_news[:self.max_items]
                
                # No fresh cache available
                return []
                
        except Exception as e:
            logger.error(f"News retrieval error for {symbol}: {e}")
            return []
    
    def _is_relevant_news(self, text: str, symbol: str) -> bool:
        """Check if news text is relevant to the symbol or general market"""
        if not text:
            return False
        
        text_upper = text.upper()
        
        # Direct symbol match
        if symbol in text_upper:
            return True
        
        # General financial keywords
        for keyword in self.financial_keywords:
            if keyword.upper() in text_upper:
                return True
        
        return False
    
    def force_refresh_async(self) -> bool:
        """Force immediate refresh of RSS feeds (non-blocking)"""
        if self._fetching:
            return False
        
        # Submit background refresh with graceful error handling  
        try:
            def safe_async_run():
                try:
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(self._fetch_all_feeds_async())
                    finally:
                        try:
                            loop.close()
                        except Exception:
                            pass
                except RuntimeError as e:
                    if "cannot schedule new futures after interpreter shutdown" in str(e):
                        logger.debug("RSS: Interpreter shutdown, skipping background fetch")
                    else:
                        logger.error(f"RSS async error: {e}")
                except Exception as e:
                    logger.error(f"RSS background error: {e}")
            
            self._background_pool.submit(safe_async_run)
            return True
        except Exception as e:
            logger.error(f"Force refresh error: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and statistics"""
        with self._cache_lock:
            cache_info = {}
            if 'global_news' in self._news_cache:
                cache_entry = self._news_cache['global_news']
                cache_info = {
                    'items_cached': cache_entry.get('items_count', 0),
                    'cache_age_seconds': time.time() - cache_entry.get('timestamp', 0),
                    'cache_fresh': time.time() - cache_entry.get('timestamp', 0) < self.cache_ttl
                }
            
            return {
                'rss_sources': self.rss_sources,
                'sources_count': len(self.rss_sources),
                'cache_ttl': self.cache_ttl,
                'max_items': self.max_items,
                'request_timeout': self.request_timeout,
                'currently_fetching': self._fetching,
                'last_fetch_time': self._last_fetch_time,
                'statistics': dict(self._stats),
                'cache_info': cache_info
            }
    
    def cleanup_old_cache(self, max_age_seconds: int = 3600):
        """Clean up old cache entries"""
        cutoff_time = time.time() - max_age_seconds
        
        with self._cache_lock:
            to_remove = []
            for cache_key, cache_entry in self._news_cache.items():
                if cache_entry.get('timestamp', 0) < cutoff_time:
                    to_remove.append(cache_key)
            
            for cache_key in to_remove:
                del self._news_cache[cache_key]
        
        if to_remove:
            logger.debug(f"🧹 Cleaned up {len(to_remove)} old RSS cache entries")


# Singleton instance
_async_rss_provider_instance = None


def get_async_rss_news_provider() -> AsyncRSSNewsProvider:
    """Get singleton async RSS news provider"""
    global _async_rss_provider_instance
    if _async_rss_provider_instance is None:
        _async_rss_provider_instance = AsyncRSSNewsProvider()
    return _async_rss_provider_instance
