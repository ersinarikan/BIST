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
from datetime import datetime  # ✅ FIX: Use for parsing pub_date
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
            """Run async coroutine in a new event loop (thread-safe)"""
            try:
                # ✅ FIX: Get current loop or create new one in this thread
                try:
                    # Try to get existing loop in this thread
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is running, we need a new one in a new thread
                        import concurrent.futures
                        
                        def _run_in_isolated_thread(coro):
                            """Run coroutine in a completely isolated thread with new loop"""
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                return new_loop.run_until_complete(coro)
                            finally:
                                try:
                                    pending = asyncio.all_tasks(new_loop)
                                    for task in pending:
                                        task.cancel()
                                    if pending:
                                        new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                                    new_loop.close()
                                    asyncio.set_event_loop(None)
                                except Exception:
                                    pass
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(_run_in_isolated_thread, coro)
                            return future.result(timeout=30)
                    else:
                        return loop.run_until_complete(coro)
                except RuntimeError:
                    # No event loop in this thread, create new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(coro)
                    finally:
                        try:
                            pending = asyncio.all_tasks(loop)
                            for task in pending:
                                task.cancel()
                            if pending:
                                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                            loop.close()
                            asyncio.set_event_loop(None)
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"RSS loop run error: {e}")
                return None
        
        def _run_in_thread_with_loop(coro):
            """Run coroutine in a completely isolated thread with new loop"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                try:
                    loop.close()
                    asyncio.set_event_loop(None)
                except Exception:
                    pass
        
        def background_fetcher():
            """Background thread that periodically fetches RSS feeds"""
            while True:
                try:
                    # Check if it's time to fetch (every 5 minutes)
                    current_time = time.time()
                    if current_time - self._last_fetch_time > 300 and not self._fetching:  # 5 minutes
                        # ✅ FIX: Use ThreadPoolExecutor for thread-safe execution
                        import concurrent.futures
                        
                        def _run_in_isolated_thread():
                            """Run in completely isolated thread with new loop"""
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                return loop.run_until_complete(self._fetch_all_feeds_async())
                            finally:
                                try:
                                    pending = asyncio.all_tasks(loop)
                                    for task in pending:
                                        task.cancel()
                                    if pending:
                                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                                    loop.close()
                                    asyncio.set_event_loop(None)
                                except Exception:
                                    pass
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            executor.submit(_run_in_isolated_thread).result(timeout=60)
                    
                    # Sleep for 30 seconds before next check
                    time.sleep(30)
                    
                except Exception as e:
                    logger.error(f"Background RSS fetcher error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start background thread
        thread = threading.Thread(target=background_fetcher, daemon=True, name="RSSBackgroundFetcher")
        thread.start()
        
        # Immediate first fetch (separate thread/loop)
        def _first_fetch():
            """Run first fetch in isolated thread"""
            import concurrent.futures
            
            def _run_in_isolated_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self._fetch_all_feeds_async())
                finally:
                    try:
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        if pending:
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        loop.close()
                        asyncio.set_event_loop(None)
                    except Exception:
                        pass
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(_run_in_isolated_thread).result(timeout=60)
        self._background_pool.submit(_first_fetch)
    
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
                    pub_date_str = ''
                    pub_timestamp = None
                    if date_elem is not None and date_elem.text:
                        pub_date_str = date_elem.text.strip()
                        # ✅ FIX: Parse pub_date to timestamp for lookback_hours filtering
                        try:
                            # Try parsing common RSS date formats
                            from email.utils import parsedate_to_datetime
                            try:
                                pub_dt = parsedate_to_datetime(pub_date_str)
                                pub_timestamp = pub_dt.timestamp()
                            except Exception:
                                # Fallback: try datetime parsing
                                try:
                                    pub_dt = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                                    pub_timestamp = pub_dt.timestamp()
                                except Exception:
                                    # If parsing fails, use current time (assume recent)
                                    pub_timestamp = time.time()
                        except Exception:
                            # If all parsing fails, use current time (assume recent)
                            pub_timestamp = time.time()
                    else:
                        # No pub_date: assume recent (use current time)
                        pub_timestamp = time.time()
                    
                    # ✅ FIX: Filter by lookback_hours (24 hours by default)
                    # Only include news items within the lookback window
                    if pub_timestamp:
                        age_hours = (time.time() - pub_timestamp) / 3600.0
                        if age_hours > self.lookback_hours:
                            # Skip news older than lookback_hours
                            continue
                    
                    # Combine title and description
                    text = f"{title}. {description}".strip()
                    
                    if text and len(text) > 20:  # Minimum text length
                        news_items.append({
                            'text': text,
                            'title': title,
                            'description': description,
                            'pub_date': pub_date_str,
                            'pub_timestamp': pub_timestamp,  # ✅ FIX: Store timestamp for filtering
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
            logger.debug("📰 RSS sources not configured")
            return []
        
        try:
            with self._cache_lock:
                # Check global news cache
                if 'global_news' in self._news_cache:
                    cached_entry = self._news_cache['global_news']
                    
                    # Check cache freshness
                    if time.time() - cached_entry['timestamp'] < self.cache_ttl:
                        all_news = cached_entry['data']
                        
                        # ✅ STRICT FIX: Filter news relevant to symbol (symbol-specific only)
                        symbol_upper = symbol.upper()
                        relevant_news = []
                        
                        for news_item in all_news:
                            # ✅ FIX: Filter by lookback_hours (24 hours) based on pub_timestamp
                            pub_timestamp = news_item.get('pub_timestamp')
                            if pub_timestamp:
                                age_hours = (time.time() - pub_timestamp) / 3600.0
                                if age_hours > self.lookback_hours:
                                    # Skip news older than lookback_hours
                                    continue
                            
                            text = news_item.get('text', '')
                            if self._is_relevant_news(text, symbol_upper):
                                relevant_news.append(text)
                                
                                if len(relevant_news) >= self.max_items:
                                    break
                        
                        # ✅ FIX: Only return if symbol-specific news found
                        # Don't fall back to general financial news
                        if relevant_news:
                            self._stats['cache_hits'] += 1
                            logger.debug(f"📰 Found {len(relevant_news)} specific news for {symbol}")
                        else:
                            # ✅ DEBUG: Log sample news items to understand why no match (INFO level for visibility)
                            if all_news and len(all_news) > 0:
                                sample_count = min(3, len(all_news))
                                logger.info(f"📰 {symbol}: No specific news found. Cache has {len(all_news)} total news. Sample titles:")
                                for i, news_item in enumerate(all_news[:sample_count], 1):
                                    title = news_item.get('title', news_item.get('text', ''))[:100]
                                    logger.info(f"   {i}. {title}...")
                            else:
                                logger.info(f"📰 {symbol}: No specific news found. Cache is empty.")
                        
                        return relevant_news[:self.max_items]
                
                # No fresh cache available
                return []
                
        except Exception as e:
            logger.error(f"News retrieval error for {symbol}: {e}")
            return []
    
    def _is_relevant_news(self, text: str, symbol: str) -> bool:
        """Check if news text is relevant to the symbol (symbol name or company name)"""
        if not text:
            return False
        
        text_upper = text.upper()
        symbol_upper = symbol.upper()
        
        # ⚠️ SPECIAL HANDLING: Symbols that are common words
        # These need stricter matching to avoid false positives
        AMBIGUOUS_SYMBOLS = {
            'HEDEF': ['HEDEF GYO', 'HEDEF GAYR', 'HEDEF.IS', 'HEDEF HOLD', 'HEDEF HİSSE'],
            'SASA': ['SASA POL', 'SASA.IS', 'SASA ŞİRKET'],
            'MERIT': ['MERIT TUR', 'MERIT.IS', 'MERIT OTEL'],
            'SELEC': ['SELEC.IS', 'SELEC ŞİRKET'],
        }
        
        # If symbol is ambiguous, require company-specific keywords
        if symbol_upper in AMBIGUOUS_SYMBOLS:
            required_keywords = AMBIGUOUS_SYMBOLS[symbol_upper]
            if any(keyword in text_upper for keyword in required_keywords):
                return True
        
        # Check symbol name with word boundaries (more precise)
        symbol_patterns = [
            f" {symbol_upper} ", f"({symbol_upper})", f"[{symbol_upper}]",
            f"{symbol_upper}:", f"{symbol_upper}-", f"{symbol_upper}.",
            f" {symbol_upper} HISSE", f" {symbol_upper} HISSESI",
            f" {symbol_upper} IS", f" {symbol_upper}.IS",
        ]
        if any(pattern in text_upper for pattern in symbol_patterns):
            return True
        
        # ✅ FIX: Check special cases BEFORE database lookup (works even without DB)
        # Special cases for well-known companies (common abbreviations/names)
        special_cases = {
            'THYAO': ['THY', 'TÜRK HAVA YOLLARI', 'TURK HAVA YOLLARI'],
            'GARAN': ['GARANTİ', 'GARANTI BANKASI', 'GARANTI BANK'],
            'AKBNK': ['AKBANK'],
            'ASELS': ['ASELSAN'],  # ✅ FIX: ASELSAN is the company name for ASELS
            'BIMAS': ['BİM', 'BIM'],
            'TUPRS': ['TÜPRAŞ', 'TUPRAS'],
            'PETKM': ['PETKİM', 'PETKIM'],
        }
        if symbol_upper in special_cases:
            special_names = special_cases[symbol_upper]
            for special_name in special_names:
                if special_name and len(special_name) > 2:
                    # Check with flexible word boundaries
                    special_patterns = [
                        f" {special_name} ", f" {special_name},", f" {special_name}.",
                        f" {special_name}:", f" {special_name}-", f"({special_name})",
                        f"[{special_name}]", f"{special_name} ", f" {special_name}"
                    ]
                    if any(pattern in text_upper for pattern in special_patterns):
                        return True
                    # Also check if special_name is a substring (for compound words, minimum 4 chars)
                    if len(special_name) > 4 and special_name in text_upper:
                        return True
        
        # ✅ FIX: Also check company name from database with flexible matching
        try:
            from flask import current_app
            try:
                app_obj = current_app._get_current_object()
                with app_obj.app_context():
                    from models import Stock
                    stock = Stock.query.filter_by(symbol=symbol_upper).first()
                    if stock and stock.name:
                        company_name = stock.name.upper()
                        
                        # Create multiple variants of company name for flexible matching
                        name_variants = []
                        
                        # Full company name
                        name_variants.append(company_name)
                        
                        # Remove common suffixes
                        base_name = company_name
                        for suffix in [' A.Ş.', ' A.S.', ' T.A.Ş.', ' T.A.S.', ' A.O.', ' A.O.', 
                                      ' ŞİRKETİ', ' ŞIRKETI', ' HOLDİNG', ' HOLDING', 
                                      ' SANAYİ', ' SANAYI', ' VE TİCARET', ' VE TICARET']:
                            base_name = base_name.replace(suffix, '').strip()
                        name_variants.append(base_name)
                        
                        # Extract key words (first 2-3 words, usually company name)
                        words = base_name.split()
                        if len(words) >= 2:
                            # First 2 words (e.g., "ASELSAN ELEKTRONİK" from "ASELSAN ELEKTRONİK SANAYİ VE TİCARET")
                            name_variants.append(' '.join(words[:2]))
                        if len(words) >= 1:
                            # First word (e.g., "ASELSAN")
                            name_variants.append(words[0])
                        
                        # Check all variants with flexible word boundaries
                        for variant in name_variants:
                            if variant and len(variant) > 2:
                                # Flexible matching: word boundaries, punctuation, etc.
                                variant_patterns = [
                                    f" {variant} ",  # Space before and after
                                    f" {variant},",  # Space before, comma after
                                    f" {variant}.",  # Space before, period after
                                    f" {variant}:",  # Space before, colon after
                                    f" {variant}-",  # Space before, dash after
                                    f"({variant})",  # In parentheses
                                    f"[{variant}]",  # In brackets
                                ]
                                
                                # Check patterns
                                if any(pattern in text_upper for pattern in variant_patterns):
                                    return True
                                
                                # Check at start/end
                                if text_upper.startswith(variant + " ") or text_upper.endswith(" " + variant):
                                    return True
                                
                                # Also check if variant is a substring (for compound words, minimum 4 chars)
                                if len(variant) > 4 and variant in text_upper:
                                    return True
            except Exception:
                pass
        except Exception:
            pass
        
        return False
    
    def force_refresh_async(self) -> bool:
        """Force immediate refresh of RSS feeds (non-blocking)"""
        if self._fetching:
            return False
        
        # ✅ FIX: Use same thread-safe loop runner as _start_background_fetching
        def _run_in_new_loop(coro):
            """Run async coroutine in a new event loop (thread-safe)"""
            try:
                # ✅ FIX: Get current loop or create new one in this thread
                try:
                    # Try to get existing loop in this thread
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is running, we need a new one in a new thread
                        import concurrent.futures
                        
                        def _run_in_isolated_thread(coro):
                            """Run coroutine in a completely isolated thread with new loop"""
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                return new_loop.run_until_complete(coro)
                            finally:
                                try:
                                    pending = asyncio.all_tasks(new_loop)
                                    for task in pending:
                                        task.cancel()
                                    if pending:
                                        new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                                    new_loop.close()
                                    asyncio.set_event_loop(None)
                                except Exception:
                                    pass
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(_run_in_isolated_thread, coro)
                            return future.result(timeout=30)
                    else:
                        return loop.run_until_complete(coro)
                except RuntimeError:
                    # No event loop in this thread, create new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(coro)
                    finally:
                        try:
                            pending = asyncio.all_tasks(loop)
                            for task in pending:
                                task.cancel()
                            if pending:
                                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                            loop.close()
                            asyncio.set_event_loop(None)
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"RSS refresh loop error: {e}")
                return None
        
        # Submit background refresh with graceful error handling  
        try:
            def safe_async_run():
                try:
                    _run_in_new_loop(self._fetch_all_feeds_async())
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
