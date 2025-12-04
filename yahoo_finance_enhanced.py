"""
Enhanced Yahoo Finance Data Fetcher with Advanced Rate Limiting Bypass
===================================================================

Bu modül curl_cffi, proxy rotation, session management ve gelişmiş
anti-detection teknikleri kullanarak Yahoo Finance'tan güvenilir veri çeker.
"""

import asyncio
import random
import time
import logging
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

logger = logging.getLogger(__name__)


class EnhancedYahooFinanceWrapper:
    """
    Gelişmiş Yahoo Finance wrapper with advanced anti-detection
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="enhanced_yf")
        self.session_pool = []
        self.session_pool_size = 5
        self.request_count = 0
        self.last_request_time = 0
        
        # Rate limiting settings
        self.min_delay = 2.0
        self.max_delay = 5.0
        self.burst_delay = 12.0
        self.burst_threshold = 3
        
        # User agent pool
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0",
        ]
        
        # Browser versions for curl_cffi
        self.browsers = ['chrome110', 'chrome101', 'chrome100', 'chrome99', 'edge99', 'safari15']
        
        logger.info("🚀 Enhanced Yahoo Finance Wrapper initialized")
        self._init_session_pool()
    
    def _init_session_pool(self):
        """Initialize session pool with different configurations"""
        try:
            from curl_cffi import requests
            
            for i in range(self.session_pool_size):
                browser = random.choice(self.browsers)
                session = requests.Session(impersonate=browser)  # type: ignore
                
                # Randomize headers slightly
                headers = self._get_random_headers()
                session.headers.update(headers)
                
                self.session_pool.append({
                    'session': session,
                    'browser': browser,
                    'last_used': 0,
                    'request_count': 0
                })
                
            logger.info(f"✅ Initialized {len(self.session_pool)} curl_cffi sessions")
            
        except ImportError:
            logger.warning("⚠️ curl_cffi not available, falling back to requests")
            self.session_pool = []
    
    def _get_random_headers(self) -> Dict[str, str]:
        """Generate randomized but realistic headers"""
        return {
            'Accept': random.choice([
                'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            ]),
            'Accept-Language': random.choice([
                'en-US,en;q=0.9',
                'en-US,en;q=0.9,tr;q=0.8',
                'en-GB,en;q=0.9,en-US;q=0.8'
            ]),
            'Accept-Encoding': 'gzip, deflate, br',
            'Cache-Control': random.choice(['no-cache', 'max-age=0', 'no-store']),
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': random.choice(['none', 'same-origin', 'cross-site']),
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'DNT': str(random.choice([0, 1])),
        }
    
    def _get_session(self):
        """Get least recently used session from pool"""
        if not self.session_pool:
            return None
        
        # Sort by last_used time and request_count
        available_sessions = sorted(
            self.session_pool, 
            key=lambda x: (x['last_used'], x['request_count'])
        )
        
        session_info = available_sessions[0]
        session_info['last_used'] = time.time()
        session_info['request_count'] += 1
        
        return session_info
    
    def _apply_rate_limiting(self):
        """Apply intelligent rate limiting"""
        current_time = time.time()
        
        # Calculate delay based on request frequency
        if self.last_request_time > 0:
            time_since_last = current_time - self.last_request_time
            
            # Burst protection
            if self.request_count % self.burst_threshold == 0:
                delay = self.burst_delay
                logger.info(f"🛡️ Burst protection: waiting {delay}s")
            else:
                # Randomized delay with jitter
                base_delay = random.uniform(self.min_delay, self.max_delay)
                # Add extra delay if requests are too frequent
                if time_since_last < 1.0:
                    base_delay *= 1.5
                delay = base_delay
            
            time.sleep(delay)
            logger.debug(f"⏳ Rate limiting delay: {delay:.2f}s")
        
        self.last_request_time = current_time
        self.request_count += 1
    
    def _sync_fetch_with_rotation(self, symbol: str, yf_symbol: str, period: str) -> Dict[str, Any]:
        """Synchronous fetch with session rotation and advanced stealth"""
        import yfinance as yf
        try:
            from bist_pattern.utils.symbols import sanitize_symbol, to_yf_symbol
            symbol_clean = sanitize_symbol(symbol)
            yf_symbol = to_yf_symbol(symbol_clean)
        except Exception as e:
            logger.debug(f"Failed to sanitize/convert symbol {symbol}: {e}")
            yf_symbol = symbol  # Fallback to original
        
        start_time = time.time()
        
        try:
            # Apply rate limiting
            self._apply_rate_limiting()
            
            # Get session from pool
            session_info = self._get_session()
            
            if session_info:
                session = session_info['session']
                browser = session_info['browser']
                logger.info(f"🎭 {symbol}: Using {browser} session (used {session_info['request_count']} times)")
                ticker = yf.Ticker(yf_symbol, session=session)
            else:
                # Fallback to requests with random user agent
                import requests
                session = requests.Session()
                ua = random.choice(self.user_agents)
                session.headers.update({
                    'User-Agent': ua,
                    **self._get_random_headers()
                })
                ticker = yf.Ticker(yf_symbol, session=session)
                logger.info(f"🔄 {symbol}: Using fallback requests session")
            
            # Fetch with multiple period fallbacks
            periods = [period, '1y', '6mo', '3mo', '1mo']
            data = None
            period_used = None
            
            for p in periods:
                try:
                    logger.info(f"📡 {symbol}: Attempting fetch with period={p}")
                    data = ticker.history(period=p, auto_adjust=True, prepost=False, timeout=15)
                    
                    if not data.empty:
                        period_used = p
                        logger.info(f"✅ {symbol}: Success with period={p}, rows={len(data)}")
                        break
                    else:
                        logger.warning(f"⚠️ {symbol}: Empty data for period={p}")
                        
                except Exception as e:
                    logger.error(f"❌ {symbol}: Period {p} failed: {e}")
                    # Add delay between period attempts to prevent rate limiting
                    time.sleep(2.0)
                    continue
            
            if data is None or data.empty:
                return {
                    'symbol': symbol,
                    'success': False,
                    'error': 'No data retrieved for any period',
                    'periods_tried': periods
                }
            
            fetch_time = time.time() - start_time
            
            return {
                'symbol': symbol,
                'success': True,
                'data': data,
                'period_used': period_used,
                'fetch_time': fetch_time,
                'rows': len(data),
                'method': 'enhanced_curl_cffi' if session_info else 'enhanced_requests'
            }
            
        except Exception as e:
            fetch_time = time.time() - start_time
            logger.error(f"💥 {symbol}: Enhanced fetch failed after {fetch_time:.2f}s: {e}")
            return {
                'symbol': symbol,
                'success': False,
                'error': str(e),
                'fetch_time': fetch_time,
                'method': 'enhanced_error'
            }
    
    async def fetch_data_enhanced(self, symbol: str, yf_symbol: str, period: str = '2y', timeout: float = 60.0) -> Dict[str, Any]:
        """Enhanced async fetch with all advanced features"""
        logger.info(f"🚀 {symbol}: Starting enhanced Yahoo Finance fetch...")
        
        try:
            # Submit to thread pool
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                self.executor,
                self._sync_fetch_with_rotation,
                symbol, yf_symbol, period
            )
            
            # Wait with timeout
            result = await asyncio.wait_for(future, timeout=timeout)
            
            if result['success']:
                logger.info(f"🎉 {symbol}: Enhanced fetch completed! Method: {result.get('method')}")
            else:
                logger.warning(f"⚠️ {symbol}: Enhanced fetch failed: {result.get('error')}")
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ {symbol}: Enhanced fetch timeout after {timeout}s")
            return {
                'symbol': symbol,
                'success': False,
                'error': f'Timeout after {timeout}s',
                'method': 'enhanced_timeout'
            }
        except Exception as e:
            logger.error(f"💥 {symbol}: Enhanced fetch error: {e}")
            return {
                'symbol': symbol,
                'success': False,
                'error': str(e),
                'method': 'enhanced_error'
            }
    
    def fetch_data_sync(self, symbol: str, yf_symbol: str, period: str = '2y', timeout: float = 60.0) -> Dict[str, Any]:
        """SYNC version for service context (no asyncio conflicts) - Thread-safe"""
        logger.info(f"🚀 {symbol}: Starting sync Yahoo Finance fetch...")
        
        # Thread-safe: Create isolated session for each call
        try:
            import threading
            thread_id = threading.get_ident()
            logger.info(f"🧵 {symbol}: Thread {thread_id} - Creating isolated session")
            
            # Create fresh session for this thread/symbol
            from curl_cffi import requests
            session = requests.Session(impersonate="chrome")
            
            try:
                import yfinance as yf
                ticker = yf.Ticker(yf_symbol)
                ticker.session = session
                
                # Try to fetch data with thread-isolated session
                df = ticker.history(period=period, auto_adjust=True, prepost=False, timeout=timeout)
                
                if not df.empty:
                    logger.info(f"✅ {symbol}: Thread-safe fetch SUCCESS, rows={len(df)}")
                    return {
                        'symbol': symbol,
                        'success': True,
                        'data': df,
                        'method': 'thread_safe_enhanced',
                        'period_used': period,
                        'thread_id': thread_id
                    }
                else:
                    logger.warning(f"⚠️ {symbol}: Thread-safe fetch empty data")
                    
            finally:
                session.close()
                logger.info(f"🧹 {symbol}: Thread {thread_id} session closed")
                
        except Exception as e:
            logger.error(f"❌ {symbol}: Thread-safe fetch error: {e}")
            
        # Fallback to original method if thread-safe fails
        return self._sync_fetch_with_rotation(symbol, yf_symbol, period)

    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up Enhanced Yahoo Finance Wrapper...")
        
        # Close curl_cffi sessions
        for session_info in self.session_pool:
            try:
                session_info['session'].close()
            except Exception as e:
                logger.warning(f"⚠️ Error closing session: {e}")
        
        self.session_pool.clear()
        self.executor.shutdown(wait=True)


# Global singleton
_enhanced_yf_wrapper = None


def get_enhanced_yahoo_finance_wrapper() -> EnhancedYahooFinanceWrapper:
    """Get singleton enhanced Yahoo Finance wrapper"""
    global _enhanced_yf_wrapper
    if _enhanced_yf_wrapper is None:
        _enhanced_yf_wrapper = EnhancedYahooFinanceWrapper()
    return _enhanced_yf_wrapper


async def fetch_yahoo_data_enhanced(symbol: str, period: str = '2y') -> Optional[pd.DataFrame]:
    """
    Convenience function to fetch Yahoo Finance data with enhanced methods
    
    Args:
        symbol: Stock symbol (e.g., 'THYAO.IS')
        period: Period for data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    wrapper = get_enhanced_yahoo_finance_wrapper()
    result = await wrapper.fetch_data_enhanced(symbol, symbol, period)
    
    if result['success']:
        return result['data']
    else:
        logger.error(f"Failed to fetch {symbol}: {result.get('error')}")
        return None


if __name__ == "__main__":
    # Test script
    async def test_enhanced_fetch():
        symbol = "THYAO.IS"
        logger.info(f"Testing enhanced fetch for {symbol}")
        
        data = await fetch_yahoo_data_enhanced(symbol, '6mo')
        
        if data is not None:
            logger.info(f"✅ Success! Got {len(data)} rows")
            logger.info(f"Latest price: {data['Close'].iloc[-1]:.2f}")
        else:
            logger.error("❌ Failed to fetch data")
    
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_enhanced_fetch())


# NOTE: Removed duplicate definition of get_enhanced_yahoo_finance_wrapper to avoid redefinition
