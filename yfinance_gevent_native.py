"""
Gevent-Native Yahoo Finance Non-blocking Wrapper
===============================================

True non-blocking implementation using gevent greenlets and polling.
Prevents any blocking of the main Gevent worker thread.
"""

import time
import random
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import gevent

logger = logging.getLogger(__name__)


class GeventNativeYFinanceWrapper:
    """
    Pure Gevent-native Yahoo Finance wrapper with polling
    """

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yf_native")
        self.session_requests = 0
        self.delay_range = (0.8, 2.0)
        logger.info("🌿 Gevent-native Yahoo Finance Wrapper initialized")

    def _sync_fetch_data(self, symbol: str, yf_symbol: str, period: str) -> Dict[str, Any]:
        """
        Synchronous fetch method for thread execution with curl_cffi support
        """
        import yfinance as yf
        try:
            from bist_pattern.utils.symbols import sanitize_symbol, to_yf_symbol
            symbol_clean = sanitize_symbol(symbol)
            yf_symbol = to_yf_symbol(symbol_clean)
        except Exception:
            pass

        try:
            # Use curl_cffi for better rate limiting bypass with advanced anti-detection
            try:
                import os as _os
                if str(_os.getenv('YF_USE_CFFI', '1')).lower() not in ('1', 'true', 'yes'):
                    raise ImportError('YF_USE_CFFI disabled')
                from curl_cffi import requests
                
                # Advanced browser impersonation with random selection
                browsers = ['chrome110', 'chrome99', 'edge99', 'safari15_5']
                selected_browser = random.choice(browsers)
                
                session = requests.Session(impersonate=selected_browser)  # type: ignore
                
                # Enhanced headers for better stealth
                enhanced_headers = {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'Accept-Language': 'en-US,en;q=0.9,tr;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Cache-Control': 'max-age=0',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'Upgrade-Insecure-Requests': '1',
                    'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"Windows"'
                }
                session.headers.update(enhanced_headers)
                
                # Random delay to mimic human behavior
                if self.session_requests > 0:
                    delay = random.uniform(1.5, 4.0)
                    logger.debug(f"⏳ {symbol}: Anti-detection delay {delay:.2f}s")
                    time.sleep(delay)
                
                ticker = yf.Ticker(yf_symbol, session=session)
                logger.info(f"🚀 {symbol}: Using curl_cffi {selected_browser} impersonation with enhanced stealth")
            except ImportError:
                # Fallback to traditional requests with User-Agent rotation
                import requests
                user_agents_env = os.getenv('YF_USER_AGENTS', '')
                user_agents = [ua.strip() for ua in user_agents_env.split('|') if ua.strip()] if user_agents_env else []

                if user_agents:
                    session = requests.Session()
                    ua = random.choice(user_agents)
                    session.headers.update({
                        'User-Agent': ua,
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                    })
                    ticker = yf.Ticker(yf_symbol, session=session)
                    logger.info(f"🌿 {symbol}: Native thread with User-Agent rotation")
                else:
                    ticker = yf.Ticker(yf_symbol)
                    logger.warning(f"⚠️ {symbol}: Native thread using default ticker")

            # Single period fetch
            logger.info(f"📡 {symbol}: Native Yahoo Finance request (period={period})...")
            start_time = time.time()

            # Use timeout parameter in history() call instead of session attribute
            data = ticker.history(period=period, auto_adjust=True, prepost=False, timeout=10)

            fetch_time = time.time() - start_time

            if data is None or data.empty:
                return {
                    'symbol': symbol,
                    'success': False,
                    'data': None,
                    'error': 'Empty data',
                    'method': 'native_empty'
                }

            logger.info(f"✅ {symbol}: Native fetch SUCCESS! (rows={len(data)}, time={fetch_time:.2f}s)")

            return {
                'symbol': symbol,
                'success': True,
                'data': data,
                'method': 'native_success',
                'fetch_time': fetch_time,
                'rows': len(data)
            }

        except Exception as e:
            logger.error(f"❌ {symbol}: Native fetch error: {e}")
            return {
                'symbol': symbol,
                'success': False,
                'data': None,
                'error': str(e),
                'method': 'native_error'
            }

    def fetch_data_native_async(self, symbol: str, yf_symbol: str, period: str = '2y', timeout: float = 10.0) -> Dict[str, Any]:
        """
        Gevent-native non-blocking fetch with aggressive timeout and verbose polling
        """
        logger.info(f"🌿 {symbol}: Starting native non-blocking fetch (timeout={timeout}s)...")

        # Submit to thread pool
        future = self.executor.submit(self._sync_fetch_data, symbol, yf_symbol, period)
        logger.info(f"🌿 {symbol}: Thread submitted, starting polling...")

        # Poll with gevent.sleep() with verbose logging
        start_time = time.time()
        poll_interval = 1.0  # 1 second polling (less aggressive)
        poll_count = 0

        while not future.done():
            elapsed = time.time() - start_time
            poll_count += 1

            # Log every 5 polls (5 seconds)
            if poll_count % 5 == 0:
                logger.info(f"🔄 {symbol}: Polling #{poll_count}, elapsed={elapsed:.1f}s, future.done()={future.done()}")

            if elapsed >= timeout:
                logger.error(f"⏰ {symbol}: TIMEOUT after {timeout}s (polls: {poll_count})")
                try:
                    future.cancel()  # Try to cancel
                except Exception as e:
                    logger.warning(f"⚠️ {symbol}: Cancel failed: {e}")
                return {
                    'symbol': symbol,
                    'success': False,
                    'data': None,
                    'error': f'Timeout after {timeout}s',
                    'method': 'native_timeout'
                }

            # Yield control to Gevent with less verbose logging
            logger.debug(f"🌿 {symbol}: gevent.sleep({poll_interval}) - yielding control...")
            gevent.sleep(poll_interval)

        # Get result (should be immediate since future.done() is True)
        try:
            result = future.result(timeout=1.0)  # Short timeout since it should be ready

            if result['success']:
                logger.info(f"✅ {symbol}: Native async completed successfully!")
            else:
                logger.warning(f"⚠️ {symbol}: Native async failed")

            return result

        except Exception as e:
            logger.error(f"💥 {symbol}: Native result error: {e}")
            return {
                'symbol': symbol,
                'success': False,
                'data': None,
                'error': str(e),
                'method': 'native_result_error'
            }

    def cleanup(self):
        """Cleanup thread pool"""
        logger.info("🧹 Shutting down Native Yahoo Finance Wrapper...")
        self.executor.shutdown(wait=True)


# Global singleton
_native_yf_wrapper = None


def get_native_yfinance_wrapper() -> GeventNativeYFinanceWrapper:
    """Get singleton native Yahoo Finance wrapper"""
    global _native_yf_wrapper
    if _native_yf_wrapper is None:
        _native_yf_wrapper = GeventNativeYFinanceWrapper()
    return _native_yf_wrapper


def get_yfinance_gevent_native_wrapper() -> GeventNativeYFinanceWrapper:
    """Alias for get_native_yfinance_wrapper (backward compatibility)"""
    return get_native_yfinance_wrapper()
