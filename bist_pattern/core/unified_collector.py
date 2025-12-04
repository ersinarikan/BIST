"""
Unified BIST Data Collector - minimal, clean
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
import random
from typing import Dict, Any, Optional
import threading

import os
import pandas as pd
import yfinance as yf

try:  # pragma: no cover
    from curl_cffi import requests as cffi_requests  # type: ignore
except Exception:  # pragma: no cover
    cffi_requests = None  # type: ignore

from config import config
from bist_pattern.utils.symbols import sanitize_symbol as _sanitize, to_yf_symbol as _to_yf
from models import db, Stock, StockPrice


logger = logging.getLogger(__name__)
DEBUG_VERBOSE = str(os.getenv('DEBUG_VERBOSE', '0')).lower() in ('1', 'true', 'yes')

# Hard safety rails for DB persistence to avoid numeric overflows
try:
    COLLECTOR_MAX_PRICE = float(os.getenv('COLLECTOR_MAX_PRICE', '999999.0'))  # numeric(10,4) => abs(value) < 1e6
except Exception as e:
    logger.debug(f"Failed to get COLLECTOR_MAX_PRICE, using 999999.0: {e}")
    COLLECTOR_MAX_PRICE = 999999.0
try:
    COLLECTOR_MIN_PRICE = float(os.getenv('COLLECTOR_MIN_PRICE', '0.0001'))
except Exception as e:
    logger.debug(f"Failed to get COLLECTOR_MIN_PRICE, using 0.0001: {e}")
    COLLECTOR_MIN_PRICE = 0.0001

# Lightweight in-process caches to reduce repeated API calls within a short window
try:
    FETCH_CACHE_TTL = int(os.getenv('COLLECTOR_FETCH_CACHE_TTL', '300'))  # seconds
except Exception as e:
    logger.debug(f"Failed to get COLLECTOR_FETCH_CACHE_TTL, using 300: {e}")
    FETCH_CACHE_TTL = 300
try:
    NO_DATA_TTL = int(os.getenv('COLLECTOR_NO_DATA_TTL_SECONDS', '600'))  # seconds
except Exception as e:
    logger.debug(f"Failed to get COLLECTOR_NO_DATA_TTL_SECONDS, using 600: {e}")
    NO_DATA_TTL = 600

# Thread-safe caches with locks
_fetch_cache: Dict[tuple[str, str], Dict[str, Any]] = {}
_no_data_until: Dict[str, float] = {}
_cache_lock = threading.RLock()  # Protect both caches


def _ddebug(msg: str) -> None:
    try:
        if DEBUG_VERBOSE:
            logger.debug(msg)
    except Exception as e:
        # Silently ignore debug logging failures
        pass


# --- Sanitization helpers (prices/volume) ---
def _isfinite(v: float) -> bool:
    try:
        return v == v and v not in (float('inf'), float('-inf'))
    except Exception as e:
        logger.debug(f"Failed to check if value is finite: {e}")
        return False


def _sanitize_prices_and_volume(row: pd.Series) -> Optional[dict]:
    """Return sanitized OHLCV or None to skip row."""
    try:
        open_v = float(row['Open'])
        high_v = float(row['High'])
        low_v = float(row['Low'])
        close_v = float(row['Close'])
        vol_v = int(row['Volume'])
    except Exception as e:
        logger.debug(f"Failed to extract OHLCV from row: {e}")
        return None

    # Basic finite checks
    for v in (open_v, high_v, low_v, close_v):
        if not _isfinite(v):
            return None

    # Range checks to prevent DB numeric(10,4) overflow
    max_p = COLLECTOR_MAX_PRICE
    min_p = COLLECTOR_MIN_PRICE
    if (
        open_v >= max_p or high_v >= max_p or low_v >= max_p or close_v >= max_p or
        open_v < min_p or high_v < min_p or low_v < min_p or close_v < min_p
    ):
        return None

    # Logical OHLC relationships
    try:
        if not (low_v <= min(open_v, close_v) <= high_v and low_v <= max(open_v, close_v) <= high_v):
            return None
        if low_v > high_v:
            return None
    except Exception as e:
        logger.debug(f"Failed to validate OHLC relationships: {e}")
        return None

    # Volume sanity (non-negative)
    if vol_v < 0:
        vol_v = 0

    return {
        'open': float(open_v),
        'high': float(high_v),
        'low': float(low_v),
        'close': float(close_v),
        'volume': int(vol_v),
    }


# silence yfinance noise
try:  # pragma: no cover
    yfl = logging.getLogger('yfinance')
    yfl.propagate = False
    yfl.setLevel(logging.CRITICAL)
except Exception as e:
    logger.debug(f"Failed to silence yfinance logger: {e}")


def _broadcast(level: str, message: str, category: str = 'collector') -> None:
    """Best-effort socket/log broadcast if app context supports it."""
    try:
        from flask import current_app
        app_obj = current_app._get_current_object()  # type: ignore[attr-defined]
        if hasattr(app_obj, 'broadcast_log'):
            # ✅ FIX: Add service identifier to distinguish from HPO logs
            app_obj.broadcast_log(level, message, category='working_automation', service='working_automation')
    except Exception as e:
        # Silently ignore if no app context/socket available
        logger.debug(f"Broadcast failed (no app context): {e}")


class UnifiedDataCollector:
    def __init__(self) -> None:
        self.max_workers = int(getattr(config['default'], 'COLLECTOR_MAX_WORKERS', 2))
        self.batch_sleep = float(getattr(config['default'], 'BATCH_SLEEP_SECONDS', 3))

    def get_bist_symbol(self, symbol: str) -> str:
        return _to_yf(symbol)

    def sanitize_symbol(self, symbol: str) -> str:
        return _sanitize(symbol)

    def _make_ticker(self, yf_symbol: str, use_cffi: bool = True):
        allow_cffi = str(os.getenv('YF_USE_CFFI', '0')).lower() in ('1', 'true', 'yes')
        if use_cffi and allow_cffi and cffi_requests is not None:
            try:
                session = cffi_requests.Session(impersonate='chrome')
                return yf.Ticker(yf_symbol, session=session)
            except Exception as e:
                logger.debug(f"Failed to create cffi session, using default: {e}")
        return yf.Ticker(yf_symbol)

    def collect_single_stock(self, symbol: str, period: str = '1mo') -> Dict[str, Any]:
        original = symbol
        symbol = self.sanitize_symbol(symbol)
        if not symbol:
            return {'symbol': original, 'success': False, 'records': 0, 'error': 'invalid_symbol'}

        yf_symbol = self.get_bist_symbol(symbol)
        if not yf_symbol:
            return {'symbol': original, 'success': False, 'records': 0, 'error': 'invalid_symbol'}

        target_period = period
        last_date: Optional[datetime] = None
        try:
            if period is None or str(period).lower() == 'auto':
                stock_obj = Stock.query.filter_by(symbol=symbol).first()
                if stock_obj is not None:
                    from sqlalchemy import desc  # type: ignore
                    last_row = (
                        StockPrice.query
                        .filter_by(stock_id=stock_obj.id)
                        .order_by(desc(StockPrice.date))
                        .first()
                    )
                    if last_row is not None:
                        last_date = getattr(last_row, 'date', None)
                if last_date is None:
                    target_period = '2y'
                else:
                    try:
                        gap = (datetime.utcnow().date() - last_date).days  # type: ignore[arg-type]
                    except Exception as e:
                        logger.debug(f"Failed to calculate date gap, using 7: {e}")
                        gap = 7
                    if gap <= 1:
                        target_period = '5d'
                    elif gap <= 30:
                        target_period = '1mo'
                    elif gap <= 90:
                        target_period = '3mo'
                    elif gap <= 180:
                        target_period = '6mo'
                    elif gap <= 365:
                        target_period = '1y'
                    else:
                        target_period = '2y'
        except Exception as e:
            logger.debug(f"Failed to determine target_period, using {period}: {e}")
            target_period = period

        def _chart_try(range_param: str) -> Optional[pd.DataFrame]:
            try:
                import requests as _rq  # type: ignore
                import pandas as _pd
                url = (
                    f"https://query2.finance.yahoo.com/v8/finance/chart/{yf_symbol}"
                    f"?range={range_param}&interval=1d&includePrePost=false&events=div%2Csplit"
                )
                headers = {'User-Agent': 'Mozilla/5.0'}
                max_retries = int(os.getenv('YF_MAX_RETRIES', '1'))
                base = float(os.getenv('YF_BACKOFF_BASE', '1.2'))
                sess = _rq.Session()
                sess.headers.update(headers)
                for attempt in range(max_retries):
                    try:
                        _ddebug(f"collector.fetch chart_try symbol={symbol} period={range_param}")
                        _broadcast('INFO', f"collector.fetch chart_try symbol={symbol} period={range_param}")
                        timeout = int(os.getenv('COLLECTOR_HTTP_TIMEOUT', '10'))
                        resp = sess.get(url, timeout=timeout)
                        if resp.status_code != 200:
                            if resp.status_code in (429, 500, 502, 503, 504):
                                wait = min(10.0, base * (2 ** attempt)) + random.uniform(0, 0.25)
                                logger.warning("collector.fetch chart_try backoff symbol=%s period=%s code=%s wait=%.2fs", symbol, range_param, resp.status_code, wait)
                                time.sleep(wait)
                                continue
                            return None
                        data = resp.json()
                        result = (data or {}).get('chart', {}).get('result')
                        if not result:
                            return None
                        res0 = result[0]
                        ts = res0.get('timestamp') or []
                        indicators = res0.get('indicators', {})
                        quote_list = indicators.get('quote') or []
                        if not quote_list:
                            return None
                        quote = quote_list[0]
                        opens = quote.get('open') or []
                        highs = quote.get('high') or []
                        lows = quote.get('low') or []
                        closes = quote.get('close') or []
                        vols = quote.get('volume') or []
                        if not ts or not closes:
                            return None
                        try:
                            idx = _pd.to_datetime(_pd.Series(ts).astype('int64'), unit='s')
                        except Exception as e:
                            logger.debug(f"Failed to parse timestamp as int64, trying fallback: {e}")
                            idx = _pd.to_datetime(ts, unit='s', errors='coerce')
                        df_chart = _pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows, 'Close': closes, 'Volume': vols}, index=idx)
                        df_chart = df_chart.dropna(how='all')
                        if df_chart.empty:
                            # treat as transient empty; retry
                            wait = min(6.0, base * (2 ** attempt)) + random.uniform(0, 0.25)
                            logger.warning("collector.fetch chart_try empty symbol=%s period=%s wait=%.2fs", symbol, range_param, wait)
                            time.sleep(wait)
                            continue
                        rows = getattr(df_chart, 'shape', (0, 0))[0]
                        logger.info("collector.fetch ok symbol=%s period=%s rows=%s source=chart", symbol, range_param, rows)
                        _broadcast('SUCCESS', f"collector.fetch ok symbol={symbol} period={range_param} rows={rows} source=chart")
                        return df_chart
                    except Exception as e:
                        logger.debug(f"Chart fetch attempt {attempt} failed: {e}")
                        wait = min(10.0, base * (2 ** attempt)) + random.uniform(0, 0.25)
                        time.sleep(wait)
                        continue
                return None
            except Exception as e:
                logger.debug(f"Chart fetch failed completely: {e}")
                return None

        def _download_try(p: str) -> Optional[pd.DataFrame]:
            try:
                days = {'5d': 7, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730}.get(p, 180)
                end_d = datetime.utcnow().date()
                start_d = end_d - timedelta(days=int(days))
                max_retries = int(os.getenv('YF_MAX_RETRIES', '1'))
                base = float(os.getenv('YF_BACKOFF_BASE', '1.2'))
                for attempt in range(max_retries):
                    try:
                        _ddebug(f"collector.fetch download_try symbol={symbol} days={days}")
                        _broadcast('INFO', f"collector.fetch download_try symbol={symbol} days={days}")
                        df_dl = yf.download(
                            yf_symbol,
                            start=start_d.isoformat(),
                            end=(end_d + timedelta(days=1)).isoformat(),
                            interval='1d',
                            auto_adjust=True,
                            progress=False,
                        )
                        if df_dl is not None and not df_dl.empty:
                            rows = getattr(df_dl, 'shape', (0, 0))[0]
                            logger.info("collector.fetch ok symbol=%s period=%s rows=%s source=download", symbol, p, rows)
                            _broadcast('SUCCESS', f"collector.fetch ok symbol={symbol} period={p} rows={rows} source=download")
                            return df_dl
                        wait = min(8.0, base * (2 ** attempt)) + random.uniform(0, 0.25)
                        logger.warning("collector.fetch download_try empty symbol=%s period=%s wait=%.2fs", symbol, p, wait)
                        time.sleep(wait)
                    except Exception as e:
                        logger.debug(f"Download fetch attempt {attempt} failed: {e}")
                        wait = min(8.0, base * (2 ** attempt)) + random.uniform(0, 0.25)
                        time.sleep(wait)
                        continue
                return None
            except Exception as e:
                logger.debug(f"Download fetch failed completely: {e}")
                return None

        def _yfinance_try(p: str) -> Optional[pd.DataFrame]:
            try:
                delay = float(os.getenv('YF_RETRY_BASE_DELAY', '1.2'))
            except Exception as e:
                logger.debug(f"Failed to get YF_RETRY_BASE_DELAY, using 1.2: {e}")
                delay = 1.2
            allow_cffi_local = str(os.getenv('YF_USE_CFFI', '0')).lower() in ('1', 'true', 'yes') and (cffi_requests is not None)
            session_order = (('cffi', True),) if allow_cffi_local else tuple()
            session_order = session_order + (('default', False),)
            # Optional: restrict to a single session attempt
            try:
                single = str(os.getenv('YF_SINGLE_SESSION', '')).lower().strip()
                if single in ('cffi', 'default'):
                    session_order = (('cffi', True),) if single == 'cffi' else (('default', False),)
            except Exception as e:
                logger.debug(f"Failed to get YF_SINGLE_SESSION: {e}")
            try:
                yfin_tries = int(os.getenv('YF_YFINANCE_TRIES', '1'))
            except Exception as e:
                logger.debug(f"Failed to get YF_YFINANCE_TRIES, using 1: {e}")
                yfin_tries = 1
            for _ in range(max(1, yfin_tries)):
                for source_name, use_cffi_flag in session_order:
                    try:
                        _ddebug(f"collector.fetch start symbol={symbol} period={p} source={source_name}")
                        _broadcast('INFO', f"collector.fetch start symbol={symbol} period={p} source={source_name}")
                        ticker = self._make_ticker(yf_symbol, use_cffi=use_cffi_flag)
                        df_local = ticker.history(period=p, auto_adjust=True, prepost=False)
                        if df_local is not None and not df_local.empty:
                            rows = getattr(df_local, 'shape', (0, 0))[0]
                            _ddebug(f"collector.fetch ok symbol={symbol} period={p} rows={rows} source={source_name}")
                            _broadcast('SUCCESS', f"collector.fetch ok symbol={symbol} period={p} rows={rows} source={source_name}")
                            return df_local
                        _broadcast('WARNING', f"collector.fetch empty symbol={symbol} period={p} wait={delay:.1f}s source={source_name}")
                        time.sleep(delay + random.uniform(0, 0.25))
                    except Exception as e:
                        logger.debug(f"yfinance fetch attempt failed: {e}")
                        time.sleep(delay + random.uniform(0, 0.25))
                        continue
                delay = min(8.0, delay + 0.6)
            return None

        def _native_try(p: str) -> Optional[pd.DataFrame]:
            try:
                allow_native = str(os.getenv('YF_NATIVE_FALLBACK', '1')).lower() in ('1', 'true', 'yes')
                if not allow_native:
                    return None
                logger.info("collector.fetch native_try symbol=%s period=%s", symbol, p)
                _broadcast('INFO', f"collector.fetch native_try symbol={symbol} period={p}")
                from yfinance_gevent_native import get_native_yfinance_wrapper  # type: ignore
                native = get_native_yfinance_wrapper()
                timeout = float(os.getenv('COLLECTOR_NATIVE_TIMEOUT', '12.0'))
                res = native.fetch_data_native_async(symbol, yf_symbol, period=p, timeout=timeout)
                if isinstance(res, dict) and res.get('success') and res.get('data') is not None:
                    df_native = res['data']
                    if not getattr(df_native, 'empty', True):
                        rows = getattr(df_native, 'shape', (0, 0))[0]
                        logger.info("collector.fetch ok symbol=%s period=%s rows=%s source=native", symbol, p, rows)
                        _broadcast('SUCCESS', f"collector.fetch ok symbol={symbol} period={p} rows={rows} source=native")
                        return df_native
            except Exception as e:
                logger.debug(f"Native fetch failed: {e}")
                return None
            return None

        # Stage order: Chart -> Download -> yfinance -> Native
        df: Optional[pd.DataFrame] = None

        # Circuit breaker: skip if symbol is under no-data cooldown (thread-safe)
        try:
            now_ts = time.time()
            with _cache_lock:
                until = float(_no_data_until.get(symbol) or 0)
                if until and now_ts < until:
                    return {'symbol': symbol, 'success': False, 'records': 0, 'error': 'no_data_cooldown', 'period': target_period}
        except Exception as e:
            logger.debug(f"Failed to check no-data cooldown: {e}")

        # Positive cache: reuse recent successful fetches for the same symbol/period (thread-safe)
        try:
            key = (symbol, target_period)
            with _cache_lock:
                entry = _fetch_cache.get(key)
                if entry and ((time.time()) - float(entry.get('ts', 0))) < FETCH_CACHE_TTL:
                    df_cached = entry.get('df')
                    if df_cached is None or getattr(df_cached, 'empty', True):
                        return {'symbol': symbol, 'success': False, 'records': 0, 'error': 'no_data_cached', 'period': target_period}
                    df = df_cached
        except Exception as e:
            logger.debug(f"Failed to check fetch cache: {e}")
        for stage in ('chart', 'download', 'yfinance', 'native'):
            if stage == 'chart':
                try:
                    allow_alt = str(os.getenv('YF_TRY_ALT_PERIODS', '0')).lower() in ('1', 'true', 'yes')
                except Exception as e:
                    logger.debug(f"Failed to get YF_TRY_ALT_PERIODS, using False: {e}")
                    allow_alt = False
                ranges = (('5d', '1mo', target_period) if allow_alt else (target_period,))
                for rp in ranges:
                    if df is None or getattr(df, 'empty', True):
                        df = _chart_try(rp)
            elif stage == 'download' and (df is None or getattr(df, 'empty', True)):
                df = _download_try(target_period)
            elif stage == 'yfinance' and (df is None or getattr(df, 'empty', True)):
                df = _yfinance_try(target_period)
            elif stage == 'native' and (df is None or getattr(df, 'empty', True)):
                df = _native_try(target_period)
            if df is not None and not getattr(df, 'empty', True):
                break

        if df is None or df.empty:
            try:
                allow_alt = str(os.getenv('YF_TRY_ALT_PERIODS', '0')).lower() in ('1', 'true', 'yes')
            except Exception as e:
                logger.debug(f"Failed to get YF_TRY_ALT_PERIODS, using False: {e}")
                allow_alt = False
            if allow_alt:
                for rp in ('1mo', '3mo', '6mo', '1y', '2y'):
                    df = _chart_try(rp) or _download_try(rp) or _yfinance_try(rp) or _native_try(rp)
                    if df is not None and not getattr(df, 'empty', True):
                        target_period = rp
                        break

        if df is None or df.empty:
            try:
                with _cache_lock:
                    _no_data_until[symbol] = time.time() + float(NO_DATA_TTL)
                    _fetch_cache[(symbol, target_period)] = {'df': None, 'ts': time.time()}
            except Exception as e:
                logger.debug(f"Failed to set no-data cache: {e}")
            return {'symbol': symbol, 'success': False, 'records': 0, 'error': 'no_data'}

        # Ensure today's row via intraday aggregation if needed
        try:
            today_date = datetime.utcnow().date()
            last_idx_date = None
            if not df.empty:
                last_index_val = df.index[-1]
                last_idx_date = last_index_val.date() if hasattr(last_index_val, 'date') else last_index_val  # type: ignore[attr-defined]
            needs_today_row = (last_idx_date is None) or (last_idx_date < today_date)
            if needs_today_row:
                intraday = None
                for interval in ('60m', '30m', '15m'):
                    try:
                        ticker = self._make_ticker(yf_symbol, use_cffi=True)
                        intraday = ticker.history(period='1d', interval=interval, auto_adjust=True, prepost=False)
                        if intraday is not None and not intraday.empty:
                            break
                    except Exception as e:
                        logger.debug(f"Failed to fetch intraday for {interval}: {e}")
                        time.sleep(0.2)
                if intraday is not None and not intraday.empty:
                    try:
                        if getattr(intraday.index, 'tz', None) is not None:
                            intraday.index = intraday.index.tz_convert('UTC').tz_localize(None)  # type: ignore[attr-defined]
                    except Exception as e:
                        logger.debug(f"Failed to convert timezone: {e}")
                    intraday['_date_only'] = intraday.index.date  # type: ignore[attr-defined]
                    today_rows = intraday[intraday['_date_only'] == today_date]
                    if not today_rows.empty:
                        agg_open = float(today_rows['Open'].iloc[0])  # type: ignore[index]
                        agg_close = float(today_rows['Close'].iloc[-1])  # type: ignore[index]
                        agg_high = float(today_rows['High'].max())
                        agg_low = float(today_rows['Low'].min())
                        agg_vol = int(today_rows['Volume'].sum())
                        df_today = pd.DataFrame({'Open': [agg_open], 'High': [agg_high], 'Low': [agg_low], 'Close': [agg_close], 'Volume': [agg_vol]}, index=pd.to_datetime([today_date]))
                        if today_date not in [x.date() if hasattr(x, 'date') else x for x in df.index]:  # type: ignore[attr-defined]
                            df = pd.concat([df, df_today])
                            target_period = 'intraday_agg'
        except Exception as e:
            logger.debug(f"Failed to aggregate intraday data: {e}")

        # Cache successful result (thread-safe)
        try:
            with _cache_lock:
                _fetch_cache[(symbol, target_period)] = {'df': df, 'ts': time.time()}
        except Exception as e:
            logger.debug(f"Failed to cache successful result: {e}")

        # Save with proper transaction management
        try:
            records_added = 0
            records_updated = 0
            
            # Use transaction context properly (avoid nested transactions)
            try:
                stock = Stock.query.filter_by(symbol=symbol).first()
                if stock is None:
                    stock = Stock(symbol=symbol, name=symbol)
                    db.session.add(stock)
                    db.session.flush()  # Get stock.id before proceeding
                
                # Bulk fetch existing records to avoid N+1 queries (vectorized approach)
                # Extract dates efficiently without iterrows
                try:
                    date_list = []
                    for idx in df.index:
                        try:
                            if hasattr(idx, 'date') and callable(getattr(idx, 'date', None)):
                                date_list.append(idx.date())  # type: ignore[attr-defined]
                            elif hasattr(idx, 'to_pydatetime'):
                                date_list.append(idx.to_pydatetime().date())  # type: ignore[attr-defined]
                            else:
                                date_list.append(idx)
                        except Exception as e:
                            logger.debug(f"Failed to extract date from index: {e}")
                            date_list.append(idx)
                except Exception as e:
                    logger.debug(f"Failed to build date_list, using df.index: {e}")
                    date_list = list(df.index)
                
                # Single query to get all existing records
                existing_records = {
                    r.date: r for r in 
                    StockPrice.query.filter_by(stock_id=stock.id).filter(StockPrice.date.in_(date_list)).all()
                }
                
                # Process all rows with vectorized operations (avoid iterrows)
                new_records = []
                now = datetime.now()
                
                skipped_invalid = 0
                for i, (idx, row) in enumerate(df.iterrows()):
                    try:
                        # Handle pandas datetime index properly
                        if hasattr(idx, 'date') and callable(getattr(idx, 'date', None)):
                            date_val = idx.date()  # type: ignore[attr-defined]
                        elif hasattr(idx, 'to_pydatetime'):
                            date_val = idx.to_pydatetime().date()  # type: ignore[attr-defined]
                        else:
                            date_val = idx
                    except Exception as e:
                        logger.debug(f"Failed to extract date_val, using idx: {e}")
                        date_val = idx
                        
                    # Sanitize OHLCV; skip out-of-range or illogical rows
                    sanitized = _sanitize_prices_and_volume(row)
                    if sanitized is None:
                        skipped_invalid += 1
                        if skipped_invalid <= 3:  # avoid noisy logs
                            logger.warning("collector.save skip invalid row symbol=%s date=%s", symbol, date_val)
                            _broadcast('WARNING', f"collector.save skip invalid row symbol={symbol} date={date_val}")
                        continue

                    existing = existing_records.get(date_val)
                    if existing:
                        # Batch update existing records
                        existing.open_price = sanitized['open']
                        existing.high_price = sanitized['high']
                        existing.low_price = sanitized['low']
                        existing.close_price = sanitized['close']
                        existing.volume = sanitized['volume']
                        existing.created_at = now
                        records_updated += 1
                    else:
                        # Prepare for bulk insert
                        new_records.append(StockPrice(
                            stock_id=stock.id,
                            date=date_val,
                            open_price=sanitized['open'],
                            high_price=sanitized['high'],
                            low_price=sanitized['low'],
                            close_price=sanitized['close'],
                            volume=sanitized['volume'],
                            created_at=now
                        ))
                        records_added += 1
                
                # Bulk insert new records
                if new_records:
                    db.session.bulk_save_objects(new_records)
                
                # Commit transaction
                db.session.commit()

                if skipped_invalid:
                    logger.warning("collector.save skipped_invalid symbol=%s count=%s", symbol, skipped_invalid)
                    _broadcast('WARNING', f"collector.save skipped_invalid symbol={symbol} count={skipped_invalid}")

            except Exception as inner_e:
                # Rollback on error
                try:
                    db.session.rollback()
                except Exception as e:
                    logger.debug(f"Failed to rollback session: {e}")
                raise inner_e
            
            return {'symbol': symbol, 'success': True, 'records': records_added, 'updated': records_updated, 'method': 'yfinance', 'period': target_period}

        except Exception as e:
            logger.error(f"db save failed {symbol}: {e}")
            return {'symbol': symbol, 'success': False, 'records': 0, 'error': f'db_error: {e}'}

    def collect_all_stocks_parallel(self, scope: str = 'ALL') -> Dict[str, Any]:
        """Legacy method - CONTINUOUS_FULL mode removed. Returns error."""
        logger.warning("collect_all_stocks_parallel called but CONTINUOUS_FULL mode removed")
        return {'success': False, 'error': 'CONTINUOUS_FULL mode removed - use SYMBOL_FLOW instead'}


_unified_collector: UnifiedDataCollector | None = None


def get_unified_collector() -> UnifiedDataCollector:
    global _unified_collector
    if _unified_collector is None:
        _unified_collector = UnifiedDataCollector()
    return _unified_collector


def get_data_collector() -> UnifiedDataCollector:
    return get_unified_collector()


def get_advanced_collector() -> UnifiedDataCollector:
    return get_unified_collector()
