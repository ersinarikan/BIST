from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import time
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from app import app
from models import db, Stock, StockPrice
from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce yfinance log noise in production
yf_logger = logging.getLogger("yfinance")
yf_logger.setLevel(logging.CRITICAL)


class AdvancedBISTCollector:
    def __init__(self):
        self.session_requests = 0

        # Configurable worker/delay
        self.max_workers = getattr(config['default'], 'COLLECTOR_MAX_WORKERS', 5)
        try:
            dmin, dmax = getattr(config['default'], 'COLLECTOR_DELAY_RANGE', '1,3').split(',')
            self.delay_range = (float(dmin), float(dmax))
        except Exception:
            self.delay_range = (1.0, 3.0)

        # Retry/backoff tuning from config/env
        try:
            self.max_retries = int(getattr(config['default'], 'YF_MAX_RETRIES', 3))
        except Exception:
            self.max_retries = 3
        try:
            self.backoff_base = float(getattr(config['default'], 'YF_BACKOFF_BASE_SECONDS', 1.0))
        except Exception:
            self.backoff_base = 1.0

        # Batch sleep seconds between batches
        try:
            self.batch_sleep_seconds = int(float(getattr(config['default'], 'BATCH_SLEEP_SECONDS', 3)))
        except Exception:
            self.batch_sleep_seconds = 3

        logger.info(f"Collector config → workers={self.max_workers}, delay_range={self.delay_range}, retries={self.max_retries}, backoff_base={self.backoff_base}, batch_sleep={self.batch_sleep_seconds}s")

    def sanitize_symbol(self, symbol: str) -> str:
        """Sembolü normalize et: sadece A-Z0-9 bırak, büyük harfe çevir"""
        if not symbol:
            return ''
        normalized = re.sub(r'[^A-Z0-9]', '', str(symbol).upper())
        return normalized

    def build_yf_session(self, yf_symbol: str) -> requests.Session:
        """Yahoo Finance için session oluştur ve header'ları ayarla"""
        session = requests.Session()
        try:
            user_agents = getattr(config['default'], 'YF_USER_AGENTS', [])
            user_agent = random.choice(user_agents) if user_agents else (
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
            )
        except Exception:
            user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
        session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.8,en-US;q=0.5,en;q=0.3',
            'Connection': 'keep-alive',
            'Referer': f'https://finance.yahoo.com/quote/{yf_symbol}?p={yf_symbol}'
        })
        return session

    def fetch_chart_api(self, yf_symbol: str, range_param: str = '3mo', interval: str = '1d') -> pd.DataFrame:
        """Yahoo chart API'den (query2→query1 fallback) veri çek ve mümkün olduğunca dayanıklı parse et"""
        common_params = f"range={range_param}&interval={interval}&includeAdjustedClose=true"
        endpoints = [
            f"https://query2.finance.yahoo.com/v8/finance/chart/{yf_symbol}?{common_params}",
            f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}?{common_params}",
        ]
        for url in endpoints:
            try:
                sess = self.build_yf_session(yf_symbol)
                resp = sess.get(url, timeout=5)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                result = (data or {}).get('chart', {}).get('result', [])
                if not result:
                    continue
                node = result[0]
                ts = node.get('timestamp') or []
                indicators = (node.get('indicators') or {})
                quotes = indicators.get('quote') or []
                if not ts:
                    continue
                q = quotes[0] if quotes else {}
                open_a = (q.get('open') or [])
                high_a = (q.get('high') or [])
                low_a = (q.get('low') or [])
                close_a = (q.get('close') or [])
                vol_a = (q.get('volume') or [])

                # Close yoksa adjusted close dene
                if not close_a:
                    adjlist = indicators.get('adjclose') or []
                    if adjlist:
                        close_a = adjlist[0].get('adjclose') or []

                n = len(ts)
                def fix_len(arr, fill=None):
                    arr = list(arr or [])
                    if len(arr) < n:
                        arr = arr + [fill] * (n - len(arr))
                    elif len(arr) > n:
                        arr = arr[:n]
                    return arr

                ts = fix_len(ts)
                close_a = fix_len(close_a, None)
                # Eğer O/H/L tamamen boşsa close'a eşitle
                def all_none(arr):
                    return not arr or all(v is None for v in arr)
                if all_none(open_a):
                    open_a = close_a
                if all_none(high_a):
                    high_a = close_a
                if all_none(low_a):
                    low_a = close_a
                if not vol_a:
                    vol_a = [0] * n
                else:
                    vol_a = fix_len(vol_a, 0)

                df = pd.DataFrame({
                    'Date': pd.to_datetime(ts, unit='s'),
                    'Open': open_a,
                    'High': high_a,
                    'Low': low_a,
                    'Close': close_a,
                    'Volume': vol_a,
                })
                # En azından Close mevcut satırları tut
                df = df.dropna(subset=['Close'])
                if not df.empty:
                    return df
            except Exception:
                continue
        return pd.DataFrame()

    def get_bist_symbol(self, symbol: str) -> str:
        return f"{symbol}.IS"

    def collect_single_stock(self, symbol: str, period: str = "3mo"):
        """Tek hisse için veri topla"""
        try:
            original_symbol = symbol
            symbol = self.sanitize_symbol(symbol)
            if not symbol:
                logger.warning(f"Geçersiz/boş sembol atlandı: {original_symbol}")
                return {'symbol': original_symbol, 'success': False, 'records': 0, 'error': 'invalid_symbol'}
            if original_symbol != symbol:
                logger.info(f"Sembol normalize edildi: {original_symbol} -> {symbol}")

            # Optional symbol validation
            if getattr(config['default'], 'VALIDATE_SYMBOLS', False):
                pattern = re.compile(getattr(config['default'], 'ALLOWED_SYMBOL_PATTERN', r'^[A-Z0-9]{3,6}$'))
                if not pattern.match(symbol):
                    logger.warning(f"Geçersiz sembol formatı: {symbol}")
                    return {'symbol': symbol, 'success': False, 'records': 0, 'error': 'invalid_symbol'}

            yf_symbol = self.get_bist_symbol(symbol)

            # Retry + backoff
            max_retries = max(1, int(self.max_retries))
            backoff_base = float(self.backoff_base)
            effective_period = period or getattr(config['default'], 'COLLECTION_PERIOD', '3mo')

            # Minumum geçmiş gün politikasını uygula (örn. 365 gün) + hibrit doğrulama
            try:
                min_history_days = int(getattr(config['default'], 'MIN_HISTORY_DAYS', 365))
                min_trading_days = int(getattr(config['default'], 'MIN_TRADING_DAYS', 240))
                min_span_days = int(getattr(config['default'], 'MIN_SPAN_DAYS', 300))
                backfill_lookback_days = int(getattr(config['default'], 'BACKFILL_LOOKBACK_DAYS', 540))
            except Exception:
                min_history_days = 365
                min_trading_days = 240
                min_span_days = 300
                backfill_lookback_days = 540

            needs_backfill = False
            try:
                with app.app_context():
                    stock_obj = Stock.query.filter_by(symbol=symbol).first()
                    existing_days = 0
                    span_days = 0
                    latest_date = None
                    if stock_obj:
                        from sqlalchemy import func
                        q = StockPrice.query.filter_by(stock_id=stock_obj.id)
                        existing_days = q.count()
                        min_d, max_d = q.with_entities(func.min(StockPrice.date), func.max(StockPrice.date)).first()
                        if min_d and max_d:
                            span_days = (max_d - min_d).days
                            latest_date = max_d
                # Hibrit kriterler: işlem günü, span ve tazelik
                if existing_days < min_trading_days or span_days < min_span_days:
                    needs_backfill = True
                else:
                    try:
                        from datetime import date, timedelta
                        if latest_date and (date.today() - latest_date).days > 5:
                            needs_backfill = True
                    except Exception:
                        pass
                if existing_days < min_history_days:
                    needs_backfill = True
                if needs_backfill:
                    effective_period = '2y'
                    logger.info(f"📈 {symbol}: hibrit backfill (count={existing_days}, span={span_days}). period='{effective_period}'")
            except Exception as _hist_err:
                logger.warning(f"Min history/hibrit backfill kontrolü başarısız ({symbol}): {_hist_err}")

            # 1) Chart-API önce dene (daha hızlı/istikrarlı)
            hist = pd.DataFrame()
            try:
                period_to_range = {
                    '1mo': '1mo', '3mo': '3mo', '6mo': '6mo', '1y': '1y',
                    '2y': '2y', '5y': '5y'
                }
                rng = period_to_range.get(effective_period, '3mo')
                hist = self.fetch_chart_api(yf_symbol, range_param=rng, interval='1d')
            except Exception as fetch_err:
                logger.warning(f"{symbol} chart-api ilk deneme hatası: {fetch_err}")

            # 2) Hâlâ boşsa yfinance denemeleri
            if hist.empty:
                for attempt in range(1, max_retries + 1):
                    try:
                        # Build a fresh session per attempt with rotated User-Agent
                        sess = self.build_yf_session(yf_symbol)
                        stock = yf.Ticker(yf_symbol, session=sess)
                        hist = stock.history(period=effective_period, interval='1d')
                        if not hist.empty:
                            break
                        logger.warning(f"{symbol} veri boş (deneme {attempt}/{max_retries})")
                    except Exception as fetch_err:
                        logger.warning(f"{symbol} veri hatası (deneme {attempt}/{max_retries}): {fetch_err}")
                    time.sleep(backoff_base * (2 ** (attempt - 1)))

            # Fallback 1: Daha uzun dönem dene (6mo)
            if hist.empty and effective_period != '6mo':
                try:
                    sess = self.build_yf_session(yf_symbol)
                    stock = yf.Ticker(yf_symbol, session=sess)
                    hist = stock.history(period='6mo', interval='1d')
                except Exception as fetch_err:
                    logger.warning(f"{symbol} fallback(6mo) hatası: {fetch_err}")

            # Fallback 2: yfinance.download kullanımı (tek sembol)
            if hist.empty:
                try:
                    dl = yf.download(tickers=yf_symbol, period=effective_period, interval='1d',
                                     auto_adjust=False, progress=False, threads=False)
                    if not dl.empty:
                        dl = dl.rename_axis('Date').reset_index()
                        # yfinance.download tek sembolde sütunlar doğrudan OHLCV
                        if all(col in dl.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                            hist = dl
                except Exception as fetch_err:
                    logger.warning(f"{symbol} download fallback hatası: {fetch_err}")

            # Fallback 3: Doğrudan chart API çağrısı
            if hist.empty:
                try:
                    # effective_period'i range'e map et
                    pmap = {'1mo': '1mo', '3mo': '3mo', '6mo': '6mo', '1y': '1y'}
                    rng = pmap.get(effective_period, '3mo')
                    hist = self.fetch_chart_api(yf_symbol, range_param=rng, interval='1d')
                except Exception as fetch_err:
                    logger.warning(f"{symbol} chart-api fallback hatası: {fetch_err}")

            # Fallback 4: Tarih bazlı backfill (gün farkını kesin tamamlamak için)
            if hist.empty or needs_backfill:
                try:
                    from datetime import date, timedelta
                    start_dt = (date.today() - timedelta(days=backfill_lookback_days)).strftime('%Y-%m-%d')
                    dl = yf.download(tickers=yf_symbol, start=start_dt, interval='1d',
                                     auto_adjust=False, progress=False, threads=False)
                    if not dl.empty:
                        dl = dl.rename_axis('Date').reset_index()
                        if all(col in dl.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                            hist = dl
                            logger.info(f"{symbol} tarih-bazlı backfill uygulandı (start={start_dt})")
                except Exception as fetch_err:
                    logger.warning(f"{symbol} tarih-bazlı backfill hatası: {fetch_err}")

            if hist.empty:
                logger.warning(f"Veri yok: {symbol}")
                return {'symbol': symbol, 'success': False, 'records': 0}

            # Yalnızca Date index ise resetle
            if 'Date' not in hist.columns:
                hist.reset_index(inplace=True)

            saved_records = self.save_stock_data(symbol, hist)
            # Progress broadcast (optional)
            try:
                if getattr(config['default'], 'BROADCAST_PROGRESS', True):
                    import requests
                    api = getattr(config['default'], 'BIST_API_URL', 'http://localhost:5000')
                    headers = {}
                    token = getattr(config['default'], 'INTERNAL_API_TOKEN', None)
                    if token:
                        headers['X-Internal-Token'] = token
                    # Localhost fallback
                    try:
                        requests.post(f"{api}/api/internal/broadcast-log", json={
                            'level': 'INFO',
                            'message': f"✅ {symbol}: {saved_records} kayıt",
                            'category': 'collector'
                        }, headers=headers, timeout=2)
                    except Exception:
                        requests.post("http://127.0.0.1:5000/api/internal/broadcast-log", json={
                            'level': 'INFO',
                            'message': f"✅ {symbol}: {saved_records} kayıt",
                            'category': 'collector'
                        }, timeout=2)
            except Exception:
                pass

            # Random polite delay
            time.sleep(random.uniform(*self.delay_range))

            return {'symbol': symbol, 'success': True, 'records': saved_records}

        except Exception as e:
            logger.error(f"Hata {symbol}: {e}")
            return {'symbol': symbol, 'success': False, 'error': str(e), 'records': 0}

    def save_stock_data(self, symbol: str, data: pd.DataFrame) -> int:
        """Veriyi PostgreSQL'e kaydet"""
        try:
            with app.app_context():
                stock_obj = Stock.query.filter_by(symbol=symbol).first()
                if not stock_obj:
                    if getattr(config['default'], 'AUTO_CREATE_STOCKS', True):
                        stock_obj = Stock(symbol=symbol, name=f"{symbol} Hisse Senedi", sector="Unknown")
                        db.session.add(stock_obj)
                        db.session.flush()
                    else:
                        logger.warning(f"AUTO_CREATE_STOCKS kapalı: {symbol} için kayıt yapılmadı")
                        return 0

                saved_count = 0
                updated_count = 0
                today = datetime.now().date()
                for _, row in data.iterrows():
                    # Veri doğrulama ve normalizasyon
                    try:
                        o = float(row['Open']) if row['Open'] is not None else None
                        h = float(row['High']) if row['High'] is not None else None
                        l = float(row['Low']) if row['Low'] is not None else None
                        c = float(row['Close']) if row['Close'] is not None else None
                        v = int(row['Volume']) if (row['Volume'] is not None and row['Volume'] == row['Volume']) else 0
                    except Exception:
                        # NaN veya tip hatası varsa bu satırı atla
                        continue

                    # Eğer OHLC tamamen yoksa atla
                    if c is None:
                        continue

                    # Aykırı değerleri engelle (DB numeric(10,4) için < 1e6 olmalı)
                    max_allowed = 10**6 - 1
                    values = [x for x in [o, h, l, c] if x is not None]
                    if any(abs(x) >= max_allowed for x in values):
                        logger.warning(f"Aykırı fiyat atlandı {symbol} {row['Date']}: {values}")
                        continue
                    dp_date = row['Date'].date()
                    existing = StockPrice.query.filter_by(stock_id=stock_obj.id, date=dp_date).first()
                    if not existing:
                        price_record = StockPrice(
                            stock_id=stock_obj.id,
                            date=dp_date,
                            open_price=o if o is not None else c,
                            high_price=h if h is not None else c,
                            low_price=l if l is not None else c,
                            close_price=c,
                            volume=v if v > 0 else 0
                        )
                        db.session.add(price_record)
                        saved_count += 1
                    else:
                        # Gün içi upsert – bugün için mevcut kaydı güncelle
                        if dp_date == today:
                            existing.open_price = o if o is not None else c
                            existing.high_price = h if h is not None else c
                            existing.low_price = l if l is not None else c
                            existing.close_price = c
                            existing.volume = v if v > 0 else 0
                            updated_count += 1

                if saved_count > 0:
                    db.session.commit()

                if updated_count > 0:
                    try:
                        db.session.commit()
                    except Exception:
                        pass

                if updated_count > 0 or saved_count > 0:
                    logger.info(f"💾 {symbol}: {saved_count} yeni, {updated_count} güncellendi")

                return saved_count + updated_count

        except Exception as e:
            logger.error(f"Kaydetme hatası {symbol}: {e}")
            try:
                # Rollback'i app context içinde yap
                from app import app as _app
                with _app.app_context():
                    db.session.rollback()
            except Exception as rollback_error:
                logger.error(f"Rollback hatası {symbol}: {rollback_error}")
            return 0

    def collect_all_stocks_parallel(self, batch_size: int = None):
        """Tüm hisseleri paralel olarak topla"""
        if batch_size is None:
            batch_size = getattr(config['default'], 'COLLECTOR_BATCH_SIZE', 50)

        with app.app_context():
            scope = getattr(config['default'], 'COLLECTION_SCOPE', 'ALL').upper()
            if scope == 'PRIORITY':
                priority = getattr(config['default'], 'PRIORITY_SYMBOLS', [])
                if not priority:
                    priority = [
                        'THYAO','AKBNK','GARAN','EREGL','ASELS','TUPRS','SAHOL','VAKBN',
                        'KOZAL','SISE','TCELL','MGROS','ARCLK','BIMAS','KRDMD','TOASO',
                        'PETKM','SASA','DOHOL','TKFEN','ENKAI','TAVHL','ISCTR','HALKB',
                        'FROTO','OTKAR','PGSUS','DOAS','GUBRF','ULKER','TTKOM','AEFES'
                    ]
                symbols = priority
            elif scope == 'DB_ACTIVE':
                symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).all()]
            else:
                symbols = [s.symbol for s in Stock.query.all()]

        # Normalize + tekilleştir
        normalized_symbols = []
        for s in symbols:
            ns = self.sanitize_symbol(s)
            if ns:
                normalized_symbols.append(ns)
        symbols = sorted(set(normalized_symbols))

        logger.info(f"Toplam {len(symbols)} hisse için veri toplama başlıyor...")

        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]

        total_success = 0
        total_failed = 0
        total_records = 0

        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Batch {batch_num}/{len(batches)} işleniyor ({len(batch)} hisse)...")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self.collect_single_stock, symbol): symbol
                    for symbol in batch
                }

                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result.get('success'):
                            total_success += 1
                            total_records += int(result.get('records', 0))
                            logger.info(f"✅ {symbol}: {result.get('records', 0)} kayıt")
                        else:
                            total_failed += 1
                            logger.warning(f"❌ {symbol}: başarısız")
                    except Exception as e:
                        total_failed += 1
                        logger.error(f"❌ {symbol}: {e}")

            if batch_num < len(batches):
                logger.info(f"Batch {batch_num} tamamlandı. {self.batch_sleep_seconds} saniye bekleniyor...")
                time.sleep(self.batch_sleep_seconds)

        logger.info(f"""
        📊 VERİ TOPLAMA RAPORU:
        ✅ Başarılı: {total_success}
        ❌ Başarısız: {total_failed}
        📈 Toplam kayıt: {total_records}
        """)

        return {
            'success_count': total_success,
            'failed_count': total_failed,
            'total_records': total_records
        }

    def collect_priority_stocks(self):
        """Öncelikli hisseleri topla (hızlı test için)"""
        priority_symbols = getattr(config['default'], 'PRIORITY_SYMBOLS', [])
        if not priority_symbols:
            priority_symbols = [
                'THYAO', 'AKBNK', 'GARAN', 'EREGL', 'ASELS', 'TUPRS', 'SAHOL', 'VAKBN',
                'KOZAL', 'SISE', 'TCELL', 'MGROS', 'ARCLK', 'BIMAS', 'KRDMD', 'TOASO',
                'PETKM', 'SASA', 'DOHOL', 'TKFEN', 'ENKAI', 'TAVHL', 'ISCTR', 'HALKB',
                'FROTO', 'OTKAR', 'PGSUS', 'DOAS', 'GUBRF', 'ULKER', 'TTKOM', 'AEFES'
            ]

        logger.info(f"Öncelikli {len(priority_symbols)} hisse için veri toplama...")

        total_success = 0
        total_failed = 0
        total_records = 0
        period = getattr(config['default'], 'PRIORITY_PERIOD', '6mo')

        for symbol in priority_symbols:
            result = self.collect_single_stock(symbol, period=period)
            if result.get('success'):
                total_success += 1
                total_records += int(result.get('records', 0))
                logger.info(f"✅ {symbol}: {result.get('records', 0)} kayıt")
            else:
                total_failed += 1
                logger.warning(f"❌ {symbol}: başarısız")

        logger.info(f"Öncelikli hisseler tamamlandı: {total_success} başarılı, {total_failed} başarısız, {total_records} kayıt")

        return {
            'success_count': total_success,
            'failed_count': total_failed,
            'total_records': total_records
        }


def main():
    collector = AdvancedBISTCollector()
    logger.info("🚀 BIST Veri Toplama Sistemi Başlatılıyor...")
    result = collector.collect_priority_stocks()
    print(f"""
    📊 SONUÇ:
    ✅ Başarılı: {result['success_count']}
    ❌ Başarısız: {result['failed_count']}
    📈 Toplam Kayıt: {result['total_records']}
    """)


if __name__ == "__main__":
    main()
