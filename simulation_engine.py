"""
Paper Trading Simulation Engine
Real-time AI Model Performance Testing System
"""
import os
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
import yfinance as yf
from sqlalchemy import desc


from models import db, SimulationSession, SimulationTrade, PortfolioSnapshot, Stock, StockPrice
# pattern_detector import'u app.py'dan gelecek

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulationEngine:
    """Paper Trading Simulation Engine"""
    
    def __init__(self):
        self.pattern_detector = None  # App context'te set edilecek
        self.commission_rate = Decimal('0.002')  # %0.2 komisyon
        self.min_trade_amount = Decimal('10.0')  # Minimum 10 TL işlem
        self.fixed_trade_amount = Decimal('10000.0')  # Parametrik trade amount
        
    def create_session(self, user_id: int, initial_balance: float = 100000.0, 
                      duration_hours: int = 48, session_name: str = "AI Performance Test") -> SimulationSession:
        """Yeni simulation session oluştur"""
        try:
            session = SimulationSession(
                user_id=user_id,
                session_name=session_name,
                initial_balance=Decimal(str(initial_balance)),
                current_balance=Decimal(str(initial_balance)),
                duration_hours=duration_hours,
                status='active'
            )
            
            db.session.add(session)
            db.session.commit()
            
            # İlk portfolio snapshot
            self._create_portfolio_snapshot(session)
            
            logger.info(f"✅ Yeni simulation session oluşturuldu: {session.id}")
            return session
            
        except Exception as e:
            logger.error(f"❌ Session oluşturma hatası: {e}")
            db.session.rollback()
            raise
    
    def process_signal(self, session_id: int, symbol: str, signal_data: Dict) -> Optional[SimulationTrade]:
        """Gelen sinyali işle ve gerekirse trade yap"""
        try:
            session = SimulationSession.query.get(session_id)
            if not session or session.status != 'active':
                return None
            
            # Session süresi dolmuş mu kontrol et
            if self._is_session_expired(session):
                self._complete_session(session)
                return None
            
            # Signal confidence kontrolü (minimum %60)
            overall_signal = signal_data.get('overall_signal', {})
            confidence = overall_signal.get('confidence', 0)
            signal_type = overall_signal.get('signal', 'NEUTRAL')
            
            # Minimum güven eşiği (env ile ayarlanabilir)
            try:
                min_conf = float(os.getenv('MIN_SIGNAL_CONFIDENCE', '0.6'))
            except Exception:
                min_conf = 0.6
            if confidence < min_conf:  # altındaki sinyalleri ignore et
                logger.info(f"📊 Düşük confidence signal ignore edildi: {symbol} - {confidence:.2%}")
                return None
            
            # Stock bilgisini al
            stock = Stock.query.filter_by(symbol=symbol).first()
            if not stock:
                logger.warning(f"⚠️ Stock bulunamadı: {symbol}")
                return None
            
            # Güncel fiyatı al
            current_price = self._get_current_price(symbol)
            if not current_price:
                logger.warning(f"⚠️ Fiyat alınamadı: {symbol}")
                return None
            
            trade = None
            
            if signal_type == 'BULLISH':
                trade = self._execute_buy_signal(session, stock, current_price, signal_data)
            elif signal_type == 'BEARISH':
                trade = self._execute_sell_signal(session, stock, current_price, signal_data)
            
            if trade:
                # Portfolio snapshot güncelle
                self._create_portfolio_snapshot(session)
                logger.info(f"📈 Trade executed: {trade.trade_type} {trade.quantity}x{symbol} @ {current_price}")
            
            return trade
            
        except Exception as e:
            logger.error(f"❌ Signal processing hatası: {e}")
            db.session.rollback()
            return None
    
    def _execute_buy_signal(self, session: SimulationSession, stock: Stock, 
                          price: Decimal, signal_data: Dict) -> Optional[SimulationTrade]:
        """Buy signal'i işle"""
        try:
            # Sabit trade amount (parametrik değer)
            trade_amount = self.fixed_trade_amount  # Her alım için sabit miktar
            
            if trade_amount < self.min_trade_amount:
                logger.info(f"💰 Trade amount çok küçük: {trade_amount} < {self.min_trade_amount}")
                return None
            
            # Komisyon hesapla
            commission = trade_amount * self.commission_rate
            net_amount = trade_amount - commission
            # Fractional shares için float quantity kullan (0.5 hisse vs.)
            quantity_float = net_amount / price
            
            if quantity_float <= 0:
                logger.info(f"💰 Quantity too small: {quantity_float}")
                return None
            
            # Trade için decimal precision kullan
            quantity = Decimal(str(round(float(quantity_float), 4)))
            
            actual_cost = (quantity * price) + commission
            
            # UNLIMITED TRADING - Bakiye kontrolü kaldırıldı
            # Sadece total investment tracking için
            
            # Trade oluştur
            trade = SimulationTrade(
                session_id=session.id,
                stock_id=stock.id,
                trade_type='BUY',
                quantity=quantity,
                price=price,
                total_amount=actual_cost,
                signal_source=signal_data.get('overall_signal', {}).get('signals', [{}])[0].get('source', 'UNKNOWN'),
                signal_confidence=Decimal(str(signal_data.get('overall_signal', {}).get('confidence', 0))),
                pattern_detected=self._extract_pattern_name(signal_data),
                status='executed'
            )
            
            # Total investment tracking (unlimited budget)
            session.total_trades += 1
            logger.info(f"💰 {stock.symbol} alındı: {quantity} adet @ {price} = {actual_cost} TL")
            
            db.session.add(trade)
            db.session.commit()
            
            # WebSocket broadcast for real-time updates
            try:
                self._broadcast_trade_update(trade, stock)
            except Exception as e:
                logger.warning(f"WebSocket broadcast hatası: {e}")
            
            return trade
            
        except Exception as e:
            logger.error(f"❌ Buy signal execution hatası: {e}")
            db.session.rollback()
            return None
    
    def _execute_sell_signal(self, session: SimulationSession, stock: Stock, 
                           price: Decimal, signal_data: Dict) -> Optional[SimulationTrade]:
        """Sell signal'i işle - mevcut pozisyonu sat"""
        try:
            # Bu stocktan açık pozisyon var mı?
            open_positions = SimulationTrade.query.filter_by(
                session_id=session.id,
                stock_id=stock.id,
                trade_type='BUY'
            ).all()
            
            if not open_positions:
                logger.info(f"📊 Satılacak pozisyon bulunamadı: {stock.symbol}")
                return None
            
            # TÜM pozisyonları sat (istediğiniz gibi)
            total_quantity = sum(pos.quantity for pos in open_positions)
            logger.info(f"💰 {stock.symbol} için {len(open_positions)} pozisyon satılıyor, toplam: {total_quantity}")
            
            # Ortalama alış fiyatını hesapla
            total_cost = sum(float(pos.total_amount) for pos in open_positions)
            # avg_buy_price hesaplandı ancak kullanılmıyor; yalnızca P/L hesaplaması yapılıyor
            
            quantity = total_quantity
            
            # Komisyon hesapla
            gross_amount = quantity * price
            commission = gross_amount * self.commission_rate
            net_amount = gross_amount - commission
            
            # Trade oluştur
            trade = SimulationTrade(
                session_id=session.id,
                stock_id=stock.id,
                trade_type='SELL',
                quantity=quantity,
                price=price,
                total_amount=net_amount,
                signal_source=signal_data.get('overall_signal', {}).get('signals', [{}])[0].get('source', 'UNKNOWN'),
                signal_confidence=Decimal(str(signal_data.get('overall_signal', {}).get('confidence', 0))),
                pattern_detected=self._extract_pattern_name(signal_data),
                status='executed'
            )
            
            # Profit/Loss hesapla (tüm pozisyonlar için toplam)
            total_profit_loss = float(net_amount) - total_cost
            trade.profit_loss = Decimal(str(total_profit_loss))
            
            if total_cost > 0:
                trade.profit_loss_percentage = Decimal(str((total_profit_loss / total_cost) * 100))
            
            # Satılan pozisyonları veritabanından sil (temizlik)
            for pos in open_positions:
                db.session.delete(pos)
            
            # Nakit geliri kaydet (unlimited budget)
            session.total_trades += 1
            logger.info(f"💰 {stock.symbol} satıldı: {quantity} adet @ {price} = {net_amount} TL, P/L: {total_profit_loss:.2f}")
            
            # Win/Loss statistics
            if total_profit_loss > 0:
                session.winning_trades += 1
            else:
                session.losing_trades += 1
            
            db.session.add(trade)
            db.session.commit()
            
            # WebSocket broadcast for real-time updates
            try:
                self._broadcast_trade_update(trade, stock)
            except Exception as e:
                logger.warning(f"WebSocket broadcast hatası: {e}")
            
            return trade
            
        except Exception as e:
            logger.error(f"❌ Sell signal execution hatası: {e}")
            db.session.rollback()
            return None
    
    def _get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Güncel hisse fiyatını al"""
        try:
            # Önce database'den güncel veriyi dene
            stock = Stock.query.filter_by(symbol=symbol).first()
            if stock:
                latest_price = StockPrice.query.filter_by(stock_id=stock.id)\
                    .order_by(desc(StockPrice.date)).first()
                if latest_price:
                    logger.info(f"📊 Database'den fiyat alındı {symbol}: {latest_price.close_price}")
                    return latest_price.close_price
            
            # Yahoo Finance'dan real-time fiyat al - debug log ekle
            try:
                from bist_pattern.utils.symbols import sanitize_symbol, to_yf_symbol
                ticker_symbol = to_yf_symbol(sanitize_symbol(symbol))
            except Exception:
                ticker_symbol = f"{symbol}.IS"
            logger.info(f"🔍 Yahoo Finance'dan fiyat aranıyor: {ticker_symbol}")
            
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                price = Decimal(str(data['Close'].iloc[-1]))
                logger.info(f"📈 Yahoo Finance'dan fiyat alındı {symbol}: {price}")
                return price
            else:
                logger.warning(f"⚠️ Yahoo Finance'dan veri gelmedi: {ticker_symbol}")
            
            # Fallback: günlük veri dene
            daily_data = ticker.history(period="1d")
            if not daily_data.empty:
                price = Decimal(str(daily_data['Close'].iloc[-1]))
                logger.info(f"📈 Yahoo Finance günlük veri {symbol}: {price}")
                return price
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Fiyat alma hatası {symbol}: {e}")
            return None
    
    def _extract_pattern_name(self, signal_data: Dict) -> Optional[str]:
        """Signal data'dan pattern adını çıkar"""
        patterns = signal_data.get('patterns', [])
        if patterns:
            return patterns[0].get('pattern', 'UNKNOWN')
        return None
    
    def _broadcast_trade_update(self, trade: SimulationTrade, stock: Stock):
        """WebSocket ile trade güncellemesini broadcast et"""
        try:
            # App context'ini import et
            from app import socketio
            
            trade_data = {
                'trade_type': trade.trade_type,
                'symbol': stock.symbol,
                'quantity': float(trade.quantity),
                'price': float(trade.price),
                'total_amount': float(trade.total_amount),
                'session_id': trade.session_id,
                'execution_time': trade.execution_time.isoformat(),
                'profit_loss': float(trade.profit_loss) if trade.profit_loss else None
            }
            
            # Admin room'a broadcast et (uygulamada 'admin' kullanılıyor)
            socketio.emit('simulation_trade', trade_data, to='admin')
            logger.info(f"📡 Trade broadcast sent: {trade.trade_type} {stock.symbol}")
            
        except Exception as e:
            logger.error(f"❌ WebSocket broadcast hatası: {e}")
    
    def _create_portfolio_snapshot(self, session: SimulationSession):
        """Portfolio snapshot oluştur"""
        try:
            # Açık pozisyonları hesapla
            open_positions = self._calculate_open_positions_value(session.id)
            total_stocks_value = sum(Decimal(str(pos['current_value'])) for pos in open_positions)
            active_positions_count = len(open_positions)
            
            # Total investment hesapla (alınan tüm hisseler)
            total_investment = self._calculate_total_investment(session.id)
            total_portfolio_value = Decimal(str(total_stocks_value))  # Sadece hisse değeri
            profit_loss = total_portfolio_value - total_investment + self._calculate_total_cash_from_sales(session.id)
            profit_loss_percentage = (profit_loss / total_investment) * Decimal('100') if total_investment > 0 else Decimal('0')
            
            snapshot = PortfolioSnapshot(
                session_id=session.id,
                cash_balance=session.current_balance,
                total_portfolio_value=total_portfolio_value,
                total_stocks_value=Decimal(str(total_stocks_value)),
                total_profit_loss=profit_loss,
                total_profit_loss_percentage=profit_loss_percentage,
                active_positions=active_positions_count
            )
            
            db.session.add(snapshot)
            db.session.commit()
            
        except Exception as e:
            logger.error(f"❌ Portfolio snapshot hatası: {e}")
            db.session.rollback()
    
    def _calculate_open_positions_value(self, session_id: int) -> List[Dict]:
        """Açık pozisyonların güncel değerini hesapla"""
        positions = []
        
        # Tüm buy trade'leri al
        buy_trades = SimulationTrade.query.filter_by(
            session_id=session_id,
            trade_type='BUY'
        ).all()
        
        # Sell trade'leri al
        sell_trades = SimulationTrade.query.filter_by(
            session_id=session_id,
            trade_type='SELL'
        ).all()
        
        # Her stock için net pozisyonu hesapla
        stock_positions = {}
        
        for trade in buy_trades:
            symbol = trade.stock.symbol
            if symbol not in stock_positions:
                stock_positions[symbol] = {'quantity': 0, 'cost_basis': 0, 'stock': trade.stock}
            stock_positions[symbol]['quantity'] += trade.quantity
            stock_positions[symbol]['cost_basis'] += float(trade.total_amount)
        
        for trade in sell_trades:
            symbol = trade.stock.symbol
            if symbol in stock_positions:
                stock_positions[symbol]['quantity'] -= trade.quantity
        
        # Güncel değerleri hesapla
        for symbol, position in stock_positions.items():
            if position['quantity'] > 0:
                current_price = self._get_current_price(symbol)
                if current_price:
                    current_value = position['quantity'] * float(current_price)
                    positions.append({
                        'symbol': symbol,
                        'quantity': position['quantity'],
                        'cost_basis': position['cost_basis'],
                        'current_price': float(current_price),
                        'current_value': current_value,
                        'profit_loss': current_value - position['cost_basis']
                    })
        
        return positions
    
    def _calculate_total_investment(self, session_id: int) -> Decimal:
        """Toplam yatırım miktarı (tüm alımlar)"""
        total_bought = SimulationTrade.query.filter_by(
            session_id=session_id,
            trade_type='BUY'
        ).all()
        return sum((trade.total_amount for trade in total_bought), Decimal('0'))
    
    def _calculate_total_cash_from_sales(self, session_id: int) -> Decimal:
        """Satışlardan gelen toplam nakit"""
        total_sold = SimulationTrade.query.filter_by(
            session_id=session_id,
            trade_type='SELL'
        ).all()
        return sum((trade.total_amount for trade in total_sold), Decimal('0'))
    
    def _is_session_expired(self, session: SimulationSession) -> bool:
        """Session süresi dolmuş mu kontrol et"""
        if session.end_time:
            return True
        
        duration = timedelta(hours=session.duration_hours)
        return datetime.utcnow() > (session.start_time + duration)
    
    def _complete_session(self, session: SimulationSession):
        """Session'ı tamamla"""
        try:
            session.status = 'completed'
            session.end_time = datetime.utcnow()
            
            # Final snapshot
            self._create_portfolio_snapshot(session)
            
            # Session metrics güncelle
            session.current_balance = self._calculate_final_balance(session.id)
            
            db.session.commit()
            logger.info(f"✅ Session tamamlandı: {session.id}")
            
        except Exception as e:
            logger.error(f"❌ Session completion hatası: {e}")
            db.session.rollback()
    
    def _calculate_final_balance(self, session_id: int) -> Decimal:
        """Final balance hesapla (açık pozisyonları likide et)"""
        session = SimulationSession.query.get(session_id)
        open_positions = self._calculate_open_positions_value(session_id)
        
        final_balance = session.current_balance
        for position in open_positions:
            final_balance += Decimal(str(position['current_value']))
        
        return final_balance
    
    def get_session_performance(self, session_id: int) -> Dict:
        """Session performans raporu"""
        try:
            session = SimulationSession.query.get(session_id)
            if not session:
                return {}
            
            # Trades listesi
            trades = SimulationTrade.query.filter_by(session_id=session_id)\
                .order_by(desc(SimulationTrade.execution_time)).all()
            
            # Portfolio snapshots
            snapshots = PortfolioSnapshot.query.filter_by(session_id=session_id)\
                .order_by(desc(PortfolioSnapshot.snapshot_time)).limit(10).all()
            
            # Açık pozisyonlar
            open_positions = self._calculate_open_positions_value(session_id)
            
            return {
                'session': session.to_dict(),
                'trades': [trade.to_dict() for trade in trades],
                'recent_snapshots': [snapshot.to_dict() for snapshot in snapshots],
                'open_positions': open_positions,
                'status': {
                    'is_active': session.status == 'active',
                    'is_expired': self._is_session_expired(session),
                    'runtime_hours': (datetime.utcnow() - session.start_time).total_seconds() / 3600
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Performance report hatası: {e}")
            return {}


# Global instance
_simulation_engine = None


def get_simulation_engine():
    """Singleton pattern for simulation engine"""
    global _simulation_engine
    if _simulation_engine is None:
        _simulation_engine = SimulationEngine()
    return _simulation_engine
