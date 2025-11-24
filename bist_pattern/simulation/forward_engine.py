"""
Forward Simulation Engine for Real-time Trading Simulation

This module implements a real-time forward simulation that:
- Starts with current highest confidence signals
- Monitors positions every cycle (10-15 times/day)
- Executes trades based on stop-loss, confidence drops, and sell signals
- Continues until the selected horizon duration
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import json
import os
import logging

logger = logging.getLogger(__name__)

STATE_FILE = 'logs/simulation_state.json'


def _read_state() -> Optional[Dict]:
    """Read simulation state from JSON file."""
    try:
        if not os.path.exists(STATE_FILE):
            return None
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading simulation state: {e}")
        return None


def _write_state(state: Dict) -> None:
    """Write simulation state to JSON file."""
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error writing simulation state: {e}")
        raise


def _delete_state() -> None:
    """Delete simulation state file."""
    try:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
    except Exception as e:
        logger.error(f"Error deleting simulation state: {e}")


def _parse_horizon(horizon: str) -> int:
    """Parse horizon string to days (e.g., '14d' -> 14)."""
    try:
        return int(horizon.replace('d', '').replace('D', ''))
    except Exception:
        return 7


def _get_eligible_horizons(max_days: int) -> List[str]:
    """Get all horizons <= max_days."""
    all_horizons = ['1d', '3d', '7d', '14d', '30d']
    return [h for h in all_horizons if _parse_horizon(h) <= max_days]


def _get_best_signals(horizon: str, topN: int, exclude_symbols: Set[str] = None) -> List[Dict]:
    """
    Get top N symbols with highest confidence from eligible horizons.
    
    Args:
        horizon: Selected horizon (e.g., '14d')
        topN: Number of symbols to return
        exclude_symbols: Symbols to exclude (already held)
    
    Returns:
        List of dicts with keys: symbol, confidence, horizon, action
    """
    from models import PredictionsLog
    from datetime import datetime, timedelta
    
    if exclude_symbols is None:
        exclude_symbols = set()
    
    max_days = _parse_horizon(horizon)
    eligible_horizons = _get_eligible_horizons(max_days)
    
    # Get recent predictions (last 2 hours)
    cutoff = datetime.utcnow() - timedelta(hours=2)
    recent_preds = PredictionsLog.query.filter(
        PredictionsLog.horizon.in_(eligible_horizons),
        PredictionsLog.ts_pred >= cutoff
    ).all()
    
    # For each symbol, keep the highest confidence prediction
    symbol_best = {}
    for pred in recent_preds:
        if pred.symbol in exclude_symbols:
            continue
        
        # Determine action from delta_pred: positive=buy, negative=sell
        delta = float(pred.delta_pred or 0.0)
        action = 'buy' if delta > 0 else 'sell' if delta < 0 else 'hold'
        
        # Convert confidence to float, default to 0.0 if None, clamp to [0, 1]
        conf = float(pred.confidence or 0.0)
        conf = max(0.0, min(1.0, conf))  # Clamp to valid range
        
        if pred.symbol not in symbol_best or conf > symbol_best[pred.symbol]['confidence']:
            symbol_best[pred.symbol] = {
                'symbol': pred.symbol,
                'confidence': conf,
                'horizon': pred.horizon,
                'action': action
            }
    
    # Filter out sell signals and hold
    symbol_best = {k: v for k, v in symbol_best.items() if v['action'] == 'buy'}
    
    # Sort by confidence and return top N
    sorted_symbols = sorted(symbol_best.values(), key=lambda x: -x['confidence'])
    return sorted_symbols[:topN]


def _get_current_price(symbol: str) -> Optional[float]:
    """Get current price from StockPrice (most recent close)."""
    try:
        from models import StockPrice, Stock
        
        # Get stock_id first
        stock = Stock.query.filter_by(symbol=symbol).first()
        if not stock:
            logger.warning(f"Stock not found: {symbol}")
            return None
        
        # Get most recent price
        sp = StockPrice.query.filter_by(stock_id=stock.id).order_by(StockPrice.date.desc()).first()
        if sp and sp.close_price and sp.close_price > 0:
            return float(sp.close_price)
        
        return None
    except Exception as e:
        logger.warning(f"Error getting price for {symbol}: {e}")
        return None


def start_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start a new forward simulation.
    
    Args:
        params: {
            'initial_capital': float,
            'horizon': str (e.g., '14d'),
            'topN': int,
            'commission': float,
            'stop_loss_pct': float,
            'relative_drop_threshold': float
        }
    
    Returns:
        Initial state dict
    """
    # Check for active simulation
    existing = _read_state()
    if existing and existing.get('active'):
        raise ValueError("An active simulation already exists. Stop it first.")
    
    # Parse parameters
    initial_capital = float(params.get('initial_capital', 100000))
    horizon = params.get('horizon', '7d')
    topN = int(params.get('topN', 10))
    commission = float(params.get('commission', 0.0005))
    stop_loss_pct = float(params.get('stop_loss_pct', 0.03))
    relative_drop_threshold = float(params.get('relative_drop_threshold', 0.20))
    
    duration_days = _parse_horizon(horizon)
    
    # Get best signals
    best_signals = _get_best_signals(horizon, topN)
    
    if not best_signals:
        raise ValueError("No buy signals found to start simulation")
    
    # Calculate total confidence for weight distribution
    total_confidence = sum(s['confidence'] for s in best_signals)
    
    # Execute initial buys
    positions = []
    trades = []
    cash = initial_capital
    
    for signal in best_signals:
        symbol = signal['symbol']
        price = _get_current_price(symbol)
        
        if not price or price <= 0:
            logger.warning(f"No valid price for {symbol}, skipping")
            continue
        
        # Calculate position size based on confidence weight
        weight = signal['confidence'] / total_confidence
        allocation = initial_capital * weight
        shares = int(allocation / price)
        
        if shares == 0:
            continue
        
        cost = shares * price
        comm = cost * commission
        total_cost = cost + comm
        
        if total_cost > cash:
            # Adjust shares to fit available cash
            shares = int(cash / (price * (1 + commission)))
            if shares == 0:
                continue
            cost = shares * price
            comm = cost * commission
            total_cost = cost + comm
        
        cash -= total_cost
        
        positions.append({
            'symbol': symbol,
            'shares': shares,
            'entry_price': price,
            'entry_cost': cost,
            'entry_confidence': signal['confidence'],
            'entry_horizon': signal['horizon'],
            'entry_time': datetime.utcnow().isoformat()
        })
        
        trades.append({
            'time': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'action': 'buy',
            'shares': shares,
            'price': price,
            'cost': cost,
            'commission': comm,
            'reason': 'initial_buy'
        })
    
    # Create initial state
    initial_equity = cash + sum(p['entry_cost'] for p in positions)
    
    state = {
        'active': True,
        'start_time': datetime.utcnow().isoformat(),
        'start_date': datetime.utcnow().date().isoformat(),
        'horizon': horizon,
        'duration_days': duration_days,
        'elapsed_days': 0,
        'params': {
            'initial_capital': initial_capital,
            'topN': topN,
            'commission': commission,
            'stop_loss_pct': stop_loss_pct,
            'relative_drop_threshold': relative_drop_threshold
        },
        'portfolio': {
            'cash': cash,
            'equity': initial_equity,
            'pnl': 0.0,
            'positions': positions
        },
        'trades': trades,
        'daily_snapshots': [{
            'date': datetime.utcnow().date().isoformat(),
            'equity': initial_equity,
            'cash': cash,
            'pnl': 0.0
        }]
    }
    
    _write_state(state)
    
    logger.info(f"✅ Forward simulation started: {len(positions)} positions, ₺{initial_equity:.2f} equity")
    
    return state


def check_and_trade() -> Dict[str, Any]:
    """
    Check current positions and execute trades if needed.
    Called by automation cycle.
    
    Returns:
        {
            'active': bool,
            'trades_made': int,
            'positions_count': int,
            'equity': float
        }
    """
    state = _read_state()
    if not state or not state.get('active'):
        return {'active': False}
    
    # Check if simulation duration has ended
    start_date = datetime.fromisoformat(state['start_date']).date()
    current_date = datetime.utcnow().date()
    elapsed_days = (current_date - start_date).days
    state['elapsed_days'] = elapsed_days
    
    if elapsed_days >= state['duration_days']:
        # Simulation ended, stop accepting new buys (but keep positions)
        logger.info(f"⏱️ Simulation duration ended ({state['duration_days']}d), no new buys")
        state['active'] = False
        _write_state(state)
        return {
            'active': False,
            'message': 'Duration ended, simulation stopped',
            'equity': state['portfolio']['equity']
        }
    
    params = state['params']
    horizon = state['horizon']
    positions = state['portfolio']['positions']
    trades = state['trades']
    cash = state['portfolio']['cash']
    
    trades_made = 0
    positions_to_remove = []
    
    # Check each position
    for i, pos in enumerate(positions):
        symbol = pos['symbol']
        current_price = _get_current_price(symbol)
        
        if not current_price or current_price <= 0:
            logger.warning(f"No price for {symbol}, skipping")
            continue
        
        should_sell = False
        sell_reason = ''
        
        # Stop-loss check
        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
        if pnl_pct <= -params['stop_loss_pct']:
            should_sell = True
            sell_reason = 'stop_loss'
        
        # Get new signal for this symbol
        if not should_sell:
            # Find signal for this symbol in recent predictions
            from models import PredictionsLog
            
            eligible_horizons = _get_eligible_horizons(_parse_horizon(horizon))
            recent = PredictionsLog.query.filter(
                PredictionsLog.symbol == symbol,
                PredictionsLog.horizon.in_(eligible_horizons),
                PredictionsLog.ts_pred >= datetime.utcnow() - timedelta(hours=2)
            ).order_by(PredictionsLog.confidence.desc()).first()
            
            if recent:
                # Determine action from delta_pred
                delta = float(recent.delta_pred or 0.0)
                action = 'buy' if delta > 0 else 'sell' if delta < 0 else 'hold'
                
                # Check for sell signal
                if action == 'sell':
                    should_sell = True
                    sell_reason = 'sell_signal'
                # Check for relative confidence drop
                elif float(recent.confidence or 0.0) < pos['entry_confidence'] * (1 - params['relative_drop_threshold']):
                    should_sell = True
                    sell_reason = 'confidence_drop'
        
        # Execute sell
        if should_sell:
            proceeds = pos['shares'] * current_price
            comm = proceeds * params['commission']
            net_proceeds = proceeds - comm
            cash += net_proceeds
            
            profit = net_proceeds - pos['entry_cost']
            
            trades.append({
                'time': datetime.utcnow().isoformat(),
                'symbol': symbol,
                'action': 'sell',
                'shares': pos['shares'],
                'price': current_price,
                'proceeds': proceeds,
                'commission': comm,
                'profit': profit,
                'reason': sell_reason
            })
            
            positions_to_remove.append(i)
            trades_made += 1
            
            logger.info(f"📉 Sold {symbol}: {pos['shares']} @ ₺{current_price:.2f}, P&L: ₺{profit:.2f} ({sell_reason})")
    
    # Remove sold positions
    for i in sorted(positions_to_remove, reverse=True):
        positions.pop(i)
    
    # Try to fill empty slots
    if len(positions) < params['topN']:
        held_symbols = {p['symbol'] for p in positions}
        available_slots = params['topN'] - len(positions)
        
        # Get new candidates
        new_candidates = _get_best_signals(horizon, params['topN'], exclude_symbols=held_symbols)
        
        # Calculate total confidence for weight distribution
        if new_candidates:
            total_confidence = sum(c['confidence'] for c in new_candidates)
            
            for candidate in new_candidates[:available_slots]:
                symbol = candidate['symbol']
                price = _get_current_price(symbol)
                
                if not price or price <= 0:
                    continue
                
                # Calculate position size
                weight = candidate['confidence'] / total_confidence
                allocation = cash * weight
                shares = int(allocation / (price * (1 + params['commission'])))
                
                if shares == 0:
                    continue
                
                cost = shares * price
                comm = cost * params['commission']
                total_cost = cost + comm
                
                if total_cost > cash:
                    shares = int(cash / (price * (1 + params['commission'])))
                    if shares == 0:
                        continue
                    cost = shares * price
                    comm = cost * params['commission']
                    total_cost = cost + comm
                
                cash -= total_cost
                
                positions.append({
                    'symbol': symbol,
                    'shares': shares,
                    'entry_price': price,
                    'entry_cost': cost,
                    'entry_confidence': candidate['confidence'],
                    'entry_horizon': candidate['horizon'],
                    'entry_time': datetime.utcnow().isoformat()
                })
                
                trades.append({
                    'time': datetime.utcnow().isoformat(),
                    'symbol': symbol,
                    'action': 'buy',
                    'shares': shares,
                    'price': price,
                    'cost': cost,
                    'commission': comm,
                    'reason': 'rotation_buy'
                })
                
                trades_made += 1
                
                logger.info(f"📈 Bought {symbol}: {shares} @ ₺{price:.2f}, conf: {candidate['confidence']:.2f}")
    
    # Calculate current equity
    position_value = sum(
        p['shares'] * (_get_current_price(p['symbol']) or p['entry_price'])
        for p in positions
    )
    current_equity = cash + position_value
    pnl = current_equity - params['initial_capital']
    
    # Update state
    state['portfolio']['cash'] = cash
    state['portfolio']['equity'] = current_equity
    state['portfolio']['pnl'] = pnl
    state['portfolio']['positions'] = positions
    state['trades'] = trades
    
    # Update daily snapshot if date changed
    last_snapshot_date = state['daily_snapshots'][-1]['date'] if state['daily_snapshots'] else None
    current_date_str = datetime.utcnow().date().isoformat()
    if last_snapshot_date != current_date_str:
        state['daily_snapshots'].append({
            'date': current_date_str,
            'equity': current_equity,
            'cash': cash,
            'pnl': pnl
        })
    
    _write_state(state)
    
    return {
        'active': True,
        'trades_made': trades_made,
        'positions_count': len(positions),
        'equity': current_equity,
        'pnl': pnl,
        'cash': cash
    }


def stop_simulation() -> Dict[str, Any]:
    """
    Stop the current simulation and return summary.
    
    Returns:
        Summary dict with final equity, P&L, trades, etc.
    """
    state = _read_state()
    if not state:
        raise ValueError("No active simulation found")
    
    # Calculate final metrics
    positions = state['portfolio']['positions']
    trades = state['trades']
    
    # Calculate final equity (mark to market)
    cash = state['portfolio']['cash']
    position_value = sum(
        p['shares'] * (_get_current_price(p['symbol']) or p['entry_price'])
        for p in positions
    )
    final_equity = cash + position_value
    pnl = final_equity - state['params']['initial_capital']
    return_pct = (pnl / state['params']['initial_capital']) * 100
    
    # Calculate hit rate (profitable trades)
    sell_trades = [t for t in trades if t['action'] == 'sell']
    profitable_trades = sum(1 for t in sell_trades if t.get('profit', 0) > 0)
    hit_rate = (profitable_trades / len(sell_trades)) if sell_trades else 0
    
    # Total commission
    total_commission = sum(t.get('commission', 0) for t in trades)
    
    summary = {
        'active': False,
        'start_time': state['start_time'],
        'end_time': datetime.utcnow().isoformat(),
        'duration_days': state['duration_days'],
        'elapsed_days': state['elapsed_days'],
        'summary': {
            'initial_capital': state['params']['initial_capital'],
            'final_equity': final_equity,
            'pnl': pnl,
            'return_pct': return_pct,
            'total_trades': len(trades),
            'profitable_trades': profitable_trades,
            'hit_rate': hit_rate,
            'total_commission': total_commission
        },
        'portfolio': {
            'cash': cash,
            'equity': final_equity,
            'positions': positions
        },
        'trades': trades,
        'daily_snapshots': state['daily_snapshots']
    }
    
    # Delete state file
    _delete_state()
    
    logger.info(f"🏁 Simulation stopped: P&L=₺{pnl:.2f} ({return_pct:.2f}%), {len(trades)} trades, hit-rate={hit_rate:.2%}")
    
    return summary


def get_simulation_status() -> Dict[str, Any]:
    """Get current simulation status."""
    state = _read_state()
    if not state or not state.get('active'):
        return {'active': False}
    
    # Calculate current equity
    positions = state['portfolio']['positions']
    cash = state['portfolio']['cash']
    position_value = sum(
        p['shares'] * (_get_current_price(p['symbol']) or p['entry_price'])
        for p in positions
    )
    current_equity = cash + position_value
    pnl = current_equity - state['params']['initial_capital']
    
    return {
        'active': True,
        'start_time': state['start_time'],
        'horizon': state['horizon'],
        'duration_days': state['duration_days'],
        'elapsed_days': state['elapsed_days'],
        'portfolio': {
            'cash': cash,
            'equity': current_equity,
            'pnl': pnl,
            'positions': positions
        },
        'trades': state['trades'],
        'daily_snapshots': state['daily_snapshots']
    }
