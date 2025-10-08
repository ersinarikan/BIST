"""
ML Backtesting System
Walk-forward backtesting for ML prediction models

Purpose: Measure real-world model performance on historical data
Metrics: Sharpe ratio, MAPE, hit rate, max drawdown
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MLBacktester:
    """
    Walk-forward backtesting for ML models
    
    Strategy:
    1. Split historical data into train/test windows
    2. Train model on T-365:T-30
    3. Predict on T-29:T
    4. Compare predictions with actual prices
    5. Calculate performance metrics
    """
    
    def __init__(self):
        # Environment-driven configuration
        try:
            self.train_window_days = int(os.getenv('BACKTEST_TRAIN_WINDOW', '365'))
        except Exception:
            self.train_window_days = 365
        
        try:
            self.test_window_days = int(os.getenv('BACKTEST_TEST_WINDOW', '30'))
        except Exception:
            self.test_window_days = 30
        
        try:
            self.min_data_days = int(os.getenv('BACKTEST_MIN_DATA', '400'))
        except Exception:
            self.min_data_days = 400
        
        try:
            self.risk_free_rate = float(os.getenv('BACKTEST_RISK_FREE_RATE', '0.08'))  # 8% (Turkish market)
        except Exception:
            self.risk_free_rate = 0.08
        
        logger.info(f"📊 ML Backtester initialized: train={self.train_window_days}d, test={self.test_window_days}d")
    
    def backtest_model(
        self, 
        symbol: str,
        model_predictor: Any,  # ML system with predict method
        historical_data: pd.DataFrame,
        horizons: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform walk-forward backtesting
        
        Args:
            symbol: Stock symbol
            model_predictor: ML prediction system (must have predict method)
            historical_data: Historical price data
            horizons: Prediction horizons to test (e.g. ['1d', '3d', '7d'])
            
        Returns:
            Backtest results with performance metrics
        """
        if horizons is None:
            horizons = ['1d', '3d', '7d', '14d', '30d']
        
        results = {
            'symbol': symbol,
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'horizons': {},
            'overall': {}
        }
        
        try:
            # Validate data
            if len(historical_data) < self.min_data_days:
                results['status'] = 'insufficient_data'
                results['message'] = f'Need {self.min_data_days} days, got {len(historical_data)}'
                return results
            
            # For each horizon, perform walk-forward test
            all_sharpe = []
            all_mape = []
            all_hit_rate = []
            
            for horizon in horizons:
                horizon_results = self._test_horizon(
                    symbol=symbol,
                    model_predictor=model_predictor,
                    data=historical_data,
                    horizon=horizon
                )
                
                results['horizons'][horizon] = horizon_results
                
                # Collect metrics for overall summary
                if horizon_results.get('sharpe_ratio') is not None:
                    all_sharpe.append(horizon_results['sharpe_ratio'])
                if horizon_results.get('mape') is not None:
                    all_mape.append(horizon_results['mape'])
                if horizon_results.get('hit_rate') is not None:
                    all_hit_rate.append(horizon_results['hit_rate'])
            
            # Overall metrics
            results['overall'] = {
                'avg_sharpe_ratio': float(np.mean(all_sharpe)) if all_sharpe else 0.0,
                'avg_mape': float(np.mean(all_mape)) if all_mape else 0.0,
                'avg_hit_rate': float(np.mean(all_hit_rate)) if all_hit_rate else 0.0,
                'horizons_tested': len(horizons)
            }
            
            # Quality assessment
            avg_sharpe = results['overall']['avg_sharpe_ratio']
            avg_hit_rate = results['overall']['avg_hit_rate']
            
            if avg_sharpe > 1.5 and avg_hit_rate > 0.6:
                results['overall']['quality'] = 'EXCELLENT'
            elif avg_sharpe > 0.8 and avg_hit_rate > 0.55:
                results['overall']['quality'] = 'GOOD'
            elif avg_sharpe > 0.3 and avg_hit_rate > 0.50:
                results['overall']['quality'] = 'ACCEPTABLE'
            else:
                results['overall']['quality'] = 'POOR'
            
            logger.info(
                f"📊 Backtest {symbol}: Sharpe={avg_sharpe:.2f}, "
                f"Hit Rate={avg_hit_rate:.1%}, Quality={results['overall']['quality']}"
            )
            
        except Exception as e:
            logger.error(f"Backtesting error for {symbol}: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def _test_horizon(
        self,
        symbol: str,
        model_predictor: Any,
        data: pd.DataFrame,
        horizon: str
    ) -> Dict[str, Any]:
        """
        Test single prediction horizon with walk-forward
        
        Returns:
            {
                'sharpe_ratio': float,
                'mape': float,
                'hit_rate': float,
                'max_drawdown': float,
                'predictions_count': int
            }
        """
        try:
            # Extract horizon days (e.g. '7d' -> 7)
            horizon_days = int(horizon.replace('d', '').replace('D', ''))
        except Exception:
            horizon_days = 7
        
        predictions = []
        actuals = []
        returns_pred = []
        returns_actual = []
        
        # Walk-forward windows
        total_data = len(data)
        start_idx = self.train_window_days
        
        while start_idx + self.test_window_days + horizon_days < total_data:
            # Prediction point (using all data up to this point for training)
            pred_point = start_idx
            pred_data = data.iloc[:pred_point + 1]
            
            # Actual future point (horizon days ahead)
            future_idx = pred_point + horizon_days
            if future_idx >= total_data:
                break
            
            # Get prediction
            try:
                if hasattr(model_predictor, 'predict_prices'):
                    # Basic ML predictor
                    preds = model_predictor.predict_prices(symbol, pred_data, None)
                    if preds and horizon in preds:
                        pred_obj = preds[horizon]
                        if isinstance(pred_obj, dict):
                            pred_price = pred_obj.get('price', pred_obj.get('prediction'))
                        else:
                            pred_price = pred_obj
                    else:
                        pred_price = None
                elif hasattr(model_predictor, 'predict_enhanced'):
                    # Enhanced ML predictor
                    preds = model_predictor.predict_enhanced(symbol, pred_data)
                    if preds and horizon in preds:
                        pred_obj = preds[horizon]
                        if isinstance(pred_obj, dict):
                            pred_price = pred_obj.get('ensemble_prediction', pred_obj.get('prediction'))
                        else:
                            pred_price = pred_obj
                    else:
                        pred_price = None
                else:
                    pred_price = None
                
                if pred_price is None or not isinstance(pred_price, (int, float)):
                    # Skip if prediction failed
                    start_idx += self.test_window_days
                    continue
                
                # Actual price
                current_price = float(data.iloc[pred_point]['close'])
                actual_price = float(data.iloc[future_idx]['close'])
                
                # Store for metrics
                predictions.append(float(pred_price))
                actuals.append(actual_price)
                
                # Returns for Sharpe ratio
                pred_return = (pred_price - current_price) / current_price
                actual_return = (actual_price - current_price) / current_price
                
                returns_pred.append(pred_return)
                returns_actual.append(actual_return)
                
            except Exception as e:
                logger.warning(f"Prediction error at step {start_idx}: {e}")
            
            # Move window forward
            start_idx += self.test_window_days
        
        # Calculate metrics
        if not predictions or len(predictions) < 3:
            return {
                'sharpe_ratio': 0.0,
                'mape': 0.0,
                'hit_rate': 0.0,
                'max_drawdown': 0.0,
                'predictions_count': 0,
                'message': 'Insufficient predictions'
            }
        
        # MAPE (Mean Absolute Percentage Error)
        mape = self._calculate_mape(actuals, predictions)
        
        # Hit Rate (% correct direction)
        hit_rate = self._calculate_hit_rate(returns_actual, returns_pred)
        
        # Sharpe Ratio
        sharpe = self._calculate_sharpe_ratio(returns_pred, returns_actual)
        
        # Max Drawdown
        max_dd = self._calculate_max_drawdown(returns_actual)
        
        return {
            'sharpe_ratio': float(sharpe),
            'mape': float(mape),
            'hit_rate': float(hit_rate),
            'max_drawdown': float(max_dd),
            'predictions_count': len(predictions)
        }
    
    def _calculate_mape(self, actuals: List[float], predictions: List[float]) -> float:
        """Mean Absolute Percentage Error"""
        try:
            errors = []
            for actual, pred in zip(actuals, predictions):
                if actual != 0:
                    errors.append(abs((actual - pred) / actual))
            
            if not errors:
                return 100.0
            
            return float(np.mean(errors) * 100.0)  # As percentage
        except Exception:
            return 100.0
    
    def _calculate_hit_rate(
        self, 
        actual_returns: List[float], 
        pred_returns: List[float]
    ) -> float:
        """Calculate % of correct directional predictions"""
        try:
            if not actual_returns or len(actual_returns) != len(pred_returns):
                return 0.0
            
            hits = 0
            for actual_ret, pred_ret in zip(actual_returns, pred_returns):
                # Same direction = hit
                if (actual_ret > 0 and pred_ret > 0) or (actual_ret < 0 and pred_ret < 0):
                    hits += 1
            
            return hits / len(actual_returns)
        except Exception:
            return 0.0
    
    def _calculate_sharpe_ratio(
        self, 
        pred_returns: List[float],
        actual_returns: List[float]
    ) -> float:
        """
        Sharpe ratio based on strategy returns
        
        Strategy: Trade based on prediction direction
        """
        try:
            if not pred_returns or not actual_returns:
                return 0.0
            
            # Strategy returns: actual return when prediction agrees
            strategy_returns = []
            for pred_ret, actual_ret in zip(pred_returns, actual_returns):
                # If prediction is bullish (>0), take long position
                # If prediction is bearish (<0), take short position
                if pred_ret > 0:
                    strategy_returns.append(actual_ret)  # Long
                elif pred_ret < 0:
                    strategy_returns.append(-actual_ret)  # Short
                else:
                    strategy_returns.append(0.0)  # No trade
            
            if not strategy_returns:
                return 0.0
            
            # Annualized Sharpe ratio
            mean_return = np.mean(strategy_returns)
            std_return = np.std(strategy_returns)
            
            if std_return == 0:
                return 0.0
            
            # Assuming daily returns, annualize (252 trading days)
            annual_return = mean_return * 252
            annual_std = std_return * np.sqrt(252)
            
            # Sharpe = (Return - RiskFreeRate) / Volatility
            sharpe = (annual_return - self.risk_free_rate) / annual_std
            
            return float(sharpe)
            
        except Exception as e:
            logger.error(f"Sharpe calculation error: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from peak"""
        try:
            if not returns:
                return 0.0
            
            # Cumulative returns
            cum_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cum_returns)
            
            # Drawdown at each point
            drawdown = (cum_returns - running_max) / (1 + running_max)
            
            # Maximum drawdown
            max_dd = np.min(drawdown)
            
            return float(abs(max_dd))
            
        except Exception:
            return 0.0


# Global singleton
_ml_backtester = None


def get_ml_backtester() -> MLBacktester:
    """Get or create ML backtester singleton"""
    global _ml_backtester
    if _ml_backtester is None:
        _ml_backtester = MLBacktester()
    return _ml_backtester
