import logging


logger = logging.getLogger(__name__)


def run_backtest(backtester, symbol: str, predictor, data, horizons: list[str]) -> dict:
    try:
        results = backtester.backtest_model(
            symbol=symbol,
            model_predictor=predictor,
            historical_data=data,
            horizons=horizons,
        )
        if results.get('status') == 'success':
            return {
                'status': 'success',
                'overall': results.get('overall', {}),
            }
        return {'status': 'failed', 'reason': results.get('reason', 'unknown')}
    except Exception as e:
        logger.warning(f"Backtest error for {symbol}: {e}")
        return {'status': 'failed', 'reason': str(e)}
