import numpy as np
import pandas as pd
import logging


logger = logging.getLogger(__name__)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply robust cleaning to feature dataframe.

    FIXED (2025-10-20):
    - Replace extreme outliers (5 sigma) with NaN
    - Clip to 0.5/99.5 percentiles
    - Replace INF with NaN (not 0!)
    - Forward fill NaN (use previous valid values)
    - For remaining NaN (at start), use MEDIAN (not 0!)
    - This prevents noise from zero-filling
    """
    try:
        df = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        # Step 1: Replace INF with NaN FIRST (before outlier detection)
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logger.warning(f"INF values detected ({inf_count}), replacing with NaN")
            df = df.replace([np.inf, -np.inf], np.nan)

        # Step 2: Outlier detection and percentile clipping
        for col in numeric_columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                continue
            
            # Replace extreme outliers (5 sigma) with NaN
            if df[col].std() > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df.loc[z_scores > 5, col] = np.nan
            
            # Clip to percentiles
            percentile_high = df[col].quantile(0.995)
            percentile_low = df[col].quantile(0.005)
            if not np.isnan(percentile_high) and not np.isnan(percentile_low):
                df[col] = df[col].clip(lower=percentile_low, upper=percentile_high)
            
            # ✅ FIX: Additional aggressive clipping for extremely large values (>1e6)
            # This prevents overflow issues that can occur even after percentile clipping
            if abs(percentile_high) > 1e6 or abs(percentile_low) > 1e6:
                # If percentiles are extremely large, use more conservative clipping
                df[col] = df[col].clip(lower=-1e6, upper=1e6)
                logger.debug(f"Applied aggressive clipping to {col} (percentiles: {percentile_low:.2e}, {percentile_high:.2e})")

        # Step 3: Forward fill (use previous valid values)
        df = df.ffill()
        
        # Step 4: For remaining NaN (at start), use MEDIAN (not 0!)
        # This is critical: using 0 creates false signals!
        for col in numeric_columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                continue
            if df[col].isna().any():
                col_median = df[col].median()
                if pd.notna(col_median):
                    df[col] = df[col].fillna(col_median)
                else:
                    # Last resort: if all NaN, use 0
                    df[col] = df[col].fillna(0)
                    logger.warning(f"Column {col} all NaN, filled with 0")
        
        return df
    except Exception as e:
        logger.error(f"clean_dataframe error: {e}")
        return df
