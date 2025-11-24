import numpy as np
import pandas as pd
import logging


logger = logging.getLogger(__name__)


def variance_and_correlation_filter(df: pd.DataFrame, candidate_cols: list[str], var_thr: float = 0.01, corr_thr: float = 0.90) -> list[str]:
    """Reduce features by variance and correlation filters.

    Returns a reduced list of columns.
    """
    try:
        if not candidate_cols:
            return []
        variances = df[candidate_cols].var()
        high_var = variances[variances > var_thr].index.tolist()
        if not high_var:
            high_var = candidate_cols

        corr_matrix = df[high_var].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [c for c in upper_triangle.columns if any(upper_triangle[c] > corr_thr)]
        reduced = [c for c in high_var if c not in to_drop]
        return reduced
    except Exception as e:
        logger.warning(f"variance/correlation filter error: {e}")
        return candidate_cols
