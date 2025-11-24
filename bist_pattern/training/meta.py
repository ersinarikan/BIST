import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def fit_meta_learner(oof_matrix: np.ndarray, y_true: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(oof_matrix)
    model = Ridge(alpha=1.0)
    model.fit(Xs, y_true)
    return model, scaler


def predict_meta_learner(model: Ridge, scaler: StandardScaler, base_returns: list[float]) -> float:
    X = np.array(base_returns, dtype=float).reshape(1, -1)
    Xs = scaler.transform(X)
    return float(model.predict(Xs)[0])
