"""Gradient boosting margin/total models with time-ordered early stopping.

Margin model is hybrid: a linear ridge prior on the rating diffs (SP+, ridge
rating, Elo) captures the full dynamic range of matchup quality — pure GBMs
compress extreme margins because leaf averages cannot extrapolate — and the
GBM learns corrections on top of that prior's residuals.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from ..config import LGB_PARAMS

PRIOR_COLS = ["sp_prior_diff", "adj_margin_diff", "elo_diff"]
ELO_TO_PTS = 1.0 / 12.0  # ~400 Elo gap ~= 33 pts


def _prior_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[PRIOR_COLS].copy()
    X["elo_diff"] = X["elo_diff"] * ELO_TO_PTS
    X = X.fillna(0.0)  # missing ratings contribute nothing; flags carry the signal
    X.insert(0, "home_adv", 1.0)
    return X.to_numpy(dtype=np.float64)


def fit_linear_prior(df: pd.DataFrame, y: np.ndarray):
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=50.0, fit_intercept=False)
    m.fit(_prior_matrix(df), y)
    return m


def apply_linear_prior(m, df: pd.DataFrame) -> np.ndarray:
    return m.predict(_prior_matrix(df))


def _fit(X, y, valid=None):
    params = dict(LGB_PARAMS)
    model = LGBMRegressor(**params)
    if valid is not None and len(valid[0]) >= 500:
        model.fit(X, y, eval_set=[valid],
                  callbacks=[early_stopping(150, verbose=False), log_evaluation(0)])
    else:
        model.fit(X, y, callbacks=[log_evaluation(0)])
    return model


def fit_pair(X, y_margin, y_total, valid_slice=None):
    """Train margin + total models. valid_slice = tail indices for early stopping."""
    if valid_slice is not None:
        n_valid = X.shape[0] - (valid_slice.start or 0) if isinstance(valid_slice, slice) else len(valid_slice)
    else:
        n_valid = 0
    if n_valid >= 500:
        Xv, ymv = X[valid_slice], y_margin[valid_slice]
        ytv = y_total[valid_slice]
        Xtr = np.delete(X, valid_slice, axis=0)
        mtr = np.delete(y_margin, valid_slice)
        ttr = np.delete(y_total, valid_slice)
        margin_model = _fit(Xtr, mtr, (Xv, ymv))
        total_model = _fit(Xtr, ttr, (Xv, ytv))
    else:
        margin_model = _fit(X, y_margin)
        total_model = _fit(X, y_total)
    return margin_model, total_model


def predict_scores(models, X, win_sigma=17.0, margin_add=None):
    """Returns dict with pred_margin, pred_total, home/away scores, win prob.
    margin_add (linear prior) is added before deriving scores/probabilities."""
    mm, tm = models
    margin = mm.predict(X)
    if margin_add is not None:
        margin = margin + margin_add
    total = np.clip(tm.predict(X), 20.0, 120.0)
    from scipy.stats import norm
    p_home = norm.cdf(margin / win_sigma)
    home = (total + margin) / 2
    away = (total - margin) / 2
    return {
        "pred_margin": margin,
        "pred_total": total,
        "pred_home_score": np.round(home, 1),
        "pred_away_score": np.round(away, 1),
        "p_home_win": p_home,
    }
