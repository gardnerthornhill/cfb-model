"""Gradient boosting margin/total models with time-ordered early stopping.

Margin model is hybrid: a linear ridge prior on the rating diffs (SP+, ridge
rating, Elo) captures the full dynamic range of matchup quality — pure GBMs
compress extreme margins because leaf averages cannot extrapolate — and the
GBM learns corrections on top of that prior's residuals.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from ..config import LGB_PARAMS, NUM_THREADS
from ..features.build import FEATURE_COLS

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


def _fit(X, y, valid=None, n_estimators=None):
    params = dict(LGB_PARAMS, n_jobs=NUM_THREADS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    model = LGBMRegressor(**params)
    if valid is not None:
        model.fit(X, y, eval_X=valid[0], eval_y=valid[1],
                  callbacks=[early_stopping(150, verbose=False), log_evaluation(0)])
    else:
        model.fit(X, y, callbacks=[log_evaluation(0)])
    return model


def temporal_split(frame):
    """Reserve the existing 6% tail, keeping all same-kickoff rows together."""
    frame = frame.sort_values(["start_date", "id"]).reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("Need at least two dated training games")
    cutoff = frame.iloc[min(int(len(frame) * .94), len(frame) - 1)]["start_date"]
    inner = frame[frame["start_date"] < cutoff]
    valid = frame[frame["start_date"] >= cutoff]
    if inner.empty or valid.empty:
        raise ValueError("Training needs separate chronological fit and validation periods")
    return frame, inner, valid


def fit_hybrid(frame):
    """Select iterations on a wholly isolated temporal tail, then refit all data.

    The inner linear prior cannot see validation outcomes. After selecting the
    tree counts, both the prior and boosting components refit on every eligible
    training row. No outer test outcomes enter either stage.
    """
    frame, inner, valid = temporal_split(frame)
    prior_inner = fit_linear_prior(inner, inner["margin"].to_numpy(dtype=np.float64))
    xi = inner[FEATURE_COLS].to_numpy(dtype=np.float32)
    xv = valid[FEATURE_COLS].to_numpy(dtype=np.float32)
    mi = inner["margin"].to_numpy(dtype=np.float32) - apply_linear_prior(prior_inner, inner)
    mv = valid["margin"].to_numpy(dtype=np.float32) - apply_linear_prior(prior_inner, valid)
    selected = (
        _fit(xi, mi, (xv, mv)),
        _fit(xi, inner["total"].to_numpy(dtype=np.float32),
             (xv, valid["total"].to_numpy(dtype=np.float32))),
    )
    iterations = [m.best_iteration_ or LGB_PARAMS["n_estimators"] for m in selected]
    prior = fit_linear_prior(frame, frame["margin"].to_numpy(dtype=np.float64))
    x = frame[FEATURE_COLS].to_numpy(dtype=np.float32)
    models = (
        _fit(x, frame["margin"].to_numpy(dtype=np.float32) - apply_linear_prior(prior, frame),
             n_estimators=iterations[0]),
        _fit(x, frame["total"].to_numpy(dtype=np.float32), n_estimators=iterations[1]),
    )
    metadata = {"training_rows": len(frame), "validation_rows": len(valid),
                "validation_start": valid["start_date"].min().isoformat(),
                "latest_training_kickoff": frame["start_date"].max().isoformat(),
                "selected_iterations": iterations}
    return models, prior, metadata


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
