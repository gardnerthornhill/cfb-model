"""Walk-forward backtest.

For every (season, week) in the test window, the model is trained ONLY on games
that kicked off before that week. Features are precomputed causally in the
feature builder, so no future information can leak. The GBM is refit every
REFIT_EVERY_WEEKS weeks (and at each season boundary) to bound compute while
keeping the model fresh.
"""
import numpy as np
import pandas as pd

from ..config import DATA_DIR, REFIT_EVERY_WEEKS, TEST_SEASONS
from ..features.build import FEATURE_COLS
from ..models.train import (apply_linear_prior, fit_linear_prior, fit_pair,
                            predict_scores)


def load_training_frame(min_season=2014):
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    df = df[(df["season"] >= min_season)].copy()
    return df.sort_values(["start_date", "id"]).reset_index(drop=True)


def market_lines() -> pd.DataFrame:
    ln = pd.read_parquet(DATA_DIR / "lines_all.parquet")
    ln = ln.explode("lines", ignore_index=True)
    x = pd.json_normalize(ln["lines"])
    ln = pd.concat([ln[["id"]], x], axis=1)
    ln = ln.dropna(subset=["spread"])
    agg = ln.groupby("id").agg(spread_home=("spread", "median"),
                               over_under=("overUnder", "median")).reset_index()
    return agg


def run(test_seasons=None, verbose=True):
    test_seasons = test_seasons or TEST_SEASONS
    df = load_training_frame()
    lines = market_lines()
    df = df.merge(lines, on="id", how="left")

    X_cols = FEATURE_COLS
    preds = []
    last_fit_key = None
    fit_season = None
    models = None

    groups = [(s, w) for s in test_seasons for w in sorted(df.loc[df["season"] == s, "week"].unique())]
    for season, week in groups:
        grp = df[(df["season"] == season) & (df["week"] == week)]
        grp = grp[grp["margin"].notna()]
        if len(grp) == 0:
            continue
        cutoff = grp["start_date"].min()
        train = df[(df["start_date"] < cutoff) & df["margin"].notna()]
        need_fit = (models is None or fit_season != season or
                    (last_fit_key is not None and
                     (season, week) != last_fit_key and
                     _weeks_between(groups, last_fit_key, (season, week)) >= REFIT_EVERY_WEEKS))
        if need_fit:
            Xtr = train[X_cols].to_numpy(dtype=np.float32)
            ym = train["margin"].to_numpy(dtype=np.float32)
            yt = train["total"].to_numpy(dtype=np.float32)
            lin = fit_linear_prior(train, train["margin"].to_numpy(dtype=np.float64))
            ym_resid = ym - apply_linear_prior(lin, train).astype(np.float32)
            tail = slice(int(len(Xtr) * 0.94), None)
            models = (fit_pair(Xtr, ym_resid, yt, valid_slice=tail), lin)
            last_fit_key = (season, week)
            fit_season = season

        X = grp[X_cols].to_numpy(dtype=np.float32)
        out = predict_scores(models[0], X,
                             margin_add=apply_linear_prior(models[1], grp))
        p = pd.DataFrame({
            "id": grp["id"].values, "season": season, "week": week,
            "home_team": grp["home_team"].values, "away_team": grp["away_team"].values,
            "start_date": grp["start_date"].values,
            "actual_margin": grp["margin"].values, "actual_total": grp["total"].values,
            "home_points": grp["home_points"].values, "away_points": grp["away_points"].values,
            "spread_home": grp["spread_home"].values, "market_total": grp["over_under"].values,
            **{k: v for k, v in out.items()},
        })
        preds.append(p)

    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(DATA_DIR / "backtest_preds.parquet", index=False)
    _report(P, verbose)
    return P


def _weeks_between(groups, a, b):
    ia, ib = groups.index(a), groups.index(b)
    return ia - ib if ia > ib else 0


def _report(P: pd.DataFrame, verbose=True):
    P["margin_err"] = (P["pred_margin"] - P["actual_margin"]).abs()
    P["total_err"] = (P["pred_total"] - P["actual_total"]).abs()
    P["actual_home_win"] = (P["actual_margin"] > 0).astype(float)
    P["model_win_hit"] = (P["p_home_win"] > 0.5).astype(float) == P["actual_home_win"]
    P["brier"] = (P["p_home_win"] - P["actual_home_win"]) ** 2

    has_line = P["spread_home"].notna()
    line_margin = -P["spread_home"]
    home_covered = P["actual_margin"] > line_margin
    P["model_cover"] = ((P["pred_margin"] > line_margin) == home_covered).astype(float)
    P["market_cover"] = ((line_margin > 0) == home_covered).astype(float)  # favorite-side rate
    P["line_margin_err"] = (P["actual_margin"] - line_margin).abs()

    F = pd.read_parquet(DATA_DIR / "features.parquet")[["id", "vs_fcs_h", "vs_fcs_a"]]
    P = P.merge(F, on="id", how="left")
    fbs_line = P[has_line & (P["vs_fcs_h"] == 0) & (P["vs_fcs_a"] == 0)]

    def agg(x):
        return pd.Series({
            "games": len(x),
            "margin_MAE": x["margin_err"].mean(),
            "total_MAE": x["total_err"].mean(),
            "win_acc": x["model_win_hit"].mean(),
            "brier": x["brier"].mean(),
        })

    if verbose:
        print("\n=== WALK-FORWARD BACKTEST (out-of-sample) ===")
        print(P.groupby("season").apply(agg, include_groups=False).round(3).to_string())
        print("\n=== OVERALL ===")
        print(agg(P).round(3).to_string())

        if len(fbs_line):
            print("\n=== vs MARKET (FBS-vs-FBS games with posted lines) ===")
            print(f"games: {len(fbs_line)}")
            print(f"model ATS cover vs posted line : {fbs_line['model_cover'].mean():.3f}")
            print(f"favorite cover rate            : {fbs_line['market_cover'].mean():.3f}")
            print(f"model margin MAE               : {fbs_line['margin_err'].mean():.2f}")
            print(f"line margin MAE                : {fbs_line['line_margin_err'].mean():.2f}")
            print("note: CFBD lines are opening/consensus (soft); ATS here is an upper bound.")

        fcs_games = P[(P["vs_fcs_h"] == 1) | (P["vs_fcs_a"] == 1)]
        if len(fcs_games):
            print("\n=== FBS vs FCS games ===")
            print(f"games: {len(fcs_games)}  "
                  f"MAE: {fcs_games['margin_err'].mean():.2f}  "
                  f"bias: {(fcs_games['pred_margin'] - fcs_games['actual_margin']).mean():+.2f}  "
                  f"win_acc: {fcs_games['model_win_hit'].mean():.3f}")
    return agg(P)


if __name__ == "__main__":
    run()
