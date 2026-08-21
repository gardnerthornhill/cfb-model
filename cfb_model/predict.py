"""Predict an upcoming week: refresh data, retrain on all completed games,
and write a CSV of every game with market lines vs model projections.

Usage:
    python -m cfb_model.predict                  # auto-detect season/week
    python -m cfb_model.predict --season 2026 --week 0
    python -m cfb_model.predict --no-refresh     # reuse cached data
"""
import argparse
import sys

import numpy as np
import pandas as pd

from . import download
from .config import OUT_DIR
from .data import cfbd
from .backtest.walk_forward import load_training_frame, market_lines
from .features.build import FEATURE_COLS, build_all
from .models.train import (apply_linear_prior, fit_linear_prior, fit_pair,
                           predict_scores)


def refresh_season(season: int):
    """Force-refetch current-season data (cache goes stale in-season), then
    re-consolidate everything else from cache."""
    print(f"refreshing {season} games/lines/stats from CFBD ...")
    cfbd.games(season, force=True)
    cfbd.lines(season, force=True)
    for w in range(21):
        try:
            cfbd.fetch_cached(f"stats_{season}_w{w}", "/games/teams",
                              {"year": season, "week": w}, force=True)
        except Exception:
            continue
    download.run()


def pick_target_week(feats: pd.DataFrame, season: int) -> int | None:
    """Earliest week that still has uncompleted games (near now or later)."""
    f = feats[(feats["season"] == season) & (~feats["completed"])]
    if not len(f):
        return None
    now = pd.Timestamp.now(tz="UTC").tz_convert(None)
    upcoming = f[f["start_date"] >= now - pd.Timedelta(hours=12)]
    pool = upcoming if len(upcoming) else f
    return int(pool["week"].min())


def run(season: int | None = None, week: int | None = None,
        refresh: bool = True) -> pd.DataFrame:
    season = season or pd.Timestamp.now().year

    if refresh:
        refresh_season(season)
    feats = build_all(save=True)

    week = week if week is not None else pick_target_week(feats, season)
    if week is None:
        sys.exit(f"no uncompleted games found for {season}; pass --season/--week explicitly")

    target = feats[(feats["season"] == season) & (feats["week"] == week) &
                   (~feats["completed"])].copy()
    if not len(target):
        sys.exit(f"no uncompleted games for {season} week {week}")

    train = load_training_frame()
    train = train[train["margin"].notna()]
    print(f"training on {len(train)} completed games ...")
    Xtr = train[FEATURE_COLS].to_numpy(dtype=np.float32)
    ym = train["margin"].to_numpy(dtype=np.float32)
    yt = train["total"].to_numpy(dtype=np.float32)
    lin = fit_linear_prior(train, train["margin"].to_numpy(dtype=np.float64))
    ym_resid = ym - apply_linear_prior(lin, train).astype(np.float32)
    models = fit_pair(Xtr, ym_resid, yt, valid_slice=slice(int(len(Xtr) * 0.94), None))

    X = target[FEATURE_COLS].to_numpy(dtype=np.float32)
    out = predict_scores(models, X, margin_add=apply_linear_prior(lin, target))

    P = pd.DataFrame({
        "id": target["id"].values,
        "kickoff_utc": target["start_date"].values,
        "season": season,
        "week": week,
        "away_team": target["away_team"].values,
        "home_team": target["home_team"].values,
        "neutral_site": target["neutral_site"].values,
        "conference_game": target["conference_game"].values,
        **out,
    })
    P = P.merge(market_lines(), on="id", how="left")
    P = P.rename(columns={"spread_home": "market_spread_home", "over_under": "market_total"})

    # Spread convention: negative = home team favored (same as sportsbooks).
    P["model_spread_home"] = -P["pred_margin"]
    # Delta vs market in points: positive = model favors the HOME side more
    # than the market, negative = model favors the AWAY side. Bigger |delta|
    # = bigger disagreement = the interesting games. Sort by this column.
    P["model_market_delta"] = P["market_spread_home"] - P["model_spread_home"]
    P["model_pick_ats"] = np.where(P["market_spread_home"].isna(), "",
                                   np.where(P["model_market_delta"] > 0,
                                            P["home_team"], P["away_team"]))
    # Total delta: positive = lean over, negative = lean under.
    P["model_market_total_delta"] = P["pred_total"] - P["market_total"]

    P = P.rename(columns={
        "pred_total": "model_total",
        "pred_home_score": "proj_home_score",
        "pred_away_score": "proj_away_score",
        "p_home_win": "home_win_prob",
    })
    P["home_win_prob"] = P["home_win_prob"].round(3)
    for c in ("model_spread_home", "model_market_delta", "model_total",
              "market_spread_home", "market_total", "model_market_total_delta"):
        P[c] = P[c].round(1)

    OUT_COLS = [
        "kickoff_utc", "season", "week", "away_team", "home_team",
        "neutral_site", "conference_game",
        "market_spread_home", "model_spread_home", "model_market_delta",
        "model_pick_ats",
        "market_total", "model_total", "model_market_total_delta",
        "home_win_prob", "proj_home_score", "proj_away_score",
    ]
    P = P.sort_values(["kickoff_utc", "home_team"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = OUT_DIR / f"predictions_{season}_w{week}.csv"
    P[OUT_COLS].to_csv(fp, index=False)
    print(f"\n=== {season} WEEK {week}: {len(P)} games -> {fp.name} ===")
    cols = ["kickoff_utc", "away_team", "home_team", "market_spread_home",
            "model_spread_home", "model_market_delta", "model_pick_ats",
            "model_total", "home_win_prob"]
    print(P[cols].to_string(index=False))
    big = P[P["model_market_delta"].abs() >= 7].sort_values("model_market_delta")
    if len(big):
        print(f"\nbiggest disagreements with market (|delta| >= 7): {len(big)}")
        print(big[["away_team", "home_team", "market_spread_home",
                   "model_spread_home", "model_market_delta", "model_pick_ats"]]
              .to_string(index=False))
    return P


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict an upcoming week of CFB games.")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip force-refreshing current-season data")
    a = ap.parse_args()
    run(a.season, a.week, refresh=not a.no_refresh)
