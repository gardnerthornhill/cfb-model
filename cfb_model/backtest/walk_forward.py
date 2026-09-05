"""Chronological FBS-only backtest with shared training and availability rules."""
import json

import numpy as np
import pandas as pd

from ..artifacts import (code_identity, new_run_dir, require_unchanged,
                         snapshot_inputs, validate_artifact, write_manifest)
from ..config import DATA_DIR, REFIT_EVERY_WEEKS, TEST_SEASONS
from ..data.eligibility import fbs_only
from ..data.timing import available_at
from ..features.build import FEATURE_COLS, feature_inputs
from ..models.train import apply_linear_prior, fit_hybrid, predict_scores


def load_training_frame(min_season=2014):
    path = DATA_DIR / "features.parquet"
    validate_artifact(path, feature_inputs())
    df = pd.read_parquet(path)
    if len(fbs_only(df)) != len(df):
        raise ValueError("Feature artifact contains non-FBS matchups; rebuild features")
    return df[df["season"] >= min_season].sort_values(["start_date", "id"]).reset_index(drop=True)


def market_lines():
    ln = pd.read_parquet(DATA_DIR / "lines_all.parquet").explode("lines", ignore_index=True)
    x = pd.json_normalize(ln["lines"])
    ln = pd.concat([ln[["id"]], x], axis=1)
    for c in ("spread", "overUnder"):
        if c not in ln:
            ln[c] = np.nan
        ln[c] = pd.to_numeric(ln[c], errors="coerce")
    # A missing spread must not discard an independently available total.
    return ln.groupby("id").agg(spread_home=("spread", "median"),
                                over_under=("overUnder", "median")).reset_index()


def week_groups(df, seasons):
    """Calendar weeks prevent reused CFBD week numbers from spanning months."""
    df = df[df["season"].isin(seasons) & df["margin"].notna()].copy()
    df["calendar_week"] = df["start_date"].dt.to_period("W-SUN").dt.start_time
    groups = list(df.groupby(["season", "season_type", "calendar_week"], sort=False))
    return sorted(groups, key=lambda item: item[1]["start_date"].min())


def _weeks_between(groups, a, b):
    return max(0, groups.index(b) - groups.index(a))


def run(test_seasons=None, verbose=True):
    seasons = TEST_SEASONS if test_seasons is None else test_seasons
    inputs = [DATA_DIR / "features.parquet", DATA_DIR / "lines_all.parquet"]
    identity = {**code_identity(), "inputs": snapshot_inputs(inputs)}
    df = load_training_frame().merge(market_lines(), on="id", how="left")
    predictions, fits = [], []
    last_fit_date = None
    fit_period = None
    for (season, season_type, calendar_week), grp in week_groups(df, seasons):
        cutoff = grp["start_date"].min()
        period = (season, season_type)
        need_fit = (last_fit_date is None or fit_period != period or
                    calendar_week - last_fit_date >= pd.Timedelta(weeks=REFIT_EVERY_WEEKS))
        if need_fit:
            train = df[df["margin"].notna() & (available_at(df["start_date"]) <= cutoff)]
            if train.empty:
                raise ValueError(f"No training games available before {cutoff}")
            models, prior, metadata = fit_hybrid(train)
            fit_period, last_fit_date = period, calendar_week
            fit_cutoff = cutoff
            fits.append({"season": int(season), "season_type": season_type,
                         "fit_cutoff_utc": cutoff.isoformat(), **metadata})
            if verbose:
                print(f"fit {season} {season_type} {calendar_week.date()}: "
                      f"{len(train)} games, trees {metadata['selected_iterations']}", flush=True)
        out = predict_scores(models, grp[FEATURE_COLS].to_numpy(dtype=np.float32),
                             margin_add=apply_linear_prior(prior, grp))
        p = grp[["id", "season", "season_type", "week", "home_team", "away_team",
                 "start_date", "home_points", "away_points", "spread_home"]].copy()
        p["calendar_week"] = calendar_week
        p["fit_cutoff_utc"] = fit_cutoff
        p["latest_training_kickoff"] = metadata["latest_training_kickoff"]
        p["actual_margin"], p["actual_total"] = grp["margin"], grp["total"]
        p["market_total"] = grp["over_under"]
        for col, values in out.items():
            p[col] = values
        predictions.append(p)
    if not predictions:
        raise ValueError("No completed FBS games in requested test seasons")
    result = pd.concat(predictions, ignore_index=True).sort_values(["start_date", "id"])
    require_unchanged(inputs, identity)
    run_dir = new_run_dir(DATA_DIR / "backtest_runs")
    archive = run_dir / "predictions.parquet"
    result.to_parquet(archive, index=False)
    details = dict(scope="FBS vs FBS", rows=len(result), test_seasons=list(seasons),
                   fit_count=len(fits), fits=fits, market_timing="unknown historical observation time")
    write_manifest(archive, inputs, **details)
    result.to_parquet(DATA_DIR / "backtest_preds.parquet", index=False)
    write_manifest(DATA_DIR / "backtest_preds.parquet", inputs, archive=str(archive), **details)
    summary = _report(result, verbose)
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return result


def load_backtest():
    load_training_frame()  # Also verify the feature table against its raw inputs.
    path = DATA_DIR / "backtest_preds.parquet"
    validate_artifact(path, [DATA_DIR / "features.parquet", DATA_DIR / "lines_all.parquet"])
    return pd.read_parquet(path)


def evaluate(P):
    """Grade independent winner, margin, ATS and total decisions, excluding pushes."""
    def bets(edge, outcome, available):
        push = available & np.isclose(outcome, 0)
        abstain = available & ~push & np.isclose(edge, 0)
        decided = available & ~push & ~abstain
        wins = int(((np.sign(edge) == np.sign(outcome)) & decided).sum())
        n = int(decided.sum())
        return dict(wins=wins, losses=n - wins, pushes=int(push.sum()),
                    abstentions=int(abstain.sum()), rate=wins / n if n else None)
    has_line, has_total = P["spread_home"].notna(), P["market_total"].notna()
    margin_error = (P["pred_margin"] - P["actual_margin"]).abs()
    total_error = (P["pred_total"] - P["actual_total"]).abs()
    return {
        "games": len(P), "margin_mae": float(margin_error.mean()),
        "total_mae": float(total_error.mean()),
        "winner_accuracy": float(((P["pred_margin"] > 0) == (P["actual_margin"] > 0)).mean()),
        "brier": float(((P["p_home_win"] - (P["actual_margin"] > 0)) ** 2).mean()),
        "games_with_spread": int(has_line.sum()),
        "paired_model_margin_mae": float(margin_error[has_line].mean()),
        "market_margin_mae": float((P.loc[has_line, "actual_margin"] + P.loc[has_line, "spread_home"]).abs().mean()),
        "games_with_total": int(has_total.sum()),
        "paired_model_total_mae": float(total_error[has_total].mean()),
        "market_total_mae": float((P.loc[has_total, "actual_total"] - P.loc[has_total, "market_total"]).abs().mean()),
        "ats": bets(P["pred_margin"] + P["spread_home"], P["actual_margin"] + P["spread_home"], has_line),
        "over_under": bets(P["pred_total"] - P["market_total"], P["actual_total"] - P["market_total"], has_total),
    }


def _report(P, verbose=True):
    summary = evaluate(P)
    if verbose:
        yearly = pd.DataFrame({s: evaluate(g) for s, g in P.groupby("season")}).T
        print(yearly[["games", "margin_mae", "total_mae", "winner_accuracy", "brier"]].to_string())
        print(json.dumps(summary, indent=2))
        print("Market lines are stored medians without historical timestamps; not verified closing or executable prices.")
    return summary


if __name__ == "__main__":
    run()
