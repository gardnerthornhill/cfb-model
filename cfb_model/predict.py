"""Issue timestamped, immutable forecasts for upcoming FBS-vs-FBS games."""
import argparse
import json
import shutil

import numpy as np
import pandas as pd

from . import download
from .artifacts import (code_identity, manifest_path, new_run_dir, require_unchanged,
                        snapshot_inputs, write_manifest)
from .config import DATA_DIR, OUT_DIR
from .data.timing import available_at
from .data.eligibility import fbs_only
from .backtest.walk_forward import load_training_frame, market_lines
from .features.build import FEATURE_COLS, build_all
from .models.train import apply_linear_prior, fit_hybrid, predict_scores


def select_target(feats, season, week=None, season_type=None, now=None):
    feats = fbs_only(feats)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    eligible = feats[(feats["season"] == season) & ~feats["completed"] & (feats["start_date"] > now)]
    if week is not None:
        eligible = eligible[eligible["week"] == week]
    if season_type is not None:
        eligible = eligible[eligible["season_type"] == season_type]
    if eligible.empty:
        raise ValueError(f"No future, uncompleted FBS matchups for season {season}, week {week}")
    first = eligible.sort_values(["start_date", "id"]).iloc[0]
    return eligible[(eligible["week"] == first["week"]) &
                    (eligible["season_type"] == first["season_type"])].copy()


def run(season=None, week=None, refresh=True, season_type=None):
    season = season or pd.Timestamp.now().year
    if refresh:
        download.run(refresh_season=season)
    data_cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None)
    feats = build_all(save=True, as_of=data_cutoff)
    inputs = [DATA_DIR / "features.parquet", DATA_DIR / "lines_all.parquet"]
    identity = {**code_identity(), "inputs": snapshot_inputs(inputs)}
    target = select_target(feats, season, week, season_type, data_cutoff)
    train = load_training_frame()
    train = train[train["margin"].notna() & (available_at(train["start_date"]) <= data_cutoff)]
    print(f"Training on {len(train)} completed FBS matchups ...", flush=True)
    models, prior, fit_metadata = fit_hybrid(train)
    # A game can start while the model fits; never publish its first forecast late.
    issued_at = pd.Timestamp.now(tz="UTC")
    target = target[target["start_date"] > issued_at.tz_localize(None)]
    if target.empty:
        raise ValueError("Target games started during training; no late forecasts were issued")
    out = predict_scores(models, target[FEATURE_COLS].to_numpy(dtype=np.float32),
                         margin_add=apply_linear_prior(prior, target))
    P = target[["id", "start_date", "season", "season_type", "week", "away_team", "home_team",
                "neutral_site", "conference_game"]].reset_index(drop=True).rename(columns={"start_date": "kickoff_utc"})
    for col, values in out.items():
        P[col] = values
    P = P.merge(market_lines(), on="id", how="left").rename(
        columns={"spread_home": "market_spread_home", "over_under": "market_total"})
    P["model_spread_home"] = -P["pred_margin"]
    P["model_market_delta"] = P["market_spread_home"] - P["model_spread_home"]
    P["model_pick_ats"] = np.where(P["market_spread_home"].isna() | np.isclose(P["model_market_delta"], 0), "",
                                    np.where(P["model_market_delta"] > 0, P["home_team"], P["away_team"]))
    P["model_market_total_delta"] = P["pred_total"] - P["market_total"]
    P = P.rename(columns={"pred_total": "model_total", "pred_home_score": "proj_home_score",
                          "pred_away_score": "proj_away_score", "p_home_win": "home_win_prob"})
    P["issued_at_utc"] = issued_at.isoformat()
    P["data_cutoff_utc"] = data_cutoff.isoformat() + "Z"
    P["home_win_prob"] = P["home_win_prob"].round(3)
    for col in ("model_spread_home", "model_market_delta", "model_total", "market_spread_home",
                "market_total", "model_market_total_delta"):
        P[col] = P[col].round(1)
    P = P.drop(columns="pred_margin").sort_values(["kickoff_utc", "id"]).reset_index(drop=True)
    require_unchanged(inputs, identity)
    run_dir = new_run_dir(OUT_DIR / "runs")
    P["run_id"] = run_dir.name
    archive = run_dir / "predictions.csv"
    P.to_csv(archive, index=False)
    manifest = write_manifest(archive, [DATA_DIR / "features.parquet", DATA_DIR / "lines_all.parquet"],
                             scope="FBS vs FBS", issued_at_utc=issued_at.isoformat(),
                             data_cutoff_utc=P["data_cutoff_utc"].iloc[0], **fit_metadata)
    # Keep a convenient latest view, while the archive and manifest remain immutable.
    label = f"{season}_w{int(P['week'].iloc[0])}"
    if P["season_type"].iloc[0] != "regular":
        label += "_" + P["season_type"].iloc[0]
    latest = OUT_DIR / f"predictions_{label}.csv"
    if latest.exists() and not manifest_path(latest).exists():
        legacy = OUT_DIR / "legacy"
        legacy.mkdir(exist_ok=True)
        destination = legacy / latest.name
        if destination.exists() and destination.read_bytes() != latest.read_bytes():
            raise ValueError("Conflicting legacy forecast; refusing to overwrite it")
        shutil.copy2(latest, destination)
        (legacy / (latest.name + ".README.txt")).write_text(
            "Preserved pre-manifest forecast. Not produced by the current FBS-only model.\n")
    shutil.copy2(archive, latest)
    manifest_path(latest).write_text(json.dumps({**manifest, "archive": str(archive)}, indent=2) + "\n")
    print(f"Issued {len(P)} games: {archive}")
    print(P[["kickoff_utc", "away_team", "home_team", "model_spread_home", "market_spread_home",
             "model_pick_ats", "model_total", "home_win_prob"]].to_string(index=False))
    return P


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--season-type", choices=["regular", "postseason", "spring_regular"])
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()
    run(args.season, args.week, not args.no_refresh, args.season_type)
