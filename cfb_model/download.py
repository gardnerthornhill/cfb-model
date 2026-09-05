"""Download and consolidate all historical data from CFBD."""
import argparse
import json
import sys
import requests

import pandas as pd

from .config import DATA_DIR, FIRST_SEASON
from .data import cfbd
from .data.eligibility import fbs_only


def _consolidate(prefix: str, frames: list[pd.DataFrame], out_name: str) -> pd.DataFrame:
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    df.to_parquet(DATA_DIR / out_name, index=False)
    print(f"{prefix}: {len(df)} rows -> {out_name}")
    return df


def run(refresh_season=None, refresh=True):
    refresh_season = refresh_season or pd.Timestamp.now().year
    last_season = max(pd.Timestamp.now().year, refresh_season)
    seasons = list(range(FIRST_SEASON, last_season + 1))
    (DATA_DIR / "source_snapshot.json").write_text(json.dumps({
        "status": "in_progress", "requested_refresh_season": refresh_season if refresh else None,
        "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }, indent=2))

    games = [cfbd.games(y, force=refresh and y == refresh_season) for y in seasons]
    g = _consolidate("games", games, "games_all.parquet")
    raw_games = g
    # Raw history is retained; all modeled rows require two season-specific FBS classifications.
    g = fbs_only(g)
    g.to_parquet(DATA_DIR / "games_fbs.parquet", index=False)
    print(f"games FBS vs FBS: {len(g)}")

    stats = []
    for y in range(2013, last_season + 1):
        weeks = []
        # The API rejects week 0. Request only actual completed schedule weeks,
        # including postseason weeks, rather than silently swallowing bad requests.
        scheduled_weeks = sorted(raw_games.loc[
            (raw_games["season"] == y) & raw_games["completed"], "week"].dropna().astype(int).unique())
        for w in scheduled_weeks:
            w = int(w)  # Stable cache keys and JSON metadata, not numpy.int64.
            try:
                dfw = cfbd.fetch_cached(f"stats_{y}_w{w}", "/games/teams", {"year": y, "week": w},
                                        force=refresh and y == refresh_season)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    continue
                raise
            if len(dfw):
                weeks.append(dfw)
        if not weeks:
            continue
        wk = pd.concat(weeks, ignore_index=True)
        wk = wk.explode("teams", ignore_index=True)
        teams = pd.json_normalize(wk["teams"])
        teams.columns = [c.replace(".", "_") for c in teams.columns]
        flat = pd.concat([wk[["id"]].reset_index(drop=True), teams.reset_index(drop=True)], axis=1)
        flat = flat.explode("stats", ignore_index=True)
        st = pd.json_normalize(flat["stats"])
        flat = pd.concat([flat.drop(columns=["stats"]).reset_index(drop=True), st.reset_index(drop=True)], axis=1)
        wide = flat.pivot_table(index=["id", "teamId", "team", "conference", "homeAway", "points"],
                                columns="category", values="stat", aggfunc="first").reset_index()
        wide.insert(1, "season", y)
        stats.append(wide)
        print(f"stats {y}: {len(wide)} team-games")
    s = pd.concat(stats, ignore_index=True)
    s.to_parquet(DATA_DIR / "stats_all.parquet", index=False)
    print(f"stats: {len(s)} rows")

    talent = [cfbd.talent(y, force=refresh and y == refresh_season) for y in seasons]
    _consolidate("talent", talent, "talent_all.parquet")

    sp = [cfbd.sp_ratings(y, force=refresh and y == refresh_season - 1)
          for y in range(2014, last_season)]
    _consolidate("sp+", sp, "sp_all.parquet")

    ln = [cfbd.lines(y, force=refresh and y == refresh_season) for y in range(2013, last_season + 1)]
    l = pd.concat([f for f in ln if len(f)], ignore_index=True)
    l.to_parquet(DATA_DIR / "lines_all.parquet", index=False)
    print(f"lines: {len(l)} rows")

    cfbd.venues().to_parquet(DATA_DIR / "venues.parquet", index=False)
    cfbd.teams().to_parquet(DATA_DIR / "teams.parquet", index=False)

    # returning-production endpoint not available on this API plan; skip gracefully
    print("returning production: unavailable, skipping")

    (DATA_DIR / "source_snapshot.json").write_text(json.dumps({
        "status": "complete",
        "consolidated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "refreshed_season": refresh_season if refresh else None,
        "refresh_completed": refresh,
        "completed_fbs_games": int(g["completed"].sum()),
        "market_observation_time": "not supplied; retrieval time is not quote time",
    }, indent=2))
    print("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh-season", type=int)
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()
    sys.exit(run(args.refresh_season, refresh=not args.no_refresh))
