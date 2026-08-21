"""Download and consolidate all historical data from CFBD."""
import sys

import pandas as pd

from .config import DATA_DIR, FIRST_SEASON
from .data import cfbd


def _consolidate(prefix: str, frames: list[pd.DataFrame], out_name: str) -> pd.DataFrame:
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    df.to_parquet(DATA_DIR / out_name, index=False)
    print(f"{prefix}: {len(df)} rows -> {out_name}")
    return df


def run():
    seasons = list(range(FIRST_SEASON, 2027))

    games = [cfbd.games(y) for y in seasons]
    g = _consolidate("games", games, "games_all.parquet")
    # keep games involving at least one FBS team
    g = g[(g["home_classification"] == "fbs") | (g["away_classification"] == "fbs")]
    g.to_parquet(DATA_DIR / "games_fbs.parquet", index=False)
    print(f"games FBS-involved: {len(g)}")

    stats = []
    for y in range(2013, 2027):
        weeks = []
        for w in range(0, 21):
            try:
                dfw = cfbd.fetch_cached(f"stats_{y}_w{w}", "/games/teams", {"year": y, "week": w})
            except Exception:
                continue
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

    talent = [cfbd.talent(y) for y in seasons]
    _consolidate("talent", talent, "talent_all.parquet")

    sp = [cfbd.sp_ratings(y) for y in range(2014, 2026)]
    _consolidate("sp+", sp, "sp_all.parquet")

    ln = [cfbd.lines(y) for y in range(2013, 2027)]
    l = pd.concat([f for f in ln if len(f)], ignore_index=True)
    l.to_parquet(DATA_DIR / "lines_all.parquet", index=False)
    print(f"lines: {len(l)} rows")

    cfbd.venues().to_parquet(DATA_DIR / "venues.parquet", index=False)
    cfbd.teams().to_parquet(DATA_DIR / "teams.parquet", index=False)

    # returning-production endpoint not available on this API plan; skip gracefully
    print("returning production: unavailable, skipping")

    print("DONE")


if __name__ == "__main__":
    sys.exit(run())
