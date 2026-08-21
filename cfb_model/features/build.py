"""Build the master causal feature table.

One row per FBS-involved game. Every feature is computed only from information
available before kickoff: Elo/ridge ratings snapshotted pre-week, rolling stat
averages shifted by one game, prior-season SP+/talent priors, rest, travel.
"""
import numpy as np
import pandas as pd

from ..config import DATA_DIR, FCS_RIDGE_PRIOR
from .ratings import EloEngine, RidgeRatings

FEATURE_COLS = [
    "elo_diff", "cfbd_elo_diff", "adj_margin_diff",
    "sp_prior_diff", "sp_prior_off_diff", "sp_prior_def_diff", "talent_diff",
    "off_ypp_h", "off_ypp_a", "def_ypp_h", "def_ypp_a",
    "off_ypp_l3_h", "off_ypp_l3_a", "def_ypp_l3_h", "def_ypp_l3_a",
    "pts_off_h", "pts_off_a", "pts_def_h", "pts_def_a",
    "tov_margin_h", "tov_margin_a", "third_down_h", "third_down_a",
    "pace_h", "pace_a", "comp_pct_h", "comp_pct_a",
    "ps_ypp_h", "ps_ypp_a", "ps_pts_h", "ps_pts_a", "ps_pts_def_h", "ps_pts_def_a",
    "rest_h", "rest_a", "opener_h", "opener_a", "bye_h", "bye_a",
    "travel_away", "neutral_site", "conference_game", "week",
    "games_played_h", "games_played_a", "vs_fcs_h", "vs_fcs_a",
    "fcs_h", "fcs_a", "sp_miss_h", "sp_miss_a", "talent_miss_h", "talent_miss_a",
]


def _parse_ratio(s: pd.Series) -> pd.Series:
    def f(v):
        try:
            a, b = str(v).split("-")
            return (float(a), float(b))
        except Exception:
            return (np.nan, np.nan)
    parsed = s.apply(f)
    return parsed


def load_games() -> pd.DataFrame:
    g = pd.read_parquet(DATA_DIR / "games_fbs.parquet")
    g = g.sort_values(["start_date", "id"]).reset_index(drop=True)
    g["start_date"] = pd.to_datetime(g["start_date"], utc=True).dt.tz_localize(None)
    g["margin"] = np.where(g["completed"], g["home_points"] - g["away_points"], np.nan)
    g["total"] = np.where(g["completed"], g["home_points"] + g["away_points"], np.nan)
    return g


def add_ratings(g: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Elo pass + ridge snapshots. Returns games with elo cols, ridge snapshots,
    and final rating dicts (for future-week prediction)."""
    fcs_h = set(g.loc[g["home_classification"] == "fcs", "home_id"].astype(int))
    fcs_a = set(g.loc[g["away_classification"] == "fcs", "away_id"].astype(int))
    fcs_teams = fcs_h | fcs_a
    elo = EloEngine(fcs_teams=fcs_teams)
    ridge = RidgeRatings()
    fcs_prior = {t: FCS_RIDGE_PRIOR for t in fcs_teams}
    elo_h = np.full(len(g), np.nan)
    elo_a = np.full(len(g), np.nan)
    adj_h = np.full(len(g), np.nan)
    adj_a = np.full(len(g), np.nan)

    key = g["season"].astype(str) + "_" + g["week"].astype(str)
    snapshots: dict[str, dict[int, float]] = {}
    cur: list[tuple[int, int, float]] = []
    cur_season = None

    done = g[g["completed"] & g["margin"].notna()]
    done_pos = set(done.index)

    def both_fcs(row):
        return (row.get("home_classification") == "fcs"
                and row.get("away_classification") == "fcs")

    for i, row in g.iterrows():
        season, week = int(row["season"]), int(row["week"])
        k = f"{season}_{week}"
        if k not in snapshots:
            elo.regress_offseason(season)
            if cur_season is not None and season != cur_season:
                cur = []
            cur_season = season
            snapshots[k] = ridge.solve(season, cur, prior=fcs_prior)
        rh, ra = elo.pregame(int(row["home_id"]), int(row["away_id"]))
        elo_h[i], elo_a[i] = rh, ra
        snap = snapshots[k]
        adj_h[i] = snap.get(int(row["home_id"]), np.nan)
        adj_a[i] = snap.get(int(row["away_id"]), np.nan)
        if i in done_pos and not both_fcs(row):
            # FCS-vs-FCS games are excluded from rating updates: they carry no
            # information about the FBS scale, and letting them vote lets FCS
            # teams cluster around average-FBS ratings.
            m = float(row["margin"])
            elo.update(int(row["home_id"]), int(row["away_id"]), m)
            ridge.add_game(season, int(row["home_id"]), int(row["away_id"]), m)
            if cur_season == season:
                cur.append((int(row["home_id"]), int(row["away_id"]), m))

    # final ratings for predicting upcoming weeks
    final_snap = ridge.solve(cur_season or 0, cur, prior=fcs_prior)
    g = g.copy()
    g["elo_home_pre"] = elo_h
    g["elo_away_pre"] = elo_a
    g["adj_home"] = adj_h
    g["adj_away"] = adj_a
    return g, snapshots, final_snap, dict(elo.rating)


def build_team_stats() -> pd.DataFrame:
    """Per team-game metrics with opponent-adjusted defensive numbers."""
    s = pd.read_parquet(DATA_DIR / "stats_all.parquet")
    s = s.rename(columns={c: "".join("_" + ch.lower() if ch.isupper() else ch for ch in c).lstrip("_")
                          for c in s.columns})
    keep = ["id", "season", "team_id", "team", "points", "total_yards", "rushing_attempts",
            "completion_attempts", "yards_per_pass", "yards_per_rush_attempt", "third_down_eff",
            "turnovers", "possession_time"]
    missing = [c for c in keep if c not in s.columns]
    if missing:
        raise KeyError(f"stats columns missing: {missing}; have {s.columns.tolist()}")
    s = s[keep].copy()

    def ratio_pair(v):
        try:
            a, b = str(v).split("-")
            return float(a), float(b)
        except Exception:
            return np.nan, np.nan

    ca = s["completion_attempts"].apply(ratio_pair)
    s["completions"] = [x[0] for x in ca]
    s["pass_att"] = [x[1] for x in ca]
    td = s["third_down_eff"].apply(ratio_pair)
    s["third_conv"] = [x[0] for x in td]
    s["third_att"] = [x[1] for x in td]
    poss = s["possession_time"].astype(str).str.split(":", expand=True)
    s["poss_min"] = pd.to_numeric(poss[0], errors="coerce") + pd.to_numeric(poss[1], errors="coerce") / 60.0
    for c in ("points", "total_yards", "rushing_attempts", "turnovers"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s["plays"] = s["rushing_attempts"] + s["pass_att"]
    s["ypp_off"] = s["total_yards"] / s["plays"]
    s["comp_pct"] = s["completions"] / s["pass_att"]
    s["third_down_pct"] = s["third_conv"] / s["third_att"]

    # pair rows of same game to get defensive side
    opp = s.merge(s, on="id", suffixes=("", "_opp"))
    opp = opp[opp["team_id"] != opp["team_id_opp"]]
    opp["ypp_def"] = opp["total_yards_opp"] / opp["plays_opp"]
    opp["third_down_def_pct"] = opp["third_conv_opp"] / opp["third_att_opp"]
    opp["tov_margin"] = opp["turnovers_opp"].fillna(0) - opp["turnovers"].fillna(0)
    opp["pts_allowed"] = pd.to_numeric(opp["points_opp"], errors="coerce")
    opp = opp.rename(columns={"points": "pts"})[
        ["id", "season", "team_id", "pts", "pts_allowed", "ypp_off", "ypp_def", "comp_pct",
         "third_down_pct", "third_down_def_pct", "tov_margin", "poss_min"]
    ].drop_duplicates(subset=["id", "team_id"])
    return opp


def add_rolling_features(g: pd.DataFrame) -> pd.DataFrame:
    ts = build_team_stats()
    game_dates = g.set_index("id")["start_date"]
    ts["date"] = ts["id"].map(game_dates)
    ts = ts.dropna(subset=["date"]).sort_values(["date", "team_id"]).reset_index(drop=True)

    met = ["pts", "pts_allowed", "ypp_off", "ypp_def", "comp_pct", "third_down_pct",
           "third_down_def_pct", "tov_margin", "poss_min"]
    out = []
    for tid, grp in ts.groupby("team_id"):
        grp = grp.sort_values("date").copy()
        for m in met:
            v = grp[m].shift(1)
            grp[f"std_{m}"] = v.expanding(min_periods=1).mean()
            grp[f"l3_{m}"] = v.rolling(3, min_periods=1).mean()
        grp["games_before"] = range(len(grp))
        grp["days_since_prev"] = grp["date"].diff().dt.days
        grp["season_opener"] = grp["season"] != grp["season"].shift(1)
        grp["prev_season"] = grp["season"].shift(1)
        out.append(grp)
    T = pd.concat(out, ignore_index=True)

    # prior-season aggregates (kept all season long as the preseason baseline;
    # current-form features capture in-season movement separately)
    agg = ts.groupby(["team_id", "season"])[met].mean().reset_index()
    agg.columns = ["team_id", "season"] + [f"ps_{m}" for m in met]
    T = T.merge(agg, left_on=["team_id", "prev_season"],
                right_on=["team_id", "season"], suffixes=("", "_agg"), how="left")

    tcols = ["id", "team_id"] + [f"std_{m}" for m in met] + [f"l3_{m}" for m in met] + \
            [f"ps_{m}" for m in met] + ["games_before", "days_since_prev", "season_opener"]
    full = T  # pre-trim copy retains date/season for the synthetic-row step
    T = T[tcols]

    # Unplayed (future) games have no stats row, so they'd otherwise lose every
    # stat feature. Give each team's next game the state of its most recent
    # played game: rolling means already shifted by one game, so they describe
    # exactly "form through last game played" — the correct causal snapshot.
    fut = g[~g["completed"]]
    if len(fut):
        fl = pd.concat([
            fut[["id", "season", "start_date", "home_id"]].rename(columns={"home_id": "team_id"}),
            fut[["id", "season", "start_date", "away_id"]].rename(columns={"away_id": "team_id"}),
        ], ignore_index=True)
        fl["team_id"] = fl["team_id"].astype(int)
        base = full.sort_values("date").groupby("team_id").tail(1)
        b = base[["team_id"] + tcols[2:] + ["date", "season"]].rename(
            columns={"date": "last_date", "season": "last_season"})
        fl = fl.merge(b, on="team_id", how="inner")
        fl["games_before"] = fl["games_before"] + 1
        fl["days_since_prev"] = (fl["start_date"] - fl["last_date"]).dt.days
        fl["season_opener"] = fl["season"] != fl["last_season"]
        fl["prev_season"] = fl["last_season"]
        # ps_* on the base row describe its *previous* season; for the synthetic
        # row the prior season is the base row's own season, so re-merge.
        fl = fl.drop(columns=[c for c in fl.columns if c.startswith("ps_")])
        fl = fl.merge(agg, left_on=["team_id", "prev_season"],
                      right_on=["team_id", "season"], suffixes=("", "_agg"), how="left")
        Ts = fl[tcols]
        T = pd.concat([T, Ts], ignore_index=True)

    h = T.rename(columns={c: f"{c}_h" for c in tcols[2:]})
    a = T.rename(columns={c: f"{c}_a" for c in tcols[2:]})
    g = g.merge(h, left_on=["id", "home_id"], right_on=["id", "team_id"], how="left").drop(columns=["team_id"])
    g = g.merge(a, left_on=["id", "away_id"], right_on=["id", "team_id"], how="left").drop(columns=["team_id"])

    # rest handling
    for side in ("h", "a"):
        d = g[f"days_since_prev_{side}"]
        opener = g[f"season_opener_{side}"].fillna(False)
        g[f"rest_{side}"] = np.where(opener, 20.0, d.clip(6, 20)).astype(float)
        g[f"opener_{side}"] = opener.astype(float)
        g[f"bye_{side}"] = ((~opener) & (d >= 13) & (d <= 40)).astype(float)
    return g


def add_priors(g: pd.DataFrame) -> pd.DataFrame:
    teams = pd.read_parquet(DATA_DIR / "teams.parquet")
    name2id = {}
    for _, r in teams.iterrows():
        if pd.notna(r["school"]):
            name2id[r["school"]] = r["id"]

    tal = pd.read_parquet(DATA_DIR / "talent_all.parquet")
    tal["team_id"] = tal["team"].map(name2id)
    tal = tal.dropna(subset=["team_id"])
    tal_map = {(int(y), int(t)): float(v) for y, t, v in zip(tal["year"], tal["team_id"], tal["talent"])}

    sp = pd.read_parquet(DATA_DIR / "sp_all.parquet")
    sp["team_id"] = sp["team"].map(name2id)
    sp = sp.dropna(subset=["team_id"])

    def talent_for(team_id, season):
        for y in range(season, 2003, -1):  # carry forward most recent available
            v = tal_map.get((y, team_id))
            if v is not None:
                return v
        return np.nan

    g["talent_home"] = [talent_for(t, s) for t, s in zip(g["home_id"], g["season"])]
    g["talent_away"] = [talent_for(t, s) for t, s in zip(g["away_id"], g["season"])]

    spi = sp.set_index(["year", "team_id"])[["rating", "offense_rating", "defense_rating"]]

    def sp_for(team_id, season):
        for y in range(season - 1, 2013, -1):  # prior-season SP+; carry forward if gap
            try:
                row = spi.loc[(y, team_id)]
                return float(row.iloc[0]), float(row.iloc[1]), float(row.iloc[2])
            except KeyError:
                continue
        return (np.nan, np.nan, np.nan)

    sph = pd.DataFrame([sp_for(t, s) for t, s in zip(g["home_id"], g["season"])],
                       columns=["sp_r_h", "sp_o_h", "sp_d_h"], index=g.index)
    spa = pd.DataFrame([sp_for(t, s) for t, s in zip(g["away_id"], g["season"])],
                       columns=["sp_r_a", "sp_o_a", "sp_d_a"], index=g.index)
    g = pd.concat([g, sph, spa], axis=1)

    # No imputation: FCS / new-FBS teams have no SP+ or talent data, and
    # filling that with league-average values told the model they were average
    # FBS teams. Missing stays NaN (trees route missing splits natively) and
    # explicit flags make the gap learnable.
    g["talent_miss_h"] = g["talent_home"].isna().astype(float)
    g["talent_miss_a"] = g["talent_away"].isna().astype(float)
    g["talent_diff"] = g["talent_home"] - g["talent_away"]
    g["sp_miss_h"] = g["sp_r_h"].isna().astype(float)
    g["sp_miss_a"] = g["sp_r_a"].isna().astype(float)
    for col, hcol, acol in (("sp_prior_diff", "sp_r_h", "sp_r_a"),
                            ("sp_prior_off_diff", "sp_o_h", "sp_o_a"),
                            ("sp_prior_def_diff", "sp_d_h", "sp_d_a")):
        g[col] = g[hcol] - g[acol]
    return g


def add_travel_rest_context(g: pd.DataFrame) -> pd.DataFrame:
    venues = pd.read_parquet(DATA_DIR / "venues.parquet")
    vcoords = venues.set_index("id")[["latitude", "longitude"]]
    teams = pd.read_parquet(DATA_DIR / "teams.parquet")
    tcoords = teams.set_index("id")[["location_latitude", "location_longitude"]]

    gv = g["venue_id"].map(vcoords["latitude"])
    g["_vlat"] = gv
    g["_vlon"] = g["venue_id"].map(vcoords["longitude"])
    g["_alat"] = g["away_id"].map(tcoords["location_latitude"])
    g["_alon"] = g["away_id"].map(tcoords["location_longitude"])
    g["_hlat"] = g["home_id"].map(tcoords["location_latitude"])
    g["_hlon"] = g["home_id"].map(tcoords["location_longitude"])

    def hav(lat1, lon1, lat2, lon2):
        r = 6371.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dp = p2 - p1
        dl = np.radians(lon2 - lon1)
        a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return 2 * r * np.arcsin(np.sqrt(a))

    away_travel = hav(g["_alat"], g["_alon"], g["_vlat"], g["_vlon"])
    home_travel = hav(g["_hlat"], g["_hlon"], g["_vlat"], g["_vlon"])
    g["neutral_site"] = g["neutral_site"].fillna(False).astype(float)
    g["travel_away"] = np.where(g["neutral_site"] > 0,
                                (away_travel.fillna(0) + home_travel.fillna(0)) / 2,
                                away_travel.fillna(0))
    g["conference_game"] = g["conference_game"].fillna(False).astype(float)
    g["week"] = g["week"].astype(float)

    cls_h = g.get("home_classification")
    cls_a = g.get("away_classification")
    g["vs_fcs_h"] = (cls_a == "fcs").astype(float)
    g["vs_fcs_a"] = (cls_h == "fcs").astype(float)
    g["fcs_h"] = (cls_h == "fcs").astype(float)
    g["fcs_a"] = (cls_a == "fcs").astype(float)

    g["games_played_h"] = g["games_before_h"]
    g["games_played_a"] = g["games_before_a"]
    drop = [c for c in g.columns if c.startswith("_")]
    return g.drop(columns=drop)


def finalize(g: pd.DataFrame) -> pd.DataFrame:
    g["elo_diff"] = g["elo_home_pre"] - g["elo_away_pre"]
    g["cfbd_elo_diff"] = g["home_pregame_elo"] - g["away_pregame_elo"]
    g["adj_margin_diff"] = g["adj_home"] - g["adj_away"]

    ren = {
        "std_ypp_off_h": "off_ypp_h", "std_ypp_off_a": "off_ypp_a",
        "std_ypp_def_h": "def_ypp_h", "std_ypp_def_a": "def_ypp_a",
        "l3_ypp_off_h": "off_ypp_l3_h", "l3_ypp_off_a": "off_ypp_l3_a",
        "l3_ypp_def_h": "def_ypp_l3_h", "l3_ypp_def_a": "def_ypp_l3_a",
        "std_pts_h": "pts_off_h", "std_pts_a": "pts_off_a",
        "std_pts_allowed_h": "pts_def_h", "std_pts_allowed_a": "pts_def_a",
        "std_tov_margin_h": "tov_margin_h", "std_tov_margin_a": "tov_margin_a",
        "std_third_down_pct_h": "third_down_h", "std_third_down_pct_a": "third_down_a",
        "std_poss_min_h": "pace_h", "std_poss_min_a": "pace_a",
        "std_comp_pct_h": "comp_pct_h", "std_comp_pct_a": "comp_pct_a",
        "ps_ypp_off_h": "ps_ypp_h", "ps_ypp_off_a": "ps_ypp_a",
        "ps_pts_allowed_h": "ps_pts_def_h", "ps_pts_allowed_a": "ps_pts_def_a",
    }
    for src, new in ren.items():
        if src in g.columns:
            g[new] = g[src]
    return g


def build_all(save=True) -> pd.DataFrame:
    g = load_games()
    g, snapshots, final_adj, final_elo = add_ratings(g)
    g = add_priors(g)
    g = add_rolling_features(g)
    g = add_travel_rest_context(g)
    g = finalize(g)
    keep_cols = list(dict.fromkeys(FEATURE_COLS + [
        "id", "season", "start_date", "home_id", "away_id",
        "home_team", "away_team", "completed",
        "home_points", "away_points", "margin", "total",
        "elo_home_pre", "elo_away_pre", "adj_home", "adj_away"]))
    feats = g[keep_cols].copy()
    if save:
        feats.to_parquet(DATA_DIR / "features.parquet", index=False)
        import json
        with open(DATA_DIR / "final_ratings.json", "w") as f:
            json.dump({"elo": {str(k): v for k, v in final_elo.items()},
                       "adj": {str(k): v for k, v in final_adj.items()},
                       "snapshots": {k: {str(a): b for a, b in v.items()} for k, v in snapshots.items()}}, f)
    return feats


if __name__ == "__main__":
    df = build_all()
    print(df.shape)
    print(df[FEATURE_COLS].isna().mean().sort_values(ascending=False).head(10))
