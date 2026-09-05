"""Build FBS-only pregame features with conservative result availability.

Live and historical rows query the same postgame state. Football weighting
formulas are unchanged; division membership comes from each season's games.
"""
import numpy as np
import pandas as pd
from collections import deque

from ..config import DATA_DIR
from ..artifacts import write_manifest
from ..data.eligibility import fbs_only
from ..data.timing import available_at
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
    "games_played_h", "games_played_a", "sp_miss_h", "sp_miss_a", "talent_miss_h", "talent_miss_a",
]


def load_games() -> pd.DataFrame:
    g = fbs_only(pd.read_parquet(DATA_DIR / "games_fbs.parquet"))
    g = g.sort_values(["start_date", "id"]).reset_index(drop=True)
    g["start_date"] = pd.to_datetime(g["start_date"], utc=True).dt.tz_localize(None)
    g["margin"] = np.where(g["completed"], g["home_points"] - g["away_points"], np.nan)
    g["total"] = np.where(g["completed"], g["home_points"] + g["away_points"], np.nan)
    return g


def add_ratings(g: pd.DataFrame, as_of=None) -> tuple[pd.DataFrame, dict, dict, dict]:
    """FBS-only ratings, with separate calendar-date and postseason snapshots."""
    if len(fbs_only(g)) != len(g):
        raise ValueError("Rating inputs must be FBS vs FBS")
    elo = EloEngine()
    ridge = RidgeRatings()
    values = []
    snapshots = {}
    current = []
    current_season = None
    pending = deque()
    for row in g.itertuples(index=False):
        season = int(row.season)
        cutoff = min(row.start_date, as_of) if as_of is not None else row.start_date
        while pending and pending[0][0] <= cutoff:
            _, played_season, game = pending.popleft()
            elo.update(*game)
            ridge.add_game(played_season, *game)
            if played_season == current_season:
                current.append(game)
        if current_season != season:
            elo.regress_offseason(season)
            current = []
            current_season = season
        # CFBD week numbers can repeat across Aug 29/Sep 3 and bowl season.
        key = f"{season}_{row.season_type}_{int(row.week)}_{row.start_date.date()}"
        if key not in snapshots:
            snapshots[key] = ridge.solve(season, current)
        rh, ra = elo.pregame(int(row.home_id), int(row.away_id))
        snap = snapshots[key]
        # Zero is the existing ridge prior for an FBS team with no observations.
        values.append((rh, ra, snap.get(int(row.home_id), 0.0),
                       snap.get(int(row.away_id), 0.0)))
        if row.completed and pd.notna(row.margin):
            game = (int(row.home_id), int(row.away_id), float(row.margin))
            pending.append((available_at(row.start_date), season, game))
    g = g.copy()
    g[["elo_home_pre", "elo_away_pre", "adj_home", "adj_away"]] = pd.DataFrame(
        values, index=g.index, columns=["elo_home_pre", "elo_away_pre", "adj_home", "adj_away"])
    return g, snapshots, ridge.solve(current_season or 0, current), dict(elo.rating)


def build_team_stats() -> pd.DataFrame:
    """Raw team-game efficiency, paired to obtain defensive statistics.

    These statistics are not opponent-adjusted; only the ridge ratings are.
    """
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


def add_rolling_features(g: pd.DataFrame, calendar: pd.DataFrame | None = None,
                         as_of=None) -> pd.DataFrame:
    """Use the same strictly-prior postgame state for historical and live rows.

    Statistical outcomes come only from completed eligible FBS matchups. The
    calendar may include excluded opponents, solely for actual rest/openers
    and season game counts; their scores never enter these averages.
    """
    met = ["pts", "pts_allowed", "ypp_off", "ypp_def", "comp_pct", "third_down_pct",
           "third_down_def_pct", "tov_margin", "poss_min"]
    ts = build_team_stats()
    done = g[g["completed"]].set_index("id")
    ts = ts[ts["id"].isin(done.index)].copy()
    ts["date"] = ts["id"].map(done["start_date"])
    ts = ts.sort_values(["date", "team_id", "id"]).drop_duplicates(["id", "team_id"])
    prior_stats = ts if as_of is None else ts[available_at(ts["date"]) <= as_of]
    agg = prior_stats.groupby(["team_id", "season"])[met].mean().add_prefix("ps_").reset_index()
    agg = agg.rename(columns={"season": "prev_season"})

    def long(frame):
        return pd.concat([
            frame[["id", "season", "start_date", "home_id"]].rename(columns={"home_id": "team_id"}),
            frame[["id", "season", "start_date", "away_id"]].rename(columns={"away_id": "team_id"}),
        ], ignore_index=True).sort_values(["start_date", "id"])

    targets = long(g)
    calendar = g if calendar is None else calendar
    played = long(calendar[calendar["completed"]]).drop_duplicates(["id", "team_id"])
    played["season_games"] = played.groupby(["team_id", "season"]).cumcount() + 1
    stat_cols = [f"{window}_{m}" for window in ("std", "l3") for m in met]
    rows = []
    for tid, target in targets.groupby("team_id", sort=False):
        target = target.sort_values("start_date").copy()
        target["query_date"] = target["start_date"] if as_of is None else target["start_date"].clip(upper=as_of)
        history = ts[ts["team_id"] == tid].sort_values("date").copy()
        for m in met:
            # State AFTER this game's outcome; query it strictly BEFORE target kickoff.
            history[f"std_{m}"] = history[m].expanding(min_periods=1).mean()
            history[f"l3_{m}"] = history[m].rolling(3, min_periods=1).mean()
        if len(history):
            history["available_at"] = available_at(history["date"])
            target = pd.merge_asof(target, history[["available_at"] + stat_cols],
                                   left_on="query_date", right_on="available_at",
                                   direction="backward", allow_exact_matches=True)
        else:
            for col in stat_cols:
                target[col] = np.nan
        context = played[played["team_id"] == tid][["start_date", "season", "season_games"]].rename(
            columns={"start_date": "last_date", "season": "last_season"})
        if len(context):
            target = pd.merge_asof(target, context.sort_values("last_date"),
                                   left_on="query_date", right_on="last_date",
                                   direction="backward", allow_exact_matches=False)
        else:
            target["last_date"] = pd.NaT
            target["last_season"] = np.nan
            target["season_games"] = 0
        target["season_opener"] = target["season"] != target["last_season"]
        target["games_before"] = np.where(target["season_opener"], 0, target["season_games"])
        target["days_since_prev"] = (target["start_date"] - target["last_date"]).dt.days
        target["prev_season"] = target["season"] - 1
        rows.append(target)
    T = pd.concat(rows, ignore_index=True).merge(agg, on=["team_id", "prev_season"], how="left")
    cols = stat_cols + [f"ps_{m}" for m in met] + ["games_before", "days_since_prev", "season_opener"]
    for side, id_col in (("h", "home_id"), ("a", "away_id")):
        state = T[["id", "team_id"] + cols].rename(columns={c: f"{c}_{side}" for c in cols})
        g = g.merge(state, left_on=["id", id_col], right_on=["id", "team_id"],
                    how="left", validate="one_to_one").drop(columns="team_id")
        d = g[f"days_since_prev_{side}"]
        opener = g[f"season_opener_{side}"].astype(bool)
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


FEATURE_INPUT_NAMES = ("games_fbs.parquet", "games_all.parquet", "stats_all.parquet",
                       "teams.parquet", "talent_all.parquet", "sp_all.parquet", "venues.parquet")


def feature_inputs():
    paths = [DATA_DIR / name for name in FEATURE_INPUT_NAMES]
    snapshot = DATA_DIR / "source_snapshot.json"
    return paths + ([snapshot] if snapshot.exists() else [])


def build_all(save=True, as_of=None) -> pd.DataFrame:
    import json
    source_snapshot = DATA_DIR / "source_snapshot.json"
    if source_snapshot.exists() and json.loads(source_snapshot.read_text()).get("status") != "complete":
        raise ValueError("Source refresh did not finish; rerun download before building or predicting")
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if as_of.tzinfo is not None:
        as_of = as_of.tz_convert("UTC").tz_localize(None)
    g = load_games()
    g, snapshots, final_adj, final_elo = add_ratings(g, as_of=as_of)
    g = add_priors(g)
    calendar = pd.read_parquet(DATA_DIR / "games_all.parquet")
    calendar["start_date"] = pd.to_datetime(calendar["start_date"], utc=True).dt.tz_localize(None)
    g = add_rolling_features(g, calendar=calendar, as_of=as_of)
    g = add_travel_rest_context(g)
    g = finalize(g)
    keep_cols = list(dict.fromkeys(FEATURE_COLS + [
        "id", "season", "season_type", "home_classification", "away_classification",
        "start_date", "home_id", "away_id",
        "home_team", "away_team", "completed",
        "home_points", "away_points", "margin", "total",
        "elo_home_pre", "elo_away_pre", "adj_home", "adj_away"]))
    feats = g[keep_cols].copy()
    if save:
        feats.to_parquet(DATA_DIR / "features.parquet", index=False)
        write_manifest(DATA_DIR / "features.parquet", feature_inputs(),
                       scope="FBS vs FBS", rows=len(feats), as_of_utc=as_of.isoformat() + "Z")
        with open(DATA_DIR / "final_ratings.json", "w") as f:
            json.dump({"elo": {str(k): v for k, v in final_elo.items()},
                       "adj": {str(k): v for k, v in final_adj.items()},
                       "snapshots": {k: {str(a): b for a, b in v.items()} for k, v in snapshots.items()}}, f)
    return feats


if __name__ == "__main__":
    df = build_all()
    print(df.shape)
    print(df[FEATURE_COLS].isna().mean().sort_values(ascending=False).head(10))
