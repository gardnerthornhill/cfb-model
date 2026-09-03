"""Regression tests for the causal feature builder."""
import pandas as pd
import pytest

from cfb_model.features import build


def _fixture():
    """Team 1 scores 10/20/30 in 2024 and 40/50/60 in 2025, then has one unplayed 2025 game.

    Game ids: 101-103 are 2024, 104-106 are 2025, 999 is the unplayed 2025 game.
    """
    games, stats, gid = [], [], 100
    for season, pts in ((2024, (10, 20, 30)), (2025, (40, 50, 60))):
        for i, p in enumerate(pts):
            gid += 1
            games.append(dict(id=gid, season=season, start_date=pd.Timestamp(f"{season}-09-{1 + 7 * i:02d}"),
                              home_id=1, away_id=2, completed=True))
            for tid, pf, pa in ((1, p, 3), (2, 3, p)):
                stats.append(dict(id=gid, season=season, team_id=tid, pts=pf, pts_allowed=pa, ypp_off=pf / 10,
                                  ypp_def=1.0, comp_pct=0.5, third_down_pct=0.4, third_down_def_pct=0.4,
                                  tov_margin=0, poss_min=30))
    games.append(dict(id=999, season=2025, start_date=pd.Timestamp("2025-09-22"),
                      home_id=1, away_id=2, completed=False))
    return pd.DataFrame(games), pd.DataFrame(stats)


@pytest.fixture
def rolled(monkeypatch):
    g, ts = _fixture()
    monkeypatch.setattr(build, "build_team_stats", lambda: ts)
    return build.add_rolling_features(g).set_index("id")


def test_prior_season_stats_come_from_previous_season_for_played_games(rolled):
    # 2025 games 2 and 3: the prior season is 2024, whose mean is 20. Keying on
    # the previous *game's* season instead returns the 2025 mean (50), which
    # includes games that have not been played yet.
    assert rolled.loc[[105, 106], "ps_pts_h"].tolist() == [20.0, 20.0]


def test_prior_season_stats_for_unplayed_game_use_previous_season(rolled):
    assert rolled.loc[999, "ps_pts_h"] == 20.0


def test_first_tracked_season_has_no_prior_season_stats(rolled):
    # 2024 has no 2023 stats, so ps_* must be missing rather than the 2024 mean.
    assert rolled.loc[[102, 103], "ps_pts_h"].isna().all()
