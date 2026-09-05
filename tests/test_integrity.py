"""Behavioral regressions for eligibility, causal state and evaluation integrity."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cfb_model import artifacts
from cfb_model.backtest import walk_forward as wf
from cfb_model.data.eligibility import fbs_only
from cfb_model.features import build
from cfb_model.models import train
from cfb_model.predict import select_target
from tests.test_features import _fixture


def game(gid, date, home=1, away=2, season=2026, week=1, kind="regular", margin=0,
         home_class="fbs", away_class="fbs", completed=True):
    return dict(id=gid, start_date=pd.Timestamp(date), home_id=home, away_id=away,
                season=season, week=week, season_type=kind, margin=margin,
                total=50, home_classification=home_class, away_classification=away_class,
                completed=completed)


def test_promotions_are_classified_in_each_season():
    games = pd.DataFrame([
        game(1, "2025-09-01", home=2449, away=16, season=2025, home_class="fcs", away_class="fcs"),
        game(2, "2026-09-01", home=2449, away=16),  # NDSU vs Sacramento State
        game(3, "2026-09-08", home=2449, away_class="fcs"),
        game(4, "2026-09-09", home_class=None),
    ])
    assert fbs_only(games).id.tolist() == [2]


def test_missing_division_schema_cannot_enter_model():
    with pytest.raises(ValueError, match="division"):
        fbs_only(pd.DataFrame({"id": [1]}))


def test_promoted_teams_start_at_existing_fbs_prior():
    raw = pd.DataFrame([
        game(1, "2025-09-01", home=2449, away=16, season=2025, home_class="fcs", away_class="fcs", margin=70),
        game(2, "2025-09-01", home=1, away=2, season=2025),
        game(3, "2026-09-01", home=2449, away=16),
    ])
    result, _, _, _ = build.add_ratings(fbs_only(raw).reset_index(drop=True))
    promoted = result[result.id == 3].iloc[0]
    assert promoted.elo_home_pre == promoted.elo_away_pre == 1500
    assert promoted.adj_home == promoted.adj_away == 0


def test_ratings_reject_cross_division_input():
    with pytest.raises(ValueError, match="FBS vs FBS"):
        build.add_ratings(pd.DataFrame([game(1, "2026-09-01", away_class="fcs")]))


def test_postseason_week_one_does_not_reuse_opener_rating():
    g = pd.DataFrame([game(1, "2026-09-01", margin=35),
                      game(2, "2026-12-20", week=1, kind="postseason", completed=False)])
    r, snapshots, _, _ = build.add_ratings(g)
    assert len(snapshots) == 2
    assert r.iloc[0].adj_home == 0
    assert r.iloc[1].adj_home > 0


def test_ratings_do_not_use_same_day_final_results():
    g = pd.DataFrame([game(1, "2026-09-01 16:00", margin=35),
                      game(2, "2026-09-01 20:00", completed=False)])
    r, _, _, _ = build.add_ratings(g)
    assert r.iloc[1].elo_home_pre == 1500


def test_live_rolling_state_includes_latest_game(monkeypatch):
    g, ts = _fixture()
    monkeypatch.setattr(build, "build_team_stats", lambda: ts)
    r = build.add_rolling_features(g).set_index("id")
    assert r.loc[999, "std_pts_h"] == 35
    assert r.loc[999, "l3_pts_h"] == 50
    assert r.loc[104, "games_before_h"] == 0  # season count, not lifetime
    assert r.loc[999, "games_before_h"] == 3


def test_live_and_historical_features_match_and_future_outcomes_cannot_leak(monkeypatch):
    g, ts = _fixture()
    monkeypatch.setattr(build, "build_team_stats", lambda: ts)
    historical = build.add_rolling_features(g).set_index("id")
    live = g[g.id <= 105].copy()
    live.loc[live.id == 105, "completed"] = False
    observed = build.add_rolling_features(live).set_index("id")
    cols = [c for c in historical if c.startswith(("std_", "l3_", "ps_", "rest_", "games_before_"))]
    pd.testing.assert_series_equal(historical.loc[105, cols], observed.loc[105, cols])
    ts.loc[ts.id >= 105, "pts"] = 9999
    mutated = build.add_rolling_features(g).set_index("id")
    pd.testing.assert_series_equal(historical.loc[105, cols], mutated.loc[105, cols])


def test_missing_history_is_preserved_without_dropping_game(monkeypatch):
    g, ts = _fixture()
    g.loc[g.id == 999, "away_id"] = 99
    monkeypatch.setattr(build, "build_team_stats", lambda: ts)
    r = build.add_rolling_features(g).set_index("id")
    assert np.isnan(r.loc[999, "std_pts_a"])
    assert r.loc[999, "opener_a"] == 1
    assert r.loc[999, "games_before_a"] == 0


def test_excluded_opponent_changes_rest_but_not_statistical_form(monkeypatch):
    g, ts = _fixture()
    excluded = g[g.id == 106].copy()
    g = g[g.id != 106]
    calendar = pd.concat([g, excluded], ignore_index=True)
    monkeypatch.setattr(build, "build_team_stats", lambda: ts)
    r = build.add_rolling_features(g, calendar=calendar).set_index("id")
    assert r.loc[999, "rest_h"] == 7
    assert r.loc[999, "games_before_h"] == 3
    assert r.loc[999, "std_pts_h"] == 30  # excludes game 106's 60 points


def test_result_availability_is_conservative(monkeypatch):
    g, ts = _fixture()
    g.loc[g.id == 999, "start_date"] = pd.Timestamp("2025-09-15 23:00")
    monkeypatch.setattr(build, "build_team_stats", lambda: ts)
    r = build.add_rolling_features(g).set_index("id")
    assert r.loc[999, "l3_pts_h"] == 40  # game 106 not available until Sep 16


def test_future_features_freeze_at_issue_time_not_scheduled_kickoff(monkeypatch):
    g, ts = _fixture()
    monkeypatch.setattr(build, "build_team_stats", lambda: ts)
    r = build.add_rolling_features(g, as_of=pd.Timestamp("2025-09-15 23:00")).set_index("id")
    assert r.loc[999, "l3_pts_h"] == 40  # Sep 22 kickoff cannot bypass Sep 15 issue cutoff
    games = pd.DataFrame([game(1, "2026-09-01 16:00", margin=35),
                          game(2, "2026-09-08 20:00", completed=False)])
    ratings, _, _, _ = build.add_ratings(games, as_of=pd.Timestamp("2026-09-02 12:00"))
    assert ratings.iloc[1].elo_home_pre == 1500


def test_forward_week_distance_and_calendar_periods():
    keys = list(range(10))
    assert wf._weeks_between(keys, 1, 5) == 4
    assert wf._weeks_between(keys, 5, 1) == 0
    df = pd.DataFrame([game(1, "2026-08-29"), game(2, "2026-09-03"),
                       game(3, "2026-12-20", kind="postseason")])
    groups = wf.week_groups(df, [2026])
    assert len(groups) == 3
    assert [x[1].id.iloc[0] for x in groups] == [1, 2, 3]


def test_validation_keeps_simultaneous_games_together():
    frame = pd.DataFrame([game(i, f"2026-09-{1+i//2:02d}") for i in range(20)])
    _, inner, valid = train.temporal_split(frame)
    assert inner.start_date.max() < valid.start_date.min()
    assert len(valid) == 2


def test_whole_pipeline_excludes_validation_then_refits_every_row(monkeypatch):
    frame = pd.DataFrame([game(i, f"2026-09-{i+1:02d}") for i in range(20)])
    for col in build.FEATURE_COLS:
        if col not in frame:
            frame[col] = 0.
    priors, fits = [], []
    def prior(df, y):
        priors.append(df.id.tolist())
        return object()
    def fit(x, y, valid=None, n_estimators=None):
        fits.append((len(x), None if valid is None else len(valid[0]), n_estimators))
        return SimpleNamespace(best_iteration_=9)
    monkeypatch.setattr(train, "fit_linear_prior", prior)
    monkeypatch.setattr(train, "apply_linear_prior", lambda model, df: np.zeros(len(df)))
    monkeypatch.setattr(train, "_fit", fit)
    _, _, metadata = train.fit_hybrid(frame)
    assert priors == [list(range(18)), list(range(20))]
    assert fits == [(18, 2, None), (18, 2, None), (20, None, 9), (20, None, 9)]
    assert metadata["training_rows"] == 20


def test_pushes_and_zero_edges_are_neither_wins_nor_losses():
    p = pd.DataFrame(dict(pred_margin=[10, 3, 4, 10], actual_margin=[7, 5, 6, 8],
                          spread_home=[-7, -3, -3, np.nan], pred_total=[50, 50, 55, 50],
                          actual_total=[50, 53, 54, 51], market_total=[50, 50, 50, np.nan],
                          p_home_win=[.8, .6, .6, .8]))
    m = wf.evaluate(p)
    assert m["ats"] == dict(wins=1, losses=0, pushes=1, abstentions=1, rate=1.)
    assert m["over_under"] == dict(wins=1, losses=0, pushes=1, abstentions=1, rate=1.)


def test_total_does_not_require_a_spread(monkeypatch):
    rows = pd.DataFrame({"id": [1], "lines": [[{"spread": None, "overUnder": 51.5}]]})
    monkeypatch.setattr(pd, "read_parquet", lambda _: rows)
    r = wf.market_lines()
    assert r.iloc[0].over_under == 51.5
    assert np.isnan(r.iloc[0].spread_home)


def test_started_games_and_other_season_types_are_not_forecast():
    df = pd.DataFrame([game(1, "2026-09-01", completed=False),
                       game(2, "2026-09-05", completed=False),
                       game(3, "2026-12-20", kind="postseason", completed=False)])
    assert select_target(df, 2026, now="2026-09-04").id.tolist() == [2]
    assert select_target(df, 2026, week=1, season_type="postseason", now="2026-09-04").id.tolist() == [3]
    with pytest.raises(ValueError, match="No future"):
        select_target(df, 2026, now="2027-01-01")


def test_artifact_checks_detect_stale_inputs_and_modified_outputs(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.write_text("a")
    output.write_text("b")
    with pytest.raises(ValueError, match="unversioned"):
        artifacts.validate_artifact(output, [source])
    artifacts.write_manifest(output, [source])
    assert artifacts.validate_artifact(output, [source])["schema_version"] == 2
    source.write_text("changed")
    with pytest.raises(ValueError, match="Stale"):
        artifacts.validate_artifact(output, [source])
    source.write_text("a")
    output.write_text("changed")
    with pytest.raises(ValueError, match="Stale"):
        artifacts.validate_artifact(output, [source])


def test_new_run_directories_never_overwrite_prior_runs(tmp_path):
    a, b = artifacts.new_run_dir(tmp_path), artifacts.new_run_dir(tmp_path)
    assert a != b and a.is_dir() and b.is_dir()


def test_incomplete_refresh_blocks_feature_build(tmp_path, monkeypatch):
    (tmp_path / "source_snapshot.json").write_text(json.dumps({"status": "in_progress"}))
    monkeypatch.setattr(build, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="refresh did not finish"):
        build.build_all()


def test_raw_force_refresh_and_failure_preserve_cache(tmp_path, monkeypatch):
    from cfb_model.data import cfbd
    monkeypatch.setattr(cfbd, "RAW_DIR", tmp_path)
    monkeypatch.setattr(cfbd, "_get", lambda *a: [{"id": 1, "homePoints": 10}])
    cfbd.fetch_cached("test", "/games", {"year": 2026})
    monkeypatch.setattr(cfbd, "_get", lambda *a: [{"id": 1, "homePoints": 20}])
    assert cfbd.fetch_cached("test", "/games", {"year": 2026}).home_points.iloc[0] == 10
    assert cfbd.fetch_cached("test", "/games", {"year": 2026}, force=True).home_points.iloc[0] == 20
    def fail(*args):
        raise RuntimeError("network unavailable")
    monkeypatch.setattr(cfbd, "_get", fail)
    with pytest.raises(RuntimeError):
        cfbd.fetch_cached("test", "/games", {"year": 2026}, force=True)
    assert cfbd.fetch_cached("test", "/games", {"year": 2026}).home_points.iloc[0] == 20
    assert len(list(tmp_path.glob("*.metadata.json"))) == 1


def test_rate_limit_retries_are_bounded(monkeypatch):
    from cfb_model.data import cfbd
    import requests
    calls = []
    response = requests.Response()
    response.status_code = 429
    monkeypatch.setattr(cfbd._session, "get", lambda *a, **kw: calls.append(1) or response)
    monkeypatch.setattr(cfbd.time, "sleep", lambda _: None)
    with pytest.raises(requests.HTTPError):
        cfbd._get("/games", {})
    assert len(calls) == 4


def test_mid_run_input_changes_cannot_be_labeled_as_current(tmp_path):
    path = tmp_path / "input"
    path.write_text("old")
    identity = {**artifacts.code_identity(), "inputs": artifacts.snapshot_inputs([path])}
    artifacts.require_unchanged([path], identity)
    path.write_text("new")
    with pytest.raises(ValueError, match="changed during"):
        artifacts.require_unchanged([path], identity)


def test_backtest_refits_after_four_calendar_weeks_and_at_postseason(tmp_path, monkeypatch):
    rows = [game(0, "2025-09-01", season=2025), game(1, "2026-08-29"),
            game(2, "2026-09-03"), game(3, "2026-09-26", week=5),
            game(4, "2026-12-20", kind="postseason")]
    frame = pd.DataFrame(rows)
    for col in build.FEATURE_COLS:
        if col not in frame:
            frame[col] = 0.
    frame["home_team"], frame["away_team"] = "Home", "Away"
    frame["home_points"], frame["away_points"] = 25, 25
    calls = []
    def fit(t):
        calls.append(t.id.tolist())
        return (None, None), None, {"latest_training_kickoff": t.start_date.max().isoformat(),
                                    "selected_iterations": [1, 1]}
    monkeypatch.setattr(wf, "DATA_DIR", tmp_path)
    for name in ("features.parquet", "lines_all.parquet"):
        (tmp_path / name).write_bytes(b"fixture")
    monkeypatch.setattr(wf, "load_training_frame", lambda: frame)
    monkeypatch.setattr(wf, "market_lines", lambda: pd.DataFrame({"id": frame.id, "spread_home": 1., "over_under": 50.}))
    monkeypatch.setattr(wf, "fit_hybrid", fit)
    monkeypatch.setattr(wf, "apply_linear_prior", lambda prior, t: np.zeros(len(t)))
    monkeypatch.setattr(wf, "predict_scores", lambda models, x, **kw: {
        "pred_margin": np.zeros(len(x)), "pred_total": np.full(len(x), 50.), "p_home_win": np.full(len(x), .5)})
    result = wf.run([2026], verbose=False)
    assert calls == [[0], [0, 1, 2], [0, 1, 2, 3]]
    assert result.id.tolist() == [1, 2, 3, 4]
    assert (pd.to_datetime(result.latest_training_kickoff) < result.fit_cutoff_utc).all()
