import pandas as pd
import pytest

from cfb_model.backtest.grade import grade_forecast, summarize


def fixture():
    p = pd.DataFrame(dict(id=[1, 2, 3], season=2026, season_type='regular', week=1,
                         home_team='Home', away_team='Away', kickoff_utc='2026-09-05T16:00:00Z',
                         issued_at_utc='2026-09-05T10:00:00Z', data_cutoff_utc='2026-09-05T09:00:00Z',
                         model_spread_home=[-7., -10., -5.], market_spread_home=[-7., -7., -3.],
                         model_pick_ats=['Home', 'Home', 'Home'], model_total=50., market_total=50.,
                         model_market_total_delta=0., home_win_prob=.7))
    g = p[['id', 'season', 'season_type', 'week', 'home_team', 'away_team']].copy()
    g['start_date'] = p['kickoff_utc']
    g['completed'] = [True, True, False]
    g['home_points'], g['away_points'] = [30, 27, 0], [20, 20, 0]
    g['home_classification'] = g['away_classification'] = 'fbs'
    return p, g


def test_pending_pushes_abstentions_and_published_rounding():
    p, g = fixture()
    r = grade_forecast(p, g)
    m = summarize(r)
    assert m['games'] == 2 and m['pending'] == 1
    assert m['ats'] == dict(wins=1, losses=0, pushes=1, abstentions=0, rate=1.)
    assert m['over_under']['abstentions'] == 2
    assert pd.isna(r.iloc[2].actual_margin)


@pytest.mark.parametrize('failure', ['late', 'duplicate', 'wrong_team', 'wrong_season', 'missing'])
def test_invalid_evaluation_is_rejected(failure):
    p, g = fixture()
    if failure == 'late':
        p.loc[0, 'issued_at_utc'] = p.loc[0, 'kickoff_utc']
    elif failure == 'duplicate':
        p.loc[1, 'id'] = 1
    elif failure == 'wrong_team':
        g.loc[0, 'home_team'] = 'Other'
    elif failure == 'wrong_season':
        g.loc[0, 'season'] = 2025
    else:
        g = g.iloc[1:]
    with pytest.raises(ValueError):
        grade_forecast(p, g)
