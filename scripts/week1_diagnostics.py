"""Reproduce Week 1 archive sensitivity, box-score diagnostics and figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from cfb_model.artifacts import sha256
from cfb_model.backtest.grade import grade_forecast, load_forecast, summarize
from cfb_model.data.eligibility import fbs_only

OUTPUT = Path('reports/2026-09-08-week1')
LATEST = Path('data_store/out/runs/20260905T104131379946Z_f2ffde92/predictions.csv')


def run():
    games = pd.read_parquet('data_store/games_all.parquet')
    p, _ = load_forecast(LATEST)
    p = grade_forecast(p, games)
    stats = pd.read_parquet('data_store/stats_all.parquet')
    stats = stats[stats.id.isin(p.id)].copy()
    numeric = ['totalYards', 'turnovers', 'rushingAttempts', 'rushingYards',
               'netPassingYards', 'sacks', 'defensiveTDs', 'kickReturnTDs', 'puntReturnTDs']
    for col in numeric:
        stats[col] = pd.to_numeric(stats[col], errors='coerce')
    stats['pass_attempts'] = pd.to_numeric(stats.completionAttempts.str.split('-').str[1], errors='coerce')
    stats['plays'] = stats.rushingAttempts + stats.pass_attempts
    stats['yards_per_play'] = stats.totalYards / stats.plays
    stats['third_down_pct'] = (pd.to_numeric(stats.thirdDownEff.str.split('-').str[0], errors='coerce') /
                               pd.to_numeric(stats.thirdDownEff.str.split('-').str[1], errors='coerce'))
    keep = ['id', 'team', 'points'] + numeric + ['plays', 'yards_per_play', 'third_down_pct', 'possessionTime']
    for side in ['home', 'away']:
        x = stats[stats.homeAway == side][keep].rename(columns={c: side + '_' + c for c in keep if c != 'id'})
        p = p.merge(x, on='id', how='left', validate='one_to_one', suffixes=('', '_stats'))
        if not p[side + '_team_stats'].eq(p[side + '_team']).all():
            raise ValueError('Box-score team mismatch')
        if not p[side + '_points_stats'].eq(p[side + '_points']).all():
            raise ValueError('Box-score score mismatch')
    p['ypp_diff'] = p.home_yards_per_play - p.away_yards_per_play
    p['turnover_margin_home'] = p.away_turnovers - p.home_turnovers
    p['home_score_error'] = p.proj_home_score - p.home_points
    p['away_score_error'] = p.proj_away_score - p.away_points
    p['market_favorite_margin_error'] = p.margin_error * np.sign(-p.spread_home)
    p.to_csv(OUTPUT / 'graded_games_with_stats.csv', index=False)
    p.nlargest(12, 'margin_abs_error').to_csv(OUTPUT / 'largest_misses.csv', index=False)
    groups = {'all': p, 'market_favorite_21_plus': p[p.spread_home.abs() >= 21],
              'market_favorite_under_21': p[p.spread_home.abs() < 21],
              'model_market_gap_7_plus': p[p.model_market_delta.abs() >= 7],
              'model_market_gap_under_7': p[p.model_market_delta.abs() < 7],
              'model_favors_home_ats': p[p.model_pick_ats == p.home_team],
              'model_favors_away_ats': p[p.model_pick_ats == p.away_team],
              'turnover_difference_0_or_1': p[p.turnover_margin_home.abs() <= 1],
              'turnover_difference_2_plus': p[p.turnover_margin_home.abs() >= 2]}
    group_metrics = {}
    for name, rows in groups.items():
        group_metrics[name] = {**summarize(rows),
            'favorite_margin_bias': float(rows.market_favorite_margin_error.mean()),
            'mean_home_ypp': float(rows.home_yards_per_play.mean()),
            'mean_away_ypp': float(rows.away_yards_per_play.mean())}
    sensitivity = {}
    for path in sorted(Path('data_store/out/runs').glob('*/predictions.csv')):
        forecast, _ = load_forecast(path)
        if not ((forecast.season == 2026) & (forecast.week == 1)).all():
            continue
        sensitivity[path.parent.name] = summarize(grade_forecast(forecast, games))
    legacy = pd.read_csv('data_store/out/legacy/predictions_2026_w1.csv')
    legacy_schedule = games[(games.season == 2026) & (games.season_type == 'regular') & (games.week == 1) &
                            ((games.home_classification == 'fbs') | (games.away_classification == 'fbs'))]
    legacy = legacy.merge(legacy_schedule[['id', 'season', 'week', 'home_team', 'away_team']],
                          on=['season', 'week', 'home_team', 'away_team'], how='left', validate='one_to_one')
    lg = grade_forecast(legacy, games, require_timestamps=False)
    legacy_metrics = {'all_legacy': summarize(lg), 'fbs_only_legacy': summarize(fbs_only(lg)),
                      'same_34_legacy': summarize(lg[lg.id.isin(p.id)]),
                      'caveat': 'Older model; no issue-time manifest. Never combine with current model record.'}
    covered = fbs_only(games[(games.season == 2026) & (games.season_type == 'regular') & (games.week == 1)])
    covered[~covered.id.isin(p.id)].to_csv(OUTPUT / 'not_forecast_by_current_model.csv', index=False)
    rng = np.random.default_rng(20260908)
    delta = (p.margin_abs_error - p.market_margin_abs_error).to_numpy()
    boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
    ci = binomtest(int((p.ats_result == 'win').sum()), int(p.ats_result.isin(['win', 'loss']).sum())).proportion_ci(method='wilson')
    diagnostic = dict(coverage=dict(fbs_week1_games=len(covered), current_forecasts=len(p),
                                   stats_games=stats.id.nunique()), groups=group_metrics,
                      archive_sensitivity=sensitivity, legacy=legacy_metrics,
                      uncertainty=dict(ats_wilson_95=[ci.low, ci.high],
                          paired_margin_mae_minus_market=float(delta.mean()),
                          paired_game_bootstrap_95=np.quantile(boot, [.025, .975]).tolist(),
                          caveat='One slate; game bootstrap ignores dependence and is descriptive.'),
                      score_bias=dict(home=float(p.home_score_error.mean()), away=float(p.away_score_error.mean())),
                      input_hashes={x: sha256(x) for x in [str(LATEST), 'data_store/games_all.parquet', 'data_store/stats_all.parquet']})
    (OUTPUT / 'diagnostics.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
    ax = axes[0]
    ax.scatter(p.actual_margin, p.pred_margin, color='#236c91', s=42, alpha=.85, label='Saved model forecast')
    ax.plot([-25, 60], [-25, 60], '--', color='#88949a', lw=1, label='Perfect margin prediction')
    ax.set(xlabel='Actual home margin (points)', ylabel='Predicted home margin (points)',
           title='Week 1: large wins were compressed', xlim=(-26, 61), ylim=(-26, 61))
    for name in ['LSU', 'Texas', 'Nevada', 'South Carolina', 'California']:
        row = p[p.home_team == name].iloc[0]
        ax.annotate(name, (row.actual_margin, row.pred_margin), xytext=(5, -14 if name == 'Nevada' else 7),
                    textcoords='offset points', fontsize=9)
    ax.legend(loc='upper left', fontsize=8)
    ax = axes[1]
    subset = p.nlargest(10, 'margin_abs_error').sort_values('margin_abs_error')
    labels = subset.away_team + ' at ' + subset.home_team
    y = np.arange(len(subset))
    ax.barh(y-.18, subset.margin_abs_error, height=.34, color='#236c91', label='Model error')
    ax.barh(y+.18, subset.market_margin_abs_error, height=.34, color='#dda653', label='Saved market error')
    ax.set_yticks(y, labels, fontsize=8)
    ax.set(xlabel='Absolute margin error (points)', title='Ten largest model misses')
    ax.legend(fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('September 5 archive: 34 pregame FBS forecasts', fontsize=15)
    fig.savefig(OUTPUT / 'week1_errors.png', dpi=170)
    plt.close(fig)
    print(json.dumps({k: diagnostic[k] for k in ['coverage', 'archive_sensitivity', 'legacy', 'uncertainty', 'score_bias']}, indent=2))
    print(pd.DataFrame(group_metrics).T[['games', 'margin_mae', 'margin_bias', 'favorite_margin_bias']].to_string())


if __name__ == '__main__':
    run()
