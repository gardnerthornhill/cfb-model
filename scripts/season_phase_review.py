"""Compare season phases in saved out-of-sample baseline predictions.

Descriptive comparisons: schedule difficulty and team composition change too.
No new model fitting or hyperparameter selection is performed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cfb_model.artifacts import sha256
from cfb_model.backtest.walk_forward import evaluate

OUTPUT = Path('reports/2026-09-08-week1')


def run():
    source = OUTPUT / 'historical_candidate_predictions.csv'
    p = pd.read_csv(source)
    p = p[(p.candidate == 'baseline') & (p.season_type == 'regular')].copy()
    p['phase'] = pd.cut(p.week, [0, 1, 4, 8, np.inf],
                        labels=['Week 1', 'Weeks 2-4', 'Weeks 5-8', 'Weeks 9+'])
    p['favorite_bias'] = (p.pred_margin - p.actual_margin) * np.sign(-p.spread_home)
    rows = []
    for sample, data in [('all_2019_2025', p), ('excluding_2020', p[p.season != 2020])]:
        for phase, group in data.groupby('phase', observed=True):
            m = evaluate(group)
            rows.append(dict(sample=sample, phase=str(phase), games=len(group),
                             margin_mae=m['margin_mae'], market_margin_mae=m['market_margin_mae'],
                             excess_margin_mae=m['margin_mae'] - m['market_margin_mae'],
                             favorite_margin_bias=float(group.favorite_bias.mean()),
                             total_mae=m['total_mae'], winner_accuracy=m['winner_accuracy'],
                             ats_wins=m['ats']['wins'], ats_losses=m['ats']['losses'],
                             ats_pushes=m['ats']['pushes'], ats_rate=m['ats']['rate']))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / 'season_phase_metrics.csv', index=False)
    p['margin_error'] = (p.pred_margin - p.actual_margin).abs()
    p['market_error'] = (p.actual_margin + p.spread_home).abs()
    p['excess_error'] = p.margin_error - p.market_error
    yearly = p.groupby(['season', 'phase'], observed=True).agg(
        games=('id', 'size'), margin_mae=('margin_error', 'mean'),
        market_mae=('market_error', 'mean'), excess_mae=('excess_error', 'mean'),
        favorite_bias=('favorite_bias', 'mean'))
    yearly.to_csv(OUTPUT / 'season_phase_by_year.csv')
    (OUTPUT / 'season_phase_provenance.json').write_text(json.dumps(dict(
        input=str(source), input_sha256=sha256(source), regular_season_games=len(p),
        method='Existing chronological out-of-sample baseline predictions; fixed CFBD week bins.',
        limitation='Different games and schedule composition; not an isolated causal estimate of added data.'
    ), indent=2) + '\n')
    print(summary.round(4).to_string(index=False))


if __name__ == '__main__':
    run()
