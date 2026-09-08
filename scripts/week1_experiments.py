"""Run the fixed chronological candidates in reports/2026-09-08-week1.

Run from the repository root: python -m scripts.week1_experiments
Does not overwrite production backtest artifacts or forecast archives.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from cfb_model.artifacts import code_identity, require_unchanged, snapshot_inputs
from cfb_model.backtest.walk_forward import evaluate, load_training_frame, market_lines, week_groups
from cfb_model.config import DATA_DIR, REFIT_EVERY_WEEKS
from cfb_model.data.timing import available_at
from cfb_model.features.build import FEATURE_COLS
from cfb_model.models.train import apply_linear_prior, fit_hybrid, predict_scores

OUTPUT = Path('reports/2026-09-08-week1')


def run():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    inputs = [DATA_DIR / 'features.parquet', DATA_DIR / 'lines_all.parquet']
    identity = {**code_identity(), 'inputs': snapshot_inputs(inputs)}
    df = load_training_frame().merge(market_lines(), on='id', how='left')
    outputs, fits = [], []
    models = {}
    last_date = period = None
    for (season, kind, week), grp in week_groups(df, list(range(2019, 2026))):
        cutoff = grp.start_date.min()
        if last_date is None or period != (season, kind) or week - last_date >= pd.Timedelta(weeks=REFIT_EVERY_WEEKS):
            train = df[df.margin.notna() & (available_at(df.start_date) <= cutoff)]
            for name in ['baseline', 'recent_six']:
                rows = train if name == 'baseline' else train[train.season >= season - 6]
                mm, prior, meta = fit_hybrid(rows)
                models[name] = (mm, prior)
                fits.append(dict(candidate=name, season=int(season), season_type=kind,
                                 cutoff=cutoff.isoformat(), prior_coefficients=prior.coef_.tolist(), **meta))
            last_date, period = week, (season, kind)
            print(f'{season} {kind} {week.date()}: baseline/recent fits complete', flush=True)
        base = grp[['id', 'season', 'season_type', 'week', 'start_date', 'home_team', 'away_team',
                    'home_points', 'away_points', 'spread_home', 'opener_h', 'opener_a']].copy()
        base['actual_margin'], base['actual_total'] = grp.margin, grp.total
        base['market_total'] = grp.over_under
        base['fit_cutoff_utc'] = cutoff if last_date == week else fits[-1]['cutoff']
        for name, (mm, prior) in models.items():
            out = predict_scores(mm, grp[FEATURE_COLS].to_numpy(dtype=np.float32),
                                 margin_add=apply_linear_prior(prior, grp))
            p = base.copy()
            for col, values in out.items():
                p[col] = values
            p['candidate'] = name
            outputs.append(p)
            if name == 'baseline':
                half = p.copy()
                half['pred_margin'] = .5 * p.pred_margin + .5 * apply_linear_prior(prior, grp)
                half['p_home_win'] = norm.cdf(half.pred_margin / 17)
                half['candidate'] = 'half_residual'
                outputs.append(half)
                blend = p.copy()
                blend['pred_margin'] = .5 * p.pred_margin - .5 * p.spread_home
                blend['pred_total'] = .5 * p.pred_total + .5 * p.market_total
                blend['p_home_win'] = norm.cdf(blend.pred_margin / 17)
                blend['candidate'] = 'market_blend_50'
                outputs.append(blend)
        # Save progress outside production artifacts; incomplete status is explicit.
        (OUTPUT / 'experiment_progress.json').write_text(json.dumps(dict(status='in_progress',
                    last_period=str(week), fits=len(fits)), indent=2))
    require_unchanged(inputs, identity)
    p = pd.concat(outputs, ignore_index=True)
    # Derived score columns would be stale for the ablations; only retain scored targets.
    p = p.drop(columns=['pred_home_score', 'pred_away_score'])
    p.to_csv(OUTPUT / 'historical_candidate_predictions.csv', index=False)
    (OUTPUT / 'experiment_fits.json').write_text(json.dumps(dict(identity=identity, fits=fits), indent=2))
    summaries = []
    for candidate, data in p.groupby('candidate'):
        scopes = {'development_2019_2023': data[data.season <= 2023],
                  'confirmation_2024_2025': data[data.season >= 2024],
                  'all_2019_2025': data,
                  'regular_week1': data[(data.season_type == 'regular') & (data.week == 1)]}
        scopes.update({str(s): g for s, g in data.groupby('season')})
        for scope, rows in scopes.items():
            m = evaluate(rows)
            summaries.append(dict(candidate=candidate, scope=scope,
                                  **{k: v for k, v in m.items() if not isinstance(v, dict)},
                                  ats_wins=m['ats']['wins'], ats_losses=m['ats']['losses'],
                                  ats_pushes=m['ats']['pushes'], ats_rate=m['ats']['rate']))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / 'historical_metrics.csv', index=False)
    (OUTPUT / 'experiment_progress.json').write_text(json.dumps(dict(status='complete',
                 fits=len(fits), rows=len(p)), indent=2))
    print(summary[summary.scope.isin(['development_2019_2023', 'confirmation_2024_2025', 'regular_week1'])]
          [['candidate', 'scope', 'games', 'margin_mae', 'total_mae', 'winner_accuracy', 'brier']].to_string(index=False))


if __name__ == '__main__':
    run()
