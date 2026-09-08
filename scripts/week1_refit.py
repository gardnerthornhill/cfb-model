"""Refit the unchanged model and record ratings weights before/after Week 1.

The before fit is a retrospective reconstruction from today's provider snapshot,
not a replacement for the original archived forecasts.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cfb_model.artifacts import code_identity, snapshot_inputs
from cfb_model.config import DATA_DIR, MODELS_DIR
from cfb_model.data.timing import available_at
from cfb_model.features.build import build_all, FEATURE_COLS, feature_inputs
from cfb_model.models.train import apply_linear_prior, fit_hybrid
from cfb_model.predict import run as forecast


def run():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    results = {}
    for label, cutoff in [('archive_cutoff_reconstruction', pd.Timestamp('2026-09-05T10:41:24.816679Z')),
                           ('after_week1', pd.Timestamp.now(tz='UTC'))]:
        feats = build_all(save=False, as_of=cutoff)
        train = feats[(feats.season >= 2014) & feats.margin.notna() &
                      (available_at(feats.start_date) <= cutoff.tz_localize(None))]
        models, prior, meta = fit_hybrid(train)
        detail = dict(cutoff_utc=cutoff.isoformat(), **meta,
                      linear_weights=dict(zip(['intercept_points', 'sp_diff', 'ridge_diff', 'elo_diff_divided_by_12'],
                                              prior.coef_.tolist())))
        if label == 'archive_cutoff_reconstruction':
            archived = pd.read_csv('data_store/out/predictions_2026_w1.csv')
            target = feats.set_index('id').loc[archived.id].reset_index()
            prior_prediction = apply_linear_prior(prior, target)
            residual = models[0].predict(target[FEATURE_COLS].to_numpy(dtype=np.float32))
            detail['max_margin_difference_from_archive'] = float(np.max(np.abs(
                prior_prediction + residual + archived.model_spread_home.to_numpy())))
            context = target[['id', 'home_team', 'away_team', 'sp_prior_diff', 'adj_margin_diff', 'elo_diff',
                              'off_ypp_h', 'off_ypp_a', 'ps_ypp_h', 'ps_ypp_a']].copy()
            context['linear_prior_margin'], context['gbm_correction'] = prior_prediction, residual
            context['reconstructed_margin'] = prior_prediction + residual
            context['archived_margin'] = -archived.model_spread_home.to_numpy()
            context.to_csv('reports/2026-09-08-week1/reconstructed_prior_context.csv', index=False)
        else:
            destination = MODELS_DIR / '2026_after_week1'
            destination.mkdir(exist_ok=True)
            for name, model in zip(['margin', 'total'], models):
                model.booster_.save_model(str(destination / (name + '.txt')))
            detail['source'] = code_identity()
            detail['inputs'] = snapshot_inputs(feature_inputs())
            (destination / 'fit.json').write_text(json.dumps(detail, indent=2) + '\n')
        results[label] = detail
        print(label, json.dumps(detail), flush=True)
    Path('reports/2026-09-08-week1/refit.json').write_text(json.dumps(results, indent=2) + '\n')
    forecast(season=2026, week=2, refresh=False)


if __name__ == '__main__':
    run()
