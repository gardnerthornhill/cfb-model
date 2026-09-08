"""Summarize paired historical experiments without selecting new parameters."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path('reports/2026-09-08-week1')


def run():
    p = pd.read_csv(OUT / 'historical_candidate_predictions.csv')
    baseline = p[p.candidate == 'baseline'].set_index('id')
    rng = np.random.default_rng(20260908)
    effects = []
    for name in ['recent_six', 'half_residual', 'market_blend_50']:
        candidate = p[p.candidate == name].set_index('id').loc[baseline.index]
        for scope, mask in [('development', baseline.season <= 2023), ('confirmation', baseline.season >= 2024),
                            ('all', baseline.season.notna())]:
            rows = baseline[mask]
            for target in ['margin', 'total']:
                delta = ((candidate.loc[rows.index, 'pred_' + target] - rows['actual_' + target]).abs() -
                         (rows['pred_' + target] - rows['actual_' + target]).abs())
                by_season = pd.DataFrame({'season': rows.season, 'delta': delta}).groupby('season').delta.agg(['sum', 'count'])
                indices = rng.integers(0, len(by_season), (10000, len(by_season)))
                draw = by_season['sum'].to_numpy()[indices].sum(axis=1) / by_season['count'].to_numpy()[indices].sum(axis=1)
                effects.append(dict(candidate=name, scope=scope, target=target, games=len(rows),
                                    delta_mae=float(delta.mean()), season_blocks=len(by_season),
                                    season_bootstrap_95=np.quantile(draw, [.025, .975]).tolist(),
                                    yearly_delta={str(k): float(v) for k, v in (by_season['sum'] / by_season['count']).items()}))
    (OUT / 'paired_effects.json').write_text(json.dumps(effects, indent=2) + '\n')
    summaries = pd.read_csv(OUT / 'historical_metrics.csv')
    decisions = {}
    for candidate in ['recent_six', 'half_residual']:
        relevant = [x for x in effects if x['candidate'] == candidate]
        dev = next(x for x in relevant if x['scope'] == 'development' and x['target'] == 'margin')
        confirm = next(x for x in relevant if x['scope'] == 'confirmation' and x['target'] == 'margin')
        totals = next(x for x in relevant if x['scope'] == 'confirmation' and x['target'] == 'total')
        promote = (dev['delta_mae'] < 0 and confirm['delta_mae'] < 0 and
                   max(confirm['yearly_delta'].values()) < 0 and confirm['season_bootstrap_95'][1] < 0 and
                   max(totals['yearly_delta'].values()) <= 0)
        decisions[candidate] = dict(promote=promote, development_margin_change=dev['delta_mae'],
                                   confirmation_margin_change=confirm['delta_mae'])
    (OUT / 'promotion_decisions.json').write_text(json.dumps(decisions, indent=2) + '\n')
    # Every year, including the adverse seasons and unusual COVID opening week.
    base = baseline.copy()
    base['favorite_margin_bias'] = (base.pred_margin - base.actual_margin) * np.sign(-base.spread_home)
    base['margin_abs_error'] = (base.pred_margin - base.actual_margin).abs()
    base['market_abs_error'] = (base.actual_margin + base.spread_home).abs()
    opening = base[(base.season_type == 'regular') & (base.week == 1)]
    opening_summary = opening.groupby('season').agg(games=('season', 'size'), margin_mae=('margin_abs_error', 'mean'),
                                  market_mae=('market_abs_error', 'mean'), favorite_bias=('favorite_margin_bias', 'mean'))
    opening_summary.to_csv(OUT / 'historical_week1_by_season.csv')
    yearly = summaries[summaries.scope.isin([str(x) for x in range(2019, 2026)])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for name, label, color in [('baseline', 'Current model', '#236c91'), ('recent_six', 'Six prior seasons', '#ba7440'),
                                ('half_residual', 'Half GBM correction', '#8b6a9c')]:
        data = yearly[yearly.candidate == name].sort_values('scope')
        axes[0].plot(data.scope.astype(int), data.margin_mae, '-o', label=label, color=color, ms=4)
    axes[0].axvspan(2023.5, 2025.4, color='#d7e6e9', alpha=.5)
    axes[0].set(title='No football candidate passes confirmation', xlabel='Test season', ylabel='Margin MAE (points)')
    axes[0].legend(fontsize=8)
    axes[1].bar(opening_summary.index, opening_summary.favorite_bias, color='#236c91')
    axes[1].axhline(0, color='#888888', lw=1)
    axes[1].set(title='Favorite margins are often too small in Week 1', xlabel='Test season', ylabel='Favorite margin bias (points)')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.ticklabel_format(useOffset=False, axis='x')
    fig.savefig(OUT / 'historical_validation.png', dpi=170)
    plt.close(fig)
    print(json.dumps(decisions, indent=2))
    print(json.dumps([x for x in effects if x['scope'] == 'confirmation' and x['target'] == 'margin'], indent=2))


if __name__ == '__main__':
    run()
