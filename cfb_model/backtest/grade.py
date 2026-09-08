"""Grade an immutable forecast against completed results, never regenerated picks."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import manifest_path, sha256
from .walk_forward import evaluate


def grade_forecast(predictions, games, *, require_timestamps=True):
    """Match IDs and teams, verify pregame issue times, and retain pending rows.

    Uses the market lines stored in the forecast, not freshly fetched lines.
    Published ATS picks/deltas resolve rounding ambiguity in displayed spreads.
    """
    p = predictions.copy()
    if p['id'].duplicated().any() or games['id'].duplicated().any():
        raise ValueError('Duplicate game IDs in forecasts or results')
    cols = ['id', 'home_team', 'away_team', 'season', 'season_type', 'week',
            'start_date', 'completed', 'home_points', 'away_points',
            'home_classification', 'away_classification']
    p = p.merge(games[cols], on='id', how='left', suffixes=('', '_result'),
                validate='one_to_one', indicator=True)
    if (p['_merge'] != 'both').any():
        raise ValueError('Forecast games missing from results feed')
    for col in ['home_team', 'away_team', 'season', 'season_type', 'week']:
        if col + '_result' in p and not p[col].eq(p[col + '_result']).all():
            raise ValueError(f'Forecast/result mismatch: {col}')
    kickoff = pd.to_datetime(p['kickoff_utc'], utc=True)
    actual_kickoff = pd.to_datetime(p['start_date'], utc=True)
    if require_timestamps:
        issued = pd.to_datetime(p['issued_at_utc'], utc=True)
        cutoff = pd.to_datetime(p['data_cutoff_utc'], utc=True)
        if issued.isna().any() or cutoff.isna().any() or not (
                (cutoff <= issued) & (issued < kickoff) & (issued < actual_kickoff)).all():
            raise ValueError('Forecast must be issued before kickoff with a valid data cutoff')
    p['graded'] = p['completed'].eq(True) & p[['home_points', 'away_points']].notna().all(axis=1)
    p['actual_margin'] = (p['home_points'] - p['away_points']).where(p['graded'])
    p['actual_total'] = (p['home_points'] + p['away_points']).where(p['graded'])
    p['pred_margin'] = -p['model_spread_home']
    p['pred_total'] = p['model_total']
    p['p_home_win'] = p['home_win_prob']
    p['spread_home'] = p['market_spread_home']
    p['margin_error'] = p['pred_margin'] - p['actual_margin']
    p['total_error'] = p['pred_total'] - p['actual_total']
    p['margin_abs_error'] = p['margin_error'].abs()
    p['total_abs_error'] = p['total_error'].abs()
    p['market_margin_abs_error'] = (p['actual_margin'] + p['spread_home']).abs()
    p['market_total_abs_error'] = (p['actual_total'] - p['market_total']).abs()
    p['winner_hit'] = (p['pred_margin'] > 0).eq(p['actual_margin'] > 0).where(p['graded'])
    ats_side = np.sign(p['pred_margin'] + p['spread_home'])
    if 'model_pick_ats' in p:
        pick = p['model_pick_ats'].fillna('')
        if not ((pick == '') | pick.eq(p['home_team']) | pick.eq(p['away_team'])).all():
            raise ValueError('ATS pick must name a participant or abstain')
        ats_side = pd.Series(np.select([pick.eq(p['home_team']), pick.eq(p['away_team'])],
                                      [1., -1.], default=0.), index=p.index)
    total_side = np.sign(p.get('model_market_total_delta', p['pred_total'] - p['market_total']))
    for name, side, outcome, line in [
        ('ats', ats_side, p['actual_margin'] + p['spread_home'], p['spread_home']),
        ('over_under', total_side, p['actual_total'] - p['market_total'], p['market_total']),
    ]:
        p[name + '_result'] = np.select(
            [~p['graded'], line.isna(), side.isna() | np.isclose(side, 0),
             np.isclose(outcome, 0), np.sign(side).eq(np.sign(outcome))],
            ['pending', 'no_line', 'abstain', 'push', 'win'], default='loss')
    return p.drop(columns='_merge')


def summarize(p):
    done = p[p['graded']]
    if done.empty:
        return dict(forecast_games=len(p), games=0, pending=len(p))
    result = evaluate(done)
    for kind in ['ats', 'over_under']:
        counts = done[kind + '_result'].value_counts()
        wins, losses = int(counts.get('win', 0)), int(counts.get('loss', 0))
        result[kind] = dict(wins=wins, losses=losses, pushes=int(counts.get('push', 0)),
                            abstentions=int(counts.get('abstain', 0)),
                            rate=wins / (wins + losses) if wins + losses else None)
    result.update(forecast_games=len(p), pending=int((~p['graded']).sum()),
                  winner_correct=int(done['winner_hit'].sum()),
                  margin_bias=float(done['margin_error'].mean()),
                  total_bias=float(done['total_error'].mean()))
    return result


def load_forecast(path):
    """Verify the original artifact hash, without requiring today's model code.

    Archived forecasts remain valid after code/input updates. Their original
    manifests describe provenance; they must not pass current-code validation.
    """
    path = Path(path)
    manifest = json.loads(manifest_path(path).read_text())
    if manifest['artifact_sha256'] != sha256(path):
        raise ValueError('Archived forecast hash mismatch')
    return pd.read_csv(path), manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--forecast', required=True, type=Path)
    ap.add_argument('--games', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    forecast, manifest = load_forecast(args.forecast)
    graded = grade_forecast(forecast, pd.read_parquet(args.games))
    args.output.mkdir(parents=True, exist_ok=True)
    graded.to_csv(args.output / 'graded_games.csv', index=False)
    payload = dict(summary=summarize(graded), forecast=str(args.forecast),
                   forecast_sha256=sha256(args.forecast), results_sha256=sha256(args.games),
                   graded_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                   original_manifest=manifest)
    (args.output / 'grading.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload['summary'], indent=2))


if __name__ == '__main__':
    main()
