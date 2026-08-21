# cfb-model

Causal college football game predictor: LightGBM margin + total models built on
strictly pre-kickoff features (Elo, schedule-adjusted ridge ratings, rolling
team stats, SP+/talent priors, rest/travel). Walk-forward backtested against
the closing-ish market on 2019–2025.

## Results (out-of-sample walk-forward, 6,004 games)

| Metric | Value |
|---|---|
| Margin MAE | 11.72 pts |
| Total MAE | 11.98 pts |
| Winner accuracy | 78.2% |
| Brier score | 0.149 |
| Margin MAE vs posted line's MAE | 11.46 vs 12.20 |

Note: CFBD lines are opening/consensus, so ATS edges vs them are an upper bound.

## Setup

Requires Python 3.13+ and a free [CFBD API key](https://collegefootballdata.com).

```bash
pip install -r requirements.txt
# .env
# CFBD_API_KEY=your_key_here
```

## Usage

All commands run from the repo root:

```bash
# one-time (cached afterwards): download all history from CFBD
python -m cfb_model.download

# rebuild the causal feature table + rating snapshots
python -m cfb_model.features.build

# walk-forward backtest report (2019-2025)
python -m cfb_model.backtest.walk_forward

# predict the next unplayed week -> CSV in data_store/out/
python -m cfb_model.predict

# or target a specific week explicitly
python -m cfb_model.predict --season 2026 --week 1

# skip the API refresh (reuse cached data)
python -m cfb_model.predict --no-refresh
```

`predict` force-refreshes current-season games/lines/stats, retrains on every
completed game, and writes `predictions_<season>_w<week>.csv` with kickoff,
market spread/total, model spread/total, predicted scores, win probability,
and the model's side vs the spread (`model_side`). Spread convention:
negative = home team favored.

Everything runs locally — GitHub Actions (below) is optional automation.

## Weekly automation (GitHub Actions)

`.github/workflows/update.yml` runs Tuesdays 14:00 UTC (10:00 ET): refreshes
data, reruns the backtest, generates the week's prediction CSV, uploads it as
an artifact, and commits it to `data_store/out/`.

First-time setup:

1. Settings → Secrets and variables → Actions → **New repository secret**
   - Name: `CFBD_API_KEY`, Value: your key
2. Actions tab → enable workflows if prompted
3. Actions → **weekly-update** → **Run workflow** for an immediate manual run;
   otherwise it fires on schedule automatically

The parquet cache (`data_store/raw/`) persists between runs via the Actions
cache, so only new weeks are downloaded each run.

## Project layout

```
cfb_model/
  config.py            paths, Elo/ridge/LightGBM hyperparameters
  download.py          bulk CFBD download + consolidation
  predict.py           weekly prediction CLI
  data/cfbd.py         API client with parquet caching
  features/
    build.py           causal feature table builder
    ratings.py         Elo engine + ridge schedule-adjusted ratings
  models/train.py      LightGBM margin/total pair + win prob
  backtest/walk_forward.py
data_store/            parquet cache, feature table, outputs (mostly gitignored)
```

## Design notes

- Every feature is computed only from games that finished before kickoff; the
  backtest retrains on a strictly expanding window.
- Ratings: margin-of-victory Elo (offseason regression toward mean) plus a
  multi-year weighted ridge regression of margins on team indicators.
- Win probability comes from a normal CDF on predicted margin (sigma = 17).
