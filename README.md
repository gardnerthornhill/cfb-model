# cfb-model

Causal college football game predictor: LightGBM margin + total models built on
strictly pre-kickoff features (Elo, schedule-adjusted ridge ratings, rolling
team stats, SP+/talent priors, rest/travel). Walk-forward backtested against
the closing-ish market on 2019–2025.

## Results (out-of-sample walk-forward, 6,004 games)

| Metric | Value |
|---|---|
| Margin MAE | 13.42 pts |
| Total MAE | 13.66 pts |
| Winner accuracy | 72.8% |
| Brier score | 0.176 |
| Margin MAE vs posted line's MAE | 13.31 vs 12.20 |
| ATS vs posted line (FBS vs FBS) | 50.0% |

Note: CFBD lines are opening/consensus, so ATS edges vs them are an upper bound.
Earlier versions reported 11.72 / 78.2% / 0.149 and a margin MAE below the
line's; those numbers came from a prior-season stat feature that was keyed on
the previous *game's* season and so leaked the current season's full-year
averages into every non-opener row. Fixed in `features/build.py`.

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
completed game, and writes `predictions_<season>_w<week>.csv`. Columns:

| Column | Meaning |
|---|---|
| `market_spread_home` / `model_spread_home` | spread from the home team's view; **negative = home favored** |
| **`model_market_delta`** | **model spread − market spread, in points. Positive → model favors the HOME side, negative → AWAY side. Sort by this (desc or asc) to find the biggest model/market disagreements.** |
| `model_pick_ats` | the team the model likes against the spread |
| `market_total` / `model_total` / `model_market_total_delta` | totals; delta positive → lean over |
| `home_win_prob` | model win probability for the home team |
| `proj_home_score` / `proj_away_score` | projected final scores |

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
- Ratings: margin-of-victory Elo (offseason regression toward a division anchor)
  plus a multi-year weighted ridge regression of margins on team indicators with
  explicit FCS priors. FCS-vs-FCS games never update ratings — they carry no
  information about the FBS scale.
- The margin model is hybrid: a linear ridge prior over the rating diffs (SP+,
  ridge rating, Elo) captures the full range of matchup quality (GBM leaf
  averages alone compress extreme margins), and LightGBM learns corrections
  on top of the prior's residuals.
- Missing SP+/talent (FCS, new-FBS programs) is left as NaN with explicit
  flags — imputing league-average values made weak teams look average.
- Unplayed games inherit each team's most recent rolling-stat snapshot, so
  week-0/1 predictions have the same feature coverage as mid-season games.
- Win probability comes from a normal CDF on predicted margin (sigma = 17).
