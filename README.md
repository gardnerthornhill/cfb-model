# cfb-model

FBS-vs-FBS college football margin, total, and win-probability forecasts.
The hybrid model combines the existing linear ratings prior with LightGBM
corrections. The integrity update fixes data and evaluation behavior without
retuning Elo, ridge, boosting, rolling-window, or win-probability parameters.

## Scope and division membership

Both participants must be classified **FBS in that game's season**. The same
filter applies to training, backtesting, predictions, internally calculated
ratings, and statistical averages. Raw downloaded games are retained for audit.

NDSU, Sacramento State, Delaware, and other promoted programs are eligible when
their current season's CFBD game records classify them as FBS. Their historical
FCS participation does not give them permanent FCS anchors. New FBS programs
use the existing FBS defaults (1500 Elo and a zero-centered ridge prior), with
missing statistical/SP+ history left explicit. No team-specific boost is applied.
Their games against an FCS opponent are still excluded.

Actual completed schedule entries, including excluded opponents, are used only
for rest, season-openers, and current-season game counts. Their scores do not
enter our statistical averages or rating updates. Existing third-party SP+,
talent and CFBD pregame Elo inputs are retained as supplied by those providers.

## Evaluation

Same **5,252 FBS-vs-FBS games**, 2019–2025. Baseline is the leakage-corrected
September 4 audit, restricted to those exact games; final scores are identical.

| Metric | Before integrity repairs | After integrity repairs |
|---|---:|---:|
| Margin MAE | 13.31 | 12.94 |
| Total MAE | 13.69 | 13.35 |
| Winner accuracy | 69.7% | 71.1% |
| Brier score | 0.1940 | 0.1873 |
| ATS, pushes excluded | 49.87% | 49.95% |

The market margin MAE is 12.20; it still beats the model.
The combined repairs improve margin MAE in six of seven seasons; 2025 worsens
by 0.11 points. This is a retrospective comparison of the full repair package,
not an isolated estimate of each fix or a new betting edge.

[Impact report and charts](reports/2026-09-05-integrity-changes/impact.md)

Compare model and market errors on **identical games**. ATS and over/under
records exclude pushes and abstain when the model has exactly zero edge.
Stored bookmaker-median lines have no historical observation timestamps and
are not necessarily executable prices. These results do not establish closing
line value, realized betting profit, or a live trading edge.

The earlier 11.74-point margin MAE / 78% winner / 62% ATS results were inflated
by a prior-season-stat leakage defect. The August 21 prediction CSV and the old
backtest were not regenerated when that defect was fixed on September 3.
Legacy forecasts are preserved in `data_store/out/legacy/` when a new forecast
replaces their latest-view filename; they are not results of this model version.

## Setup

Python 3.11+ is supported. Direct dependency versions are pinned. Tests require
no API key and no downloaded data.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
```

Create `.env` with `CFBD_API_KEY=your_key_here`; never commit that file. Obtain a
key from [CollegeFootballData](https://collegefootballdata.com).

## Refresh, evaluate, and predict

```bash
# Refresh the current season and consolidate cached history.
python -m cfb_model.download

# Rebuild features and their provenance manifest.
python -m cfb_model.features.build

# Recreate the chronological 2019–2025 evaluation.
python -m cfb_model.backtest.walk_forward

# Use that same refreshed snapshot to forecast the next eligible week.
python -m cfb_model.predict --no-refresh

# Or refresh and explicitly target a season/week.
python -m cfb_model.predict --season 2026 --week 2
python -m cfb_model.predict --season 2026 --week 1 --season-type postseason
```

`download` refreshes current games, completed schedule weeks' statistics, lines,
talent, and the latest prior-season SP+; older seasons use the cache. It requests
actual schedule weeks, not invalid week-zero requests. API failures propagate,
rate-limit retries are bounded, and an incomplete refresh blocks feature
building. `download --no-refresh` explicitly reconsolidates cached data.

`predict` defaults to a refresh. It rejects games that have already kicked off,
even when the results cache incorrectly calls them uncompleted. Regular and
postseason week numbers are selected separately. Games that start during fitting
are removed before issuing forecasts. No historical prediction is manufactured
by invoking this CLI after the game.

## Timing and validation

- Historical and upcoming rows query the same **postgame** rolling state, so the
  latest eligible game is included once available. Upcoming features freeze at
  the forecast issue cutoff, even when kickoff is days later. Future game outcomes and
  incomplete-game box scores cannot enter a target's features.
- Because historical records have kickoff timestamps rather than verified final
  publication timestamps, results become eligible **24 hours after kickoff** in
  features and training. This is a fixed conservative information-availability
  rule, not a fitted football parameter. Recently finished games can therefore
  remain excluded until that buffer expires.
- Internally calculated Elo updates respect this availability rule. Ridge
  snapshots are distinct by season, season type, week and calendar date; bowl
  games cannot reuse an opening-week snapshot.
- Backtest groups are chronological calendar weeks. The learner refits every
  four calendar weeks and at season/season-type boundaries. Features continue
  to evolve between fits. Each prediction stores its learner's training cutoff.
- The existing 6% chronological validation tail is always used, including early
  historical folds with fewer than 500 validation rows. Same-kickoff rows stay
  together. The inner linear prior cannot see validation labels.
- After selecting tree counts, **both** the linear prior and boosting models
  refit on all eligible training rows. The outer test period remains excluded.
- Raw historical provider data were not archived at each original prediction
  time. Historical testing remains retrospective; immutable future forecasts
  are the final check on generalization.

## Output and reproducibility

Every prediction run creates a unique directory:

```text
data_store/out/runs/<UTC-run-id>/predictions.csv
                              predictions.csv.manifest.json
```

Each forecast includes game ID, season type, issue time, data cutoff and run ID.
The manifest records the code hash, Git commit/dirty state, input-file hashes,
package versions, training/validation sizes and selected tree counts. The CSV
`predictions_<season>_w<week>.csv` is a convenient **latest view**; the run archive
is the permanent record. Choose a grading run-selection rule before outcomes
are known; never select the best-looking archived run after a game.
Backtest archives and fit metadata are stored under
`data_store/backtest_runs/`.

Feature and backtest readers reject missing manifests, changed source, changed
inputs or modified artifacts. Runs refuse to label results if model code or
consumed feature/line files changed while fitting. Rebuild after such changes.
`CFB_NUM_THREADS` controls runtime concurrency (default 4), not model weights.

| Column | Meaning |
|---|---|
| `model_spread_home`, `market_spread_home` | Negative means home favored |
| `model_market_delta` | **Market spread − model spread**; positive leans home |
| `model_pick_ats` | ATS team; blank if no line or exactly zero edge |
| `model_total`, `market_total` | Predicted and market combined points |
| `model_market_total_delta` | Positive leans over |
| `home_win_prob` | Home win probability from the existing sigma=17 rule |
| `proj_home_score`, `proj_away_score` | Projected scores |

## Automation

The existing Tuesday workflow now runs tests, refreshes once, builds features,
backtests, and forecasts from that same snapshot. It uploads forecast archives
and manifests with the backtest, then commits the forecast outputs using its
existing automated workflow. Configure `CFBD_API_KEY` in GitHub Actions secrets.

The separate test workflow runs on pushes and pull requests for Python 3.11
and 3.13. No network/data credentials are required for the regression suite.

## Deliberately deferred

No new roster/QB/coach inputs, time-decay weights, opponent adjustments, FCS
strength model, betting thresholds or calibration tuning were introduced.
Lifetime averages and last-three windows retain their existing formulas; the
legacy `pace_*` fields still mean time of possession, not actual play pace.
Any future modeling change should be predeclared, tested separately across
chronological seasons, and checked on new forecasts issued before kickoff.
