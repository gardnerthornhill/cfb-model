# Week 1 review: fixed experiment plan

Recorded before running the candidate backtests on September 8, 2026.
Week 1's current 34-game results have already been inspected. It is diagnostic
data, not an independent test for hypotheses suggested by these misses.

Baseline: GitHub main 988f404's FBS-only hybrid and four-calendar-week refits.
No changes to production model defaults are made for this experiment.

Two fixed football candidates, without a hyperparameter search:

1. **Recent six seasons:** fit the identical hybrid using only the six prior
   seasons plus completed current-season games. Tests whether old training
   relationships dilute current football. Inner chronological validation and
   all availability rules remain identical.
2. **Half residual:** retain the baseline linear prior but multiply its learned
   LightGBM margin correction by 0.5. Totals remain the baseline prediction.
   Tests whether nonlinear corrections are harming margin forecasts.

A fixed 50/50 model/market blend is a comparator only. Historical line timestamps
are unknown, so it cannot establish deployable or executable betting performance.
Market-only is also reported on exactly the same games.

Use 2019–2023 for development; 2024–2025 for confirmation without further
candidate changes. These seasons have been studied in earlier integrity work,
so confirmation remains retrospective, not an untouched prospective holdout.
Report every season and regular-season Week 1 separately. Do not choose only
favorable subgroups. Report paired season-bootstrap error differences.

Promotion gate: a football candidate must improve development and confirmation
margin MAE, improve both confirmation seasons, and have a confirmation
season-bootstrap interval below zero. For the recency refit, total MAE must
not worsen in either confirmation season. Otherwise retain baseline weights.
Do not increase confidence or tune team-specific factors from one opening week.

Archive the latest September 5 forecast as the primary grading sample (the
repository's published latest view). Also grade the earlier run as a sensitivity
check. This rule is adopted during this review, after outcomes occurred; no
earlier run-selection policy was recorded. Never pool model versions or fill
the 17 uncovered FBS games with predictions generated after kickoff.
