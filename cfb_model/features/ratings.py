"""Causal team rating engines: Elo (game-by-game) and ridge schedule-adjusted margins.

Everything here is strictly causal: a game only ever uses information from games
that finished before it kicked off.
"""
import numpy as np
import pandas as pd

from ..config import ELO_HOME_ADV, ELO_INIT, ELO_K, ELO_REGRESS, FCS_ELO_ANCHOR, FCS_PRIOR_WEIGHT, FCS_RIDGE_PRIOR, RIDGE_LAMBDA


class EloEngine:
    """Margin-of-victory Elo with offseason regression toward a division anchor.

    FBS teams regress toward ELO_INIT; teams in `fcs_teams` regress toward
    FCS_ELO_ANCHOR, so the cross-division gap is baked into the scale instead
    of being learned (badly) from sparse interdivision games.
    """

    def __init__(self, fcs_teams: set[int] | None = None):
        self.rating: dict[int, float] = {}
        self.fcs_teams = fcs_teams or set()
        self._season = None

    def _anchor(self, tid: int) -> float:
        return FCS_ELO_ANCHOR if tid in self.fcs_teams else ELO_INIT

    def _get(self, tid: int) -> float:
        return self.rating.get(tid, self._anchor(tid))

    def regress_offseason(self, season: int):
        if self._season is not None and season > self._season:
            for t in self.rating:
                self.rating[t] = self.rating[t] + (self._anchor(t) - self.rating[t]) * ELO_REGRESS
        self._season = season

    def pregame(self, home_id: int, away_id: int) -> tuple[float, float]:
        return self._get(home_id), self._get(away_id)

    def update(self, home_id: int, away_id: int, margin: float):
        """margin = home_points - away_points."""
        rh, ra = self.pregame(home_id, away_id)
        diff = rh + ELO_HOME_ADV - ra
        expected = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        actual = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        mov = np.log1p(abs(margin))
        k = ELO_K * (1 + mov / 12.0)
        delta = k * (actual - expected)
        self.rating[home_id] = rh + delta
        self.rating[away_id] = ra - delta


class RidgeRatings:
    """Multi-year weighted ridge regression of margins on team indicators.

    r solves  min ||W(Xr - m)||^2 + lambda*||r||^2   where X has +1 home / -1 away.
    Prior seasons enter with geometrically decaying weights; `current` holds this
    season's completed games so far. Week-0 ratings rest on prior years only.
    """

    PRIOR_WEIGHTS = {1: 0.30, 2: 0.08}

    def __init__(self):
        self._history: dict[int, list[tuple[int, int, float]]] = {}

    def add_game(self, season: int, home_id: int, away_id: int, margin: float):
        self._history.setdefault(season, []).append((home_id, away_id, margin))

    def solve(self, season: int, current: list[tuple[int, int, float]] | None = None,
              prior: dict[int, float] | None = None) -> dict[int, float]:
        """Prior: optional {team_id: mean_margin} pseudo-observations (weighted
        FCS_PRIOR_WEIGHT) that anchor teams with little FBS-scale evidence."""
        rows_h, rows_a, vals, wts = [], [], [], []
        for lag, w in self.PRIOR_WEIGHTS.items():
            for h, a, m in self._history.get(season - lag, []):
                rows_h.append(h); rows_a.append(a); vals.append(m); wts.append(w)
        for h, a, m in (current or []):
            rows_h.append(h); rows_a.append(a); vals.append(m); wts.append(1.0)
        if not rows_h:
            return {}
        teams = sorted(set(rows_h) | set(rows_a))
        idx = {t: i for i, t in enumerate(teams)}
        n_t = len(teams)
        X = np.zeros((len(rows_h), n_t))
        for g in range(len(rows_h)):
            X[g, idx[rows_h[g]]] = 1.0
            X[g, idx[rows_a[g]]] = -1.0
        if prior:
            for t, p in prior.items():
                if t in idx:
                    row = np.zeros((1, n_t))
                    row[0, idx[t]] = 1.0
                    X = np.vstack([X, row])
                    vals.append(p)
                    wts.append(FCS_PRIOR_WEIGHT)
        w = np.sqrt(np.array(wts, dtype=float))[:, None]
        A = X * w
        b = np.array(vals, dtype=float) * w.ravel()
        sol = np.linalg.solve(A.T @ A + RIDGE_LAMBDA * np.eye(n_t), A.T @ b)
        return {t: float(sol[idx[t]]) for t in teams}
