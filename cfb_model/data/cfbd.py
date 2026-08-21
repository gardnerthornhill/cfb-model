"""CFBD API client with local parquet caching."""
import hashlib
import json
import time

import pandas as pd
import requests

from ..config import CFBD_API_KEY, CFBD_BASE, RAW_DIR

_session = requests.Session()
_session.headers.update({"Authorization": f"Bearer {CFBD_API_KEY}"})
_last_call = 0.0


def _get(path: str, params: dict) -> list:
    global _last_call
    wait = 0.35 - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()
    r = _session.get(f"{CFBD_BASE}{path}", params=params, timeout=60)
    if r.status_code == 429:
        time.sleep(5)
        return _get(path, params)
    r.raise_for_status()
    return r.json()


def fetch_cached(endpoint_name: str, path: str, params: dict, force: bool = False) -> pd.DataFrame:
    key = hashlib.md5(json.dumps([endpoint_name, sorted(params.items())], default=str).encode()).hexdigest()[:12]
    fp = RAW_DIR / f"{endpoint_name}_{key}.parquet"
    if fp.exists() and not force:
        df = pd.read_parquet(fp)
        return _normalize(df)
    rows = _get(path, params)
    df = _normalize(pd.json_normalize(rows))
    df.to_parquet(fp, index=False)
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: _snake(c.replace(".", "_")) for c in df.columns})
    return df


def _snake(c: str) -> str:
    out = []
    for i, ch in enumerate(c):
        if ch.isupper() and i and (c[i - 1].islower() or c[i - 1] == "_"):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


# ---- typed endpoints -------------------------------------------------------

def games(season: int, force: bool = False) -> pd.DataFrame:
    return fetch_cached(f"games_{season}", "/games", {"year": season}, force=force)


def team_game_stats(season: int, force: bool = False) -> pd.DataFrame:
    return fetch_cached(f"stats_{season}", "/games/teams", {"year": season}, force=force)


def lines(season: int, force: bool = False) -> pd.DataFrame:
    return fetch_cached(f"lines_{season}", "/lines", {"year": season, "seasonType": "both"}, force=force)


def talent(season: int) -> pd.DataFrame:
    return fetch_cached(f"talent_{season}", "/talent", {"year": season})


def sp_ratings(season: int) -> pd.DataFrame:
    try:
        return fetch_cached(f"sp_{season}", "/ratings/sp", {"year": season})
    except Exception:
        return pd.DataFrame()


def returning_production(season: int) -> pd.DataFrame:
    try:
        return fetch_cached(f"retprod_{season}", "/teams/returning-production", {"year": season})
    except Exception:
        return pd.DataFrame()


def venues() -> pd.DataFrame:
    return fetch_cached("venues", "/venues", {})


def teams() -> pd.DataFrame:
    return fetch_cached("teams", "/teams", {})
