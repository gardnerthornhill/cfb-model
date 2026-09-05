"""Game-time division eligibility; never infer membership from a team's past."""
import pandas as pd


def fbs_only(games: pd.DataFrame) -> pd.DataFrame:
    """Keep matchups whose two participants are FBS in the game's season.

    CFBD classifies transitioning programs on their season's game records.
    Historical FCS games cannot permanently label NDSU, Sacramento State, etc.
    Unknown classifications fail closed instead of silently entering the model.
    """
    cols = ["home_classification", "away_classification"]
    missing = set(cols) - set(games.columns)
    if missing:
        raise ValueError(f"Missing division fields: {sorted(missing)}; refresh games")
    eligible = games[cols].apply(lambda s: s.str.lower().eq("fbs")).all(axis=1)
    return games.loc[eligible].copy()
