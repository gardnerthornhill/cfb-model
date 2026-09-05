"""Conservative, shared result availability for live features and evaluation."""
import pandas as pd

from ..config import RESULT_DELAY_HOURS


def available_at(kickoff):
    return kickoff + pd.Timedelta(hours=RESULT_DELAY_HOURS)
