from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_store"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "out"
MODELS_DIR = DATA_DIR / "models"
for _d in (DATA_DIR, RAW_DIR, OUT_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")
CFBD_BASE = "https://api.collegefootballdata.com"

# History windows
FIRST_SEASON = 2004          # game results for Elo warm-up
STATS_FIRST_SEASON = 2013    # team game stats available (rolling features)
TEST_SEASONS = list(range(2019, 2026))  # walk-forward backtest window

# Elo parameters
ELO_INIT = 1500.0
ELO_K = 24.0
ELO_HOME_ADV = 55.0
ELO_REGRESS = 0.33           # fraction regressed to mean each offseason

# Ridge schedule-adjusted margin rating
RIDGE_LAMBDA = 8.0

# LightGBM
LGB_PARAMS = dict(
    objective="regression",
    metric="l1",
    learning_rate=0.02,
    num_leaves=31,
    min_child_samples=60,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    n_estimators=4000,
    verbose=-1,
)
REFIT_EVERY_WEEKS = 4
