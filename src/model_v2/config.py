"""
config.py — Greyhound Prediction System v2

Central configuration for paths, constants, and environment variables.
All paths, date boundaries, and thresholds are defined here.

Authoritative source: PLAN.md
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ── Project Paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/home/matt_sent/projects/grsa")
DATA_RESULTS_DIR = Path("/media/matt_sent/vault/dishlicker_data/data/results/")
MODEL_DIR = Path("/media/matt_sent/vault/dishlicker_data/models/v2")
BETFAIR_CERT_PATH = PROJECT_ROOT / "certs"

# ── Data Split Boundaries (Strict — Never Shuffle) ─────────────────────────────
# These dates define chronological train/validation/test splits
# Test set is LOCKED until all modelling decisions are final

TRAIN_END_DATE = "2025-12-15"      # Training: Sep 30 – Dec 15, 2025 (~70%)
VAL_END_DATE = "2026-01-15"        # Validation: Dec 16 – Jan 15, 2026 (~15%)
# Test: Jan 16 – Feb 9, 2026 (~15%)

# ── Betfair API (from .env) ────────────────────────────────────────────────────

BETFAIR_API_KEY = os.getenv("BETFAIR_API_KEY")
BETFAIR_USERNAME = os.getenv("BETFAIR_USERNAME")
BETFAIR_PASSWORD = os.getenv("BETFAIR_PASSWORD")

# ── Telegram (reuse existing) ──────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ── Value Alerting Thresholds ──────────────────────────────────────────────────
# Updated based on simulation results (Step 10):
# - Profitability concentrated at short odds ($2-$10) with high edge (7%+)
# - Longer odds show model overconfidence; negative ROI at $10-$20 band
# - "Uncapped" positive ROI is driven by variance, not signal

MIN_EDGE_THRESHOLD = 0.07      # 7% minimum edge to trigger alert (was 5%)
MIN_MODEL_CONFIDENCE = 0.25    # 25% minimum model probability
ODDS_RANGE_MIN = 2.0           # Ignore below minimum odds
ODDS_RANGE_MAX = 10.0          # Ignore longshots (was 15.0)

# ── Model Training Parameters ──────────────────────────────────────────────────

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# ── Evaluation Targets (must all pass before live betting) ─────────────────────

TARGET_LOG_LOSS = 0.65          # Quality of probability estimates
TARGET_BRIER_SCORE = 0.20       # Mean squared error of probabilities
TARGET_AUC_ROC = 0.65           # Ranking: winners vs losers
TARGET_CALIBRATION_ERROR = 0.05 # Predicted 30% wins 25–35% of the time
MAX_FEATURE_IMPORTANCE = 0.30   # No single feature > 30% importance

# ── Grade Encoding ─────────────────────────────────────────────────────────────
# Normalised grade values for feature engineering

GRADE_ENCODING = {
    "M": 0,       # Maiden
    "5": 1,       # Grade 5
    "4/5": 2,     # Grade 4/5
    "4": 3,       # Grade 4
    "3/4": 4,     # Grade 3/4
    "3": 5,       # Grade 3
    "2/3": 6,     # Grade 2/3
    "2": 7,       # Grade 2
    "1": 8,       # Grade 1
    "FFA": 9,     # Free For All (highest)
    "Maiden": 0,  # Alternative naming
    "J/M": 0.5,   # Juvenile/Maiden
}

# ── Commission Rate ────────────────────────────────────────────────────────────

BETFAIR_COMMISSION = 0.05  # 5% standard Betfair commission on winnings

# ── Schedule-Based Polling ─────────────────────────────────────────────────────

# Alert timing windows (minutes before race start)
# User can modify this list to change alert timing
ALERT_WINDOWS = [45, 20, 10, 5, 2]  # Minutes before race

# Sleep notifications
SLEEP_ALERT = True  # Set False to disable end-of-day sleep messages

# Polling intervals
SCHEDULE_CHECK_INTERVAL = 60  # Seconds between schedule checks when waiting
NO_RACES_POLL_INTERVAL = 1800  # Poll every 30min when no races scheduled

# ── API Rate Limiting ──────────────────────────────────────────────────────
# Betfair enforces strict rate limits to prevent abuse
# Exceeding these limits can result in account suspension

MAX_API_CALLS_PER_HOUR = 1000   # Betfair hard limit (sliding 1-hour window)
MAX_API_CALLS_PER_DAY = 10000   # Betfair hard limit (resets at midnight)

# Warning thresholds (as percentages of max)
API_WARN_THRESHOLD_80 = 0.80    # Log warning at 80% of limit
API_WARN_THRESHOLD_90 = 0.90    # Send Telegram alert at 90% of limit
API_EMERGENCY_THRESHOLD = 1.00  # Emergency shutdown at 100% of limit

# ── Prediction Logging ─────────────────────────────────────────────────────
# Dual-output prediction logging for model analysis and ROI tracking
#
# Output 1: predictions.jsonl - ALL predictions for analysis
# Output 2: value_bets.csv - Value bets only, matching bet_results.xlsx format
#
# Directory structure:
#   /base_dir/YYYYMMDD/predictions.jsonl
#   /base_dir/YYYYMMDD/value_bets.csv

# Base directory for prediction logs
PREDICTION_LOG_DIR = Path("/media/matt_sent/vault/dishlicker_data/data/analysis/")

# Enable/disable prediction logging (set False to disable without removing code)
PREDICTION_LOGGING_ENABLED = True

# Check window labels for metadata
# Maps minutes before race to human-readable label
ALERT_WINDOW_LABELS = {
    45: "45min",
    20: "20min",
    10: "10min",
    5: "5min",
    2: "2min",
}

# ── Forbidden Feature Names ────────────────────────────────────────────────────
# If any of these strings appear in features.py, the implementation is invalid
# These must NEVER be in feature_columns.json

FORBIDDEN_FEATURE_SUBSTRINGS = [
    "betfair",
    "odds",
    "price",
    "implied",
    "market",
    "edge",
    "sp",
    "bsp",
]

# ── Verification ───────────────────────────────────────────────────────────────

def verify_paths():
    """Verify all required paths exist. Raises FileNotFoundError if not."""
    paths_to_check = [
        ("DATA_RESULTS_DIR", DATA_RESULTS_DIR),
        ("MODEL_DIR", MODEL_DIR),
        ("BETFAIR_CERT_PATH", BETFAIR_CERT_PATH),
    ]

    missing = []
    for name, path in paths_to_check:
        if not path.exists():
            missing.append(f"{name}: {path}")

    if missing:
        raise FileNotFoundError(f"Missing paths:\n" + "\n".join(missing))

    return True


def verify_env_vars():
    """Verify required environment variables are set. Returns list of missing vars."""
    # Telegram is required for alerts
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")

    # Betfair is optional until Phase 8
    betfair_missing = []
    if not BETFAIR_API_KEY:
        betfair_missing.append("BETFAIR_API_KEY")
    if not BETFAIR_USERNAME:
        betfair_missing.append("BETFAIR_USERNAME")
    if not BETFAIR_PASSWORD:
        betfair_missing.append("BETFAIR_PASSWORD")

    return missing, betfair_missing


if __name__ == "__main__":
    print("=" * 60)
    print("CONFIG VERIFICATION")
    print("=" * 60)

    # Check paths
    print("\n[Paths]")
    try:
        verify_paths()
        print(f"  DATA_RESULTS_DIR:   {DATA_RESULTS_DIR}")
        print(f"  MODEL_DIR:          {MODEL_DIR}")
        print(f"  BETFAIR_CERT_PATH:  {BETFAIR_CERT_PATH}")
        print("  All paths verified.")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")

    # Check environment variables
    print("\n[Environment Variables]")
    missing, betfair_missing = verify_env_vars()

    if missing:
        print(f"  ERROR - Missing required: {', '.join(missing)}")
    else:
        print("  TELEGRAM_BOT_TOKEN: Set")
        print("  TELEGRAM_CHAT_ID:   Set")

    if betfair_missing:
        print(f"\n  [Betfair - Optional until Phase 8]")
        print(f"  Missing: {', '.join(betfair_missing)}")
    else:
        print("  BETFAIR_API_KEY:    Set")
        print("  BETFAIR_USERNAME:   Set")
        print("  BETFAIR_PASSWORD:   Set")

    # Check data date range
    print("\n[Data Split Boundaries]")
    print(f"  Training:    Sep 30 – Dec 15, 2025")
    print(f"  Validation:  Dec 16 – Jan 15, 2026")
    print(f"  Test:        Jan 16 – Feb 9, 2026 (LOCKED)")

    print("\n[Evaluation Targets]")
    print(f"  Log Loss:         < {TARGET_LOG_LOSS}")
    print(f"  Brier Score:      < {TARGET_BRIER_SCORE}")
    print(f"  AUC-ROC:          > {TARGET_AUC_ROC}")
    print(f"  Calibration:      within {TARGET_CALIBRATION_ERROR * 100:.0f}%")
    print(f"  Max Feature Imp:  < {MAX_FEATURE_IMPORTANCE * 100:.0f}%")

    print("\n" + "=" * 60)
    print("Step 1 gate condition: All paths and env vars confirmed.")
    print("=" * 60)
