"""
Greyhound Prediction System v2

This package implements a market-agnostic prediction model for greyhound racing.
See PLAN.md for full specification.
"""

from .config import (
    # Paths
    PROJECT_ROOT,
    DATA_RESULTS_DIR,
    MODEL_DIR,
    BETFAIR_CERT_PATH,
    # Date boundaries
    TRAIN_END_DATE,
    VAL_END_DATE,
    # Betfair
    BETFAIR_API_KEY,
    BETFAIR_USERNAME,
    BETFAIR_PASSWORD,
    # Telegram
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    # Thresholds
    MIN_EDGE_THRESHOLD,
    MIN_MODEL_CONFIDENCE,
    ODDS_RANGE_MIN,
    ODDS_RANGE_MAX,
    # Model
    XGBOOST_PARAMS,
    # Targets
    TARGET_LOG_LOSS,
    TARGET_BRIER_SCORE,
    TARGET_AUC_ROC,
    TARGET_CALIBRATION_ERROR,
    MAX_FEATURE_IMPORTANCE,
    # Grade encoding
    GRADE_ENCODING,
    # Commission
    BETFAIR_COMMISSION,
    # Forbidden
    FORBIDDEN_FEATURE_SUBSTRINGS,
    # Verification
    verify_paths,
    verify_env_vars,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_RESULTS_DIR",
    "MODEL_DIR",
    "BETFAIR_CERT_PATH",
    "TRAIN_END_DATE",
    "VAL_END_DATE",
    "BETFAIR_API_KEY",
    "BETFAIR_USERNAME",
    "BETFAIR_PASSWORD",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "MIN_EDGE_THRESHOLD",
    "MIN_MODEL_CONFIDENCE",
    "ODDS_RANGE_MIN",
    "ODDS_RANGE_MAX",
    "XGBOOST_PARAMS",
    "TARGET_LOG_LOSS",
    "TARGET_BRIER_SCORE",
    "TARGET_AUC_ROC",
    "TARGET_CALIBRATION_ERROR",
    "MAX_FEATURE_IMPORTANCE",
    "GRADE_ENCODING",
    "BETFAIR_COMMISSION",
    "FORBIDDEN_FEATURE_SUBSTRINGS",
    "verify_paths",
    "verify_env_vars",
]
