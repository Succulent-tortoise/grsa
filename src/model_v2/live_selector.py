"""
live_selector.py — Greyhound Prediction System v2

Main orchestrator for live value bet detection.

Fetches upcoming greyhound markets from Betfair, generates model predictions,
matches runners, identifies value bets, and sends alerts.

Workflow:
1. Connect to Betfair API
2. Fetch greyhound markets starting within the next 60 minutes
3. For each market:
   - Parse venue, race number, distance from market data
   - Build runner features from available data
   - Generate model predictions
   - Match runners to Betfair odds
   - Calculate edge and identify value bets
4. Send Telegram alerts for qualifying value bets (unless --dry-run)
5. Log all activity to dated log file

Error handling:
- Never crash on a single race failure
- Log errors and continue to next race
- Track failed races for reporting

Usage:
    python live_selector.py --once --dry-run    # Single run, no alerts
    python live_selector.py --once              # Single run with alerts
    python live_selector.py --continuous        # Schedule-based polling
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, List

import numpy as np
import pandas as pd
import xgboost as xgb

from betfair_client import BetfairClient, BetfairMarket, BetfairRunner, RateLimitExceeded
from config import (
    GRADE_ENCODING,
    MIN_EDGE_THRESHOLD,
    MIN_MODEL_CONFIDENCE,
    MODEL_DIR,
    ODDS_RANGE_MAX,
    ODDS_RANGE_MIN,
    PREDICTION_LOG_DIR,
    PREDICTION_LOGGING_ENABLED,
    ALERT_WINDOW_LABELS,
)
from predict import load_feature_columns, load_frozen_model
from prediction_logger import PredictionLogger, PredictionRecord, ValueBetRecord
from runner_matcher import (
    MatchResult,
    RunnerMatcher,
    extract_distance,
    extract_grade,
    extract_race_number,
    normalize_venue,
    parse_runner_info,
)
from value_alerter import ValueAlerter, ValueBet
from jsonl_loader import JSONLLoader

# ============================================================================
# VERSION TRACKING
# ============================================================================

__version__ = "1.2.1"
__version_name__ = "Session Refresh Hotfix"
__version_date__ = "2026-02-26"

# Version History:
# 1.0.0 (2026-02-15) - Initial production release
# 1.1.0 (2026-02-19) - Prediction data logging (dual output)
# 1.2.0 (2026-02-21) - Check window labeling fix + Telegram display
# 1.2.1 (2026-02-26) - Hotfix: Session refresh no longer blocked by expired session
#                      - find_safe_refresh_window falls back to schedule-based lookup on API failure
#                      - get_upcoming_markets detects 400/Bad Request as session expiry

VERSION_INFO = {
    "version": __version__,
    "name": __version_name__,
    "date": __version_date__,
    "features": [
        "Schedule-based polling (45/20/10/5/2 min windows)",
        "Prediction logging (JSONL + CSV)",
        "Value bet detection with change tracking",
        "Session auto-refresh (10h threshold)",
        "API call tracking and rate limiting",
        "Check window labeling (fixed in 1.2.0)",
    ]
}

# Configure logging to both console and file
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)


def setup_logging(dry_run: bool = False, append: bool = True) -> logging.Logger:
    """
    Set up logging to both file and console.

    Args:
        dry_run: If True, adds DRY-RUN prefix to logs
        append: If True, appends to existing log file (prevents overwrites)
               If False, creates timestamped log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("live_selector")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()
    logger.propagate = False

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Create log filename
    today = datetime.now().strftime("%Y-%m-%d")

    if append:
        # Append mode: Use simple date-based filename
        # Multiple runs on same day will append to same file
        log_file = log_dir / f"live_selector_{today}.log"
        file_mode = 'a'  # Append mode
    else:
        # Timestamp mode: Create unique file for each run
        # Useful for testing or when you want separate logs per run
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_file = log_dir / f"live_selector_{timestamp}.log"
        file_mode = 'w'  # Write mode (new file)

    # Remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File handler (append or write based on mode)
    file_handler = logging.FileHandler(log_file, mode=file_mode, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Enhanced startup messages
    mode_text = "DRY-RUN" if dry_run else "LIVE"
    append_text = "APPEND" if append and file_mode == 'a' else "NEW"

    logger.info("=" * 60)
    logger.info(f"LOGGING INITIALIZED - {mode_text} MODE ({append_text} FILE)")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"File mode: {'Appending to existing' if file_mode == 'a' else 'Creating new file'}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return logger


@dataclass
class ScheduledCheck:
    """Represents a scheduled check for a specific race."""
    market_id: str
    venue: str
    race_number: int
    race_start_time: datetime
    check_times: list[datetime]  # When to check this race
    last_checked: Optional[datetime] = None
    last_alert_data: Optional[dict] = None  # For comparison
    completed_windows: set = field(default_factory=set)  # Tracks processed window indices
    current_window_index: Optional[int] = None  # Track which window is currently active


@dataclass
class RacePrediction:
    """Predictions for a single race."""
    venue: str
    race_number: int
    race_time: str
    market_id: str
    runners: list[dict] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class LiveSelectorStats:
    """Statistics for a live selector run."""
    start_time: datetime
    end_time: Optional[datetime] = None
    markets_fetched: int = 0
    markets_within_window: int = 0
    runners_processed: int = 0
    runners_with_predictions: int = 0
    runners_with_odds: int = 0
    value_bets_found: int = 0
    alerts_sent: int = 0
    races_failed: int = 0
    errors: list[str] = field(default_factory=list)


def encode_grade(grade: str | None) -> float:
    """
    Encode grade string to numeric value.
    Higher values = better grades.
    """
    if not grade:
        return np.nan

    grade = grade.strip()

    # Direct lookup
    if grade in GRADE_ENCODING:
        return float(GRADE_ENCODING[grade])

    # Try partial matches for variations
    grade_upper = grade.upper()

    # Handle common Betfair grade formats
    if "MDN" in grade_upper or "MAIDEN" in grade_upper:
        return 0.0
    if "FFA" in grade_upper:
        return 9.0
    if "GR5" in grade_upper or "GRADE 5" in grade_upper:
        return 1.0
    if "GR4" in grade_upper or "GRADE 4" in grade_upper:
        return 3.0
    if "GR3" in grade_upper or "GRADE 3" in grade_upper:
        return 5.0
    if "GR2" in grade_upper or "GRADE 2" in grade_upper:
        return 7.0
    if "GR1" in grade_upper or "GRADE 1" in grade_upper:
        return 8.0

    # Try general partial matches
    for key, value in GRADE_ENCODING.items():
        if key.upper() in grade_upper or grade_upper in key.upper():
            return float(value)

    # Unknown grade
    return np.nan


def build_runner_features(
    runner_info: dict,
    race_context: dict,
) -> dict:
    """
    Build feature dict for a single runner from live data.

    Available from Betfair:
    - box: from runner name prefix (e.g., "1. Dog Name")
    - distance: from market name
    - grade: from market name
    - field_size: from number of runners

    Not available (set to NaN, XGBoost handles missing values):
    - ewma_position, trend_slope, form_volatility (form data)
    - days_since_last_run (historical data)
    - num_races (form data)
    - best_t_d_rank (time data)

    Args:
        runner_info: Dict with box, runner_name
        race_context: Dict with distance, grade_encoded, field_size

    Returns:
        Dict with all feature columns
    """
    features = {
        # Available from live data
        "box": runner_info.get("box", 0),
        "distance": race_context.get("distance", 0),
        "grade_encoded": race_context.get("grade_encoded", np.nan),
        "field_size": race_context.get("field_size", 0),
        # Not available in live data - XGBoost will handle NaN
        "ewma_position": np.nan,
        "trend_slope": np.nan,
        "form_volatility": np.nan,
        "days_since_last_run": np.nan,
        "num_races": 0,  # No form data available
        "best_t_d_rank": np.nan,
    }

    return features


class LiveSelector:
    """
    Main orchestrator for live value bet detection.
    """

    def __init__(
        self,
        dry_run: bool = False,
        lookahead_minutes: int = 60,
        min_edge: float = MIN_EDGE_THRESHOLD,
        odds_min: float = ODDS_RANGE_MIN,
        odds_max: float = ODDS_RANGE_MAX,
    ):
        # Helper method to generate a unique market key for schedule matching
        def _get_market_key(self, market: BetfairMarket) -> str:
            """Extract venue_rN key from market for schedule matching."""
            venue = normalize_venue(market.event_name or market.venue)
            race_number = extract_race_number(market.market_name) or 0
            return f"{venue}_r{race_number}"

        # Helper method to determine current check window based on scheduled races
        def _determine_check_window(self, scheduled_races: Optional[list[ScheduledCheck]]) -> int:
            """Determine which check window we're in (45, 20, 10, 5, or 2 minutes)."""
            from config import ALERT_WINDOWS
            if not scheduled_races:
                return 0
            now = datetime.now()
            nearest_window = 0
            min_diff = float('inf')
            for sc in scheduled_races:
                for check_time in sc.check_times:
                    diff = abs((now - check_time).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        time_to_race = (sc.race_start_time - check_time).total_seconds() / 60
                        for window in ALERT_WINDOWS:
                            if abs(time_to_race - window) < 1:
                                nearest_window = int(window)
            return nearest_window

        self.dry_run = dry_run
        self.lookahead_minutes = lookahead_minutes
        self.min_edge = min_edge
        self.odds_min = odds_min
        self.odds_max = odds_max

        self.logger = setup_logging(dry_run)

        # Log version information
        self.logger.info(f"\n[VERSION] {__version__} - {__version_name__}")
        self.logger.info(f"[VERSION] Released: {__version_date__}")
        self.logger.info(f"[VERSION] Features enabled:")
        for feature in VERSION_INFO["features"]:
            self.logger.info(f"[VERSION]   - {feature}")

        # Components
        self.betfair_client: Optional[BetfairClient] = None
        self.runner_matcher: Optional[RunnerMatcher] = None
        self.value_alerter: Optional[ValueAlerter] = None
        self.model: Optional[xgb.Booster] = None
        self.feature_columns: Optional[list[str]] = None

        # Prediction logger (dual-output: JSONL for all, CSV for value bets)
        self.prediction_logger = (
            PredictionLogger(PREDICTION_LOG_DIR)
            if PREDICTION_LOGGING_ENABLED
            else None
        )
        self.poll_number = 0  # Track poll number for JSONL metadata

        # Stats
        self.stats = LiveSelectorStats(start_time=datetime.now())

        # Schedule
        self.daily_schedule: dict[str, ScheduledCheck] = {}

        # API call counters
        self.api_call_count = 0
        self.api_call_count_daily = 0
        self.api_call_reset_date = datetime.now().date()

        # Session management
        self.session_start_time: Optional[datetime] = None
        self.last_refresh_time: Optional[datetime] = None
        self.next_refresh_time: Optional[datetime] = None
        self.refresh_threshold = 10 * 3600  # 10 hours in seconds (2hr safety margin before 12hr timeout)

    def initialize(self) -> bool:
        """
        Initialize all components and load JSONL data.
        """
        self.logger.info("=" * 60)
        self.logger.info("LIVE SELECTOR INITIALIZATION")
        self.logger.info("=" * 60)

        errors = []

        # Load model
        try:
            self.logger.info("\n[1] Loading frozen model...")
            self.model = load_frozen_model()
            self.feature_columns = load_feature_columns()
            self.logger.info(f"  Model loaded: {len(self.feature_columns)} features")
        except Exception as e:
            errors.append(f"Model loading failed: {e}")
            self.logger.error(f"  ✗ {e}")

        # Load JSONL data
        try:
            self.logger.info("\n[2] Loading JSONL data...")
            today = datetime.now().strftime('%Y%m%d')
            jsonl_loader = JSONLLoader()
            self.jsonl_data = jsonl_loader.load_daily_data(today)
            self.logger.info(f"  Loaded JSONL data for today: {today}")
        except Exception as e:
            errors.append(f"JSONL loading failed: {e}")
            self.logger.error(f"  ✗ {e}")

        # Load daily schedule
        try:
            self.logger.info("\n[2b] Building daily race schedule...")
            today = datetime.now().strftime('%Y%m%d')
            self.daily_schedule = self.build_daily_schedule(today)
            self.logger.info(f"  Loaded schedule: {len(self.daily_schedule)} races")
        except Exception as e:
            errors.append(f"Schedule building failed: {e}")
            self.logger.error(f"  ✗ {e}")

        # Initialize Betfair client
        try:
            self.logger.info("\n[3] Connecting to Betfair...")
            self.betfair_client = BetfairClient()
            self.betfair_client.login()

            # IMPORTANT: Set session start time immediately after login
            self.session_start_time = datetime.now()

            self.logger.info("  ✓ Betfair connection established")
            self.logger.info(f"  [SESSION] Started at {self.session_start_time.strftime('%I:%M %p')}")
        except Exception as e:
            errors.append(f"Betfair connection failed: {e}")
            self.logger.error(f"  ✗ {e}")

        # Initialize runner matcher
        try:
            self.logger.info("\n[4] Initializing runner matcher...")
            self.runner_matcher = RunnerMatcher()
            self.logger.info("  ✓ Runner matcher initialized")
        except Exception as e:
            errors.append(f"Runner matcher init failed: {e}")
            self.logger.error(f"  ✗ {e}")

        # Initialize value alerter
        try:
            self.logger.info("\n[5] Initializing value alerter...")
            self.value_alerter = ValueAlerter(
                min_edge=self.min_edge,
                odds_min=self.odds_min,
                odds_max=self.odds_max,
            )
            self.logger.info(f"  ✓ Value alerter initialized")
            self.logger.info(f"    - Min edge: {self.min_edge:.0%}")
            self.logger.info(f"    - Odds range: ${self.odds_min:.2f} - ${self.odds_max:.2f}")
        except Exception as e:
            errors.append(f"Value alerter init failed: {e}")
            self.logger.error(f"  ✗ {e}")

        if errors:
            self.logger.error(f"\n✗ Initialization failed with {len(errors)} error(s)")
            for err in errors:
                self.logger.error(f"  - {err}")
            return False

        self.logger.info("\n✓ All components initialized successfully")
        return True

    def check_session_age(self) -> bool:
        """
        Check if session needs refresh based on age.

        Returns:
            True if session age >= 10 hours and refresh needed
            False if session is still fresh or not started
        """
        if not self.session_start_time:
            return False

        age_seconds = (datetime.now() - self.session_start_time).total_seconds()
        age_hours = age_seconds / 3600

        if age_seconds >= self.refresh_threshold:
            self.logger.info(f"[SESSION] Age check: {age_hours:.1f}h - Refresh needed")
            return True

        return False

    def refresh_session(self) -> bool:
        """
        Refresh Betfair session by reconnecting.

        This should be called during safe windows (gaps between races).
        Updates session tracking variables on success.

        Returns:
            True if refresh successful, False if failed
        """
        start_time = datetime.now()

        try:
            self.logger.info("[SESSION] Beginning refresh...")

            # Reconnect to Betfair
            if not self.betfair_client:
                self.logger.error("[SESSION] ✗ No Betfair client initialized")
                return False

            self.betfair_client.login()

            # Calculate refresh duration
            duration = (datetime.now() - start_time).total_seconds()

            # Update session tracking
            self.session_start_time = datetime.now()
            self.last_refresh_time = datetime.now()
            self.next_refresh_time = None  # Will be rescheduled

            self.logger.info(f"[SESSION] ✓ Refresh complete ({duration:.1f} seconds)")

            return True

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"[SESSION] ✗ Refresh failed after {duration:.1f}s: {e}")

            # Schedule retry in 5 minutes
            self.next_refresh_time = datetime.now() + timedelta(minutes=5)
            self.logger.info(f"[SESSION] Retry scheduled for {self.next_refresh_time.strftime('%I:%M %p')}")

            return False

    def get_session_status(self) -> dict:
        """
        Get current session status information.

        Returns:
            Dict with session age, last refresh, next refresh times
        """
        status = {
            'session_started': None,
            'session_age_hours': 0.0,
            'last_refresh': None,
            'next_refresh': None,
        }

        if self.session_start_time:
            status['session_started'] = self.session_start_time.strftime('%I:%M %p')
            age_seconds = (datetime.now() - self.session_start_time).total_seconds()
            status['session_age_hours'] = age_seconds / 3600

        if self.last_refresh_time:
            status['last_refresh'] = self.last_refresh_time.strftime('%I:%M %p')

        if self.next_refresh_time:
            status['next_refresh'] = self.next_refresh_time.strftime('%I:%M %p')

        return status

    def build_daily_schedule(self, date_str: str) -> dict[str, ScheduledCheck]:
        """
        Build complete daily race schedule from JSONL files.

        For each race in JSONL files for the given date:
        - Calculate check times based on ALERT_WINDOWS
        - Create ScheduledCheck object
        - Return dict keyed by market_id

        Args:
            date_str: Date in YYYYMMDD format (e.g., "20260215")

        Returns:
            Dict of {market_id: ScheduledCheck}
        """
        from config import ALERT_WINDOWS
        import json
        from pathlib import Path

        schedule = {}

        try:
            # JSONL directory for this date
            jsonl_dir = Path(f"/media/matt_sent/vault/dishlicker_data/data/jsonl/{date_str}")

            if not jsonl_dir.exists():
                self.logger.warning(f"[SCHEDULE] JSONL directory not found: {jsonl_dir}")
                return schedule

            # Process each JSONL file (one per venue)
            jsonl_files = list(jsonl_dir.glob("*.jsonl"))

            if not jsonl_files:
                self.logger.warning(f"[SCHEDULE] No JSONL files found for {date_str}")
                self.logger.warning(f"[SCHEDULE] Directory: {jsonl_dir}")
                self.logger.warning(f"[SCHEDULE] System will poll for races every 30 minutes")
                return schedule

            self.logger.info(f"[SCHEDULE] Loading schedule from {len(jsonl_files)} JSONL files")

            for jsonl_file in jsonl_files:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        race_data = json.loads(line)

                        # Extract race info
                        venue = race_data.get('venue', '')
                        race_number = int(race_data.get('race_number', 0))
                        date = race_data.get('date', '')
                        time_str = race_data.get('time', '')

                        # Parse race start time
                        # Format in JSONL: "11:34am" or "2:30pm"
                        try:
                            # Convert to 24-hour format and parse
                            race_datetime_str = f"{date} {time_str}"
                            # Parse (handle am/pm)
                            time_24h = datetime.strptime(time_str, "%I:%M%p").strftime("%H:%M")
                            race_start_time = datetime.strptime(f"{date} {time_24h}", "%Y-%m-%d %H:%M")

                        except Exception as e:
                            self.logger.warning(f"[SCHEDULE] Could not parse time for {venue} R{race_number}: {e}")
                            continue

                        # Calculate check times (ALERT_WINDOWS minutes before race)
                        check_times = []
                        for minutes_before in ALERT_WINDOWS:
                            check_time = race_start_time - timedelta(minutes=minutes_before)
                            # Only schedule if in the future
                            if check_time > datetime.now():
                                check_times.append(check_time)

                        # Skip if no valid check times (race already started or too soon)
                        if not check_times:
                            continue

                        # Create market_id (we'll match this to Betfair later)
                        # For now, use venue_raceNumber as key
                        market_id = f"{venue}_r{race_number}"

                        # Create scheduled check
                        scheduled_check = ScheduledCheck(
                            market_id=market_id,
                            venue=venue,
                            race_number=race_number,
                            race_start_time=race_start_time,
                            check_times=check_times,
                        )

                        schedule[market_id] = scheduled_check

            self.logger.info(f"[SCHEDULE] Built schedule: {len(schedule)} races")

            # Log summary
            if schedule:
                first_race = min(s.race_start_time for s in schedule.values())
                last_race = max(s.race_start_time for s in schedule.values())
                self.logger.info(f"[SCHEDULE] First race: {first_race.strftime('%I:%M %p')}")
                self.logger.info(f"[SCHEDULE] Last race: {last_race.strftime('%I:%M %p')}")

                # Count total checks
                total_checks = sum(len(s.check_times) for s in schedule.values())
                self.logger.info(f"[SCHEDULE] Total checks scheduled: {total_checks}")

            return schedule

        except Exception as e:
            self.logger.error(f"[SCHEDULE] Error building schedule: {e}")
            return {}

    def get_next_check_time(self) -> Optional[datetime]:
        """
        Find the next scheduled check time across all races.

        Returns:
            datetime of next check, or None if no checks scheduled
        """
        if not self.daily_schedule:
            return None

        now = datetime.now()
        upcoming_checks = []

        for scheduled_check in self.daily_schedule.values():
            for check_time in scheduled_check.check_times:
                if check_time > now:
                    upcoming_checks.append(check_time)

        if not upcoming_checks:
            return None

        next_check = min(upcoming_checks)
        return next_check

    def get_races_to_check_now(self) -> list[ScheduledCheck]:
        """
        Get all races that need checking right now.

        A race needs checking if:
        1. Any of its check_times is within the ±120s tolerance window AND
        2. That specific window index has NOT already been completed

        Uses completed_windows set on each ScheduledCheck to prevent the
        same check window from triggering twice across consecutive runs.

        Returns:
            List of ScheduledCheck objects that need checking now
        """
        if not self.daily_schedule:
            return []

        now = datetime.now()
        to_check = []

        for scheduled_check in self.daily_schedule.values():
            for window_index, check_time in enumerate(scheduled_check.check_times):

                # Step 1: Is this check_time due? (within ±120s tolerance)
                time_diff = (now - check_time).total_seconds()
                is_due = -120 <= time_diff <= 120

                if not is_due:
                    continue

                # Step 2: Has this specific window already been processed?
                if window_index in scheduled_check.completed_windows:
                    self.logger.debug(
                        f"  [SCHEDULE] Skipping {scheduled_check.venue} R{scheduled_check.race_number} "
                        f"window {window_index} - already completed"
                    )
                    continue

                # Step 3: Not yet processed - mark as completed and add to list
                scheduled_check.completed_windows.add(window_index)
                scheduled_check.last_checked = now
                scheduled_check.current_window_index = window_index  # Store the window index

                from config import ALERT_WINDOWS
                window_minutes = ALERT_WINDOWS[window_index] if window_index < len(ALERT_WINDOWS) else 0

                self.logger.info(
                    f"  [SCHEDULE] Window {window_index} due: "
                    f"{scheduled_check.venue} R{scheduled_check.race_number} "
                    f"({window_minutes}min check)"
                )

                to_check.append(scheduled_check)
                break  # Don't add same race multiple times

        return to_check

    def cleanup_completed_races(self) -> None:
        """
        Remove races that have started from the schedule.

        Races are removed if:
        - Race start time has passed
        - All check times are in the past
        """
        now = datetime.now()
        to_remove = []

        for market_id, scheduled_check in self.daily_schedule.items():
            # Race has started
            if scheduled_check.race_start_time < now:
                to_remove.append(market_id)
                continue

            # All check times are past
            if all(check_time < now for check_time in scheduled_check.check_times):
                to_remove.append(market_id)

        for market_id in to_remove:
            del self.daily_schedule[market_id]

        if to_remove:
            self.logger.info(f"[SCHEDULE] Cleaned up {len(to_remove)} completed races")

    def find_safe_refresh_window(self) -> Optional[datetime]:
        """
        Find a safe time window for session refresh.

        Safe window criteria:
        - At least 10 minutes between races
        - Not within 5 minutes before a race starts
        - At least 2 minutes after previous race ends

        Returns:
            datetime of safe refresh time, or None if no suitable window in next 2 hours
            If no races found, returns datetime 5 minutes from now
        """
        try:
            # Get upcoming markets
            markets = self.get_upcoming_markets()

            if not markets:
                # No races - safe to refresh anytime
                safe_time = datetime.now() + timedelta(minutes=5)
                self.logger.info("[SESSION] No upcoming races - can refresh anytime")
                return safe_time

            # Extract and sort race start times (converted to local naive)
            race_times = []
            for market in markets:
                if hasattr(market, "start_time") and market.start_time:
                    # Convert to local naive for comparison with datetime.now()
                    local_time = market.start_time.astimezone().replace(tzinfo=None)
                    race_times.append(local_time)

            if not race_times:
                safe_time = datetime.now() + timedelta(minutes=5)
                return safe_time

            race_times.sort()
            now = datetime.now()

            # Look for gaps between consecutive races
            for i in range(len(race_times) - 1):
                # Gap starts 2 minutes after race i ends
                gap_start = race_times[i] + timedelta(minutes=2)
                # Gap ends 5 minutes before race i+1 starts
                gap_end = race_times[i + 1] - timedelta(minutes=5)

                # Calculate gap duration in minutes
                gap_duration_minutes = (gap_end - gap_start).total_seconds() / 60

                # Check if this gap is suitable
                if gap_start > now and gap_duration_minutes >= 10:
                    self.logger.info(
                        f"[SESSION] Found safe window: {gap_start.strftime('%I:%M %p')} "
                        f"({gap_duration_minutes:.0f} min gap)"
                    )
                    return gap_start

            # No suitable gap found in next races - schedule after last race
            last_race = race_times[-1]
            safe_time = last_race + timedelta(minutes=5)

            # Only schedule if within next 2 hours (don't schedule too far ahead)
            if safe_time < now + timedelta(hours=2):
                self.logger.info(
                    f"[SESSION] No gaps found - scheduling after last race: "
                    f"{safe_time.strftime('%I:%M %p')}"
                )
                return safe_time

            # Can't find suitable window in reasonable timeframe
            self.logger.warning("[SESSION] No suitable refresh window found in next 2 hours")
            return None

        except Exception as e:
            self.logger.error(f"[SESSION] Error finding safe window: {e}")
            # API call failed - likely session already expired.
            # Fall back to schedule-based window lookup so refresh isn't permanently blocked.
            self.logger.info("[SESSION] Falling back to schedule-based safe window search")
            return self._find_safe_window_from_schedule()

    def _find_safe_window_from_schedule(self) -> Optional[datetime]:
        """
        Fallback safe window finder using local schedule (no API call needed).

        Used when the Betfair session is already expired and we can't call
        listMarketCatalogue to find race times. Reads self.daily_schedule instead.

        Returns:
            datetime of safe refresh time, or datetime 2 minutes from now if no schedule
        """
        now = datetime.now()

        if not self.daily_schedule:
            # No schedule at all - safe to refresh now
            self.logger.info("[SESSION] No schedule found - refreshing immediately")
            return now + timedelta(minutes=2)

        # Collect upcoming race start times from the local schedule
        race_times = sorted(
            sc.race_start_time
            for sc in self.daily_schedule.values()
            if sc.race_start_time > now
        )

        if not race_times:
            # No upcoming races in schedule - safe to refresh
            self.logger.info("[SESSION] No upcoming races in schedule - refreshing immediately")
            return now + timedelta(minutes=2)

        # Check if we're currently in a safe gap
        next_race = race_times[0]
        minutes_to_next = (next_race - now).total_seconds() / 60

        if minutes_to_next >= 5:
            # Safe to refresh now (more than 5 mins to next race)
            safe_time = now + timedelta(minutes=1)
            self.logger.info(
                f"[SESSION] Schedule-based window: refresh in 1 min "
                f"({minutes_to_next:.0f} min until next race)"
            )
            return safe_time

        # Too close to next race - look for a gap after it
        for i in range(len(race_times) - 1):
            gap_start = race_times[i] + timedelta(minutes=2)
            gap_end = race_times[i + 1] - timedelta(minutes=5)
            gap_minutes = (gap_end - gap_start).total_seconds() / 60
            if gap_start > now and gap_minutes >= 5:
                self.logger.info(
                    f"[SESSION] Schedule-based window: {gap_start.strftime('%I:%M %p')} "
                    f"({gap_minutes:.0f} min gap)"
                )
                return gap_start

        # No gap found - schedule after last race
        safe_time = race_times[-1] + timedelta(minutes=5)
        self.logger.info(
            f"[SESSION] Schedule-based window: after last race at "
            f"{safe_time.strftime('%I:%M %p')}"
        )
        return safe_time

    def is_safe_to_refresh_now(self) -> bool:
        """
        Check if current time is safe for immediate refresh.

        Safe if:
        - No races starting in next 5 minutes
        - No races that started in last 2 minutes

        Returns:
            True if safe to refresh immediately, False otherwise
        """
        try:
            markets = self.get_upcoming_markets()

            if not markets:
                return True  # No races, always safe

            now = datetime.now()

            for market in markets:
                if not hasattr(market, "start_time") or not market.start_time:
                    continue

                # Convert to local naive for comparison
                local_start = market.start_time.astimezone().replace(tzinfo=None)
                time_until_race = (local_start - now).total_seconds() / 60

                # Too close to race start
                if 0 < time_until_race < 5:
                    self.logger.info(
                        f"[SESSION] Cannot refresh - race in {time_until_race:.1f} minutes"
                    )
                    return False

                # Race just started
                if -2 < time_until_race < 0:
                    self.logger.info("[SESSION] Cannot refresh - race just started")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"[SESSION] Error checking safety: {e}")
            return False  # Err on side of caution

    def get_upcoming_markets(self) -> list[BetfairMarket]:
        """
        Fetch greyhound markets starting within the lookahead window.
        Includes session error handling and automatic retry on expiry.

        Returns:
            List of BetfairMarket objects
        """
        if not self.betfair_client:
            raise RuntimeError("Betfair client not initialized")

        # Increment API call counters
        self.api_call_count += 1

        # Reset daily counter at midnight
        if datetime.now().date() != self.api_call_reset_date:
            self.logger.info(f"[API] Yesterday's calls: {self.api_call_count_daily}")
            self.api_call_count_daily = 0
            self.api_call_reset_date = datetime.now().date()

        self.api_call_count_daily += 1

        # Log every 10 calls
        if self.api_call_count % 10 == 0:
            self.logger.info(f"[API] Total calls today: {self.api_call_count_daily}")

        self.logger.info(f"\n[MARKETS] Fetching markets")
        self.logger.info(f"[MARKETS]   Lookahead: {self.lookahead_minutes} minutes")

        try:
            # Fetch all Australian greyhound markets
            all_markets = self.betfair_client.list_greyhound_markets(max_results=100)
        except Exception as e:
            error_msg = str(e).lower()

            # Detect session expiry errors.
            # 400 Bad Request is returned by Betfair when the session token has fully expired,
            # so we include it here alongside the standard session-related keywords.
            is_session_error = (
                'session' in error_msg
                or 'invalid' in error_msg
                or 'token' in error_msg
                or '400' in error_msg
                or 'bad request' in error_msg
            )
            if is_session_error:
                self.logger.error(f"[SESSION] Session expired during API call: {e}")

                # Try immediate refresh if safe
                if self.is_safe_to_refresh_now():
                    self.logger.info("[SESSION] Attempting immediate refresh...")
                    if self.refresh_session():
                        # Retry the API call
                        all_markets = self.betfair_client.list_greyhound_markets(max_results=100)
                    else:
                        raise
                else:
                    self.logger.warning("[SESSION] Cannot refresh now - not safe (race proximity)")
                    raise
            else:
                # Re-raise if not a session error
                raise

        self.stats.markets_fetched = len(all_markets)
        self.logger.info(f"[MARKETS]   Total found: {len(all_markets)}")

        # Filter to markets starting within our time window
        now = datetime.now(timezone.utc)
        window_end = now + timedelta(minutes=self.lookahead_minutes)

        upcoming = []
        for market in all_markets:
            try:
                # Parse market start time (ISO format with Z suffix)
                start_time_str = market.market_start_time
                if start_time_str.endswith("Z"):
                    start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                else:
                    start_time = datetime.fromisoformat(start_time_str)
                    if start_time.tzinfo is None:
                        start_time = start_time.replace(tzinfo=timezone.utc)

                # Check if within window (must be in future, within lookahead)
                if now <= start_time <= window_end:
                    market.start_time = start_time
                    upcoming.append(market)
                elif start_time > now - timedelta(minutes=5) and start_time < now:
                    # Include races that started in last 5 minutes (may still have markets open)
                    market.start_time = start_time
                    upcoming.append(market)

            except Exception as e:
                self.logger.warning(f"  Could not parse start time for {market.market_name}: {e}")

        self.stats.markets_within_window = len(upcoming)
        self.logger.info(f"[MARKETS]   Within window: {len(upcoming)}")

        return upcoming

    def process_market(self, market: BetfairMarket, check_window: int = 0) -> RacePrediction:
        """
        Process a single market: generate predictions and identify value bets.

        Args:
            market: BetfairMarket to process
            check_window: Minutes before race (45, 20, 10, 5, 2) for logging

        Returns:
            RacePrediction with runner predictions
        """
        prediction = RacePrediction(
            venue=normalize_venue(market.venue or market.event_name),
            race_number=extract_race_number(market.market_name) or 0,
            race_time=market.market_start_time,
            market_id=market.market_id,
        )

        try:
            # Update market with current odds
            self.betfair_client.update_market_with_odds(market)

            # Extract race context
            distance = extract_distance(market.market_name) or 0
            grade = extract_grade(market.market_name)
            grade_encoded = encode_grade(grade)
            field_size = len(market.runners)

            race_context = {
                "distance": distance,
                "grade_encoded": grade_encoded,
                "field_size": field_size,
            }

            self.logger.info(f"\n[RACE] {prediction.venue} R{prediction.race_number}")
            self.logger.info(f"[RACE]   Distance: {distance}m | Grade: {grade} | Field: {field_size}")

            # Process each runner
            for runner in market.runners:
                self.stats.runners_processed += 1

                # Parse runner info
                parsed = parse_runner_info(runner)
                runner_info = {
                    "box": parsed.box_number,
                    "runner_name": parsed.runner_name_normalized,
                }

                # Build features
                features = build_runner_features(runner_info, race_context)

                # Fetch form features using form_lookup
                df = pd.DataFrame([features])

                # Generate prediction
                try:
                    dmatrix = xgb.DMatrix(
                        df[self.feature_columns].values,
                        feature_names=self.feature_columns,
                    )
                    model_proba = float(self.model.predict(dmatrix)[0])
                    self.stats.runners_with_predictions += 1
                except Exception as e:
                    self.logger.warning(f"    Prediction failed for {parsed.runner_name_normalized}: {e}")
                    continue

                # Get odds
                back_odds = runner.back_odds
                if back_odds:
                    self.stats.runners_with_odds += 1

                # Build runner result
                runner_result = {
                    "venue": prediction.venue,
                    "race_number": prediction.race_number,
                    "race_time": prediction.race_time,
                    "runner_name": parsed.runner_name_normalized,
                    "box": parsed.box_number,
                    "model_proba": model_proba,
                    "back_odds": back_odds,
                    "selection_id": runner.selection_id,
                }

                # Calculate edge if odds available
                if back_odds and back_odds > 1:
                    implied = 1.0 / back_odds
                    edge = model_proba - implied
                    runner_result["implied_proba"] = implied
                    runner_result["edge"] = edge

                    # Check if value bet
                    if self.value_alerter.is_value_bet(model_proba, back_odds):
                        runner_result["is_value_bet"] = True
                        self.stats.value_bets_found += 1

                        self.logger.info(
                            f"[VALUE BET] ★ {parsed.runner_name_normalized} "
                            f"(Box {parsed.box_number}) "
                            f"| Model: {model_proba:.1%} | Odds: ${back_odds:.2f} | Edge: +{edge:.1%}"
                        )
                    else:
                        runner_result["is_value_bet"] = False
                else:
                    runner_result["is_value_bet"] = False

                prediction.runners.append(runner_result)

                # Store additional metadata for logging
                runner_result["distance"] = distance
                runner_result["grade"] = grade
                runner_result["field_size"] = field_size
                runner_result["race_start_time"] = market.market_start_time
                runner_result["check_window"] = check_window  # Store check window for alerts
                runner_result["meets_min_edge"] = (
                    edge >= self.min_edge if (back_odds and back_odds > 1 and "edge" in runner_result) else False
                )
                runner_result["meets_min_confidence"] = model_proba >= MIN_MODEL_CONFIDENCE
                runner_result["meets_odds_range"] = (
                    (back_odds and self.odds_min <= back_odds <= self.odds_max)
                    if back_odds else False
                )

        except Exception as e:
            prediction.error = str(e)
            self.stats.races_failed += 1
            self.stats.errors.append(f"{prediction.venue} R{prediction.race_number}: {e}")
            self.logger.error(f"  ✗ Failed to process market: {e}")

        # Log ALL predictions to JSONL for analysis
        if self.prediction_logger and prediction.runners:
            prediction_records = self._build_prediction_records(prediction, check_window)
            self.prediction_logger.log_all_predictions(prediction_records)
            self.logger.info(f"[PREDICTION LOG] Logged {len(prediction_records)} predictions to JSONL")

        return prediction

    # ========================================================================
    # Prediction Logging Methods
    # ========================================================================

    def _build_prediction_records(
        self,
        prediction: RacePrediction,
        check_window: int
    ) -> list[PredictionRecord]:
        """
        Convert race prediction to PredictionRecord format for JSONL logging.

        Args:
            prediction: RacePrediction object with all runners
            check_window: Minutes before race (45, 20, 10, 5, 2)

        Returns:
            List of PredictionRecord objects for JSONL logging
        """
        records = []

        for runner_data in prediction.runners:
            # Get check window label
            window_label = ALERT_WINDOW_LABELS.get(check_window, "unknown")

            # Calculate minutes to start
            race_time = runner_data.get("race_start_time", prediction.race_time)

            # Handle race_time being a string or datetime
            race_time_str_for_json = str(race_time)  # Default to string
            if isinstance(race_time, str):
                # Parse string to datetime for calculation
                try:
                    race_time = datetime.fromisoformat(race_time.replace('Z', '+00:00'))
                    race_time_str_for_json = race_time.isoformat()
                except:
                    race_time = datetime.now()  # Fallback
            elif hasattr(race_time, 'isoformat'):
                race_time_str_for_json = race_time.isoformat()

            now = datetime.now()
            if hasattr(race_time, 'tzinfo') and race_time.tzinfo is not None:
                now = now.replace(tzinfo=race_time.tzinfo)
            minutes_to_start = (race_time - now).total_seconds() / 60

            record = PredictionRecord(
                timestamp=datetime.now().isoformat(),
                poll_number=self.poll_number,
                check_window=window_label,
                market_id=prediction.market_id,
                venue=prediction.venue,
                race_number=prediction.race_number,
                race_time=race_time_str_for_json,
                minutes_to_start=minutes_to_start,
                distance=runner_data.get("distance", 0),
                grade=str(runner_data.get("grade", "")),
                field_size=runner_data.get("field_size", 0),
                runner_name=runner_data.get("runner_name", ""),
                box=runner_data.get("box", 0),
                model_probability=runner_data.get("model_proba", 0.0),
                back_odds=runner_data.get("back_odds", 0.0),
                implied_probability=runner_data.get("implied_proba", 0.0),
                edge=runner_data.get("edge", 0.0),
                is_value_bet=runner_data.get("is_value_bet", False),
                meets_min_edge=runner_data.get("meets_min_edge", False),
                meets_min_confidence=runner_data.get("meets_min_confidence", False),
                meets_odds_range=runner_data.get("meets_odds_range", False),
                alert_sent=False  # Will be updated after alert sending
            )
            records.append(record)

        return records

    def _build_value_bet_record(
        self,
        runner_data: dict,
        prediction: RacePrediction
    ) -> ValueBetRecord:
        """
        Convert runner data to ValueBetRecord format for CSV logging.

        Formats data to match bet_results.xlsx structure exactly.

        Args:
            runner_data: Dictionary with runner prediction data
            prediction: RacePrediction object with race metadata

        Returns:
            ValueBetRecord object for CSV logging
        """
        race_time = runner_data.get("race_start_time", prediction.race_time)

        # Format venue name (title case with spaces)
        venue_display = prediction.venue.replace("_", " ").replace("-", " ").title()

        # Format race time
        if hasattr(race_time, 'strftime'):
            race_time_str = race_time.strftime("%H:%M")
            race_date_str = race_time.strftime("%d/%m/%Y")
        else:
            race_time_str = str(race_time)
            race_date_str = str(datetime.now().date())

        return ValueBetRecord(
            date=race_date_str,
            time=race_time_str,
            venue=venue_display,
            race=f"R{prediction.race_number}",
            race_time=race_time_str,
            runner=runner_data.get("runner_name", ""),
            box=runner_data.get("box", 0),
            model_prob=f"{runner_data.get('model_proba', 0.0)*100:.2f}%",
            back_odds=f"${runner_data.get('back_odds', 0.0):.2f}",
            implied_prob="",  # Leave empty per user's sheet
            edge=f"{runner_data.get('edge', 0.0)*100:.2f}%",
            alert_time="",  # Will be set by logger
            notes=""
        )

    def _determine_check_window(self, scheduled_races: Optional[list[ScheduledCheck]] = None) -> int:
        """
        DEPRECATED: This method returns the first window found, which causes
        incorrect labels when multiple races are processed together.

        Use _get_window_for_race() instead for per-race window lookup.

        Determine which check window we're in using the stored window index.

        Previously this method tried to calculate the window based on time-to-race,
        but that was unreliable. Now we use the window_index that was set when
        get_races_to_check_now() identified the race as due for checking.

        Args:
            scheduled_races: List of races scheduled for checking now

        Returns:
            Minutes before race (45, 20, 10, 5, or 2), or 0 if unknown
        """
        from config import ALERT_WINDOWS

        # First try: use scheduled_races if provided
        races_to_check = scheduled_races

        # Second try: fallback to daily_schedule
        if not races_to_check and self.daily_schedule:
            races_to_check = list(self.daily_schedule.values())

        if not races_to_check:
            return 0

        # Get the window index from any scheduled race
        # (They should all have the same window if checked together)
        now = datetime.now()
        for scheduled_check in races_to_check:
            if scheduled_check.current_window_index is not None:
                window_idx = scheduled_check.current_window_index
                if window_idx < len(ALERT_WINDOWS):
                    return ALERT_WINDOWS[window_idx]

            # Also check if any check_time is near now
            for window_index, check_time in enumerate(scheduled_check.check_times):
                time_diff = abs((now - check_time).total_seconds())
                if time_diff < 120:  # Within 2 minutes
                    if window_index < len(ALERT_WINDOWS):
                        return ALERT_WINDOWS[window_index]

        # Fallback: return 0 (unknown)
        self.logger.warning("[SCHEDULE] Could not determine check window - using 0")
        return 0

    def _get_window_for_race(
        self,
        market: BetfairMarket,
        scheduled_races: Optional[list[ScheduledCheck]] = None
    ) -> int:
        """
        Get the check window for a specific race.

        When multiple races are checked together (e.g., bendigo R1 at 10min,
        geelong R5 at 45min), we need to look up the window for THIS specific
        race, not just grab the first window we find.

        Args:
            market: The Betfair market for this race
            scheduled_races: List of races scheduled for checking now

        Returns:
            Minutes before race (45, 20, 10, 5, or 2), or 0 if unknown
        """
        from config import ALERT_WINDOWS

        # Fallback to daily_schedule if no scheduled_races provided (ONCE mode)
        races_to_check = scheduled_races
        if not races_to_check and self.daily_schedule:
            races_to_check = list(self.daily_schedule.values())

        if not races_to_check:
            return 0

        # Extract venue and race number from this market
        venue_norm = self._normalize_venue(market.venue or market.event_name)
        race_num = self._extract_race_number(market.market_name)

        if not race_num:
            return 0

        # Find the matching scheduled check for THIS specific race
        now = datetime.now()
        for scheduled_check in races_to_check:

            if (scheduled_check.venue == venue_norm and
                scheduled_check.race_number == race_num):

                # Found it! First try current_window_index (set in CONTINUOUS mode)
                if scheduled_check.current_window_index is not None:
                    window_idx = scheduled_check.current_window_index
                    if window_idx < len(ALERT_WINDOWS):
                        return ALERT_WINDOWS[window_idx]

                # Fallback: check if any check_time is near now (ONCE mode)
                for window_index, check_time in enumerate(scheduled_check.check_times):
                    time_diff = abs((now - check_time).total_seconds())
                    if time_diff < 120:  # Within 2 minutes
                        if window_index < len(ALERT_WINDOWS):
                            return ALERT_WINDOWS[window_index]

                # If we found the race but couldn't determine window from check_times,
                # calculate window based on time-to-race
                if scheduled_check.race_start_time:
                    minutes_to_race = (scheduled_check.race_start_time - now).total_seconds() / 60

                    # Find the closest ALERT_WINDOW to the time-to-race
                    if minutes_to_race > 0:
                        # Find the closest window
                        closest_window = min(ALERT_WINDOWS, key=lambda w: abs(w - minutes_to_race))
                        # Only use if within reasonable range (within 10 minutes of a window)
                        if abs(closest_window - minutes_to_race) <= 10:
                            return closest_window

                # If we found the race but couldn't determine window, still return 0
                break

        # Fallback: couldn't find this race in scheduled list
        self.logger.warning(
            f"[SCHEDULE] Could not find window for {venue_norm} R{race_num}"
        )
        return 0

    def _normalize_venue(self, venue: str) -> str:
        """
        Normalize venue name for matching.

        Uses the same normalization as the schedule keys to ensure matching.
        The schedule stores raw venue names from JSONL, so we need to also
        normalize market venue names the same way.

        Args:
            venue: Raw venue name

        Returns:
            Normalized venue
        """
        if not venue:
            return ""

        # Use the imported normalize_venue from runner_matcher
        # This ensures consistent matching between markets and schedule
        return normalize_venue(venue)

    def _extract_race_number(self, market_name: str) -> Optional[int]:
        """
        Extract race number from market name.

        Args:
            market_name: Betfair market name (e.g., "R5 450m Gr5")

        Returns:
            Race number as integer, or None if not found
        """
        import re
        match = re.search(r'R(\d+)', market_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def compare_alerts(
        self,
        current: dict,
        previous: Optional[dict]
    ) -> tuple[str, str]:
        """
        Compare current alert with previous to determine header and details.

        Args:
            current: Current runner data with odds, edge, etc.
            previous: Previous alert data (or None if first alert)

        Returns:
            tuple of (header_emoji, change_text)

        Examples:
            ("🆕", "") - First detection
            ("📈", "Odds: $11.50 (+$1.50) | Edge: +16.1% (+0.9pp)")
            ("📉", "Odds: $9.00 (-$1.00) | Edge: +14.0% (-1.2pp)")
        """
        if not previous:
            return ("🆕 VALUE BET DETECTED", "NEW")

        # Get current values
        curr_odds = current.get('back_odds', 0)
        curr_edge = current.get('edge', 0)

        # Get previous values
        prev_odds = previous.get('back_odds', 0)
        prev_edge = previous.get('edge', 0)

        # Calculate changes
        odds_change = curr_odds - prev_odds
        edge_change = curr_edge - prev_edge

        # Determine header based on primary change
        if abs(odds_change) >= 0.10:  # Odds changed by at least $0.10
            if odds_change > 0:
                header = "📈 VALUE BET - ODDS INCREASED"
            else:
                header = "📉 VALUE BET - ODDS DECREASED"
        elif abs(edge_change) >= 0.01:  # Edge changed by at least 1pp
            if edge_change > 0:
                header = "✓ VALUE BET - EDGE INCREASED"
            else:
                header = "✗ VALUE BET - EDGE DECREASED"
        else:
            # No significant change, but still value
            header = "🔄 VALUE BET - STILL VALUE"

        # Build change text
        change_parts = []

        if abs(odds_change) >= 0.10:
            sign = "+" if odds_change > 0 else ""
            change_parts.append(f"Odds: ${curr_odds:.2f} ({sign}${odds_change:.2f})")

        if abs(edge_change) >= 0.005:  # 0.5pp
            sign = "+" if edge_change > 0 else ""
            change_parts.append(f"Edge: {curr_edge:+.1%} ({sign}{edge_change:+.1%})")

        change_text = " | ".join(change_parts) if change_parts else "No significant change"

        return (header, change_text)

    def send_alerts(self, predictions: list[RacePrediction]) -> int:
        """
        Send Telegram alerts for all value bets with change detection.

        Args:
            predictions: List of RacePrediction objects

        Returns:
            Number of alerts sent
        """
        if self.dry_run:
            self.logger.info("\n[DRY-RUN] Skipping alert sending")
            return 0

        alerts_sent = 0

        for pred in predictions:
            # Find scheduled check for this race (to store alert data)
            market_key = f"{pred.venue}_r{pred.race_number}"
            scheduled_check = self.daily_schedule.get(market_key)

            for runner in pred.runners:
                if runner.get("is_value_bet"):
                    try:
                        # Get previous alert data for this runner
                        previous_alert = None
                        if scheduled_check and scheduled_check.last_alert_data:
                            # Check if we have data for this specific runner
                            runner_key = runner.get('runner_name', '')
                            previous_alert = scheduled_check.last_alert_data.get(runner_key)

                        # Compare with previous alert
                        header, change_text = self.compare_alerts(runner, previous_alert)

                        # Create value bet
                        value_bet = ValueBet(
                            venue=runner["venue"],
                            race_number=runner["race_number"],
                            race_time=runner["race_time"],
                            runner_name=runner["runner_name"],
                            box=runner["box"],
                            model_proba=runner["model_proba"],
                            back_odds=runner["back_odds"],
                            implied_proba=runner["implied_proba"],
                            edge=runner["edge"],
                        )

                        # Send alert with header and change info
                        # Get check window from runner data
                        check_window = runner.get("check_window", 0)

                        self.value_alerter.send_value_alert(
                            value_bet,
                            header=header,
                            change_text=change_text,
                            check_window=check_window
                        )
                        alerts_sent += 1

                        # Log value bet to CSV for ROI tracking
                        if self.prediction_logger:
                            value_bet_record = self._build_value_bet_record(runner, pred)
                            alert_time = datetime.now().strftime("%H:%M")
                            self.prediction_logger.log_value_bet(value_bet_record, alert_time)
                            runner_name = runner.get('runner_name', 'Unknown')
                            self.logger.info(f"[PREDICTION LOG] Logged value bet to CSV: {runner_name}")

                        # Store this alert data for next comparison
                        if scheduled_check:
                            if not scheduled_check.last_alert_data:
                                scheduled_check.last_alert_data = {}

                            runner_key = runner.get('runner_name', '')
                            scheduled_check.last_alert_data[runner_key] = {
                                'back_odds': runner['back_odds'],
                                'edge': runner['edge'],
                                'model_proba': runner['model_proba'],
                                'alert_time': datetime.now(),
                            }

                        self.logger.info(f"[ALERT] ✓ Sent: {runner['runner_name']} ({header})")

                        # Small delay to avoid rate limiting
                        time.sleep(0.5)

                    except Exception as e:
                        self.logger.error(f"  ✗ Failed to send alert: {e}")

        self.stats.alerts_sent = alerts_sent
        return alerts_sent

    def run_once(self, scheduled_races: Optional[list[ScheduledCheck]] = None) -> LiveSelectorStats:
        """
        Run a single value detection cycle.

        Returns:
            LiveSelectorStats with run statistics
        """
        self.stats = LiveSelectorStats(start_time=datetime.now())

        # Increment poll number for prediction logging
        self.poll_number += 1

        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"LIVE SELECTOR RUN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.dry_run:
            self.logger.info("*** DRY-RUN MODE - No alerts will be sent ***")
        self.logger.info("=" * 60)

        # Get upcoming markets
        try:
            markets = self.get_upcoming_markets()
        except Exception as e:
            self.logger.error(f"Failed to fetch markets: {e}")
            self.stats.errors.append(f"Market fetch failed: {e}")
            self.stats.end_time = datetime.now()
            return self.stats

        # ADD DEBUG LOGGING AFTER MARKET FILTERING
        self.logger.info(f"\n[DEBUG] Market Processing")
        self.logger.info(f"  scheduled_races parameter: {len(scheduled_races) if scheduled_races else 'None'}")
        self.logger.info(f"  Total markets fetched: {len(markets)}")
        self.logger.info(f"  self.daily_schedule: {len(self.daily_schedule)} races")

        # Show schedule keys
        if self.daily_schedule:
            self.logger.info(f"  Schedule keys (first 10):")
            for i, (key, sc) in enumerate(list(self.daily_schedule.items())[:10]):
                self.logger.info(f"    - {key} ({sc.venue} R{sc.race_number})")

        # Show market keys
        if markets:
            self.logger.info(f"  Market keys (first 10):")
            for m in markets[:10]:
                venue_norm = normalize_venue(m.venue or m.event_name)
                race_num = extract_race_number(m.market_name) or 0
                key = f"{venue_norm}_r{race_num}"
                self.logger.info(f"    - {key} (event: {m.event_name})")

        if not markets:
            self.logger.info("\nNo markets within time window")
            self.stats.end_time = datetime.now()
            return self.stats

        # ============================================================
        # MARKET FILTERING: Determine which races to process
        # ============================================================
        if scheduled_races is not None:
            # CONTINUOUS MODE: Only process races scheduled for checking NOW
            scheduled_keys = {f"{s.venue}_r{s.race_number}" for s in scheduled_races}
            self.logger.info(f"\n[Mode: CONTINUOUS]")
            self.logger.info(f"  Races due to check NOW: {len(scheduled_races)}")
            for sr in scheduled_races:
                self.logger.info(f"    - {sr.venue}_r{sr.race_number}")

        elif self.daily_schedule:
            # ONCE MODE: Process all scheduled races
            scheduled_keys = set(self.daily_schedule.keys())
            self.logger.info(f"\n[Mode: ONCE]")
            self.logger.info(f"  Processing all scheduled races: {len(scheduled_keys)}")

        else:
            # NO SCHEDULE: Process all markets (fallback)
            scheduled_keys = None
            self.logger.info(f"\n[Mode: NO SCHEDULE]")
            self.logger.info(f"  Processing all markets")

        # Apply filtering
        if scheduled_keys is not None:
            markets_before = len(markets)

            # Filter to only scheduled markets
            filtered_markets = []
            for m in markets:
                venue_norm = normalize_venue(m.venue or m.event_name)
                race_num = extract_race_number(m.market_name) or 0
                market_key = f"{venue_norm}_r{race_num}"
                if market_key in scheduled_keys:
                    filtered_markets.append(m)

            markets = filtered_markets

            self.logger.info(f"  Filtered: {markets_before} → {len(markets)} markets")

            if len(markets) == 0:
                self.logger.info("  No markets match scheduled races")
        else:
            self.logger.info(f"  No filtering - processing all {len(markets)} markets")

        # Process each market
        self.logger.info(f"\n[Processing {len(markets)} Markets]")
        predictions = []

        # Process each market with its own check window
        for market in markets:
            # Look up the window for THIS specific race
            check_window = self._get_window_for_race(market, scheduled_races)

            pred = self.process_market(market, check_window)
            predictions.append(pred)

        # Send alerts
        if self.stats.value_bets_found > 0:
            self.logger.info(f"\n[ALERT] Sending alerts - {self.stats.value_bets_found} value bets found")
            self.send_alerts(predictions)
        else:
            self.logger.info("\n[VALUE BET] None found")

        # Summary
        self.stats.end_time = datetime.now()
        self.print_summary()

        return self.stats

    def print_summary(self) -> None:
        """Print run summary."""
        duration = (self.stats.end_time - self.stats.start_time).total_seconds()

        self.logger.info("\n" + "=" * 60)
        self.logger.info("RUN SUMMARY")
        self.logger.info("=" * 60)

        self.logger.info(f"\n[Markets]")
        self.logger.info(f"  Fetched:          {self.stats.markets_fetched}")
        self.logger.info(f"  Within window:    {self.stats.markets_within_window}")
        self.logger.info(f"  Failed:           {self.stats.races_failed}")

        self.logger.info(f"\n[RUNNERS]")
        self.logger.info(f"[RUNNERS]   Processed: {self.stats.runners_processed}")
        self.logger.info(f"[RUNNERS]   With predictions: {self.stats.runners_with_predictions}")
        self.logger.info(f"[RUNNERS]   With odds: {self.stats.runners_with_odds}")

        self.logger.info(f"\n[Value Bets]")
        self.logger.info(f"  Found:            {self.stats.value_bets_found}")
        self.logger.info(f"  Alerts sent:      {self.stats.alerts_sent}")

        if self.stats.errors:
            self.logger.info(f"\n[Errors: {len(self.stats.errors)}]")
            for err in self.stats.errors[:5]:
                self.logger.info(f"  - {err}")
            if len(self.stats.errors) > 5:
                self.logger.info(f"  ... and {len(self.stats.errors) - 5} more")

        self.logger.info(f"\n[Duration: {duration:.1f}s]")

        if self.session_start_time:
            session_age = (datetime.now() - self.session_start_time).total_seconds() / 3600
            self.logger.info(f"\n[Session]")
            self.logger.info(f"  Age: {session_age:.1f} hours")

            if self.last_refresh_time:
                self.logger.info(f"  Last refresh: {self.last_refresh_time.strftime('%I:%M %p')}")
            else:
                self.logger.info(f"  Last refresh: Never")

            if self.next_refresh_time:
                self.logger.info(f"  Next refresh: {self.next_refresh_time.strftime('%I:%M %p')}")
            else:
                needs_refresh = "Yes" if self.check_session_age() else "No"
                self.logger.info(f"  Needs refresh: {needs_refresh}")

        if self.daily_schedule:
            races_remaining = len(self.daily_schedule)
            self.logger.info(f"\n[Schedule]")
            self.logger.info(f"  Races remaining: {races_remaining}")

            if races_remaining > 0:
                next_race = min(s.race_start_time for s in self.daily_schedule.values())
                self.logger.info(f"  Next race: {next_race.strftime('%I:%M %p')}")

        self.logger.info(f"\n[API Calls]")
        self.logger.info(f"  Today: {self.api_call_count_daily}")
        self.logger.info(f"  This session: {self.api_call_count}")

        if self.dry_run:
            self.logger.info("\n*** DRY-RUN MODE - No alerts were sent ***")

    def run_continuous(self) -> None:
        """
        Run continuous value detection with schedule-based polling.

        Checks races ONLY at scheduled windows defined by ALERT_WINDOWS
        in config.py (default: 45, 20, 10, 5, 2 minutes before race).

        Sleeps precisely until the next scheduled check window.
        Only uses NO_RACES_SLEEP when no schedule is loaded (e.g. before
        JSONL files are created in the morning, or after all races complete).
        """
        from config import ALERT_WINDOWS, SLEEP_ALERT, NO_RACES_POLL_INTERVAL

        # Sleep duration when no races scheduled (30 min)
        # Used ONLY in two cases:
        # 1. Before morning JSONL files are created
        # 2. After all races for the day are complete
        NO_RACES_SLEEP = NO_RACES_POLL_INTERVAL  # From config (default 1800s)

        self.logger.info(f"\n[Configuration]")
        self.logger.info(f"  Alert windows: {ALERT_WINDOWS} minutes before race")
        self.logger.info(f"  Sleep alerts: {'Enabled' if SLEEP_ALERT else 'Disabled'}")
        self.logger.info(f"  No-race polling: {NO_RACES_POLL_INTERVAL}s")
        self.logger.info(f"  Session refresh: {self.refresh_threshold / 3600:.0f} hours")

        self.logger.info(f"\n[Continuous Mode - Schedule-Based]")
        self.logger.info(f"  Schedule loaded: {len(self.daily_schedule)} races")
        self.logger.info(f"  Session refresh: Auto (every 10 hours)")

        if self.session_start_time:
            self.logger.info(f"  Session started: {self.session_start_time.strftime('%I:%M %p')}")

        while True:
            try:
                # --- Clean up completed races ---
                self.cleanup_completed_races()

                if not self.daily_schedule:
                    # No schedule loaded - try to rebuild from JSONL files
                    self.logger.info("[SCHEDULE] No schedule loaded - checking for JSONL files")

                    today = datetime.now().strftime('%Y%m%d')
                    self.daily_schedule = self.build_daily_schedule(today)

                    if not self.daily_schedule:
                        # Still no schedule - sleep 30min and try again
                        # This happens before morning JSONL files are created
                        self.logger.info(
                            f"[SCHEDULE] No races found - sleeping "
                            f"{NO_RACES_SLEEP/60:.0f}min until next check"
                        )
                        time.sleep(NO_RACES_SLEEP)
                        continue

                # --- Check if refresh needed NOW ---
                if self.next_refresh_time:
                    now = datetime.now()
                    if now >= self.next_refresh_time:
                        self.logger.info(f"\n[SESSION] Refresh time reached")
                        self.refresh_session()

                # --- Check if refresh should be SCHEDULED ---
                if self.check_session_age() and not self.next_refresh_time:
                    window = self.find_safe_refresh_window()
                    if window:
                        self.next_refresh_time = window
                        self.logger.info(
                            f"[SESSION] Refresh scheduled for {window.strftime('%I:%M %p')} "
                            f"(session age: {(datetime.now() - self.session_start_time).total_seconds() / 3600:.1f}h)"
                        )

                # --- Process races that need checking NOW ---
                races_to_check = self.get_races_to_check_now()

                if races_to_check:
                    self.logger.info(f"\n[SCHEDULE] {len(races_to_check)} races need checking now")

                    # Process each scheduled race
                    for scheduled_check in races_to_check:
                        self.logger.info(
                            f"  Checking: {scheduled_check.venue} R{scheduled_check.race_number} "
                            f"(race at {scheduled_check.race_start_time.strftime('%I:%M %p')})"
                        )

                        # Mark as checked
                        scheduled_check.last_checked = datetime.now()

                    # Run the actual market processing (pass scheduled races)
                    self.run_once(scheduled_races=races_to_check)

                # --- Determine next wake time ---
                next_check = self.get_next_check_time()

                if next_check:
                    # Sleep PRECISELY until next scheduled check
                    sleep_seconds = (next_check - datetime.now()).total_seconds()

                    # Allow up to 1 hour sleep, minimum 10s buffer for processing time
                    # NOTE: No artificial 60s minimum - we trust the schedule completely
                    sleep_seconds = max(10, min(sleep_seconds, 3600))

                    self.logger.info(
                        f"\n[SCHEDULE] Next check: {next_check.strftime('%I:%M %p')} "
                        f"(sleeping {sleep_seconds/60:.1f} min)"
                    )
                    time.sleep(sleep_seconds)

                else:
                    # No races scheduled - check if we should shutdown or keep polling
                    if SLEEP_ALERT:
                        self.logger.info("\n[SCHEDULE] No more races today")
                        self.logger.info("🌙 All races complete. System entering sleep mode.")

                        # Send Telegram notification if alerter available
                        if self.value_alerter:
                            try:
                                # Get daily API call count from Betfair client
                                daily_api_calls = self.betfair_client.get_daily_api_calls()

                                # Send sleep notification with API stats
                                self.value_alerter.send_sleep_notification(
                                    api_calls_today=daily_api_calls
                                )
                            except Exception as e:
                                self.logger.error(f"Could not send sleep notification: {e}")

                    # Sleep for 30 minutes and check for new races
                    self.logger.info(f"[SCHEDULE] Sleeping {NO_RACES_POLL_INTERVAL}s, will check for new races")
                    time.sleep(NO_RACES_POLL_INTERVAL)

                    # Rebuild schedule (in case new JSONL file appeared)
                    today = datetime.now().strftime('%Y%m%d')
                    self.daily_schedule = self.build_daily_schedule(today)

            except KeyboardInterrupt:
                self.logger.info("\n[Interrupted by user]")
                break
            except RateLimitExceeded as e:
                self.logger.critical(f"🚨 EMERGENCY SHUTDOWN: API rate limit exceeded")
                self.logger.critical(f"Error: {e}")

                # Send emergency Telegram alert
                if self.value_alerter and not self.dry_run:
                    try:
                        self.value_alerter.send_emergency_alert(
                            f"🚨 API RATE LIMIT EXCEEDED\n\n"
                            f"Live selector shutting down to prevent account suspension.\n\n"
                            f"Error: {e}\n\n"
                            f"Please check logs and restart manually after investigating."
                        )
                    except Exception as alert_error:
                        self.logger.error(f"Could not send emergency alert: {alert_error}")

                # Exit with error code
                sys.exit(1)
            except Exception as e:
                self.logger.error(f"\n[Error in continuous run: {e}]")
                self.logger.info(f"[Retrying in 60s...]")
                time.sleep(60)

    def close(self) -> None:
        """Clean up resources."""
        self.logger.info("\n[Closing connections]")
        # Betfair client doesn't have explicit logout, but we could add it
        self.logger.info("  Done")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Live value bet selector for greyhound racing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously, polling for new markets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending Telegram alerts",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=60,
        help="Minutes ahead to look for markets (default: 60)",
    )

    args = parser.parse_args()

    if not args.once and not args.continuous:
        parser.print_help()
        print("\nError: Must specify --once or --continuous")
        return 1

    # Create selector
    selector = LiveSelector(
        dry_run=args.dry_run,
        lookahead_minutes=args.lookahead,
    )

    try:
        # Initialize
        if not selector.initialize():
            return 1

        # Run
        if args.once:
            stats = selector.run_once()
            # Return non-zero if there were errors but we found value bets
            return 0 if stats.value_bets_found > 0 or stats.errors == [] else 0
        else:
            selector.run_continuous()
            return 0

    except KeyboardInterrupt:
        selector.logger.info("\n[Interrupted]")
        return 0
    except Exception as e:
        selector.logger.error(f"\n[Fatal error: {e}]")
        return 1
    finally:
        selector.close()


if __name__ == "__main__":
    sys.exit(main())
