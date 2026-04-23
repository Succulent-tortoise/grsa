"""
api_call_tracker.py — Betfair API Rate Limit Tracker

Tracks API calls and enforces rate limits to prevent account suspension.

Betfair limits:
- 1,000 calls per hour (hard limit)
- 10,000 calls per day (hard limit)

This module is MARKET-SIDE ONLY (handles Betfair API interaction).
Never touches model training or feature engineering.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import logging


@dataclass
class RateLimitStatus:
    """Status snapshot of API rate limits."""
    hourly_count: int
    hourly_max: int
    hourly_percent: float
    daily_count: int
    daily_max: int
    daily_percent: float
    timestamp: datetime


class APICallTracker:
    """
    Tracks Betfair API calls and enforces rate limits.

    Prevents account suspension by monitoring hourly and daily call counts
    against Betfair's hard limits.
    """

    def __init__(
        self,
        max_per_hour: int = 1000,
        max_per_day: int = 10000
    ):
        """
        Initialize API call tracker.

        Args:
            max_per_hour: Maximum API calls allowed per hour (Betfair: 1000)
            max_per_day: Maximum API calls allowed per day (Betfair: 10000)
        """
        self.max_per_hour = max_per_hour
        self.max_per_day = max_per_day

        # Track call timestamps
        self.hourly_calls: list[datetime] = []
        self.daily_calls: list[datetime] = []

        # Track start of current day
        self.start_of_day = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        self.logger = logging.getLogger("api_call_tracker")

    def register_call(self) -> tuple[int, int]:
        """
        Register an API call.

        Returns:
            Tuple of (hourly_count, daily_count) after registration
        """
        now = datetime.now()

        # Check if new day started
        if now.date() > self.start_of_day.date():
            self._reset_daily_counter(now)

        # Clean up old hourly calls (older than 1 hour)
        one_hour_ago = now - timedelta(hours=1)
        self.hourly_calls = [t for t in self.hourly_calls if t > one_hour_ago]

        # Add new call
        self.hourly_calls.append(now)
        self.daily_calls.append(now)

        return len(self.hourly_calls), len(self.daily_calls)

    def get_status(self) -> RateLimitStatus:
        """
        Get current rate limit status.

        Returns:
            RateLimitStatus with current counts and percentages
        """
        # Clean old hourly calls first
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        self.hourly_calls = [t for t in self.hourly_calls if t > one_hour_ago]

        hourly = len(self.hourly_calls)
        daily = len(self.daily_calls)

        return RateLimitStatus(
            hourly_count=hourly,
            hourly_max=self.max_per_hour,
            hourly_percent=hourly / self.max_per_hour if self.max_per_hour > 0 else 0,
            daily_count=daily,
            daily_max=self.max_per_day,
            daily_percent=daily / self.max_per_day if self.max_per_day > 0 else 0,
            timestamp=now,
        )

    def check_limits(self) -> Optional[str]:
        """
        Check if rate limits are exceeded.

        Returns:
            Error message if limit violated, None otherwise
        """
        status = self.get_status()

        if status.hourly_percent >= 1.0:
            return f"HOURLY LIMIT EXCEEDED ({status.hourly_count}/{status.hourly_max})"

        if status.daily_percent >= 1.0:
            return f"DAILY LIMIT EXCEEDED ({status.daily_count}/{status.daily_max})"

        return None

    def get_warning_level(self) -> Optional[str]:
        """
        Get warning level based on current usage.

        Returns:
            "CRITICAL" (90%+), "WARNING" (80%+), or None
        """
        status = self.get_status()

        # Check hourly first (more immediate)
        if status.hourly_percent >= 0.90:
            return "CRITICAL"
        if status.hourly_percent >= 0.80:
            return "WARNING"

        # Check daily
        if status.daily_percent >= 0.90:
            return "CRITICAL"
        if status.daily_percent >= 0.80:
            return "WARNING"

        return None

    def _reset_daily_counter(self, now: datetime):
        """Reset daily counter at midnight."""
        self.daily_calls = []
        self.start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self.logger.info(f"[API TRACKER] Daily counter reset at {now}")

    def get_daily_total(self) -> int:
        """Get total API calls made today."""
        return len(self.daily_calls)

    def format_status(self) -> str:
        """
        Format status as human-readable string.

        Returns:
            Formatted status string for logging
        """
        status = self.get_status()

        return (
            f"Hourly: {status.hourly_count}/{status.hourly_max} "
            f"({status.hourly_percent:.1%}) | "
            f"Daily: {status.daily_count}/{status.daily_max} "
            f"({status.daily_percent:.1%})"
        )


class RateLimitExceeded(Exception):
    """Raised when Betfair API rate limit is exceeded."""
    pass


# Test harness
if __name__ == "__main__":
    print("=" * 60)
    print("API CALL TRACKER TEST")
    print("=" * 60)

    # Create tracker with low limits for testing
    tracker = APICallTracker(max_per_hour=10, max_per_day=50)

    print("\n[Test 1: Register calls]")
    for i in range(15):
        hourly, daily = tracker.register_call()
        status = tracker.get_status()
        warning = tracker.get_warning_level()

        print(f"Call {i+1:2d}: {tracker.format_status()}", end="")
        if warning:
            print(f" [{warning}]", end="")

        # Check if limit exceeded
        error = tracker.check_limits()
        if error:
            print(f" ⚠️ {error}")
            break
        else:
            print()

    print("\n[Test 2: Daily total]")
    print(f"Total API calls today: {tracker.get_daily_total()}")

    print("\n" + "=" * 60)
    print("Tests complete. Module ready for integration.")
    print("=" * 60)
