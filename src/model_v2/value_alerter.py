"""
value_alerter.py — Greyhound Prediction System v2

Identifies value bets and sends Telegram alerts.

Computes edge = model probability - implied probability (1 / back_odds)
Filters for edge >= MIN_EDGE_THRESHOLD and odds in ODDS_RANGE_MIN to ODDS_RANGE_MAX
Sends formatted Telegram alerts for each value bet.

Config thresholds (from config.py):
- MIN_EDGE_THRESHOLD: 7%
- ODDS_RANGE_MIN: $2.00
- ODDS_RANGE_MAX: $10.00

Usage:
    python value_alerter.py --test     # Send test alert with sample data
    python value_alerter.py --run      # Run live value detection (requires model + Betfair)
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import requests

from config import (
    BETFAIR_COMMISSION,
    MIN_EDGE_THRESHOLD,
    MIN_MODEL_CONFIDENCE,
    MODEL_DIR,
    ODDS_RANGE_MAX,
    ODDS_RANGE_MIN,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Telegram API endpoint
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


@dataclass
class ValueBet:
    """Represents a detected value bet."""
    venue: str
    race_number: int
    race_time: str
    runner_name: str
    box: int
    model_proba: float
    back_odds: float
    implied_proba: float
    edge: float

    def __post_init__(self):
        # Edge is already computed, but ensure it's consistent
        if self.implied_proba == 0:
            self.edge = self.model_proba
        else:
            self.edge = self.model_proba - self.implied_proba


class TelegramError(Exception):
    """Raised when Telegram API call fails."""
    pass


class ValueAlerter:
    """
    Identifies value bets and sends Telegram alerts.
    """

    def __init__(
        self,
        min_edge: float = MIN_EDGE_THRESHOLD,
        odds_min: float = ODDS_RANGE_MIN,
        odds_max: float = ODDS_RANGE_MAX,
        min_confidence: float = MIN_MODEL_CONFIDENCE,
    ):
        self.min_edge = min_edge
        self.odds_min = odds_min
        self.odds_max = odds_max
        self.min_confidence = min_confidence

        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")

    def compute_edge(self, model_proba: float, back_odds: float) -> float:
        """
        Compute edge = model probability - implied probability.

        Implied probability = 1 / back_odds (uses BACK odds only, not lay)

        Args:
            model_proba: Model's predicted win probability
            back_odds: Best available back odds from Betfair

        Returns:
            Edge value (positive = model thinks runner is undervalued)
        """
        if back_odds <= 1.0:
            return 0.0

        implied_proba = 1.0 / back_odds
        return model_proba - implied_proba

    def is_value_bet(self, model_proba: float, back_odds: float) -> bool:
        """
        Check if a runner qualifies as a value bet.

        Criteria:
        1. back_odds in range [ODDS_RANGE_MIN, ODDS_RANGE_MAX]
        2. edge >= MIN_EDGE_THRESHOLD
        3. model_proba >= MIN_MODEL_CONFIDENCE (optional)

        Args:
            model_proba: Model's predicted win probability
            back_odds: Best available back odds from Betfair

        Returns:
            True if runner qualifies as a value bet
        """
        # Check odds range
        if back_odds < self.odds_min or back_odds > self.odds_max:
            return False

        # Check model confidence
        if model_proba < self.min_confidence:
            return False

        # Check edge
        edge = self.compute_edge(model_proba, back_odds)
        return edge >= self.min_edge

    def find_value_bets(self, runners: list[dict]) -> list[ValueBet]:
        """
        Find all value bets from a list of runners.

        Args:
            runners: List of dicts with keys:
                - venue: str
                - race_number: int
                - race_time: str (optional)
                - runner_name: str
                - box: int
                - model_proba: float
                - back_odds: float

        Returns:
            List of ValueBet objects that meet criteria
        """
        value_bets = []

        for runner in runners:
            model_proba = runner.get("model_proba", 0)
            back_odds = runner.get("back_odds", 0)

            if self.is_value_bet(model_proba, back_odds):
                implied_proba = 1.0 / back_odds if back_odds > 1 else 0
                edge = model_proba - implied_proba

                value_bet = ValueBet(
                    venue=runner.get("venue", "Unknown"),
                    race_number=runner.get("race_number", 0),
                    race_time=runner.get("race_time", ""),
                    runner_name=runner.get("runner_name", "Unknown"),
                    box=runner.get("box", 0),
                    model_proba=model_proba,
                    back_odds=back_odds,
                    implied_proba=implied_proba,
                    edge=edge,
                )
                value_bets.append(value_bet)

        # Sort by edge descending
        value_bets.sort(key=lambda x: x.edge, reverse=True)

        return value_bets

    def format_race_time_adelaide(self, race_time: str) -> str:
        """
        Convert UTC race time to Adelaide local time for display.

        Args:
            race_time: UTC time string (e.g., "2026-02-13T02:23:00.000Z")

        Returns:
            Formatted Adelaide time (e.g., "13 Feb 2026 12:53 PM ACDT")
            Returns original string if parsing fails.
        """
        if not race_time:
            return ""

        try:
            # Parse UTC time (handles both 'Z' suffix and '+00:00')
            if race_time.endswith('Z'):
                utc_time = datetime.fromisoformat(race_time.replace('Z', '+00:00'))
            else:
                utc_time = datetime.fromisoformat(race_time)

            # Ensure UTC timezone is set
            if utc_time.tzinfo is None:
                utc_time = utc_time.replace(tzinfo=timezone.utc)

            # Convert to Adelaide time (handles ACDT/ACST daylight savings automatically)
            adelaide_tz = ZoneInfo('Australia/Adelaide')
            local_time = utc_time.astimezone(adelaide_tz)

            # Format nicely: "13 Feb 2026 12:53 PM ACDT"
            return local_time.strftime('%d %b %Y %I:%M %p %Z')

        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse race_time '{race_time}': {e}")
            return race_time  # Return original if parsing fails

    def format_alert(self, bet: ValueBet, header: str = "🐕 <b>VALUE BET DETECTED</b>", change_text: str = "", window_minutes: int = 0) -> str:
        """
        Format a value bet as a Telegram message.

        Format:
        🐕 VALUE BET DETECTED (20min check)

        📍 Taree R4
        ⏰ 13 Feb 2026 12:53 PM ACDT
        🐾 Raccoon Roger (Box 5)
        📊 Model: 35.0% | Odds: $2.80 | Edge: +10.7%

        """
        # Ensure header stays bold if passed from live_selector
        if header and "<b>" not in header:
            header = f"<b>{header}</b>"

        # Add check window to header if provided
        if window_minutes > 0:
            window_text = f" ({window_minutes}min check)"
            header = f"{header[:-4]}{window_text}</b>" if header.endswith("</b>") else f"{header}{window_text}"

        lines = [
            header,
            "",
            f"📍 {bet.venue} R{bet.race_number}",
        ]

        if bet.race_time:
            formatted_time = self.format_race_time_adelaide(bet.race_time)
            lines.append(f"⏰ {formatted_time}")

        lines.extend([
            f"🐾 {bet.runner_name} (Box {bet.box})",
            f"📊 Model: {bet.model_proba:.1%} | Odds: ${bet.back_odds:.2f} | Edge: <b>{bet.edge:+.1%}</b>",
        ])

        if change_text and change_text != "NEW":
            lines.append(f"\n<i>{change_text}</i>")

        return "\n".join(lines)

    def send_telegram_alert(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message via Telegram Bot API.

        Args:
            message: Message text to send
            parse_mode: "HTML" or "Markdown" for formatting

        Returns:
            True if message sent successfully

        Raises:
            TelegramError: If API call fails
        """
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": parse_mode,
        }

        try:
            response = requests.post(
                TELEGRAM_API_URL,
                json=payload,
                timeout=30,
            )

            response.raise_for_status()
            result = response.json()

            if result.get("ok"):
                logger.info("Telegram alert sent successfully")
                return True
            else:
                raise TelegramError(f"Telegram API error: {result.get('description', 'Unknown error')}")

        except requests.exceptions.RequestException as e:
            raise TelegramError(f"Failed to send Telegram alert: {e}")

    def send_value_alert(
        self,
        bet: ValueBet,
        header: str = "🐕 <b>VALUE BET DETECTED</b>",
        change_text: str = "",
        check_window: int = 0
    ) -> bool:
        """
        Send a value bet alert to Telegram.

        Args:
            bet: ValueBet object to alert
            header: Alert header (includes emoji and change type)
            change_text: Description of what changed
            check_window: Minutes before race (45, 20, 10, 5, 2)

        Returns:
            True if alert sent successfully
        """
        message = self.format_alert(bet, header=header, change_text=change_text, window_minutes=check_window)
        return self.send_telegram_alert(message)

    def send_sleep_notification(self, api_calls_today: int = 0) -> bool:
        """
        Send end-of-day sleep notification to Telegram.

        Args:
            api_calls_today: Total Betfair API calls made today

        Returns:
            True if notification sent successfully
        """
        from config import SLEEP_ALERT

        if not SLEEP_ALERT:
            return False

        message = (
            "🌙 <b>System Sleep Mode</b>\n\n"
            "All races for today have been processed.\n"
            "System entering sleep mode until tomorrow's races.\n\n"
            f"📊 <b>Daily Stats:</b>\n"
            f"• Betfair API calls: {api_calls_today:,}\n\n"
            "Will check for new races periodically.\n"
            "Goodnight! 😴"
        )

        return self.send_telegram_alert(message)

    def send_emergency_alert(self, message: str) -> bool:
        """
        Send critical emergency alert to Telegram.

        Used for system failures, API limit violations, or other critical issues
        that require immediate attention.

        Args:
            message: Emergency message to send

        Returns:
            True if message sent successfully
        """
        emergency_message = (
            "🚨 <b>EMERGENCY ALERT</b> 🚨\n\n"
            f"{message}"
        )

        return self.send_telegram_alert(emergency_message)

    def send_all_alerts(self, bets: list[ValueBet], max_alerts: int = 10) -> int:
        """
        Send alerts for all value bets.

        Args:
            bets: List of ValueBet objects
            max_alerts: Maximum number of alerts to send

        Returns:
            Number of alerts successfully sent
        """
        sent = 0
        for bet in bets[:max_alerts]:
            try:
                self.send_value_alert(bet)
                sent += 1
            except TelegramError as e:
                logger.error(f"Failed to send alert: {e}")

        return sent


def get_sample_value_bet() -> ValueBet:
    """
    Create a sample value bet for testing.

    This is a HARDCODED sample, not live market data.
    """
    return ValueBet(
        venue="Taree",
        race_number=4,
        race_time="14:22",
        runner_name="Raccoon Roger",
        box=5,
        model_proba=0.35,
        back_odds=2.80,
        implied_proba=1.0 / 2.80,  # 35.7%
        edge=0.35 - (1.0 / 2.80),  # -0.7% -- not a value bet
    )


def get_sample_value_bet_passing() -> ValueBet:
    """
    Create a sample value bet that PASSES all filters.

    Edge calculation:
    - model_proba = 0.40 (40%)
    - back_odds = $3.50
    - implied_proba = 1/3.50 = 28.6%
    - edge = 40% - 28.6% = 11.4% (passes 7% threshold)
    - odds $3.50 is in range [$2, $10]
    """
    return ValueBet(
        venue="Taree",
        race_number=4,
        race_time="14:22",
        runner_name="Raccoon Roger",
        box=5,
        model_proba=0.40,
        back_odds=3.50,
        implied_proba=1.0 / 3.50,
        edge=0.40 - (1.0 / 3.50),
    )


def run_gate_check() -> dict:
    """
    Run gate check for value alerter.

    Tests:
    1. Edge calculation uses back_odds only
    2. Filter thresholds match config values
    3. Telegram test alert is sent and received
    """
    logger.info("=" * 60)
    logger.info("VALUE ALERTER GATE CHECK")
    logger.info("=" * 60)

    results = {
        "status": None,
        "edge_calculation": {},
        "filter_thresholds": {},
        "telegram_test": {},
    }

    alerter = ValueAlerter()

    # Test 1: Edge calculation uses back_odds
    logger.info("\n[Test 1] Edge calculation uses back_odds only")

    model_proba = 0.40
    back_odds = 3.50
    expected_implied = 1.0 / back_odds
    expected_edge = model_proba - expected_implied
    computed_edge = alerter.compute_edge(model_proba, back_odds)

    edge_test_passed = abs(computed_edge - expected_edge) < 0.0001

    results["edge_calculation"] = {
        "passed": edge_test_passed,
        "model_proba": model_proba,
        "back_odds": back_odds,
        "expected_implied": expected_implied,
        "expected_edge": expected_edge,
        "computed_edge": computed_edge,
        "uses_back_odds_only": True,
    }

    if edge_test_passed:
        logger.info(f"  ✓ Edge calculation correct: {model_proba:.1%} - 1/{back_odds} = {computed_edge:.1%}")
    else:
        logger.error(f"  ✗ Edge calculation incorrect: expected {expected_edge:.4f}, got {computed_edge:.4f}")

    # Test 2: Filter thresholds match config
    logger.info("\n[Test 2] Filter thresholds match config")

    thresholds_match = (
        alerter.min_edge == MIN_EDGE_THRESHOLD and
        alerter.odds_min == ODDS_RANGE_MIN and
        alerter.odds_max == ODDS_RANGE_MAX
    )

    results["filter_thresholds"] = {
        "passed": thresholds_match,
        "min_edge": alerter.min_edge,
        "expected_min_edge": MIN_EDGE_THRESHOLD,
        "odds_min": alerter.odds_min,
        "expected_odds_min": ODDS_RANGE_MIN,
        "odds_max": alerter.odds_max,
        "expected_odds_max": ODDS_RANGE_MAX,
    }

    if thresholds_match:
        logger.info(f"  ✓ Thresholds match config:")
        logger.info(f"    - Min edge: {alerter.min_edge:.0%} (>= 7%)")
        logger.info(f"    - Odds range: ${alerter.odds_min:.2f} - ${alerter.odds_max:.2f}")
    else:
        logger.error("  ✗ Thresholds do not match config")

    # Test 3: Telegram test alert
    logger.info("\n[Test 3] Send Telegram test alert")

    sample_bet = get_sample_value_bet_passing()
    message = alerter.format_alert(sample_bet)

    results["telegram_test"]["message_format"] = message

    logger.info(f"  Message format:\n{'-' * 40}")
    for line in message.split("\n"):
        logger.info(f"    {line}")
    logger.info("-" * 40)

    try:
        alerter.send_telegram_alert(message)
        results["telegram_test"]["sent"] = True
        results["telegram_test"]["error"] = None
        logger.info("  ✓ Test alert sent to Telegram")
    except TelegramError as e:
        results["telegram_test"]["sent"] = False
        results["telegram_test"]["error"] = str(e)
        logger.error(f"  ✗ Failed to send test alert: {e}")

    # Overall status
    all_passed = (
        edge_test_passed and
        thresholds_match and
        results["telegram_test"]["sent"]
    )
    results["status"] = "PASSED" if all_passed else "FAILED"

    return results


def print_gate_report(results: dict) -> None:
    """Print formatted gate check report."""
    print("\n" + "=" * 60)
    print("VALUE ALERTER GATE CHECK REPORT")
    print("=" * 60)

    # Edge calculation
    ec = results["edge_calculation"]
    print(f"\n[1] Edge Calculation (uses back_odds only)")
    print(f"    Model prob:  {ec['model_proba']:.1%}")
    print(f"    Back odds:   ${ec['back_odds']:.2f}")
    print(f"    Implied:     {ec['expected_implied']:.1%} (= 1/{ec['back_odds']})")
    print(f"    Edge:        {ec['computed_edge']:.1%} (= {ec['model_proba']:.1%} - {ec['expected_implied']:.1%})")
    print(f"    Status:      {'✓ PASSED' if ec['passed'] else '✗ FAILED'}")

    # Filter thresholds
    ft = results["filter_thresholds"]
    print(f"\n[2] Filter Thresholds")
    print(f"    Min edge:    {ft['min_edge']:.0%} (config: {ft['expected_min_edge']:.0%})")
    print(f"    Odds range:  ${ft['odds_min']:.2f} - ${ft['odds_max']:.2f}")
    print(f"    Status:      {'✓ PASSED' if ft['passed'] else '✗ FAILED'}")

    # Telegram
    tg = results["telegram_test"]
    print(f"\n[3] Telegram Test Alert")
    print(f"    Status:      {'✓ SENT' if tg['sent'] else '✗ FAILED'}")
    if tg.get("error"):
        print(f"    Error:       {tg['error']}")

    print(f"\n[Message Format]")
    print("-" * 40)
    print(tg["message_format"])
    print("-" * 40)

    print("\n" + "=" * 60)
    if results["status"] == "PASSED":
        print("✓ GATE CHECK PASSED")
    else:
        print("✗ GATE CHECK FAILED")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Value bet alerter")
    parser.add_argument("--test", action="store_true", help="Run gate check (send test alert)")
    parser.add_argument("--run", action="store_true", help="Run live value detection")
    args = parser.parse_args()

    if args.test:
        results = run_gate_check()
        print_gate_report(results)
        return 0 if results["status"] == "PASSED" else 1

    elif args.run:
        logger.info("Live value detection not implemented yet - requires full integration")
        logger.info("Use live_selector.py for complete value detection pipeline")
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    exit(main())
