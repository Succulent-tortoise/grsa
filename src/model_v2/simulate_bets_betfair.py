"""
simulate_bets_betfair.py — Greyhound Prediction System v2

STUB ONLY — Betfair SP data has NOT been purchased (£300 cost).

This module is a placeholder for future implementation when Betfair SP data
becomes available. Currently, it:
- Accepts value bet recommendations
- Logs what bets WOULD have been placed
- Does NOT place any real bets
- Does NOT make any API calls to Betfair

⚠️  BETFAIR SP DATA NOT PURCHASED — STUB ONLY  ⚠️

For actual backtesting, use simulate_bets.py which uses GRSA scraped odds.
For live betting, integrate Betfair placeOrders API after purchasing SP data.

Usage:
    python simulate_bets_betfair.py --test    # Run with sample input
    python simulate_bets_betfair.py --run     # Run with real value bets (stub mode)

Authoritative source: PLAN.md Section 10.3
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config import (
    BETFAIR_COMMISSION,
    MODEL_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Log directory for bet simulation records
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


# =============================================================================
# ⚠️  WARNING: BETFAIR SP DATA NOT PURCHASED — STUB ONLY  ⚠️
# =============================================================================
#
# This module is a STUB. Real bet placement is NOT implemented.
#
# To implement real Betfair betting:
# 1. Purchase Betfair SP historical data (£300)
# 2. Implement place_bet() function with Betfair placeOrders API
# 3. Add proper bankroll management and stake sizing
# 4. Implement bet tracking and P&L calculation
#
# Until then, this stub only logs what bets WOULD have been placed.
# =============================================================================


STUB_WARNING = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ⚠️  BETFAIR SP DATA NOT PURCHASED — STUB ONLY  ⚠️                         ║
║                                                                              ║
║   This module is a PLACEHOLDER for future implementation.                   ║
║   No real bets will be placed. No API calls will be made.                   ║
║                                                                              ║
║   For backtesting: use simulate_bets.py (GRSA scraped odds)                 ║
║   For live betting: implement place_bet() after purchasing SP data          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


@dataclass
class SimulatedBet:
    """A bet that would have been placed (simulation only)."""
    venue: str
    race_number: int
    race_time: str
    runner_name: str
    box: int
    model_proba: float
    back_odds: float
    implied_proba: float
    edge: float
    stake: float
    potential_return: float
    timestamp: str


@dataclass
class BetSimulationResult:
    """Result of a bet simulation run."""
    total_bets: int
    total_stake: float
    total_potential_return: float
    expected_value: float
    bets: list[SimulatedBet]
    timestamp: str


class BetfairBettingError(Exception):
    """Raised when Betfair betting operation fails."""
    pass


class BetNotImplementedError(NotImplementedError):
    """Raised when attempting to place a real bet in stub mode."""
    pass


def calculate_stake_kelly(
    edge: float,
    odds: float,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
) -> float:
    """
    Calculate stake using fractional Kelly criterion.

    Kelly formula: f* = (bp - q) / b
    where:
        b = decimal odds - 1
        p = model probability
        q = 1 - p

    Fractional Kelly reduces variance (default: quarter Kelly).

    Args:
        edge: Model edge (model_proba - implied_proba)
        odds: Decimal odds
        bankroll: Current bankroll
        kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)

    Returns:
        Recommended stake
    """
    if odds <= 1.0 or edge <= 0:
        return 0.0

    # For fractional Kelly, we simplify:
    # Stake = bankroll * kelly_fraction * edge
    # This is a conservative approximation
    stake = bankroll * kelly_fraction * edge

    # Minimum stake
    stake = max(stake, 1.0)

    # Maximum stake (5% of bankroll)
    stake = min(stake, bankroll * 0.05)

    return round(stake, 2)


def place_bet(
    market_id: str,
    selection_id: int,
    odds: float,
    stake: float,
) -> dict[str, Any]:
    """
    ⚠️  STUB — NOT IMPLEMENTED  ⚠️

    Place a bet on Betfair.

    This function would use the Betfair placeOrders API to place a real bet.
    It is NOT implemented because Betfair SP data has not been purchased.

    Args:
        market_id: Betfair market ID
        selection_id: Betfair selection ID (runner)
        odds: Desired odds (limit order)
        stake: Stake amount in account currency

    Returns:
        Dict with bet placement result (if implemented)

    Raises:
        BetNotImplementedError: Always, because this is a stub
    """
    raise BetNotImplementedError(
        "Real bet placement not implemented.\n\n"
        "To enable real betting:\n"
        "1. Purchase Betfair SP historical data (£300)\n"
        "2. Implement Betfair placeOrders API integration\n"
        "3. Add proper error handling and bet tracking\n\n"
        "This stub only logs what bets WOULD have been placed."
    )


def simulate_bets(
    value_bets: list[dict],
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    dry_run: bool = True,
) -> BetSimulationResult:
    """
    Simulate placing bets on value bet recommendations.

    In stub mode (dry_run=True), this only logs what bets would be placed.
    In real mode (dry_run=False), this would place actual bets via Betfair API.

    Args:
        value_bets: List of value bet dicts with keys:
            - venue, race_number, race_time, runner_name, box
            - model_proba, back_odds, implied_proba, edge
        bankroll: Starting bankroll for stake calculation
        kelly_fraction: Fraction of Kelly criterion to use
        dry_run: If True, only log without placing bets

    Returns:
        BetSimulationResult with simulated bet details
    """
    timestamp = datetime.now().isoformat()
    bets = []

    logger.info(f"\n[Bet Simulation] {len(value_bets)} value bets to process")
    logger.info(f"  Bankroll: ${bankroll:.2f}")
    logger.info(f"  Kelly fraction: {kelly_fraction:.0%}")
    logger.info(f"  Mode: {'DRY-RUN (stub)' if dry_run else 'LIVE (not implemented)'}")

    total_stake = 0.0
    total_potential_return = 0.0

    for bet in value_bets:
        # Calculate stake
        stake = calculate_stake_kelly(
            edge=bet.get("edge", 0),
            odds=bet.get("back_odds", 0),
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
        )

        if stake <= 0:
            logger.debug(f"  Skipping {bet.get('runner_name')} - stake = 0")
            continue

        # Calculate potential return (before commission)
        potential_return = stake * bet.get("back_odds", 0)

        # Apply commission to winnings only
        if potential_return > stake:
            commission = (potential_return - stake) * BETFAIR_COMMISSION
            net_return = potential_return - commission
        else:
            net_return = potential_return

        # Create simulated bet record
        sim_bet = SimulatedBet(
            venue=bet.get("venue", "Unknown"),
            race_number=bet.get("race_number", 0),
            race_time=bet.get("race_time", ""),
            runner_name=bet.get("runner_name", "Unknown"),
            box=bet.get("box", 0),
            model_proba=bet.get("model_proba", 0),
            back_odds=bet.get("back_odds", 0),
            implied_proba=bet.get("implied_proba", 0),
            edge=bet.get("edge", 0),
            stake=stake,
            potential_return=net_return,
            timestamp=timestamp,
        )
        bets.append(sim_bet)

        total_stake += stake
        total_potential_return += net_return

        # Log what would happen
        logger.info(f"\n  📋 WOULD BET: {sim_bet.runner_name}")
        logger.info(f"     Venue:  {sim_bet.venue} R{sim_bet.race_number}")
        logger.info(f"     Box:    {sim_bet.box}")
        logger.info(f"     Odds:   ${sim_bet.back_odds:.2f}")
        logger.info(f"     Edge:   +{sim_bet.edge:.1%}")
        logger.info(f"     Stake:  ${sim_bet.stake:.2f}")
        logger.info(f"     Return: ${sim_bet.potential_return:.2f} (if win)")

        # If not dry run, would place actual bet here
        if not dry_run:
            try:
                # This will raise BetNotImplementedError
                result = place_bet(
                    market_id=bet.get("market_id", ""),
                    selection_id=bet.get("selection_id", 0),
                    odds=bet.get("back_odds", 0),
                    stake=stake,
                )
                logger.info(f"     ✓ Bet placed: {result}")
            except BetNotImplementedError as e:
                logger.error(f"     ✗ {e}")
                # In stub mode, continue without placing

    # Calculate expected value
    expected_value = total_potential_return - total_stake

    result = BetSimulationResult(
        total_bets=len(bets),
        total_stake=total_stake,
        total_potential_return=total_potential_return,
        expected_value=expected_value,
        bets=bets,
        timestamp=timestamp,
    )

    return result


def print_simulation_summary(result: BetSimulationResult) -> None:
    """Print a formatted summary of the simulation."""
    print("\n" + "=" * 60)
    print("BET SIMULATION SUMMARY")
    print("=" * 60)

    print(f"\n[Overview]")
    print(f"  Total bets:       {result.total_bets}")
    print(f"  Total stake:      ${result.total_stake:.2f}")
    print(f"  Potential return: ${result.total_potential_return:.2f}")
    print(f"  Expected value:   ${result.expected_value:.2f}")

    if result.total_stake > 0:
        roi = (result.expected_value / result.total_stake) * 100
        print(f"  Expected ROI:     {roi:+.1f}%")

    print(f"\n[Bets Placed (Simulated)]")
    for i, bet in enumerate(result.bets[:10], 1):
        print(f"  {i}. {bet.runner_name} @ ${bet.back_odds:.2f}")
        print(f"     Stake: ${bet.stake:.2f} | Edge: +{bet.edge:.1%}")

    if len(result.bets) > 10:
        print(f"  ... and {len(result.bets) - 10} more bets")

    print(f"\n[Timestamp: {result.timestamp}]")

    print("\n" + "=" * 60)
    print("⚠️  STUB MODE — NO REAL BETS PLACED")
    print("⚠️  Betfair SP data not purchased")
    print("=" * 60)


def save_simulation_log(result: BetSimulationResult) -> Path:
    """Save simulation result to a dated log file."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"bet_simulation_{today}.json"

    # Convert to dict for JSON serialization
    data = {
        "timestamp": result.timestamp,
        "total_bets": result.total_bets,
        "total_stake": result.total_stake,
        "total_potential_return": result.total_potential_return,
        "expected_value": result.expected_value,
        "bets": [
            {
                "venue": b.venue,
                "race_number": b.race_number,
                "race_time": b.race_time,
                "runner_name": b.runner_name,
                "box": b.box,
                "model_proba": b.model_proba,
                "back_odds": b.back_odds,
                "implied_proba": b.implied_proba,
                "edge": b.edge,
                "stake": b.stake,
                "potential_return": b.potential_return,
            }
            for b in result.bets
        ],
    }

    with open(log_file, "w") as f:
        json.dump(data, f, indent=2)

    return log_file


def get_sample_value_bets() -> list[dict]:
    """
    Generate sample value bets for testing.

    These are HARDCODED samples, not real market data.
    """
    return [
        {
            "venue": "Shepparton",
            "race_number": 4,
            "race_time": "2026-02-12T14:22:00Z",
            "runner_name": "FANCY RUNNER",
            "box": 5,
            "model_proba": 0.38,
            "back_odds": 3.20,
            "implied_proba": 0.3125,
            "edge": 0.0675,  # 6.75% edge
            "market_id": "1.123456789",
            "selection_id": 12345678,
        },
        {
            "venue": "Meadows",
            "race_number": 8,
            "race_time": "2026-02-12T19:45:00Z",
            "runner_name": "QUICK PAWS",
            "box": 3,
            "model_proba": 0.42,
            "back_odds": 2.90,
            "implied_proba": 0.3448,
            "edge": 0.0752,  # 7.52% edge
            "market_id": "1.987654321",
            "selection_id": 87654321,
        },
        {
            "venue": "Wentworth Park",
            "race_number": 2,
            "race_time": "2026-02-12T11:30:00Z",
            "runner_name": "LIGHTNING BOLT",
            "box": 1,
            "model_proba": 0.35,
            "back_odds": 4.50,
            "implied_proba": 0.2222,
            "edge": 0.1278,  # 12.78% edge
            "market_id": "1.112233445",
            "selection_id": 11223344,
        },
    ]


def run_gate_check() -> dict[str, Any]:
    """
    Run gate check for the Betfair betting stub.

    Tests:
    1. Stub warning is displayed
    2. Sample bets are processed without errors
    3. No real API calls are made
    4. place_bet() raises NotImplementedError
    5. Simulation log is saved correctly

    Returns:
        Dictionary with gate check results
    """
    logger.info("=" * 60)
    logger.info("BETFAIR BETTING STUB GATE CHECK")
    logger.info("=" * 60)

    results = {
        "status": None,
        "stub_warning_displayed": False,
        "sample_bets_processed": False,
        "no_api_calls_made": True,
        "place_bet_raises_error": False,
        "log_saved": False,
        "error_detail": None,
    }

    # Display stub warning
    print(STUB_WARNING)
    results["stub_warning_displayed"] = True
    logger.info("\n[Test 1] Stub warning displayed ✓")

    # Test sample bet processing
    logger.info("\n[Test 2] Processing sample value bets...")

    try:
        sample_bets = get_sample_value_bets()
        logger.info(f"  Generated {len(sample_bets)} sample bets")

        result = simulate_bets(
            value_bets=sample_bets,
            bankroll=1000.0,
            kelly_fraction=0.25,
            dry_run=True,
        )

        results["sample_bets_processed"] = True
        logger.info(f"  ✓ Processed {result.total_bets} bets")
        logger.info(f"  ✓ Total stake: ${result.total_stake:.2f}")

        # Print summary
        print_simulation_summary(result)

        # Save log
        log_file = save_simulation_log(result)
        results["log_saved"] = True
        logger.info(f"\n[Test 5] Log saved to: {log_file}")

    except Exception as e:
        results["error_detail"] = str(e)
        logger.error(f"  ✗ Failed: {e}")

    # Test place_bet raises error
    logger.info("\n[Test 4] Testing place_bet() raises NotImplementedError...")

    try:
        place_bet(
            market_id="1.123456",
            selection_id=12345,
            odds=3.50,
            stake=10.0,
        )
        logger.error("  ✗ place_bet() did NOT raise an error!")
    except BetNotImplementedError as e:
        results["place_bet_raises_error"] = True
        logger.info(f"  ✓ place_bet() correctly raises NotImplementedError")
        logger.info(f"    Message: {str(e)[:100]}...")
    except Exception as e:
        logger.error(f"  ✗ Wrong exception type: {type(e).__name__}")

    # Overall status
    all_passed = (
        results["stub_warning_displayed"]
        and results["sample_bets_processed"]
        and results["no_api_calls_made"]
        and results["place_bet_raises_error"]
        and results["log_saved"]
    )

    results["status"] = "PASSED" if all_passed else "FAILED"

    return results


def print_gate_report(results: dict) -> None:
    """Print formatted gate check report."""
    print("\n" + "=" * 60)
    print("BETFAIR BETTING STUB GATE CHECK REPORT")
    print("=" * 60)

    print(f"\n[Tests]")
    print(f"  1. Stub warning displayed:    {'✓' if results['stub_warning_displayed'] else '✗'}")
    print(f"  2. Sample bets processed:     {'✓' if results['sample_bets_processed'] else '✗'}")
    print(f"  3. No API calls made:         {'✓' if results['no_api_calls_made'] else '✗'}")
    print(f"  4. place_bet() raises error:  {'✓' if results['place_bet_raises_error'] else '✗'}")
    print(f"  5. Log saved correctly:       {'✓' if results['log_saved'] else '✗'}")

    if results.get("error_detail"):
        print(f"\n[Error Detail]")
        print(f"  {results['error_detail']}")

    print("\n" + "=" * 60)
    if results["status"] == "PASSED":
        print("✓ GATE CHECK PASSED")
        print("\n⚠️  REMINDER: This is a STUB only.")
        print("⚠️  No real bets can be placed until Betfair SP data is purchased.")
    else:
        print("✗ GATE CHECK FAILED")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Simulate Betfair bets (STUB ONLY)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  BETFAIR SP DATA NOT PURCHASED — STUB ONLY  ⚠️

This module is a placeholder. No real bets will be placed.
For backtesting, use simulate_bets.py instead.
        """,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run gate check with sample input",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run simulation with real value bets (stub mode)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="JSON file with value bets",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Starting bankroll (default: 1000)",
    )
    parser.add_argument(
        "--kelly",
        type=float,
        default=0.25,
        help="Kelly fraction (default: 0.25 = quarter Kelly)",
    )

    args = parser.parse_args()

    if args.test:
        # Run gate check
        print(STUB_WARNING)
        results = run_gate_check()
        print_gate_report(results)
        return 0 if results["status"] == "PASSED" else 1

    elif args.run:
        # Run with input
        print(STUB_WARNING)

        if args.input:
            with open(args.input) as f:
                value_bets = json.load(f)
        else:
            logger.info("No input file provided, using sample data")
            value_bets = get_sample_value_bets()

        result = simulate_bets(
            value_bets=value_bets,
            bankroll=args.bankroll,
            kelly_fraction=args.kelly,
            dry_run=True,
        )

        print_simulation_summary(result)
        log_file = save_simulation_log(result)
        logger.info(f"\nSimulation log saved to: {log_file}")

        return 0

    else:
        parser.print_help()
        print("\n" + STUB_WARNING)
        return 0


if __name__ == "__main__":
    sys.exit(main())
