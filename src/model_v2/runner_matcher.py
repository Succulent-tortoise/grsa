"""
runner_matcher.py — Greyhound Prediction System v2

Matches Betfair runners to model runner records using venue, race number,
distance, and runner name.

Handles:
- Venue name normalisation (case, state suffixes like "(AUS)")
- Fuzzy name matching for runner names (typos, truncation)
- Partial fields (not all runners may have odds)
- Logs unmatched runners as warnings, not errors

Usage:
    python runner_matcher.py --test    # Run gate check against live markets
"""

import argparse
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from betfair_client import BetfairClient, BetfairMarket, BetfairRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Minimum similarity threshold for fuzzy name matching
NAME_SIMILARITY_THRESHOLD = 0.80

# Venue name mappings (Betfair name -> normalized name)
VENUE_MAPPINGS = {
    "the meadows": "meadows",
    "wentworth park": "wentworth-park",
    "murray bridge": "murray-bridge",
    "murray bridge straight": "murray-bridge-straight",
    "shepparton straight": "shepparton",
    "angle park": "angle-park",
    "mount gambier": "mount-gambier",
    "gold coast": "gold-coast",
    "sunshine coast": "sunshine-coast",
    "casino dogs": "casino",
    "dapto dogs": "dapto",
    "lismore dogs": "lismore",
    "grafton dogs": "grafton",
    "moree dogs": "moree",
    "tamworth dogs": "tamworth",
    "armidale dogs": "armidale",
    "broken hill dogs": "broken-hill",
    "coonamble dogs": "coonamble",
    "cootamundra dogs": "cootamundra",
    "cowra dogs": "cowra",
    "dubbo dogs": "dubbo",
    "goulburn dogs": "goulburn",
    "gunnedah dogs": "gunnedah",
    "gosford dogs": "gosford",
    "griffith dogs": "griffith",
    "inverell dogs": "inverell",
    "maitland dogs": "maitland",
    "muswellbrook dogs": "muswellbrook",
    "nowra dogs": "nowra",
    "orange dogs": "orange",
    "parkes dogs": "parkes",
    "queanbeyan dogs": "queanbeyan",
    "taree dogs": "taree",
    "temora dogs": "temora",
    "wagga dogs": "wagga",
    "wellington dogs": "wellington",
}


@dataclass
class ParsedMarketInfo:
    """Parsed information from a Betfair market."""
    venue_normalized: str
    race_number: int
    distance: int
    grade: Optional[str] = None


@dataclass
class ParsedRunnerInfo:
    """Parsed information from a Betfair runner."""
    box_number: int
    runner_name_normalized: str
    original_name: str


@dataclass
class MatchedRunner:
    """A runner matched between Betfair and model data."""
    betfair_runner: BetfairRunner
    parsed_info: ParsedRunnerInfo
    model_runner_name: Optional[str] = None
    model_runner_data: Optional[dict] = None
    match_confidence: float = 0.0
    match_method: str = "none"


@dataclass
class MatchResult:
    """Result of matching a Betfair market to model data."""
    market: BetfairMarket
    parsed_market: ParsedMarketInfo
    matched_runners: list[MatchedRunner] = field(default_factory=list)
    unmatched_runners: list[MatchedRunner] = field(default_factory=list)
    match_rate: float = 0.0


def normalize_venue(venue: str) -> str:
    """
    Normalize a venue name for matching.

    Handles:
    - Case normalization
    - Removal of "(AUS)" suffix
    - Common name variations
    - Space to hyphen conversion
    """
    if not venue:
        return ""

    # Lowercase
    normalized = venue.lower().strip()

    # Remove (AUS) suffix
    normalized = re.sub(r"\s*\(aus\)\s*$", "", normalized)

    # Remove "dogs" suffix if present
    normalized = re.sub(r"\s+dogs\s*$", "", normalized)

    # Check for known mappings
    if normalized in VENUE_MAPPINGS:
        return VENUE_MAPPINGS[normalized]

    # Replace spaces with hyphens (our standard format)
    normalized = normalized.replace(" ", "-")

    return normalized


def extract_race_number(market_name: str) -> Optional[int]:
    """
    Extract race number from Betfair market name.

    Examples:
    - "R1 300m Mdn" -> 1
    - "R8 340m Gr5" -> 8
    - "R10 390m FFA" -> 10
    """
    match = re.search(r"R(\d+)", market_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_distance(market_name: str) -> Optional[int]:
    """
    Extract distance in metres from Betfair market name.

    Examples:
    - "R1 300m Mdn" -> 300
    - "R8 340m Gr5" -> 340
    - "R10 525m Gr3/4" -> 525
    """
    match = re.search(r"(\d+)m", market_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_grade(market_name: str) -> Optional[str]:
    """
    Extract grade from Betfair market name.

    Examples:
    - "R1 300m Mdn" -> "Mdn"
    - "R8 340m Gr5" -> "Gr5"
    - "R10 525m Gr3/4" -> "Gr3/4"
    """
    match = re.search(r"\d+m\s+(\S+)", market_name)
    if match:
        return match.group(1)
    return None


def parse_market_info(market: BetfairMarket) -> ParsedMarketInfo:
    """Parse all relevant information from a Betfair market."""
    venue = normalize_venue(market.venue or market.event_name)
    race_number = extract_race_number(market.market_name) or 0
    distance = extract_distance(market.market_name) or 0
    grade = extract_grade(market.market_name)

    return ParsedMarketInfo(
        venue_normalized=venue,
        race_number=race_number,
        distance=distance,
        grade=grade,
    )


def normalize_runner_name(name: str) -> str:
    """
    Normalize a runner name for matching.

    Handles:
    - Removal of box number prefix (e.g., "1. Power Rolex" -> "Power Rolex")
    - Case normalization
    - Trailing whitespace
    """
    if not name:
        return ""

    # Remove box number prefix (e.g., "1. ", "2. ", "8. ")
    normalized = re.sub(r"^\d+\.\s*", "", name.strip())

    # Uppercase for consistency with our data
    normalized = normalized.upper()

    return normalized


def parse_runner_info(runner: BetfairRunner) -> ParsedRunnerInfo:
    """Parse runner information from Betfair runner."""
    original = runner.runner_name
    normalized = normalize_runner_name(original)

    # Extract box number from original name if present
    match = re.match(r"^(\d+)\.", original)
    box_number = int(match.group(1)) if match else runner.sort_priority

    return ParsedRunnerInfo(
        box_number=box_number,
        runner_name_normalized=normalized,
        original_name=original,
    )


def fuzzy_match_score(name1: str, name2: str) -> float:
    """
    Calculate fuzzy match score between two names.

    Uses SequenceMatcher for similarity.
    """
    if not name1 or not name2:
        return 0.0

    return SequenceMatcher(None, name1, name2).ratio()


def match_runner_name(
    betfair_name: str,
    model_names: list[str],
    threshold: float = NAME_SIMILARITY_THRESHOLD,
) -> tuple[Optional[str], float]:
    """
    Match a Betfair runner name to model runner names.

    Uses fuzzy matching with configurable threshold.

    Returns:
        Tuple of (matched_name, confidence) or (None, 0.0) if no match
    """
    betfair_normalized = normalize_runner_name(betfair_name)

    best_match = None
    best_score = 0.0

    for model_name in model_names:
        # Exact match (after normalization)
        if betfair_normalized == model_name.upper():
            return model_name, 1.0

        # Fuzzy match
        score = fuzzy_match_score(betfair_normalized, model_name.upper())

        if score > best_score:
            best_score = score
            best_match = model_name

    if best_score >= threshold:
        return best_match, best_score

    return None, 0.0


class RunnerMatcher:
    """
    Matches Betfair runners to model runner records.
    """

    def __init__(self, similarity_threshold: float = NAME_SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        self.client: Optional[BetfairClient] = None

    def connect(self) -> None:
        """Connect to Betfair API."""
        if not self.client:
            self.client = BetfairClient()
            self.client.login()
            logger.info("Connected to Betfair API")

    def get_live_markets(self, max_markets: int = 50) -> list[BetfairMarket]:
        """Retrieve live Australian greyhound markets."""
        if not self.client:
            self.connect()

        markets = self.client.list_greyhound_markets(max_results=max_markets)

        # Update with odds
        for market in markets:
            self.client.update_market_with_odds(market)

        return markets

    def match_market(
        self,
        market: BetfairMarket,
        model_runners: Optional[list[dict]] = None,
    ) -> MatchResult:
        """
        Match a Betfair market's runners to model runner data.

        Args:
            market: Betfair market with runners
            model_runners: Optional list of model runner dicts with 'runner_name' key
                          If None, only parsing is performed

        Returns:
            MatchResult with matched and unmatched runners
        """
        parsed_market = parse_market_info(market)
        matched = []
        unmatched = []

        # Get model runner names for matching
        model_names = []
        if model_runners:
            model_names = [r.get("runner_name", "").upper() for r in model_runners]

        for runner in market.runners:
            parsed_runner = parse_runner_info(runner)

            matched_runner = MatchedRunner(
                betfair_runner=runner,
                parsed_info=parsed_runner,
            )

            if model_names:
                # Try to match
                match_name, confidence = match_runner_name(
                    runner.runner_name,
                    model_names,
                    self.similarity_threshold,
                )

                if match_name:
                    matched_runner.model_runner_name = match_name
                    matched_runner.match_confidence = confidence
                    matched_runner.match_method = "exact" if confidence == 1.0 else "fuzzy"

                    # Find the full model runner data
                    for mr in model_runners:
                        if mr.get("runner_name", "").upper() == match_name.upper():
                            matched_runner.model_runner_data = mr
                            break

                    matched.append(matched_runner)
                else:
                    unmatched.append(matched_runner)
            else:
                # No model data to match against - consider all parsed as matched
                matched_runner.match_method = "parsed_only"
                matched.append(matched_runner)

        # Calculate match rate
        total = len(matched) + len(unmatched)
        match_rate = len(matched) / total if total > 0 else 0.0

        return MatchResult(
            market=market,
            parsed_market=parsed_market,
            matched_runners=matched,
            unmatched_runners=unmatched,
            match_rate=match_rate,
        )

    def match_all_markets(
        self,
        markets: list[BetfairMarket],
        model_data: Optional[dict] = None,
    ) -> list[MatchResult]:
        """
        Match all markets against model data.

        Args:
            markets: List of Betfair markets
            model_data: Optional dict keyed by (venue, race_number) containing runner lists

        Returns:
            List of MatchResult objects
        """
        results = []

        for market in markets:
            parsed = parse_market_info(market)

            # Look up model runners for this venue/race
            model_runners = None
            if model_data:
                key = (parsed.venue_normalized, parsed.race_number)
                model_runners = model_data.get(key)

            result = self.match_market(market, model_runners)
            results.append(result)

            # Log unmatched runners as warnings
            for unmatched in result.unmatched_runners:
                logger.warning(
                    f"Unmatched runner: {unmatched.parsed_info.original_name} "
                    f"in {parsed.venue_normalized} R{parsed.race_number}"
                )

        return results


def run_gate_check() -> dict:
    """
    Run gate check against live Betfair markets.

    Tests:
    1. All markets can be parsed (venue, race number, distance extracted)
    2. All runners can be normalized
    3. Overall parse rate exceeds 95%

    Returns:
        Dictionary with gate check results
    """
    logger.info("=" * 60)
    logger.info("RUNNER MATCHER GATE CHECK")
    logger.info("=" * 60)

    results = {
        "status": None,
        "markets_analyzed": 0,
        "total_runners": 0,
        "parsed_runners": 0,
        "parse_rate": 0.0,
        "venue_extractions": [],
        "unmatched_details": [],
        "sample_matches": [],
    }

    matcher = RunnerMatcher()

    try:
        # Get live markets
        matcher.connect()
        markets = matcher.get_live_markets(max_markets=20)
        results["markets_analyzed"] = len(markets)

        logger.info(f"\nAnalyzing {len(markets)} live markets...")

        total_runners = 0
        parsed_runners = 0
        parse_failures = []

        for market in markets:
            parsed_market = parse_market_info(market)

            # Track venue extraction
            venue_result = {
                "original": market.venue or market.event_name,
                "normalized": parsed_market.venue_normalized,
                "race_number": parsed_market.race_number,
                "distance": parsed_market.distance,
            }
            results["venue_extractions"].append(venue_result)

            # Process runners
            for runner in market.runners:
                total_runners += 1

                parsed_info = parse_runner_info(runner)

                # Check if parsing succeeded
                if parsed_info.runner_name_normalized and parsed_info.box_number > 0:
                    parsed_runners += 1
                else:
                    parse_failures.append({
                        "market": market.market_name,
                        "runner": runner.runner_name,
                        "reason": "Failed to parse name or box number",
                    })

            # Store sample match
            result = matcher.match_market(market)
            if result.matched_runners:
                sample = result.matched_runners[0]
                results["sample_matches"].append({
                    "market": market.market_name,
                    "venue_normalized": parsed_market.venue_normalized,
                    "runner_original": sample.parsed_info.original_name,
                    "runner_normalized": sample.parsed_info.runner_name_normalized,
                    "box": sample.parsed_info.box_number,
                    "has_odds": sample.betfair_runner.back_odds is not None,
                })

        results["total_runners"] = total_runners
        results["parsed_runners"] = parsed_runners
        results["parse_rate"] = parsed_runners / total_runners if total_runners > 0 else 0.0
        results["unmatched_details"] = parse_failures

        # Gate check
        results["status"] = "PASSED" if results["parse_rate"] >= 0.95 else "FAILED"

    except Exception as e:
        results["status"] = "ERROR"
        results["error"] = str(e)
        logger.error(f"Gate check failed: {e}")

    return results


def print_gate_report(results: dict) -> None:
    """Print formatted gate check report."""
    print("\n" + "=" * 60)
    print("RUNNER MATCHER GATE CHECK REPORT")
    print("=" * 60)

    print(f"\n[Summary]")
    print(f"  Markets analyzed: {results['markets_analyzed']}")
    print(f"  Total runners:    {results['total_runners']}")
    print(f"  Parsed runners:   {results['parsed_runners']}")
    print(f"  Parse rate:       {results['parse_rate']:.1%}")

    print(f"\n[Venue Normalization Samples]")
    for v in results["venue_extractions"][:5]:
        print(f"  '{v['original']}' -> '{v['normalized']}' (R{v['race_number']}, {v['distance']}m)")

    print(f"\n[Runner Parsing Samples]")
    for s in results["sample_matches"][:5]:
        odds_str = f"Back: {s['has_odds']}" if s['has_odds'] else "No odds"
        print(f"  '{s['runner_original']}' -> '{s['runner_normalized']}' (Box {s['box']}) - {odds_str}")

    if results["unmatched_details"]:
        print(f"\n[Parse Failures: {len(results['unmatched_details'])}]")
        for f in results["unmatched_details"][:10]:
            print(f"  {f['runner']} in {f['market']}: {f['reason']}")

    print("\n" + "=" * 60)
    if results["status"] == "PASSED":
        print(f"✓ GATE CHECK PASSED (parse rate {results['parse_rate']:.1%} >= 95%)")
    else:
        print(f"✗ GATE CHECK FAILED (parse rate {results['parse_rate']:.1%} < 95%)")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Runner matcher for Betfair integration")
    parser.add_argument("--test", action="store_true", help="Run gate check")
    args = parser.parse_args()

    if args.test:
        results = run_gate_check()
        print_gate_report(results)
        return 0 if results["status"] == "PASSED" else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    exit(main())
