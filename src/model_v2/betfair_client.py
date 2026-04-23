"""
betfair_client.py — Greyhound Prediction System v2

Betfair API client for retrieving Australian greyhound markets.

Handles:
- SSL certificate authentication
- Session management
- Market data retrieval
- Graceful error handling

Usage:
    python betfair_client.py --test      # Test API connection
    python betfair_client.py --markets   # List today's greyhound markets

Requirements:
- Betfair API key, username, password in .env
- SSL client certificates in certs/ directory:
  - client-2048.crt (client certificate)
  - client-2048.key (private key)
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from api_call_tracker import APICallTracker, RateLimitExceeded
from config import (
    BETFAIR_API_KEY,
    BETFAIR_CERT_PATH,
    BETFAIR_PASSWORD,
    BETFAIR_USERNAME,
    MAX_API_CALLS_PER_HOUR,
    MAX_API_CALLS_PER_DAY,
    API_WARN_THRESHOLD_80,
    API_WARN_THRESHOLD_90,
    API_EMERGENCY_THRESHOLD,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Betfair API endpoints
BETFAIR_LOGIN_URL = "https://identitysso-cert.betfair.com/api/certlogin"
BETFAIR_API_URL = "https://api.betfair.com/exchange/betting/rest/v1.0"
BETFAIR_NAVIGATION_URL = "https://api.betfair.com/exchange/betting/rest/v1.0/navigation/menu.json"

# Australian greyhound event type ID
GREYHOUND_EVENT_TYPE_ID = "4339"

# Australian jurisdiction
AUS_JURISDICTION = "AUS"


@dataclass
class BetfairRunner:
    """Represents a runner in a Betfair market."""
    selection_id: int
    runner_name: str
    handicap: float
    sort_priority: int
    back_odds: Optional[float] = None
    lay_odds: Optional[float] = None


@dataclass
class BetfairMarket:
    """Represents a Betfair market with runners."""
    market_id: str
    event_name: str
    market_name: str
    market_start_time: str
    venue: Optional[str] = None
    runners: list[BetfairRunner] = None

    def __post_init__(self):
        if self.runners is None:
            self.runners = []


class BetfairAuthError(Exception):
    """Raised when Betfair authentication fails."""
    pass


class BetfairApiError(Exception):
    """Raised when Betfair API call fails."""
    pass


class CertificateMissingError(BetfairAuthError):
    """Raised when SSL certificates are not found."""
    pass


class BetfairClient:
    """
    Betfair API client for Australian greyhound markets.

    Uses certificate-based authentication.
    """

    def __init__(self):
        self.session_token: Optional[str] = None
        self.session: Optional[requests.Session] = None
        self._cert_files: Optional[tuple[str, str]] = None
        self.logger = logging.getLogger("betfair_client")

        # Initialize API call tracker
        self.api_tracker = APICallTracker(
            max_per_hour=MAX_API_CALLS_PER_HOUR,
            max_per_day=MAX_API_CALLS_PER_DAY
        )
        self.logger.info("[API TRACKER] Initialized with hourly={}, daily={}".format(
            MAX_API_CALLS_PER_HOUR, MAX_API_CALLS_PER_DAY
        ))

    def _get_cert_files(self) -> tuple[str, str]:
        """Locate and validate SSL certificate files."""
        cert_dir = Path(BETFAIR_CERT_PATH)

        # Check common certificate file names
        cert_names = [
            ("client-2048.crt", "client-2048.key"),
            ("client.crt", "client.key"),
            ("betfair.crt", "betfair.key"),
        ]

        for cert_name, key_name in cert_names:
            cert_path = cert_dir / cert_name
            key_path = cert_dir / key_name

            if cert_path.exists() and key_path.exists():
                logger.info(f"Found certificates: {cert_path}, {key_path}")
                return (str(cert_path), str(key_path))

        # No certificates found
        raise CertificateMissingError(
            f"SSL certificates not found in {cert_dir}.\n"
            f"Required files:\n"
            f"  - client-2048.crt (or client.crt)\n"
            f"  - client-2048.key (or client.key)\n\n"
            f"To obtain certificates:\n"
            f"  1. Log in to https://www.betfair.com.au\n"
            f"  2. Go to My Account > Security > API-NG\n"
            f"  3. Generate and download client certificates\n"
            f"  4. Place in: {cert_dir}/"
        )

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()

        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)

        return session

    def _track_api_call(self, endpoint: str) -> None:
        """
        Track an API call and check rate limits.

        Logs the call, checks limits, and sends warnings/alerts as needed.
        Raises RateLimitExceeded if limit is violated.

        Args:
            endpoint: API endpoint being called (for logging)

        Raises:
            RateLimitExceeded: If hourly or daily limit exceeded
        """
        # Check limits BEFORE making call
        error = self.api_tracker.check_limits()
        if error:
            self.logger.critical(f"🚨 API LIMIT VIOLATED: {error}")
            raise RateLimitExceeded(error)

        # Register the call
        hourly, daily = self.api_tracker.register_call()
        status = self.api_tracker.get_status()

        # Log with running totals
        self.logger.info(
            f"[API Call #{daily}] {endpoint} | "
            f"{self.api_tracker.format_status()}"
        )

        # Check warning thresholds
        warning_level = self.api_tracker.get_warning_level()

        if warning_level == "CRITICAL":
            # 90% threshold - CRITICAL
            if status.hourly_percent >= API_WARN_THRESHOLD_90:
                self.logger.critical(
                    f"⚠️ CRITICAL: {status.hourly_percent:.1%} of hourly API limit used! "
                    f"({status.hourly_count}/{status.hourly_max})"
                )
            if status.daily_percent >= API_WARN_THRESHOLD_90:
                self.logger.critical(
                    f"⚠️ CRITICAL: {status.daily_percent:.1%} of daily API limit used! "
                    f"({status.daily_count}/{status.daily_max})"
                )

        elif warning_level == "WARNING":
            # 80% threshold - WARNING
            if status.hourly_percent >= API_WARN_THRESHOLD_80:
                self.logger.warning(
                    f"⚠️ WARNING: {status.hourly_percent:.1%} of hourly API limit used "
                    f"({status.hourly_count}/{status.hourly_max})"
                )
            if status.daily_percent >= API_WARN_THRESHOLD_80:
                self.logger.warning(
                    f"⚠️ WARNING: {status.daily_percent:.1%} of daily API limit used "
                    f"({status.daily_count}/{status.daily_max})"
                )

    def get_daily_api_calls(self) -> int:
        """Get total API calls made today."""
        return self.api_tracker.get_daily_total()

    def login(self) -> bool:
        """
        Authenticate with Betfair using SSL certificates.

        Returns:
            True if authentication successful

        Raises:
            CertificateMissingError: If certificates not found
            BetfairAuthError: If authentication fails
        """
        # Track this API call
        self._track_api_call("certlogin")

        # Validate credentials exist
        if not BETFAIR_API_KEY:
            raise BetfairAuthError("BETFAIR_API_KEY not set in .env")
        if not BETFAIR_USERNAME:
            raise BetfairAuthError("BETFAIR_USERNAME not set in .env")
        if not BETFAIR_PASSWORD:
            raise BetfairAuthError("BETFAIR_PASSWORD not set in .env")

        # Get certificate files
        self._cert_files = self._get_cert_files()

        # Create session
        self.session = self._create_session()

        # Prepare login request
        headers = {
            "X-Application": BETFAIR_API_KEY,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {
            "username": BETFAIR_USERNAME,
            "password": BETFAIR_PASSWORD,
        }

        logger.info("Attempting Betfair login...")

        try:
            response = self.session.post(
                BETFAIR_LOGIN_URL,
                data=data,
                headers=headers,
                cert=self._cert_files,
                timeout=30,
            )

            response.raise_for_status()
            result = response.json()

            if "sessionToken" in result:
                self.session_token = result["sessionToken"]
                logger.info("Betfair login successful")
                return True
            else:
                error_msg = result.get("error", "Unknown error")
                raise BetfairAuthError(f"Login failed: {error_msg}")

        except requests.exceptions.SSLError as e:
            raise BetfairAuthError(f"SSL error during login: {e}")
        except requests.exceptions.RequestException as e:
            raise BetfairAuthError(f"Request error during login: {e}")

    def _get_headers(self) -> dict[str, str]:
        """Get headers for authenticated API requests."""
        if not self.session_token:
            raise BetfairAuthError("Not authenticated - call login() first")

        return {
            "X-Application": BETFAIR_API_KEY,
            "X-Authentication": self.session_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def list_event_types(self) -> list[dict]:
        """List all available event types."""
        # Track this API call
        self._track_api_call("listEventTypes")

        if not self.session:
            raise BetfairAuthError("Not authenticated")

        url = f"{BETFAIR_API_URL}/listEventTypes/"

        payload = {
            "filter": {},
        }

        response = self.session.post(
            url,
            headers=self._get_headers(),
            json=payload,
            timeout=30,
        )

        response.raise_for_status()
        return response.json()

    def list_greyhound_markets(
        self,
        country: str = "AU",
        max_results: int = 100,
    ) -> list[BetfairMarket]:
        """
        List Australian greyhound markets.

        Args:
            country: Country code (default: AU)
            max_results: Maximum markets to return

        Returns:
            List of BetfairMarket objects
        """
        # Track this API call
        self._track_api_call("listMarketCatalogue")

        if not self.session:
            raise BetfairAuthError("Not authenticated")

        url = f"{BETFAIR_API_URL}/listMarketCatalogue/"

        # Filter for Australian greyhounds
        payload = {
            "filter": {
                "eventTypeIds": [GREYHOUND_EVENT_TYPE_ID],
                "marketTypeCodes": ["WIN"],
                "marketCountries": [country],
            },
            "maxResults": max_results,
            "marketProjection": [
                "EVENT",
                "EVENT_TYPE",
                "MARKET_START_TIME",
                "RUNNER_DESCRIPTION",
            ],
        }

        logger.info(f"Fetching greyhound markets for {country}...")

        response = self.session.post(
            url,
            headers=self._get_headers(),
            json=payload,
            timeout=30,
        )

        if response.status_code == 401:
            raise BetfairAuthError("Session expired - re-authenticate")

        response.raise_for_status()
        data = response.json()

        markets = []
        for item in data:
            market = BetfairMarket(
                market_id=item["marketId"],
                event_name=item["event"]["name"],
                market_name=item["marketName"],
                market_start_time=item["marketStartTime"],
                venue=item["event"].get("venue"),
                runners=[],
            )

            for runner in item.get("runners", []):
                market.runners.append(BetfairRunner(
                    selection_id=runner["selectionId"],
                    runner_name=runner["runnerName"],
                    handicap=runner.get("handicap", 0),
                    sort_priority=runner.get("sortPriority", 0),
                ))

            markets.append(market)

        logger.info(f"Found {len(markets)} greyhound markets")
        return markets

    def get_market_odds(
        self,
        market_id: str,
    ) -> dict[str, Any]:
        """
        Get current odds for a market.

        Args:
            market_id: Betfair market ID

        Returns:
            Dictionary with runner odds
        """
        # Track this API call
        self._track_api_call("listMarketBook")

        if not self.session:
            raise BetfairAuthError("Not authenticated")

        url = f"{BETFAIR_API_URL}/listMarketBook/"

        payload = {
            "marketIds": [market_id],
            "priceProjection": {
                "priceData": ["EX_BEST_OFFERS"],
                "exBestOffersOverrides": {
                    "bestPricesDepth": 1,
                },
            },
        }

        response = self.session.post(
            url,
            headers=self._get_headers(),
            json=payload,
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()

        if not data:
            return {}

        result = {}
        market_book = data[0]

        for runner in market_book.get("runners", []):
            selection_id = str(runner["selectionId"])

            ex = runner.get("ex", {})
            back_prices = ex.get("availableToBack", [])
            lay_prices = ex.get("availableToLay", [])

            back_odds = back_prices[0]["price"] if back_prices else None
            lay_odds = lay_prices[0]["price"] if lay_prices else None

            result[selection_id] = {
                "back_odds": back_odds,
                "lay_odds": lay_odds,
                "total_matched": runner.get("totalMatched", 0),
            }

        return result

    def update_market_with_odds(self, market: BetfairMarket) -> BetfairMarket:
        """Update a market with current odds."""
        odds = self.get_market_odds(market.market_id)

        for runner in market.runners:
            runner_odds = odds.get(str(runner.selection_id), {})
            runner.back_odds = runner_odds.get("back_odds")
            runner.lay_odds = runner_odds.get("lay_odds")

        return market


def test_connection() -> dict[str, Any]:
    """
    Test Betfair API connection.

    Returns:
        Dictionary with test results
    """
    result = {
        "status": None,
        "message": None,
        "certificates_found": False,
        "authentication_successful": False,
        "markets_retrieved": False,
        "sample_market": None,
        "error_detail": None,
    }

    client = BetfairClient()

    # Check for certificates
    try:
        client._get_cert_files()
        result["certificates_found"] = True
        logger.info("✓ SSL certificates found")
    except CertificateMissingError as e:
        result["status"] = "PENDING"
        result["message"] = "SSL certificates required"
        result["error_detail"] = str(e)
        logger.warning(f"✗ {e}")
        return result

    # Attempt authentication
    try:
        client.login()
        result["authentication_successful"] = True
        logger.info("✓ Authentication successful")
    except BetfairAuthError as e:
        result["status"] = "FAILED"
        result["message"] = "Authentication failed"
        result["error_detail"] = str(e)
        logger.error(f"✗ {e}")
        return result

    # Retrieve markets
    try:
        markets = client.list_greyhound_markets(max_results=5)
        result["markets_retrieved"] = True
        logger.info(f"✓ Retrieved {len(markets)} markets")

        if markets:
            # Update first market with odds
            sample = client.update_market_with_odds(markets[0])
            result["sample_market"] = {
                "market_id": sample.market_id,
                "event_name": sample.event_name,
                "market_name": sample.market_name,
                "market_start_time": sample.market_start_time,
                "venue": sample.venue,
                "runner_count": len(sample.runners),
                "runners": [
                    {
                        "selection_id": r.selection_id,
                        "runner_name": r.runner_name,
                        "back_odds": r.back_odds,
                        "lay_odds": r.lay_odds,
                    }
                    for r in sample.runners[:5]  # First 5 runners only
                ],
            }

        result["status"] = "PASSED"
        result["message"] = "API connection successful"

    except (BetfairApiError, BetfairAuthError) as e:
        result["status"] = "FAILED"
        result["message"] = "Market retrieval failed"
        result["error_detail"] = str(e)
        logger.error(f"✗ {e}")

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Betfair API client")
    parser.add_argument("--test", action="store_true", help="Test API connection")
    parser.add_argument("--markets", action="store_true", help="List today's greyhound markets")
    parser.add_argument("--max", type=int, default=10, help="Max markets to display")
    args = parser.parse_args()

    if args.test:
        print("\n" + "=" * 60)
        print("BETFAIR API CONNECTION TEST")
        print("=" * 60)

        result = test_connection()

        print(f"\n[Status] {result['status']}")
        print(f"[Message] {result['message']}")
        print(f"\nCertificates found: {result['certificates_found']}")
        print(f"Authentication: {result['authentication_successful']}")
        print(f"Markets retrieved: {result['markets_retrieved']}")

        if result["sample_market"]:
            print("\n[Sample Market]")
            market = result["sample_market"]
            print(f"  Market ID: {market['market_id']}")
            print(f"  Event: {market['event_name']}")
            print(f"  Market: {market['market_name']}")
            print(f"  Start: {market['market_start_time']}")
            print(f"  Runners: {market['runner_count']}")
            print("\n  Sample runners:")
            for r in market["runners"]:
                odds_str = f"Back: {r['back_odds']}, Lay: {r['lay_odds']}" if r['back_odds'] else "No odds"
                print(f"    - {r['runner_name']}: {odds_str}")

        if result["error_detail"]:
            print(f"\n[Error Detail]\n{result['error_detail']}")

        print("\n" + "=" * 60)

        if result["status"] == "PASSED":
            print("✓ GATE CHECK PASSED")
        elif result["status"] == "PENDING":
            print("⏳ GATE CHECK PENDING - SSL certificates required")
        else:
            print("✗ GATE CHECK FAILED")
        print("=" * 60)

        sys.exit(0 if result["status"] in ["PASSED", "PENDING"] else 1)

    elif args.markets:
        client = BetfairClient()
        client.login()

        markets = client.list_greyhound_markets(max_results=args.max)

        print(f"\n[Australian Greyhound Markets - {len(markets)} found]\n")

        for market in markets[:args.max]:
            print(f"{market.event_name} - {market.market_name}")
            print(f"  Start: {market.market_start_time}")
            print(f"  Market ID: {market.market_id}")
            print(f"  Runners: {len(market.runners)}")
            print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
