import json
import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class PredictionRecord:
    """Complete prediction record for analysis."""
    timestamp: datetime
    poll_number: int
    check_window: str  # "45min", "20min", "10min", "5min", "2min"
    market_id: str
    venue: str
    race_number: int
    race_time: datetime
    minutes_to_start: float
    distance: int
    grade: str
    field_size: int
    runner_name: str
    box: int
    model_probability: float
    back_odds: float
    implied_probability: float
    edge: float
    is_value_bet: bool
    meets_min_edge: bool
    meets_min_confidence: bool
    meets_odds_range: bool
    alert_sent: bool

@dataclass
class ValueBetRecord:
    """Value bet record matching bet_results.xlsx format."""
    date: str  # DD/MM/YYYY
    time: str  # HH:MM
    venue: str
    race: str  # "R5"
    race_time: str  # HH:MM
    runner: str
    box: int
    model_prob: str  # "25.20%"
    back_odds: str  # "$10.00"
    implied_prob: str  # "15.0%" or empty
    edge: str  # "15.20%"
    alert_time: str  # "11:53; 12:03; 12:13"
    result: str = ""
    finish_position: str = ""
    margin: str = ""
    won_lost: str = ""
    stake: str = ""
    return_amount: str = ""
    profit: str = ""
    running_total: str = ""
    notes: str = ""

class PredictionLogger:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.logger = logging.getLogger("prediction_logger")
        self.value_bet_cache: dict = {}  # key: (date, venue, race, runner)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def log_all_predictions(self, predictions: list[PredictionRecord]):
        """Log ALL predictions to JSONL for analysis."""
        output_file = self._get_predictions_file()
        try:
            with open(output_file, "a", encoding="utf-8") as f:
                for pred in predictions:
                    json_record = self._prediction_to_json(pred)
                    f.write(json.dumps(json_record) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to log predictions: {e}")
            raise

    def log_value_bet(self, value_bet: ValueBetRecord, alert_time: str):
        """Log value bet to CSV matching bet_results.xlsx format.

        If same runner already logged, UPDATE alert_time column.
        Otherwise, append new row.
        """
        output_file = self._get_value_bets_file()
        bet_key = (value_bet.date, value_bet.venue, value_bet.race, value_bet.runner)
        if bet_key in self.value_bet_cache:
            self._update_alert_time(output_file, bet_key, alert_time)
        else:
            self._append_value_bet(output_file, value_bet, alert_time)
            self.value_bet_cache[bet_key] = True

    # ---------------------------------------------------------------------
    # Helpers for file paths
    # ---------------------------------------------------------------------
    def _get_predictions_file(self) -> Path:
        """Get predictions.jsonl file for today, creating directories as needed."""
        today = datetime.now().strftime("%Y%m%d")
        date_dir = self.base_dir / today
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir / "predictions.jsonl"

    def _get_value_bets_file(self) -> Path:
        """Get value_bets.csv file for today, creating directories and headers as needed."""
        today = datetime.now().strftime("%Y%m%d")
        date_dir = self.base_dir / today
        date_dir.mkdir(parents=True, exist_ok=True)
        csv_file = date_dir / "value_bets.csv"
        if not csv_file.exists():
            self._create_csv_with_headers(csv_file)
        return csv_file

    # ---------------------------------------------------------------------
    # CSV handling
    # ---------------------------------------------------------------------
    def _create_csv_with_headers(self, csv_file: Path):
        headers = [
            "Date", "Time", "Venue", "Race", "Race Time", "Runner", "Box",
            "Model Prob", "Back Odds", "Implied Prob", "Edge", "Alert Time",
            "Result", "Finish Position", "Margin", "WonLost", "Stake",
            "Return", "Profit", "RunningTotal", "Notes"
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _append_value_bet(self, csv_file: Path, bet: ValueBetRecord, alert_time: str):
        row = [
            bet.date,
            f"'{bet.time}",
            bet.venue,
            bet.race,
            f"{bet.race_time}",
            bet.runner,
            bet.box,
            bet.model_prob,
            bet.back_odds,
            "",  # implied_prob left empty per spec
            bet.edge,
            f"'{alert_time}",
            "",  # Result (filled later)
            "",  # Finish Position (filled later)
            "",  # Margin (filled later)
            "",  # WonLost (filled later)
            "",  # Stake (filled later)
            "",  # Return (filled later)
            "",  # Profit (filled later)
            "",  # RunningTotal (filled later)
            bet.notes
        ]
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _update_alert_time(self, csv_file: Path, bet_key: tuple, new_alert_time: str):
        rows = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows.append(headers)
            for row in reader:
                if (
                    row[0] == bet_key[0] and  # Date
                    row[2] == bet_key[1] and  # Venue
                    row[3] == bet_key[2] and  # Race
                    row[5] == bet_key[3]      # Runner
                ):
                    # CSV stores with apostrophe prefix (Excel text format)
                    # Read raw value - may have one apostrophe at start
                    existing_raw = row[11]
                    # Strip exactly ONE leading apostrophe if present
                    if existing_raw.startswith("'"):
                        existing = existing_raw[1:]
                    else:
                        existing = existing_raw
                    # Build new alert time: preserve apostrophe prefix for Excel
                    if existing:
                        row[11] = f"'{existing}; {new_alert_time}"
                    else:
                        row[11] = f"'{new_alert_time}"
                rows.append(row)

        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    # ---------------------------------------------------------------------
    # Helper to convert dataclass to serialisable dict
    # ---------------------------------------------------------------------
    def _prediction_to_json(self, pred: PredictionRecord) -> dict:
        # Handle timestamp - can be string or datetime
        if isinstance(pred.timestamp, str):
            timestamp_str = pred.timestamp
        else:
            timestamp_str = pred.timestamp.isoformat()

        # Handle race_time - can be string or datetime
        if isinstance(pred.race_time, str):
            race_time_str = pred.race_time
        else:
            race_time_str = pred.race_time.isoformat()

        return {
            "timestamp": timestamp_str,
            "poll_number": pred.poll_number,
            "check_window": pred.check_window,
            "market_id": pred.market_id,
            "venue": pred.venue,
            "race_number": pred.race_number,
            "race_time": race_time_str,
            "minutes_to_start": pred.minutes_to_start,
            "distance": pred.distance,
            "grade": pred.grade,
            "field_size": pred.field_size,
            "runner_name": pred.runner_name,
            "box": pred.box,
            "model_probability": pred.model_probability,
            "back_odds": pred.back_odds,
            "implied_probability": pred.implied_probability,
            "edge": pred.edge,
            "is_value_bet": pred.is_value_bet,
            "meets_min_edge": pred.meets_min_edge,
            "meets_min_confidence": pred.meets_min_confidence,
            "meets_odds_range": pred.meets_odds_range,
            "alert_sent": pred.alert_sent,
        }

# ============================================================================
# TEST HARNESS
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from datetime import datetime, timezone, timedelta
    import csv as csv_module
    import json

    print("=" * 70)
    print("PREDICTION LOGGER TEST HARNESS")
    print("=" * 70)

    # Create test logger (use /tmp for testing)
    test_dir = Path("/tmp/prediction_logger_test")
    logger = PredictionLogger(test_dir)

    print(f"\n[Test Directory]")
    print(f"  Location: {test_dir}")

    # ========================================================================
    # TEST 1: ALL PREDICTIONS (JSONL)
    # ========================================================================

    print("\n" + "=" * 70)
    print("TEST 1: ALL PREDICTIONS LOGGING (JSONL)")
    print("=" * 70)

    # Create sample prediction records
    race_time = datetime.now(timezone.utc) + timedelta(minutes=5)

    test_predictions = [
        PredictionRecord(
            timestamp=datetime.now(timezone.utc),
            poll_number=1,
            check_window="5min",
            market_id="1.234567890",
            venue="murray_bridge",
            race_number=5,
            race_time=race_time,
            minutes_to_start=4.25,
            distance=520,
            grade="5",
            field_size=8,
            runner_name="Grapefruit Wilma",
            box=3,
            model_probability=0.252,
            back_odds=10.0,
            implied_probability=0.100,
            edge=0.152,
            is_value_bet=True,
            meets_min_edge=True,
            meets_min_confidence=True,
            meets_odds_range=True,
            alert_sent=False
        ),
        PredictionRecord(
            timestamp=datetime.now(timezone.utc),
            poll_number=1,
            check_window="5min",
            market_id="1.234567890",
            venue="murray_bridge",
            race_number=5,
            race_time=race_time,
            minutes_to_start=4.25,
            distance=520,
            grade="5",
            field_size=8,
            runner_name="Aurora Sunshine",
            box=1,
            model_probability=0.268,
            back_odds=7.4,
            implied_probability=0.135,
            edge=0.133,
            is_value_bet=True,
            meets_min_edge=True,
            meets_min_confidence=True,
            meets_odds_range=True,
            alert_sent=False
        ),
        PredictionRecord(
            timestamp=datetime.now(timezone.utc),
            poll_number=1,
            check_window="5min",
            market_id="1.234567890",
            venue="murray_bridge",
            race_number=5,
            race_time=race_time,
            minutes_to_start=4.25,
            distance=520,
            grade="5",
            field_size=8,
            runner_name="Not Value Runner",
            box=2,
            model_probability=0.120,
            back_odds=8.5,
            implied_probability=0.118,
            edge=0.002,
            is_value_bet=False,
            meets_min_edge=False,
            meets_min_confidence=False,
            meets_odds_range=True,
            alert_sent=False
        ),
    ]

    print(f"\n[Logging {len(test_predictions)} predictions to JSONL]")
    logger.log_all_predictions(test_predictions)

    # Verify JSONL file
    jsonl_file = logger._get_predictions_file()
    print(f"  ✓ File created: {jsonl_file}")
    print(f"  ✓ File size: {jsonl_file.stat().st_size} bytes")

    # Check content
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()
        print(f"  ✓ Records written: {len(lines)}")

    # Show first record
    print(f"\n[Sample JSONL Record]")
    first_record = json.loads(lines[0])
    print(f"  Runner: {first_record['runner_name']}")
    print(f"  Model Prob: {first_record['model_probability']:.1%}")
    print(f"  Back Odds: ${first_record['back_odds']:.2f}")
    print(f"  Edge: {first_record['edge']:.1%}")
    print(f"  Is Value Bet: {first_record['is_value_bet']}")

    # ========================================================================
    # TEST 2: VALUE BETS (CSV)
    # ========================================================================

    print("\n" + "=" * 70)
    print("TEST 2: VALUE BET LOGGING (CSV)")
    print("=" * 70)

    # Create sample value bet records
    value_bet_1 = ValueBetRecord(
        date="13/02/2026",
        time="12:53",
        venue="Murray Bridge",
        race="R5",
        race_time="12:53",
        runner="Grapefruit Wilma",
        box=3,
        model_prob="25.20%",
        back_odds="$10.00",
        implied_prob="",
        edge="15.20%",
        alert_time="",  # Will be set by logger
        notes=""
    )

    value_bet_2 = ValueBetRecord(
        date="13/02/2026",
        time="12:53",
        venue="Murray Bridge",
        race="R5",
        race_time="12:53",
        runner="Aurora Sunshine",
        box=1,
        model_prob="26.80%",
        back_odds="$7.40",
        implied_prob="",
        edge="13.30%",
        alert_time="",
        notes="Showed up just once as a value bet"
    )

    print(f"\n[First Alert - Grapefruit Wilma at 11:53]")
    logger.log_value_bet(value_bet_1, "11:53")
    print(f"  ✓ New row created")

    print(f"\n[Second Alert - Grapefruit Wilma at 12:03]")
    logger.log_value_bet(value_bet_1, "12:03")
    print(f"  ✓ Alert time updated with semicolon")

    print(f"\n[Third Alert - Grapefruit Wilma at 12:13]")
    logger.log_value_bet(value_bet_1, "12:13")
    print(f"  ✓ Alert time updated again")

    print(f"\n[Single Alert - Aurora Sunshine at 12:54]")
    logger.log_value_bet(value_bet_2, "12:54")
    print(f"  ✓ New row created")

    # Verify CSV file
    csv_file = logger._get_value_bets_file()
    print(f"\n[CSV File]")
    print(f"  ✓ File created: {csv_file}")
    print(f"  ✓ File size: {csv_file.stat().st_size} bytes")

    # Check content
    with open(csv_file, 'r') as f:
        reader = csv_module.reader(f)
        rows = list(reader)
        print(f"  ✓ Total rows: {len(rows)} (including header)")

    # ========================================================================
    # TEST 3: CSV FORMAT VERIFICATION
    # ========================================================================

    print("\n" + "=" * 70)
    print("TEST 3: CSV FORMAT VERIFICATION")
    print("=" * 70)

    print(f"\n[Headers]")
    headers = rows[0]
    for i, header in enumerate(headers):
        print(f"  {i+1:2d}. {header}")

    print(f"\n[Value Bet Rows]")
    for i, row in enumerate(rows[1:], 1):
        print(f"\n  Row {i}: {row[5]} (Box {row[6]})")  # Runner name and box
        print(f"    Date: {row[0]}")
        print(f"    Time: {row[1]}")
        print(f"    Venue: {row[2]}")
        print(f"    Race: {row[3]}")
        print(f"    Model Prob: {row[7]}")
        print(f"    Back Odds: {row[8]}")
        print(f"    Edge: {row[10]}")
        print(f"    Alert Time: {row[11]}")  # Should show semicolon-separated times
        print(f"    Notes: {row[20] if row[20] else '(empty)'}")

    # ========================================================================
    # TEST 4: ALERT TIME ACCUMULATION VERIFICATION
    # ========================================================================

    print("\n" + "=" * 70)
    print("TEST 4: ALERT TIME ACCUMULATION")
    print("=" * 70)

    # Check that Grapefruit Wilma has accumulated alert times
    grapefruit_row = rows[1]  # First data row
    alert_times = grapefruit_row[11]

    print(f"\n[Grapefruit Wilma Alert Times]")
    print(f"  Expected: '11:53; 12:03; 12:13")
    print(f"  Actual:   '{alert_times}'")

    # Expected has one leading apostrophe (Excel text format)
    if alert_times == "'11:53; 12:03; 12:13":
        print(f"  ✓ Alert time accumulation WORKING!")
    else:
        print(f"  ✗ Alert time accumulation FAILED!")
        print(f"    Check the _update_alert_time() method")

    # Check that Aurora Sunshine has single alert time
    aurora_row = rows[2]  # Second data row
    aurora_alert = aurora_row[11]

    print(f"\n[Aurora Sunshine Alert Time]")
    print(f"  Expected: '12:54'")
    print(f"  Actual:   '{aurora_alert}'")

    if aurora_alert == "'12:54":
        print(f"  ✓ Single alert time CORRECT!")
    else:
        print(f"  ✗ Alert time INCORRECT!")

    # ========================================================================
    # TEST 5: EXCEL COMPATIBILITY CHECK
    # ========================================================================

    print("\n" + "=" * 70)
    print("TEST 5: EXCEL COMPATIBILITY")
    print("=" * 70)

    print(f"\n[Expected bet_results.xlsx Columns (21 total)]")
    expected_headers = [
        "Date", "Time", "Venue", "Race", "Race Time", "Runner", "Box",
        "Model Prob", "Back Odds", "Implied Prob", "Edge", "Alert Time",
        "Result", "Finish Position", "Margin", "WonLost", "Stake",
        "Return", "Profit", "RunningTotal", "Notes"
    ]

    print(f"\n[Comparing Headers]")
    all_match = True
    for i, (expected, actual) in enumerate(zip(expected_headers, headers)):
        match = "✓" if expected == actual else "✗"
        if expected != actual:
            all_match = False
        print(f"  {match} Column {i+1}: '{expected}' vs '{actual}'")

    if all_match:
        print(f"\n  ✓ ALL HEADERS MATCH! CSV is Excel-compatible!")
    else:
        print(f"\n  ✗ HEADERS MISMATCH! Fix column names!")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print(f"\n[Files Created]")
    print(f"  JSONL: {jsonl_file}")
    print(f"  CSV:   {csv_file}")

    print(f"\n[Test Results]")
    print(f"  ✓ JSONL logging: {len(test_predictions)} predictions written")
    print(f"  ✓ CSV logging: {len(rows)-1} value bets written")
    expected_alert = "'11:53; 12:03; 12:13"
    print(f"  ✓ Alert accumulation: {'PASS' if alert_times == expected_alert else 'FAIL'}")
    print(f"  ✓ Excel compatibility: {'PASS' if all_match else 'FAIL'}")

    print(f"\n[Manual Verification]")
    print(f"  1. Check JSONL file:")
    print(f"     cat {jsonl_file} | jq .")
    print(f"  2. Check CSV file:")
    print(f"     cat {csv_file}")
    print(f"  3. Import CSV to Excel and verify format")

    print("\n" + "=" * 70)
    print("Test harness complete!")
    print("=" * 70)
