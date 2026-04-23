"""
Results Analyzer - Match Predictions to Race Results

Reads predictions from predictions.jsonl and matches them to race results
to generate analysis reports for model calibration and ROI tracking.
"""

import json
import csv
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MatchedPrediction:
    """
    A prediction matched with its actual race result.

    Used for model calibration, ROI analysis, and performance evaluation.
    """
    # Prediction data
    timestamp: str
    poll_number: int
    check_window: str
    venue: str
    race_number: int
    race_time: str
    runner_name: str
    box: int
    model_probability: float
    back_odds: float
    implied_probability: float
    edge: float
    is_value_bet: bool

    # Result data
    final_position: Optional[int] = None
    won: bool = False
    margin: Optional[float] = None
    finishing_time: Optional[float] = None
    odds_final: Optional[float] = None  # BSP (Betfair Starting Price)
    was_scratched: bool = False

    # Analysis fields
    prediction_correct: bool = False  # For calibration (did win prediction match outcome?)
    roi: float = 0.0  # If staked $1, what's the return?
    matched: bool = False  # Was result found for this prediction?


@dataclass
class AnalysisSummary:
    """
    Summary statistics for daily analysis report.
    """
    date: str
    total_predictions: int = 0
    total_matched: int = 0
    total_unmatched: int = 0

    # Calibration (by probability bucket)
    calibration: Dict[str, Dict] = field(default_factory=dict)
    # Example: {"20-25%": {"predictions": 150, "wins": 35, "win_rate": 0.233}}

    # ROI by edge threshold
    roi_by_edge: Dict[str, Dict] = field(default_factory=dict)
    # Example: {"5%": {"bets": 45, "wins": 12, "roi": 0.082}}

    # Check window performance
    window_performance: Dict[str, Dict] = field(default_factory=dict)
    # Example: {"45min": {"bets": 15, "wins": 5, "avg_odds": 8.2, "roi": 0.12}}

    # Overall metrics
    total_value_bets: int = 0
    value_bets_won: int = 0
    value_bet_win_rate: float = 0.0
    overall_roi: float = 0.0


# ============================================================================
# RESULTS ANALYZER CLASS
# ============================================================================

class ResultsAnalyzer:
    """
    Match predictions to race results and generate analysis reports.

    Reads:
    - predictions.jsonl (from prediction_logger)
    - venue_YYYY-MM-DD_results.jsonl files (from your daily results scraping)

    Outputs:
    - analysis_report.csv (matched predictions with results)
    - summary.json (calibration, ROI, performance metrics)
    """

    def __init__(self, analysis_dir: Path, results_dir: Path):
        """
        Initialize results analyzer.

        Args:
            analysis_dir: Base directory for predictions and analysis
                         (e.g., /media/matt_sent/vault/dishlicker_data/data/analysis/)
            results_dir: Base directory for race results
                        (e.g., /media/matt_sent/vault/dishlicker_data/data/results/)
        """
        self.analysis_dir = Path(analysis_dir)
        self.results_dir = Path(results_dir)
        self.logger = logging.getLogger("results_analyzer")

    def analyze_date(self, date_str: str) -> AnalysisSummary:
        """
        Analyze all predictions for a given date.

        Args:
            date_str: Date in YYYYMMDD format (e.g., "20260219")

        Returns:
            AnalysisSummary object with all metrics
        """
        self.logger.info(f"[ANALYSIS] Analyzing predictions for {date_str}")

        # Load predictions
        predictions = self._load_predictions(date_str)
        if not predictions:
            self.logger.warning(f"[ANALYSIS] No predictions found for {date_str}")
            return AnalysisSummary(date=date_str)

        # Load results
        results = self._load_results(date_str)
        if not results:
            self.logger.warning(f"[ANALYSIS] No results found for {date_str}")
            return AnalysisSummary(date=date_str)

        # Match predictions to results
        matched_predictions = self._match_predictions_to_results(predictions, results)

        # Generate analysis
        summary = self._calculate_summary(matched_predictions, date_str)

        # Save outputs
        self._save_analysis_report(matched_predictions, date_str)
        self._save_summary(summary, date_str)

        self.logger.info(
            f"[ANALYSIS] Complete: {summary.total_matched}/{summary.total_predictions} "
            f"predictions matched"
        )

        return summary

    # -------------------------------------------------------------------------
    # Data Loading Methods
    # -------------------------------------------------------------------------

    def _load_predictions(self, date_str: str) -> List[dict]:
        """
        Load predictions from predictions.jsonl for given date.

        Args:
            date_str: Date in YYYYMMDD format

        Returns:
            List of prediction dictionaries
        """
        predictions_file = self.analysis_dir / date_str / "predictions.jsonl"

        if not predictions_file.exists():
            self.logger.warning(f"[ANALYSIS] Predictions file not found: {predictions_file}")
            return []

        predictions = []
        with open(predictions_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    pred = json.loads(line.strip())
                    predictions.append(pred)
                except json.JSONDecodeError as e:
                    self.logger.error(f"[ANALYSIS] Invalid JSON in predictions: {e}")
                    continue

        self.logger.info(f"[ANALYSIS] Loaded {len(predictions)} predictions")
        return predictions

    def _load_results(self, date_str: str) -> Dict[str, List[dict]]:
        """
        Load race results from all venue files for given date.

        Args:
            date_str: Date in YYYYMMDD format

        Returns:
            Dictionary mapping venue to list of race results
            Format: {"angle_park": [race1_dict, race2_dict, ...], ...}
        """
        results_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"  # YYYY-MM-DD
        results_dir = self.results_dir / date_str

        if not results_dir.exists():
            self.logger.warning(f"[ANALYSIS] Results directory not found: {results_dir}")
            return {}

        all_results = {}

        # Find all result files for this date
        result_files = list(results_dir.glob(f"*_{results_date}_results.jsonl"))

        if not result_files:
            self.logger.warning(f"[ANALYSIS] No result files found in {results_dir}")
            return {}

        for result_file in result_files:
            # Extract venue from filename: "angle-park_2026-02-19_results.jsonl"
            venue = result_file.stem.split('_')[0]
            venue_normalized = self._normalize_venue(venue)

            # Load races from this venue
            races = []
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        race = json.loads(line.strip())
                        races.append(race)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"[ANALYSIS] Invalid JSON in results: {e}")
                        continue

            all_results[venue_normalized] = races
            self.logger.info(f"[ANALYSIS] Loaded {len(races)} races from {venue}")

        return all_results

    # -------------------------------------------------------------------------
    # Normalization Methods
    # -------------------------------------------------------------------------

    def _normalize_venue(self, venue: str) -> str:
        """
        Normalize venue name for matching.

        Handles variations:
        - "angle-park" → "angle_park"
        - "Angle Park" → "angle_park"
        - "angle_park" → "angle_park"

        Args:
            venue: Raw venue name

        Returns:
            Normalized venue name (lowercase with underscores)
        """
        return venue.lower().replace('-', '_').replace(' ', '_')

    def _normalize_runner_name(self, name: str) -> str:
        """
        Normalize runner name for matching.

        Handles variations:
        - Case differences: "GIGGLE MONSTER" → "giggle monster"
        - Extra spaces: "Giggle  Monster" → "giggle monster"
        - Common abbreviations (future: "St." → "Saint", etc.)

        Args:
            name: Raw runner name

        Returns:
            Normalized runner name (lowercase, single spaces)
        """
        return ' '.join(name.lower().split())

    # -------------------------------------------------------------------------
    # Matching Methods
    # -------------------------------------------------------------------------

    def _match_predictions_to_results(
        self,
        predictions: List[dict],
        results: Dict[str, List[dict]]
    ) -> List[MatchedPrediction]:
        """
        Match predictions to actual race results.

        Matching strategy:
        1. Match on venue + race_number
        2. Within race, match on runner name (normalized)
        3. Fallback to box number if name doesn't match
        4. Mark unmatched predictions

        Args:
            predictions: List of prediction dictionaries
            results: Dictionary of venue → races

        Returns:
            List of MatchedPrediction objects
        """
        matched = []
        unmatched_count = 0

        for pred in predictions:
            # Find the race in results
            venue_norm = self._normalize_venue(pred['venue'])
            race_number = int(pred['race_number'])

            if venue_norm not in results:
                # Venue not in results (no races or different venue name)
                matched_pred = self._create_unmatched_prediction(pred)
                matched.append(matched_pred)
                unmatched_count += 1
                continue

            # Find race by race number
            race = None
            for r in results[venue_norm]:
                if int(r['race_number']) == race_number:
                    race = r
                    break

            if not race:
                # Race not found in results
                matched_pred = self._create_unmatched_prediction(pred)
                matched.append(matched_pred)
                unmatched_count += 1
                continue

            # Find runner in race
            runner_norm = self._normalize_runner_name(pred['runner_name'])
            runner_result = None

            # Try matching by name first
            for runner in race['runners']:
                if self._normalize_runner_name(runner['name']) == runner_norm:
                    runner_result = runner
                    break

            # Fallback: match by box if name didn't work
            if not runner_result:
                for runner in race['runners']:
                    if runner.get('run_box') == pred['box']:
                        runner_result = runner
                        self.logger.debug(
                            f"[ANALYSIS] Matched by box: {pred['runner_name']} "
                            f"(box {pred['box']}) → {runner['name']}"
                        )
                        break

            if not runner_result:
                # Runner not found (scratched before results file created?)
                matched_pred = self._create_unmatched_prediction(pred)
                matched.append(matched_pred)
                unmatched_count += 1
                continue

            # Create matched prediction
            matched_pred = self._create_matched_prediction(pred, runner_result)
            matched.append(matched_pred)

        self.logger.info(
            f"[ANALYSIS] Matched {len(matched) - unmatched_count}/{len(predictions)} predictions"
        )

        return matched

    def _create_matched_prediction(
        self,
        prediction: dict,
        result: dict
    ) -> MatchedPrediction:
        """
        Create MatchedPrediction from prediction and result data.

        Args:
            prediction: Prediction dictionary from predictions.jsonl
            result: Runner result dictionary from results.jsonl

        Returns:
            MatchedPrediction object with both prediction and result data
        """
        won = result.get('final_position') == 1
        was_scratched = result.get('is_scratched', False)

        # Calculate ROI (if staked $1)
        roi = 0.0
        if not was_scratched:
            if won:
                # Won: return is back_odds
                roi = prediction['back_odds'] - 1.0  # Subtract stake
            else:
                # Lost: return is -1 (lost stake)
                roi = -1.0

        # Prediction correctness (for calibration)
        # This is subjective - we'll mark as "correct" if outcome matches probability bucket
        # For now, just check if won matches high probability (>30%)
        prediction_correct = (won and prediction['model_probability'] > 0.30) or \
                            (not won and prediction['model_probability'] <= 0.30)

        return MatchedPrediction(
            timestamp=prediction['timestamp'],
            poll_number=prediction['poll_number'],
            check_window=prediction['check_window'],
            venue=prediction['venue'],
            race_number=prediction['race_number'],
            race_time=prediction['race_time'],
            runner_name=prediction['runner_name'],
            box=prediction['box'],
            model_probability=prediction['model_probability'],
            back_odds=prediction['back_odds'],
            implied_probability=prediction['implied_probability'],
            edge=prediction['edge'],
            is_value_bet=prediction['is_value_bet'],
            final_position=result.get('final_position'),
            won=won,
            margin=result.get('margin'),
            finishing_time=result.get('finishing_time'),
            odds_final=result.get('odds_final'),
            was_scratched=was_scratched,
            prediction_correct=prediction_correct,
            roi=roi,
            matched=True
        )

    def _create_unmatched_prediction(self, prediction: dict) -> MatchedPrediction:
        """
        Create MatchedPrediction for predictions with no result found.

        Args:
            prediction: Prediction dictionary from predictions.jsonl

        Returns:
            MatchedPrediction object with matched=False
        """
        return MatchedPrediction(
            timestamp=prediction['timestamp'],
            poll_number=prediction['poll_number'],
            check_window=prediction['check_window'],
            venue=prediction['venue'],
            race_number=prediction['race_number'],
            race_time=prediction['race_time'],
            runner_name=prediction['runner_name'],
            box=prediction['box'],
            model_probability=prediction['model_probability'],
            back_odds=prediction['back_odds'],
            implied_probability=prediction['implied_probability'],
            edge=prediction['edge'],
            is_value_bet=prediction['is_value_bet'],
            matched=False
        )

    # -------------------------------------------------------------------------
    # Summary and Output Methods
    # -------------------------------------------------------------------------

    def _save_analysis_report(
        self,
        matched_predictions: List[MatchedPrediction],
        date_str: str
    ) -> None:
        """
        Save matched predictions to analysis_report.csv.

        Args:
            matched_predictions: List of MatchedPrediction objects
            date_str: Date in YYYYMMDD format
        """
        output_file = self.analysis_dir / date_str / "analysis_report.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # CSV headers
        headers = [
            "timestamp", "poll_number", "check_window",
            "venue", "race_number", "runner_name", "box",
            "model_probability", "back_odds", "edge",
            "is_value_bet", "final_position", "won",
            "margin", "odds_final", "was_scratched",
            "roi", "matched"
        ]

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for mp in matched_predictions:
                row = {
                    "timestamp": mp.timestamp,
                    "poll_number": mp.poll_number,
                    "check_window": mp.check_window,
                    "venue": mp.venue,
                    "race_number": mp.race_number,
                    "runner_name": mp.runner_name,
                    "box": mp.box,
                    "model_probability": f"{mp.model_probability:.4f}" if mp.model_probability is not None else "",
                    "back_odds": f"{mp.back_odds:.2f}" if mp.back_odds is not None else "",
                    "edge": f"{mp.edge:.4f}" if mp.edge is not None else "",
                    "is_value_bet": mp.is_value_bet,
                    "final_position": mp.final_position if mp.matched else "",
                    "won": mp.won if mp.matched else "",
                    "margin": f"{mp.margin:.2f}" if mp.matched and mp.margin else "",
                    "odds_final": f"{mp.odds_final:.2f}" if mp.matched and mp.odds_final else "",
                    "was_scratched": mp.was_scratched,
                    "roi": f"{mp.roi:.4f}" if mp.matched else "",
                    "matched": mp.matched
                }
                writer.writerow(row)

        self.logger.info(f"[ANALYSIS] Saved analysis report: {output_file}")

    def _calculate_summary(
        self,
        matched_predictions: List[MatchedPrediction],
        date_str: str
    ) -> AnalysisSummary:
        """
        Calculate summary statistics from matched predictions.

        This method will be implemented in Chunk 3.
        For now, just return basic counts.

        Args:
            matched_predictions: List of MatchedPrediction objects
            date_str: Date in YYYYMMDD format

        Returns:
            AnalysisSummary with basic metrics
        """
        total_matched = sum(1 for mp in matched_predictions if mp.matched)
        total_unmatched = len(matched_predictions) - total_matched

        return AnalysisSummary(
            date=date_str,
            total_predictions=len(matched_predictions),
            total_matched=total_matched,
            total_unmatched=total_unmatched
        )

    def _save_summary(self, summary: AnalysisSummary, date_str: str) -> None:
        """
        Save summary statistics to summary.json.

        This method will be implemented in Chunk 3.
        For now, just log the counts.

        Args:
            summary: AnalysisSummary object
            date_str: Date in YYYYMMDD format
        """
        self.logger.info(
            f"[ANALYSIS] Summary: {summary.total_matched} matched, "
            f"{summary.total_unmatched} unmatched"
        )


# ============================================================================
# TEST HARNESS
# ============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("RESULTS ANALYZER TEST HARNESS")
    print("=" * 70)

    # Use actual paths
    analysis_dir = Path("/media/matt_sent/vault/dishlicker_data/data/analysis")
    results_dir = Path("/media/matt_sent/vault/dishlicker_data/data/results")

    analyzer = ResultsAnalyzer(analysis_dir, results_dir)

    # Test with yesterday's date (or specify date)
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        # Default to yesterday
        from datetime import datetime, timedelta
        yesterday = datetime.now() - timedelta(days=1)
        date_str = yesterday.strftime("%Y%m%d")

    print(f"\n[Testing with date: {date_str}]")

    # Run analysis
    summary = analyzer.analyze_date(date_str)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total predictions: {summary.total_predictions}")
    print(f"Matched: {summary.total_matched}")
    print(f"Unmatched: {summary.total_unmatched}")

    if summary.total_predictions > 0:
        match_rate = summary.total_matched / summary.total_predictions * 100
        print(f"Match rate: {match_rate:.1f}%")

    print("\n" + "=" * 70)
