"""
jsonl_loader.py — Greyhound Prediction System v2

Module to load JSONL data for live predictions.

This module handles the following:
1. Load today's pre-race JSONL files from the defined path.
2. Parse runner data and compute missing features required for model predictions.
3. Returns structured data for usage in runner matching.
4. Caches data in memory for efficiency.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from runner_matcher import normalize_venue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
JSONL_DATA_DIR = Path("/media/matt_sent/vault/dishlicker_data/data/jsonl/")

class JSONLLoader:
    """Load and cache JSONL files for a given date."""

    def __init__(self) -> None:
        """Initialise the in-memory cache."""
        self.data: Dict[tuple, List[Dict[str, Any]]] = {}

    def load_daily_data(self, date: str) -> Dict[tuple, List[Dict[str, Any]]]:
        """Load all JSONL files for *date* and cache the processed data."""
        logger.info(f"Loading pre-race JSONL files for date: {date}")
        jsonl_path = JSONL_DATA_DIR / date

        if not jsonl_path.exists():
            logger.warning(f"JSONL path does not exist: {jsonl_path}")
            return {}

        for jsonl_file in jsonl_path.glob("*.jsonl"):
            logger.info(f"  Found JSONL file: {jsonl_file}")
            try:
                with jsonl_file.open() as f:
                    for line in f:
                        data = json.loads(line)
                        venue = normalize_venue(data['venue'])
                        race_number = int(data['race_number'])
                        runners = data['runners']
                        self.data[(venue, race_number)] = self.compute_runner_features(runners)
            except Exception as e:
                logger.error(f"Error loading JSONL file {jsonl_file}: {e}")

        return self.data

    def compute_runner_features(self, runners: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute the feature set for each runner."""
        computed_features: List[Dict[str, Any]] = []

        for runner in runners:
            if runner.get("is_scratched"):
                continue
            last_4 = runner["last_4"]
            ewma_position = self.compute_ewma_position(last_4)
            trend_slope = self.compute_trend_slope(last_4)
            form_volatility = self.compute_form_volatility(last_4)
            num_races = self.count_races(last_4)
            best_t_d_rank = self.compute_best_t_d_rank(runners)

            computed_features.append({
                "name": runner["name"],
                "ewma_position": ewma_position,
                "trend_slope": trend_slope,
                "form_volatility": form_volatility,
                "num_races": num_races,
                "best_t_d_rank": best_t_d_rank,
            })

        return computed_features

    def compute_ewma_position(self, last_4: str) -> float:
        """Compute the exponentially weighted moving average position."""
        if not last_4:
            return np.nan
        positions = [int(pos) for pos in last_4 if pos.isdigit()]
        if not positions:
            return np.nan
        ewma = pd.Series(positions).ewm(span=3, adjust=False).mean().iloc[-1]
        return ewma

    def compute_trend_slope(self, last_4: str) -> float:
        """Compute the trend slope from the last 4 positions."""
        positions = [int(pos) for pos in last_4 if pos.isdigit()]
        if len(positions) < 2:
            return np.nan
        x = np.arange(len(positions))
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), positions)
        return model.coef_[0]

    def compute_form_volatility(self, last_4: str) -> float:
        """Compute the form volatility (standard deviation of positions)."""
        positions = [int(pos) for pos in last_4 if pos.isdigit()]
        if len(positions) < 2:
            return np.nan
        return np.std(positions, ddof=0)

    def count_races(self, last_4: str) -> int:
        """Count the number of races represented in the last_4 string."""
        return len([pos for pos in last_4 if pos.isdigit()])

    def compute_best_t_d_rank(self, runners: List[Dict[str, Any]]) -> float:
        """Assign rank based on best time within the race.

        Returns rank of the current runner (first in list).
        Lower best_t_d = better (faster) = rank 1.
        """
        # Extract best_t_d values, converting None to np.nan explicitly
        best_t_d_values = []
        for runner in runners:
            val = runner.get('best_t_d')
            if val is None:
                best_t_d_values.append(np.nan)
            else:
                best_t_d_values.append(float(val))

        # Get valid (non-NaN) values for ranking
        valid_t_d = [(i, v) for i, v in enumerate(best_t_d_values) if pd.notna(v)]

        if not valid_t_d:
            return np.nan

        # Rank only valid values (lower time = better rank)
        valid_indices, valid_values = zip(*valid_t_d)
        ranks = stats.rankdata(valid_values, method='min')

        # Return rank of first runner if they have a valid time
        if pd.notna(best_t_d_values[0]):
            first_val = best_t_d_values[0]
            for (idx, val), rank in zip(valid_t_d, ranks):
                if idx == 0:
                    return float(rank)
        return np.nan
