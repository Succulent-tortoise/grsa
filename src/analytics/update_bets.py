#!/usr/bin/env python3
"""
Script to update daily bets logging CSV with settled outcomes from bookmaker export.
Usage: python -m src.analytics.update_bets --date 2025-10-21
Assumes config.py in src/utils for paths.
Includes relaxed matching for time/race shifts.
"""

import argparse
from datetime import datetime
import pandas as pd
import re
from pathlib import Path
import sys

# Import your config
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.utils.config import DATA_ROOT

# Add bets subpaths
BETS_DIR = DATA_ROOT / "logs" / "bets"
DAILY_DIR = BETS_DIR / "daily"
SETTLED_DIR = BETS_DIR / "settled"
DAILY_UPDATED_DIR = BETS_DIR / "daily_updated"

# Headers for logging file
LOG_HEADERS = [
    "date", "venue", "race", "time", "runner", "box",
    "recommended_odds", "actual_odds", "stake", "placed",
    "type", "sweet_spot_priority", "outcome", "return_amount", "profit"
]

def parse_settled_date(placed_str: str) -> str:
    if not placed_str:
        return None
    try:
        date_part = placed_str.split()[0]
        dt = datetime.strptime(date_part, "%d-%b-%y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None

def normalize_time(time_str: str) -> str:
    time_str = str(time_str).strip().lower()
    if 'pm' in time_str and '12' not in time_str:
        time_str = time_str.replace('pm', '').strip()
        try:
            parts = time_str.split(':')
            hr = int(parts[0]) + 12
            min_ = int(parts[1])
            return f"{hr:02d}:{min_:02d}"
        except (ValueError, IndexError):
            pass
    elif 'am' in time_str:
        time_str = time_str.replace('am', '').strip()
        try:
            parts = time_str.split(':')
            hr = int(parts[0])
            if hr == 12:
                hr = 0
            min_ = int(parts[1])
            return f"{hr:02d}:{min_:02d}"
        except (ValueError, IndexError):
            pass
    for fmt in ["%H:%M", "%I:%M%p", "%I:%M %p", "%H:%M:%S", "%I:%M"]:
        try:
            dt = datetime.strptime(time_str, fmt)
            return dt.strftime("%H:%M")
        except ValueError:
            continue
    return time_str

def normalize_runner(runner: str) -> str:
    """Lowercase, strip, remove apostrophes for flexible matching."""
    return re.sub(r"['']", '', str(runner).strip().lower())

def parse_settled_description(desc: str) -> dict:
    if not desc:
        return None
    desc = str(desc).strip()
    # Pattern for full parse
    pattern = r'(\d{1,2}[:\.]?\d{2})\s+([A-Za-z0-9\s\&\'-]+?)\s*(\d+)\s*\.\s*([A-Za-z0-9\s\&\'-]+)\s*-\s*(Win|Back|Place)\s*(?:\|\s*)?'
    match = re.search(pattern, desc, re.IGNORECASE)
    if not match:
        return None
    groups = match.groups()
    if len(groups) < 5:
        return None
    time, venue, race, runner, bet_type = groups
    norm_time = normalize_time(time)
    norm_venue = re.sub(r'\s+', '-', re.sub(r'[^a-zA-Z0-9\s-]', '', venue).strip().lower())
    norm_runner = normalize_runner(runner)
    if len(norm_runner.split()) < 1 or len(norm_runner) < 2:
        return None
    return {
        "time": norm_time,
        "venue": norm_venue,
        "race": race.strip(),
        "runner": norm_runner
    }

def time_to_minutes(time_str: str) -> int:
    """Convert HH:MM to minutes since midnight for diff calc."""
    try:
        hr, min_ = map(int, time_str.split(':'))
        return hr * 60 + min_
    except ValueError:
        return 0

def find_relaxed_match(log_key: tuple, settled_lookup: dict) -> tuple:
    """Find best settled match by runner + venue + date, with time proximity check."""
    target_date, target_venue, _, target_time, target_runner = log_key
    norm_runner = normalize_runner(target_runner)
    candidates = []
    for s_key, s_data in settled_lookup.items():
        s_date, s_venue, _, s_time, s_runner = s_key
        if (s_date == target_date and s_venue == target_venue and 
            normalize_runner(s_runner) == norm_runner):
            time_diff = abs(time_to_minutes(s_time) - time_to_minutes(target_time))
            candidates.append((s_key, s_data, time_diff))
    if not candidates:
        return None, None
    # Pick closest time
    best = min(candidates, key=lambda x: x[2])
    time_diff = best[2]
    if time_diff > 60:  # >1 hour diff? Flag but still match if runner exact
        print(f"Warning: Large time diff {time_diff} min for '{target_runner}' (planned {target_time}, actual {best[0][3]})")
    return best[0], best[1]

def main(date_str: str):
    daily_file = DAILY_DIR / f"{date_str}_bets.csv"
    settled_file = SETTLED_DIR / f"settled_{date_str.replace('-', '')}.csv"
    updated_file = DAILY_UPDATED_DIR / f"{date_str}_bets_updated.csv"

    DAILY_UPDATED_DIR.mkdir(parents=True, exist_ok=True)

    if not daily_file.exists():
        print(f"Error: Logging file not found: {daily_file}")
        return
    if not settled_file.exists():
        print(f"Error: Settled file not found: {settled_file}")
        return

    # Read logging
    try:
        log_df = pd.read_csv(daily_file, header=0)
        if log_df.columns.tolist() != LOG_HEADERS:
            log_df.columns = LOG_HEADERS
    except pd.errors.ParserError:
        log_df = pd.read_csv(daily_file, sep=':', header=None, names=LOG_HEADERS)
    log_df = log_df.fillna('')
    print(f"Loaded {len(log_df)} logging entries.")

    # Read settled
    settled_df = pd.read_csv(settled_file)
    print(f"Loaded {len(settled_df)} settled bets.")

    # Build settled lookup (exact keys)
    settled_lookup = {}
    skipped_settled = 0
    date_matched = 0
    for idx, row in settled_df.iterrows():
        placed_date = parse_settled_date(str(row.get('Placed', '')))
        if placed_date != date_str:
            continue
        date_matched += 1
        parsed = parse_settled_description(str(row.get('Description', '')))
        if not parsed:
            skipped_settled += 1
            continue

        odds_str = str(row.get('Odds', '1.0'))
        if '(BSP)' in odds_str:
            odds_str = odds_str.replace('(BSP)', '').strip()
        try:
            actual_odds = float(odds_str) if odds_str else 1.0
        except ValueError:
            skipped_settled += 1
            continue

        stake = float(str(row.get('Stake (AUD)', '0.0')).replace('(AUD)', '', 1).strip() or 0.0)
        profit = float(str(row.get('Profit/Loss', '0.0')).replace('$', '').strip() or 0.0)
        status = str(row.get('Status', '')).strip()
        outcome = 'Won' if status == 'Won' or profit > 0 else 'Lost'

        key = (
            placed_date,
            parsed['venue'],
            parsed['race'],
            parsed['time'],
            parsed['runner']
        )
        settled_lookup[key] = {
            'actual_odds': actual_odds,
            'stake': stake,
            'outcome': outcome,
            'profit': profit,
            'return_amount': stake * actual_odds if outcome == 'Won' else 0.0,
            'status': status,
            'original_time': parsed['time'],
            'original_race': parsed['race']
        }

    print(f"Processed {len(settled_lookup)} date-matching settled bets (skipped {skipped_settled} unparseable).")

    # Update logging (exact first, then relaxed)
    updated_count = 0
    total_profit = 0.0
    for idx, log_row in log_df.iterrows():
        norm_time = normalize_time(str(log_row['time']))
        norm_venue = str(log_row['venue']).lower().replace(' ', '-').strip()
        norm_runner = normalize_runner(log_row['runner'])
        key = (
            str(log_row['date']).strip(),
            norm_venue,
            str(log_row['race']).strip(),
            norm_time,
            norm_runner
        )
        print(f"Debug: Logging key {idx}: {key}")  # Keep for now

        # Exact match first
        matched_key = None
        if key in settled_lookup:
            matched_key = key
            print(f"Exact match for {log_row['runner']} (race {log_row['race']}, {norm_time})")
        else:
            # Relaxed match (runner + venue + date)
            matched_key, settled_data = find_relaxed_match(key, settled_lookup)
            if settled_data:
                s_time = settled_data.get('original_time', '')
                s_race = settled_data.get('original_race', '')
                print(f"Relaxed match for {log_row['runner']}: Planned race {log_row['race']} {norm_time} vs Actual race {s_race} {s_time}")

        if matched_key and settled_data is None:  # From exact
            settled_data = settled_lookup[matched_key]

        if matched_key:
            log_df.at[idx, 'placed'] = 'Yes'
            if pd.isna(log_row['actual_odds']) or str(log_row['actual_odds']).strip() == '':
                log_df.at[idx, 'actual_odds'] = settled_data['actual_odds']
            log_df.at[idx, 'stake'] = settled_data['stake']
            log_df.at[idx, 'outcome'] = settled_data['outcome']
            log_df.at[idx, 'return_amount'] = settled_data['return_amount']
            log_df.at[idx, 'profit'] = settled_data['profit']
            updated_count += 1
            total_profit += settled_data['profit']
            print(f"Updated: {log_row['runner']} - {settled_data['outcome']}, Profit: {settled_data['profit']:.2f}")

    # Save
    log_df.to_csv(updated_file, index=False)
    print(f"\nSummary: Updated {updated_count}/{len(log_df)} bets. Total profit for date: {total_profit:.2f}")
    print(f"Updated file saved: {updated_file}")

    relevant_settled = date_matched
    unmatched_settled = relevant_settled - len(settled_lookup) - skipped_settled  # Adjusted
    if unmatched_settled > 0 or skipped_settled > 0:
        print(f"Warning: {unmatched_settled} settled bets didn't match logging ({skipped_settled} skipped).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update bets logging with settled data.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Date to process (YYYY-MM-DD)")
    args = parser.parse_args()
    main(args.date)
