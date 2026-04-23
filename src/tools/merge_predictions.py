#!/usr/bin/env python3

import argparse
import csv
import datetime
import json
import logging
import os
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Merge daily prediction JSONL files into a sorted CSV.")
    parser.add_argument("--date", type=str, default=datetime.date.today().strftime("%Y-%m-%d"),
                        help="Date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--no-sort", action="store_true",
                        help="Disable default sorting (output in file discovery order)")
    return parser.parse_args()

def get_input_dir(date_str):
    # Convert YYYY-MM-DD to YYYYMMDD for directory name
    yyyymmdd = date_str.replace("-", "")
    return Path("/media/matt_sent/vault/dishlicker_data/data/predictions") / yyyymmdd

def get_output_base(date_str):
    return f"all-races_{date_str}.csv"

def find_matching_files(input_dir):
    if not input_dir.exists():
        return None
    pattern = f"*_{args.date}_predictions.jsonl"
    return list(input_dir.glob(pattern))

def extract_venue(filename):
    # e.g., "broken-hill_2025-11-16_predictions.jsonl" -> "broken-hill"
    stem = filename.stem
    parts = stem.rsplit("_predictions", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid filename format: {filename}")
    date_part = f"_{args.date}"
    if not parts[0].endswith(date_part):
        raise ValueError(f"Date mismatch in filename: {filename}")
    return parts[0][:-len(date_part)]

def parse_time_to_sortable(time_str):
    if not time_str:
        return "99:99"  # Sort last
    try:
        # Assume format like "7:09pm" or "12:32pm" (24hr parsing for sortable)
        # Use strptime to parse %I:%M%p, format to %H:%M
        parsed = datetime.datetime.strptime(time_str.strip(), "%I:%M%p")
        return parsed.strftime("%H:%M")
    except ValueError:
        logging.warning(f"Unparsable time '{time_str}' - using original for output, sort last")
        return "99:99"

def process_race_line(line_json, venue):
    race_number = line_json.get("race_number", "")
    race_name = line_json.get("race_name", "")
    time_str = line_json.get("time", "")
    sortable_time_str = parse_time_to_sortable(time_str)
    
    rows = []
    runners = line_json.get("runners", [])
    if not runners:
        logging.warning(f"No runners in race {race_number} for {venue}")
        return rows
    
    for runner in runners:
        if not isinstance(runner, dict):
            continue  # Skip invalid runner
        row = {
            "Venue": venue,
            "time": time_str,
            "sortable_time": sortable_time_str,
            "race_number": race_number,
            "race_name": race_name,
            "name": runner.get("name", ""),
            "drawn_box": runner.get("drawn_box", ""),
            "odds": runner.get("odds", ""),
            "winner_prob": runner.get("winner_prob", ""),
            "predicted_winner": str(runner.get("predicted_winner", False)).lower(),
            "is_scratched": str(runner.get("is_scratched", False)).lower(),
        }
        rows.append(row)
    
    return rows

def sort_rows(rows):
    def sort_key(row):
        try:
            st = row["sortable_time"]
            time_obj = datetime.datetime.strptime(st, "%H:%M").time()
        except ValueError:
            time_obj = datetime.time(99, 99)  # Last
        return (time_obj, row["Venue"], row["race_number"], row["drawn_box"])
    
    return sorted(rows, key=sort_key)

def get_next_version(base_path):
    base_stem = base_path.stem
    counter = 1
    while base_path.exists():
        versioned = base_path.parent / f"{base_stem}_{counter:02d}{base_path.suffix}"
        if not versioned.exists():
            return versioned
        counter += 1
    return base_path  # If none exist, use base

def setup_logging(log_path):
    os.makedirs(log_path.parent, exist_ok=True)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stderr)  # Also log warnings to stderr if verbose needed
        ]
    )

def write_csv(rows, csv_path):
    if not rows:
        return
    
    headers = ["Venue", "time", "sortable_time", "race_number", "race_name", 
               "name", "drawn_box", "odds", "winner_prob", "predicted_winner", "is_scratched"]
    
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"CSV saved: {csv_path}")

def main():
    global args
    args = parse_args()
    
    input_dir = get_input_dir(args.date)
    files = find_matching_files(input_dir)
    
    if not files:
        print(f"ERROR: No prediction JSONL files found for {args.date}. Ensure the workflow has run first—aborting.")
        sys.exit(1)
    
    log_base = f"all-races_{args.date}.log"
    log_dir = Path("/media/matt_sent/vault/dishlicker_data/data/logs/prediction_csv")
    log_path = get_next_version(log_dir / log_base)  # Version log too?
    
    setup_logging(log_path)
    
    all_rows = []
    total_races = 0
    successful_races = 0
    
    for file_path in files:
        try:
            venue = extract_venue(file_path)
            print(f"Processing {file_path.name} ({venue})...")
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        line_json = json.loads(line)
                        race_rows = process_race_line(line_json, venue)
                        total_races += 1
                        if race_rows:
                            successful_races += 1
                            all_rows.extend(race_rows)
                        else:
                            logging.warning(f"Race {line_num} in {file_path} yielded no valid rows")
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON error on line {line_num} in {file_path}: {e}")
                    except Exception as e:
                        logging.error(f"Unexpected error on line {line_num} in {file_path}: {e}")
                        
        except Exception as e:
            logging.error(f"Failed to process file {file_path}: {e}")
            continue
    
    if not all_rows:
        print(f"ERROR: No valid races processed for {args.date}. Check logs and files—aborting.")
        sys.exit(1)
    
    if not args.no_sort:
        all_rows = sort_rows(all_rows)
    
    csv_base = input_dir / get_output_base(args.date)
    csv_path = get_next_version(csv_base)
    
    write_csv(all_rows, csv_path)
    
    log_msg = f", log saved: {log_path}" if log_path.exists() and log_path.stat().st_size > 0 else ""
    print(f"{successful_races} of {total_races} races processed{log_msg}, CSV saved: {csv_path}")

if __name__ == "__main__":
    main()