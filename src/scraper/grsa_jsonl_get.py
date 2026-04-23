import os
import sys
import csv
import json
import argparse
from datetime import datetime, date, timedelta
import logging

from pathlib import Path

# Add root to path for imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, root_dir)

from src.utils.config import DATA_ROOT, get_index_path, JSONL_DIR

def parse_csv_to_races(csv_path, venue_slug, date):
    races = []
    current_race = None
    runners_section = False

    with open(csv_path, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not any(row):
                continue

            # Detect race header
            if row[0].startswith("Race"):
                if current_race:
                    races.append(current_race)
                current_race = {
                    "venue": venue_slug,
                    "date": date,
                    "race_number": row[1],
                    "race_name": row[2],
                    "grade": None,
                    "distance": None,
                    "time": None,
                    "runners": [],
                    "results": None  # placeholder for later
                }
                runners_section = False
                continue

            if not current_race:
                continue

            # Metadata
            if row[0] == "Grade":
                current_race["grade"] = row[1]
            elif row[0] == "Distance":
                try:
                    current_race["distance"] = int(row[1])
                except ValueError:
                    current_race["distance"] = None
            elif row[0] == "Time":
                current_race["time"] = row[1]
            elif row[0] == "Drawn Box":
                runners_section = True
                continue

            # Runner rows
            if runners_section and len(row) >= 9:
                # Normalise types
                drawn_box = int(row[0]) if row[0].isdigit() else None
                run_box = None  # to be filled later
                is_scratched = row[8].strip().upper() == "TRUE"

                best_t_d = None
                if row[6] and row[6].replace('.', '', 1).isdigit():
                    best_t_d = float(row[6])

                odds = None
                try:
                    odds = float(row[7]) if row[7] else None
                except ValueError:
                    odds = None

                runner = {
                    "drawn_box": drawn_box,
                    "run_box": run_box,
                    "name": row[3],
                    "trainer": row[4],
                    "last_4": row[5] if row[5] else None,
                    "best_t_d": best_t_d,
                    "odds": odds,
                    "is_scratched": is_scratched,
                    "finishing_time": None,
                    "margin": None,
                    "final_position": None,
                    "odds_final": None
                }
                current_race["runners"].append(runner)

        if current_race:
            races.append(current_race)

    return races

def write_jsonl(races, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for race in races:
            f.write(json.dumps(race) + "\n")

def batch_process(index_path, output_dir, logger):
    """
    Process CSV files to JSONL based on index file.
    
    Args:
        index_path: Path to index JSONL file (e.g., index_YYYY-MM-DD_csv.jsonl)
        output_dir: Directory to write race JSONL files
        logger: Logger instance
    """
    # Corresponding JSON path: index_YYYY-MM-DD.json (fallback)
    json_path = Path(str(index_path).replace('_csv.jsonl', '.json'))
    loaded_from_jsonl = False
    index_data = None

    # Try JSONL first (preferred format)
    if index_path.exists():
        try:
            temp_data = []
            with open(index_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        temp_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Skipping malformed line {line_num} in {index_path}: {e}")
                        continue
            if temp_data:
                index_data = temp_data
                loaded_from_jsonl = True
            else:
                raise ValueError(f"No valid data in {index_path}")
        except Exception as e:
            logger.warning(f"⚠️ Error loading JSONL {index_path}: {e}, falling back to JSON")

    # Fallback to JSON
    if index_data is None:
        try:
            with open(json_path, encoding="utf-8") as f:
                index_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"❌ Error: Index file not found: {json_path}")
            return
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error: Invalid JSON in index file: {e}")
            return
        except Exception as e:
            logger.error(f"❌ Error reading index file: {e}")
            return

    counters = {"converted": 0, "skipped": 0, "failed": 0}

    for entry in index_data:
        csv_path = entry.get("csv_path")
        venue_slug = entry.get("venue_slug", "unknown")
        date = entry.get("date", "unknown")

        reprocess = False
        if entry.get("jsonl_created", False):
            race_jsonl_path = entry.get("jsonl_path")
            if race_jsonl_path is None or not os.path.exists(race_jsonl_path):
                logger.warning(f"JSONL file missing for {venue_slug} on {date}, reprocessing CSV")
                entry["jsonl_created"] = False
                reprocess = True
            else:
                logger.info(f"[SKIP] Already converted JSONL for {venue_slug} on {date}")
                counters["skipped"] += 1
                continue

        if csv_path is None or not csv_path:
            logger.warning(f"⚠️  Skipping entry with null/empty csv_path for venue {venue_slug} on {date}")
            now = datetime.utcnow().isoformat()
            entry["jsonl_created"] = False
            entry["jsonl_path"] = None
            entry["jsonl_timestamp"] = now
            entry["jsonl_status"] = "failed"
            entry["error"] = "No CSV path"
            counters["failed"] += 1
            continue

        if not os.path.exists(csv_path):
            logger.warning(f"⚠️  CSV not found: {csv_path}, skipping")
            now = datetime.utcnow().isoformat()
            entry["jsonl_created"] = False
            entry["jsonl_path"] = None
            entry["jsonl_timestamp"] = now
            entry["jsonl_status"] = "failed"
            entry["error"] = f"CSV not found: {csv_path}"
            counters["failed"] += 1
            continue

        try:
            races = parse_csv_to_races(csv_path, venue_slug, date)
        except Exception as e:
            logger.error(f"❌ Error parsing CSV {csv_path}: {e}")
            now = datetime.utcnow().isoformat()
            entry["jsonl_created"] = False
            entry["jsonl_path"] = None
            entry["jsonl_timestamp"] = now
            entry["jsonl_status"] = "failed"
            entry["error"] = f"Parse error: {str(e)}"
            counters["failed"] += 1
            continue

        output_filename = f"{venue_slug}_{date}_prerace.jsonl"
        output_path = output_dir / output_filename

        try:
            write_jsonl(races, output_path)
            now = datetime.utcnow().isoformat()
            entry["jsonl_created"] = True
            entry["jsonl_path"] = str(output_path)
            entry["jsonl_timestamp"] = now
            entry["jsonl_status"] = "converted"
            counters["converted"] += 1
            if reprocess:
                logger.info(f"✅ Reprocessed {csv_path} -> {output_path}")
            else:
                logger.info(f"✅ Processed {csv_path} -> {output_path}")
        except Exception as e:
            logger.error(f"❌ Error writing JSONL {output_path}: {e}")
            now = datetime.utcnow().isoformat()
            entry["jsonl_created"] = False
            entry["jsonl_path"] = None
            entry["jsonl_timestamp"] = now
            entry["jsonl_status"] = "failed"
            entry["error"] = f"Write error: {str(e)}"
            counters["failed"] += 1
            continue

    logger.info(f"[SUMMARY] {counters['converted']} converted, {counters['skipped']} skipped, {counters['failed']} failed")

    # CRITICAL FIX: Save updated INDEX file, not race JSONL files!
    # Save to the same path we loaded from (index file)
    index_save_path = index_path
    try:
        # Atomic write with temp file
        temp_path = index_save_path.with_suffix('.jsonl.tmp')
        with open(temp_path, "w", encoding="utf-8") as f:
            for entry in index_data:
                f.write(json.dumps(entry) + "\n")
        temp_path.rename(index_save_path)
        logger.info(f"✅ Index file updated: {index_save_path}")
    except Exception as e:
        logger.warning(f"⚠️  Warning: Could not update index file: {e}")

if __name__ == "__main__":
    # Setup logging
    logs_dir = DATA_ROOT / "logs" / "index"
    logs_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    year, week_num, _ = now.isocalendar()
    log_filename = f"index_log_W{week_num:02d}-{year}.log"
    log_path = logs_dir / log_filename

    logger = logging.getLogger('grsa_jsonl_get')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # File handler - append mode
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser(description="Process GRSA CSV files to JSONL for a single date or date range.")
    parser.add_argument(
        "--date",
        nargs="+",
        default=[datetime.today().strftime("%Y-%m-%d")],
        help="Date(s) to process. Provide one date for single day (YYYY-MM-DD or YYYYMMDD) or two for range (start end). Defaults to today."
    )
    args = parser.parse_args()

    def parse_date_input(input_str):
        """Parse date input as YYYY-MM-DD or YYYYMMDD, return datetime.date object."""
        if len(input_str) == 10 and input_str.count("-") == 2:
            return datetime.strptime(input_str, "%Y-%m-%d").date()
        elif len(input_str) == 8 and input_str.isdigit():
            return datetime.strptime(input_str, "%Y%m%d").date()
        else:
            raise ValueError(f"Invalid date format: {input_str}. Use YYYY-MM-DD or YYYYMMDD.")

    try:
        date_inputs = args.date
        if len(date_inputs) == 1:
            start_date = end_date = parse_date_input(date_inputs[0])
        elif len(date_inputs) == 2:
            start_date = parse_date_input(date_inputs[0])
            end_date = parse_date_input(date_inputs[1])
            if start_date > end_date:
                raise ValueError("Start date must be before or equal to end date.")
        else:
            raise ValueError("Provide 1 or 2 dates with --date.")

        # Generate list of dates
        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date)
            current_date += timedelta(days=1)

        # Paths
        base_jsonl_dir = JSONL_DIR

        for dt in date_list:
            date_str = dt.strftime("%Y-%m-%d")
            date_folder = dt.strftime("%Y%m%d")

            # Index file using config helper
            index_path = get_index_path(date_str, "csv")

            # Output folder per date - uses YYYYMMDD format
            output_dir = base_jsonl_dir / date_folder
            output_dir.mkdir(exist_ok=True)

            logger.info(f"Processing date: {date_str}")
            logger.info(f"Using index file: {index_path}")
            logger.info(f"Output dir: {output_dir}")
            batch_process(index_path, output_dir, logger)
            logger.info("-" * 50)

    except ValueError as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)