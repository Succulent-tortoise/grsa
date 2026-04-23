import os
import json
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
import ssl
import certifi
import sys
import shutil
import argparse
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(os.path.abspath("/home/matt_sent/projects/grsa/src"))
from src.utils.config import JSONL_DIR, MODEL_DATA_DIR, RESULTS_DIR

# --------------------
# CONFIGURATION
# --------------------
INDEX_BASE_DIR = Path("/media/matt_sent/vault/dishlicker_data/data/index")
LOG_BASE_DIR = Path("/media/matt_sent/vault/dishlicker_data/data/logs/index")
RESULTS_URL_BASE = "https://greyhoundracingsa.com.au/racing/meetingdetails"

VENUE_URL_EXCEPTIONS = {
    "q1-lakeside": "ladbrokes-q1-lakeside",
    "q-straight": "ladbrokes-q-straight",
    "q2-parklands": "ladbrokes-q2-parklands"
}

# --------------------
# LOGGING
# --------------------
def get_log_file_path(date_obj):
    """Get weekly log file path based on ISO week number."""
    week_num = date_obj.isocalendar()[1]
    year = date_obj.year
    log_filename = f"index_log_W{week_num:02d}-{year}.log"
    LOG_BASE_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_BASE_DIR / log_filename

def log_message(level, message, log_file=None):
    """Log message to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {level}: {message}"
    print(log_line)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

# --------------------
# Cloudflare SSL adapter
# --------------------
class CloudflareSSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.set_ciphers('DEFAULT@SECLEVEL=1')
        context.load_verify_locations(cafile=certifi.where())
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

def create_cloudflare_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = CloudflareSSLAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    })
    return session

# --------------------
# SCRAPING
# --------------------
def scrape_meet_results(session, meet_url):
    """Scrape results from a meeting page. Returns race dict keyed by race_number."""
    resp = session.get(meet_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results_by_race = {}
    race_blocks = soup.select("div.fields-race")

    for race_block in race_blocks:
        race_number_tag = race_block.select_one(".race-number")
        if not race_number_tag:
            continue
        race_number = race_number_tag.text.strip().replace("Race ", "").replace("R", "")

        rows = race_block.select("table.fields-results-table tbody tr")
        if not rows:
            continue

        finishing_order = []
        runners_data = []

        for row in rows:
            # Extract place/position
            place_td = row.select_one("td.place")
            if not place_td:
                continue
            place_text = place_td.text.strip()
            is_scratched = place_text.upper() == "SCR"
            
            # Parse final position
            final_position = None
            if not is_scratched:
                try:
                    final_position = int(place_text)
                except ValueError:
                    pass

            # Extract run box from image alt attribute
            box_img = row.select_one("td.box-number img")
            run_box = None
            if box_img and 'alt' in box_img.attrs:
                try:
                    run_box = int(box_img['alt'])
                except ValueError:
                    pass

            # Extract dog name
            name_tag = row.select_one("td.name b a")
            if not name_tag:
                continue
            dog_name = name_tag.text.strip()

            # Extract trainer
            trainer_tag = row.select_one("td.trainer-column a")
            trainer_name = trainer_tag.text.strip() if trainer_tag else None

            # Extract all data columns
            data_tds = row.select("td.data")
            
            # Finishing time (first data column)
            finishing_time = None
            if len(data_tds) > 0:
                time_text = data_tds[0].text.strip()
                if time_text:
                    try:
                        finishing_time = float(time_text)
                    except ValueError:
                        pass

            # Margin (second data column, inside <text> tag)
            margin = None
            if len(data_tds) > 1:
                margin_text_tag = data_tds[1].find("text")
                if margin_text_tag:
                    margin_text = margin_text_tag.text.strip()
                    if margin_text and margin_text != "-":
                        try:
                            margin = float(margin_text)
                        except ValueError:
                            pass

            # Odds final (third data column, inside span)
            odds_final = None
            if len(data_tds) > 2:
                odds_span = data_tds[2].find("span", class_="hidden-on-scratched")
                if odds_span:
                    odds_text = odds_span.text.strip().replace("$", "")
                    if odds_text and odds_text != "0.00":
                        try:
                            odds_final = float(odds_text)
                        except ValueError:
                            pass

            runner_info = {
                "name": dog_name,
                "run_box": run_box,
                "trainer": trainer_name,
                "is_scratched": is_scratched,
                "finishing_time": finishing_time,
                "margin": margin,
                "final_position": final_position,
                "odds_final": odds_final
            }

            runners_data.append(runner_info)
            if not is_scratched and run_box is not None:
                finishing_order.append(run_box)

        results_by_race[race_number] = {
            "runners": runners_data,
            "results": finishing_order
        }

    return results_by_race

# --------------------
# VALIDATION
# --------------------
def validate_finishing_times(race, race_number, log_file):
    """Validate that faster times correspond to better positions."""
    placed_runners = [
        r for r in race["runners"]
        if not r.get("is_scratched", False)
        and r.get("finishing_time") is not None
        and r.get("final_position") is not None
    ]
    
    if len(placed_runners) < 2:
        return
    
    placed_runners.sort(key=lambda x: x["final_position"])
    
    issues = []
    for i in range(len(placed_runners) - 1):
        curr = placed_runners[i]
        next_runner = placed_runners[i + 1]
        
        if curr["finishing_time"] > next_runner["finishing_time"]:
            issues.append(
                f"Position {curr['final_position']} ({curr['name']}: {curr['finishing_time']}s) "
                f"slower than position {next_runner['final_position']} "
                f"({next_runner['name']}: {next_runner['finishing_time']}s)"
            )
    
    if issues:
        log_message("WARNING", f"Race {race_number} timing issues:", log_file)
        for issue in issues:
            log_message("WARNING", f"  - {issue}", log_file)
    
    if placed_runners:
        winner = placed_runners[0]
        if winner.get("margin") is not None:
            log_message("WARNING", 
                       f"Race {race_number}: Winner {winner['name']} has margin "
                       f"{winner['margin']} (should be null)", log_file)

def validate_race_completeness(prerace_jsonl_path, scraped_results):
    """Check if all races from pre-race file have results. Returns (is_complete, expected, actual)."""
    race_numbers_prerace = set()
    
    with open(prerace_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            race = json.loads(line)
            race_numbers_prerace.add(race["race_number"])
    
    race_numbers_scraped = set(scraped_results.keys())
    
    is_complete = race_numbers_prerace == race_numbers_scraped
    return is_complete, len(race_numbers_prerace), len(race_numbers_scraped)

# --------------------
# JSONL PROCESSING
# --------------------
def update_jsonl_with_results(input_path, output_path, meet_results, log_file):
    """Update JSONL file with scraped results."""
    updated_races = []
    total_matched = 0
    total_runners = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            race = json.loads(line)
            race_number = race["race_number"]
            total_runners += len(race["runners"])

            if race_number in meet_results:
                res_data = meet_results[race_number]
                matched_count = 0
                
                for runner in race["runners"]:
                    matched_runner = None
                    
                    # Primary match: by name
                    matched_runner = next(
                        (r for r in res_data["runners"]
                         if r["name"].strip().lower() == runner["name"].strip().lower()),
                        None
                    )
                    
                    # Fallback match: by drawn_box
                    if not matched_runner and runner.get("drawn_box") is not None:
                        matched_runner = next(
                            (r for r in res_data["runners"]
                             if r.get("run_box") == runner.get("drawn_box")),
                            None
                        )
                    
                    if matched_runner:
                        runner.update({
                            "run_box": matched_runner.get("run_box"),
                            "is_scratched": matched_runner.get("is_scratched", False),
                            "finishing_time": matched_runner.get("finishing_time"),
                            "margin": matched_runner.get("margin"),
                            "final_position": matched_runner.get("final_position"),
                            "odds_final": matched_runner.get("odds_final")
                        })
                        matched_count += 1
                
                race["results"] = res_data["results"]
                validate_finishing_times(race, race_number, log_file)
                
                total_matched += matched_count
                log_message("INFO", f"  Race {race_number}: {matched_count}/{len(race['runners'])} runners matched", log_file)

            updated_races.append(race)

    # Write result JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for race in updated_races:
            f.write(json.dumps(race) + "\n")
    
    log_message("INFO", f"  ✅ Created: {output_path}", log_file)
    log_message("INFO", f"  Total: {total_matched}/{total_runners} runners matched", log_file)

# --------------------
# INDEX MANAGEMENT
# --------------------
def load_index(index_path):
    """Load index file into list of dicts."""
    if not index_path.exists():
        return []
    
    index_data = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            index_data.append(json.loads(line))
    return index_data

def save_index(index_path, index_data):
    """Atomically save index file."""
    temp_path = index_path.with_suffix(".jsonl.tmp")
    
    with open(temp_path, "w", encoding="utf-8") as f:
        for entry in index_data:
            f.write(json.dumps(entry) + "\n")
    
    # Atomic rename
    temp_path.rename(index_path)

def update_index_entry(entry, result_path, status, timestamp=None):
    """Update a single index entry with result metadata."""
    entry["result_created"] = (status == "success")
    entry["result_path"] = str(result_path) if result_path else None
    entry["result_timestamp"] = timestamp or datetime.now().isoformat()
    entry["result_status"] = status
    return entry

def reconcile_filesystem(results_dir, index_data, log_file):
    """Check for orphaned files and missing files."""
    if not results_dir.exists():
        return [], []
    
    # Get all result files in directory
    result_files = {f.name for f in results_dir.glob("*_results.jsonl")}
    
    # Get all result files in index
    index_files = set()
    index_venues = {}
    for entry in index_data:
        if entry.get("result_path"):
            filename = Path(entry["result_path"]).name
            index_files.add(filename)
            index_venues[filename] = entry["venue_slug"]
    
    # Find orphans and missing
    orphaned = result_files - index_files
    missing = []
    
    for entry in index_data:
        if entry.get("result_created") and entry.get("result_path"):
            if not Path(entry["result_path"]).exists():
                missing.append(entry["venue_slug"])
    
    if orphaned:
        log_message("WARNING", f"Orphaned files found: {len(orphaned)}", log_file)
        for filename in sorted(orphaned):
            log_message("WARNING", f"  - {results_dir / filename}", log_file)
    
    if missing:
        log_message("WARNING", f"Missing files: {len(missing)}", log_file)
        for venue in missing:
            log_message("WARNING", f"  - {venue} (index shows success but file missing)", log_file)
    
    return list(orphaned), missing

# --------------------
# DATE VALIDATION
# --------------------
def check_date_safety(date_str, log_file):
    """Warn if processing today or future dates."""
    processing_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    today = datetime.now().date()
    
    if processing_date >= today:
        log_message("WARNING", 
                   f"⚠️  WARNING: Attempting to process {date_str} which is "
                   f"{'TODAY' if processing_date == today else 'in the FUTURE'}", 
                   log_file)
        response = input("Results may not be published yet. Continue? [y/N]: ")
        if response.lower() != 'y':
            log_message("INFO", "Processing aborted by user.", log_file)
            sys.exit(0)

# --------------------
# MAIN PROCESSING
# --------------------
def process_venue(venue_entry, date_str, date_folder_format, session, 
                 jsonl_base_dir, results_base_dir, force, log_file):
    """Process a single venue. Returns updated entry and status."""
    venue_slug = venue_entry["venue_slug"]
    
    # Check if should skip
    current_status = venue_entry.get("result_status", "not_attempted")
    if not force and current_status == "success":
        log_message("INFO", f"[SKIP] {venue_slug} - already completed", log_file)
        return venue_entry, "skipped"
    
    # Build paths
    url_slug = VENUE_URL_EXCEPTIONS.get(venue_slug, venue_slug)
    prerace_jsonl_path = jsonl_base_dir / date_folder_format / f"{venue_slug}_{date_str}_prerace.jsonl"
    result_jsonl_path = results_base_dir / date_folder_format / f"{venue_slug}_{date_str}_results.jsonl"
    meet_url = f"{RESULTS_URL_BASE}/{url_slug}/{date_str}"
    
    log_message("INFO", f"Processing: {venue_slug}", log_file)
    log_message("INFO", f"  URL: {meet_url}", log_file)
    
    # Check pre-race file exists
    if not prerace_jsonl_path.exists():
        log_message("ERROR", f"  Pre-race JSONL not found: {prerace_jsonl_path}", log_file)
        return update_index_entry(venue_entry, result_jsonl_path, "failed"), "failed"
    
    try:
        # Scrape results
        meet_results = scrape_meet_results(session, meet_url)
        
        if not meet_results:
            log_message("WARNING", f"  No results found on page", log_file)
            return update_index_entry(venue_entry, result_jsonl_path, "no_results"), "no_results"
        
        # Validate completeness (all-or-nothing)
        is_complete, expected_races, actual_races = validate_race_completeness(
            prerace_jsonl_path, meet_results
        )
        
        if not is_complete:
            log_message("WARNING", 
                       f"  Incomplete results: {actual_races}/{expected_races} races found", 
                       log_file)
            return update_index_entry(venue_entry, result_jsonl_path, "no_results"), "no_results"
        
        # Update JSONL
        result_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        update_jsonl_with_results(prerace_jsonl_path, result_jsonl_path, meet_results, log_file)
        
        # Update index entry
        timestamp = datetime.now().isoformat()
        return update_index_entry(venue_entry, result_jsonl_path, "success", timestamp), "success"
        
    except requests.RequestException as e:
        log_message("ERROR", f"  Network error: {e}", log_file)
        return update_index_entry(venue_entry, result_jsonl_path, "failed"), "failed"
    except Exception as e:
        log_message("ERROR", f"  Unexpected error: {e}", log_file)
        return update_index_entry(venue_entry, result_jsonl_path, "failed"), "failed"

def process_date(date_str, force=False):
    """Process all venues for a given date."""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    date_folder_format = date_str.replace("-", "")
    
    # Setup logging
    log_file = get_log_file_path(date_obj)
    
    log_message("INFO", "=" * 64, log_file)
    log_message("INFO", f"Processing Results for {date_str}", log_file)
    log_message("INFO", "=" * 64, log_file)
    
    # Check date safety
    check_date_safety(date_str, log_file)
    
    # Load index
    index_path = INDEX_BASE_DIR / f"index_{date_str}_csv.jsonl"
    if not index_path.exists():
        log_message("ERROR", f"Index file not found: {index_path}", log_file)
        return
    
    log_message("INFO", f"Using index file: {index_path}", log_file)
    index_data = load_index(index_path)
    
    # Reconcile filesystem
    results_dir = RESULTS_DIR / date_folder_format
    log_message("INFO", f"Results directory: {results_dir}", log_file)
    orphaned, missing = reconcile_filesystem(results_dir, index_data, log_file)
    
    # Process venues
    session = create_cloudflare_session()
    stats = {"success": 0, "skipped": 0, "no_results": 0, "failed": 0}
    
    for i, entry in enumerate(index_data):
        updated_entry, status = process_venue(
            entry, date_str, date_folder_format, session,
            JSONL_DIR, RESULTS_DIR, force, log_file
        )
        index_data[i] = updated_entry
        stats[status] += 1
        
        # Save index after each venue
        save_index(index_path, index_data)
    
    # Summary
    log_message("INFO", "", log_file)
    log_message("INFO", "PROCESSING SUMMARY:", log_file)
    log_message("INFO", f"  ✅ Success: {stats['success']} venues", log_file)
    log_message("INFO", f"  ⏭️  Skipped: {stats['skipped']} venues (already completed)", log_file)
    log_message("INFO", f"  ⚠️  No Results: {stats['no_results']} venues (incomplete races)", log_file)
    log_message("INFO", f"  ❌ Failed: {stats['failed']} venues (errors)", log_file)
    log_message("INFO", "", log_file)
    log_message("INFO", f"UPDATED INDEX: {index_path}", log_file)
    log_message("INFO", f"LOG FILE: {log_file}", log_file)
    log_message("INFO", "=" * 64, log_file)

# --------------------
# CLI
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update JSONL files with race results and update index metadata."
    )
    parser.add_argument(
        '--date', 
        nargs='+', 
        required=True, 
        help="Date(s) in YYYYMMDD format: single date or start end for range"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force reprocessing of all venues (overwrite existing results)"
    )
    args = parser.parse_args()

    dates = args.date

    if len(dates) == 1:
        date_str = dates[0]
        if len(date_str) == 8:
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        process_date(date_str, force=args.force)
        
    elif len(dates) == 2:
        start_str = dates[0]
        end_str = dates[1]
        if len(start_str) != 8 or len(end_str) != 8:
            raise ValueError("For date range, provide two YYYYMMDD dates.")
        
        start_date = datetime.strptime(start_str, '%Y%m%d')
        end_date = datetime.strptime(end_str, '%Y%m%d')
        current = start_date
        
        while current <= end_date:
            date_str = current.strftime('%Y-%m-%d')
            process_date(date_str, force=args.force)
            current += timedelta(days=1)
    else:
        raise ValueError("Provide exactly one date or two dates (start and end) for the range.")
