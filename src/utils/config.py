from pathlib import Path
import os

# Base path for all data (external volume)
DATA_ROOT = Path("/media/matt_sent/vault/dishlicker_data/data")

# Subfolders
CSV_DIR = DATA_ROOT / "csvs"
JSONL_DIR = DATA_ROOT / "jsonl"
MODEL_DATA_DIR = DATA_ROOT / "model_data"
PREDICTIONS_DIR = DATA_ROOT / "predictions"
ASSESSMENTS_DIR = DATA_ROOT / "assessments"
TRACK_STATISTICS_DIR = DATA_ROOT / "track_statistics"
PROCESSED_DIR = DATA_ROOT / "processed"
RESULTS_DIR = DATA_ROOT / "results"
INDEX_DIR = DATA_ROOT / "index"
BACKUP_DIR = INDEX_DIR / "backup"
BETS_DIR = DATA_ROOT / "logs" / "bets"
DAILY_DIR = BETS_DIR / "daily"
SETTLED_DIR = BETS_DIR / "settled"
DAILY_UPDATED_DIR = BETS_DIR / "daily_updated"

# Logs and assessments (remain on main vault)
LOGS_DIR = Path("/media/matt_sent/vault/dishlicker_data/logs/assessments")

# Web scraping constants
BASE_URL = "https://greyhoundracingsa.com.au"
STATES = ["SA", "NSW", "NT", "QLD", "TAS", "VIC", "WA"]

# Date utils
DATE_FORMAT_FOLDER = "%Y%m%d"   # folder name YYYYMMDD
DATE_FORMAT_FILE = "%Y-%m-%d"   # file names YYYY-MM-DD

# Optional: fallback via env var
DATA_ROOT = Path(os.getenv("DMP_DATA_PATH", DATA_ROOT))
import json
import shutil
from tempfile import NamedTemporaryFile

def get_index_path(date_str: str, filetype: str = "csv", backup: bool = False) -> Path:
    """Return the full path to an index file for the given date."""
    folder = BACKUP_DIR if backup else INDEX_DIR
    suffix = "_csv.jsonl" if filetype == "csv" else ".json"
    return folder / f"index_{date_str}{suffix}"


def verify_jsonl(file_path: Path) -> bool:
    """Quick integrity check for JSONL validity."""
    try:
        with open(file_path, "r") as f:
            first_line = f.readline().strip()
            json.loads(first_line)
        return True
    except Exception:
        return False


def write_index(data, date_str: str, filetype="csv", backup=True, logger=None):
    """Write and optionally back up an index file atomically."""
    main_path = get_index_path(date_str, filetype)
    backup_path = get_index_path(date_str, filetype, backup=True)

    main_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.parent.mkdir(parents=True, exist_ok=True)

    # Write atomically to a temporary file first
    with NamedTemporaryFile("w", delete=False, dir=main_path.parent) as tmp:
        for record in data:
            tmp.write(json.dumps(record) + "\n")
        tmp.flush()
        shutil.move(tmp.name, main_path)

    if backup:
        shutil.copy(main_path, backup_path)

    if logger:
        logger.info(f"[INDEX] Wrote {main_path.name}, backup={backup_path.exists()}")

    return verify_jsonl(main_path)


def update_index_partial(date_str: str, updates, key_field="race_id"):
    """Safely merge updates into an existing JSONL index file."""
    path = get_index_path(date_str)
    existing = {}
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                rec = json.loads(line)
                existing[rec[key_field]] = rec
    for update in updates:
        existing[update[key_field]] = update
    write_index(existing.values(), date_str)
