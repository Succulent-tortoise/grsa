#!/usr/bin/env python3
"""
Consolidated daily pipeline for GRSA greyhound racing analytics.

Runs the full morning pipeline end-to-end:
1. Download pre-race CSV data
2. Convert CSVs to JSONL
3. Run ML predictions
4. Analyze for edge bets
5. Auto-generate bet logger CSV (if bets found)

Designed to be scheduled as a cron job at 6:30 AM daily.
"""

import argparse
import importlib
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_PATH = Path("/media/matt_sent/vault/dishlicker_data/models/random_forest_baseline.pkl")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ──── Logging ─────────────────────────────────────────────
def log(msg: str, *, header: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{header}] " if header else ""
    print(f"[{ts}] {prefix}{msg}")


def send_telegram(text: str):
    """Send a plain-text message via Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram config missing — skipping notification")
        return
    import requests
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        if resp.status_code == 200 and resp.json().get("ok"):
            log("Telegram notification sent")
        else:
            log(f"Telegram send failed: {resp.text}")
    except Exception as e:
        log(f"Telegram send error: {e}")


# ──── Step runner ─────────────────────────────────────────
def run_module(module_path: str, args: list[str], step_label: str) -> None:
    """Invoke a module's __main__ with patched sys.argv via subprocess."""
    cmd = [sys.executable, "-m", module_path] + args
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Module {module_path} exited with code {result.returncode}")


# ──── Pre-flight ──────────────────────────────────────────
def check_model() -> None:
    log("Checking ML model exists…", header="PREFLIGHT")
    if not MODEL_PATH.exists():
        msg = (
            "Pipeline pre-flight failed: ML model not found at\n"
            f"{MODEL_PATH}\n"
            "The pipeline cannot continue."
        )
        log(msg, header="PREFLIGHT")
        send_telegram(msg)
        sys.exit(1)
    log(f"Model found: {MODEL_PATH}", header="PREFLIGHT")


def check_no_racing(date: str) -> None:
    """Send a Telegram message that no racing is scheduled today."""
    msg = f"No racing data available today ({date}). This typically happens only on major public holidays."
    log(msg, header="PIPELINE")
    send_telegram(msg)


def check_no_bets(date: str) -> None:
    """Send a Telegram message that ML ran but no bets matched criteria."""
    msg = f"Model predictions ran for {date}, but no smart bets were identified today."
    log(msg, header="PIPELINE")
    send_telegram(msg)


# ──── Pipeline logic ──────────────────────────────────────
def run_pipeline(date: str) -> None:
    args = [f"--date={date}"]

    # Step 0 — Get yesterday's results
    from datetime import datetime, timedelta
    yesterday = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    log(f"Fetching results for {yesterday}…", header="STEP 0/4")
    try:
        run_module("src.scraper.grsa_results_get", [f"--date={yesterday}"], "STEP 0/4")
    except Exception as e:
        log(f"Results fetch failed: {e}", header="STEP 0/4 FAILED")
        send_telegram(f"Pipeline warning: results fetch failed for {yesterday}: {e}")
        # Non-fatal — continue pipeline

    # Step 1 — Download CSV data
    log("Downloading pre-race CSV data…", header="STEP 1/4")
    try:
        run_module("src.scraper.grsa_csv_get", args, "STEP 1/4")
    except Exception as e:
        log(f"CSV download failed: {e}", header="STEP 1/4 FAILED")
        send_telegram(f"Pipeline failed at Step 1 (CSV download): {e}")
        sys.exit(1)
    log("CSV download complete", header="STEP 1/4 COMPLETE")

    # Step 2 — Convert CSV to JSONL
    log("Converting CSVs to JSONL…", header="STEP 2/4")
    try:
        run_module("src.scraper.grsa_jsonl_get", args, "STEP 2/4")
    except Exception as e:
        log(f"JSONL conversion failed: {e}", header="STEP 2/4 FAILED")
        send_telegram(f"Pipeline failed at Step 2 (CSV→JSONL conversion): {e}")
        sys.exit(1)
    log("JSONL conversion complete", header="STEP 2/4 COMPLETE")

    # Step 3 — Run ML predictions
    log("Running ML predictions…", header="STEP 3/4")
    try:
        run_module("src.scraper.grsa_predict_meeting", args, "STEP 3/4")
    except Exception as e:
        log(f"ML predictions failed: {e}", header="STEP 3/4 FAILED")
        send_telegram(f"Pipeline failed at Step 3 (ML predictions): {e}")
        sys.exit(1)
    log("ML predictions complete", header="STEP 3/4 COMPLETE")

    # Step 4 — Analyze for edge bets
    log("Analyzing for edge bets…", header="STEP 4/4")
    try:
        subprocess.run([sys.executable, "-m", "src.analytics.bet_smart_box_bias", date],
                       check=True)
    except subprocess.CalledProcessError as e:
        log(f"Edge bet analysis failed: {e}", header="STEP 4/4 FAILED")
        send_telegram(f"Pipeline failed at Step 4 (edge bet analysis): {e}")
        sys.exit(1)
    log("Edge bet analysis complete", header="STEP 4/4 COMPLETE")

    # Step 5 — Auto-generate bet logger template
    # bet_smart_box_bias already prints "NO BETTING OPPORTUNITIES TODAY"
    # and sends a Telegram in that case, so we only run the logger if
    # the betting slip file exists.
    date_str = date
    betting_slip = Path("/media/matt_sent/vault/dishlicker_data/data/bets") / f"betting_slip_{date_str}.txt"
    if betting_slip.exists():
        log("Generating bet logger template…", header="STEP 5")
        _generate_bet_logger(date_str)
        log("Bet logger template generated", header="STEP 5 COMPLETE")
    else:
        log("No betting slip found — skipping bet logger", header="STEP 5")
        check_no_bets(date_str)


def _generate_bet_logger(date: str):
    """Non-interactive template generation (equivalent to 'generate' mode with $1 stake)."""
    import json
    from src.analytics.bet_logger_csv import generate_daily_template

    generate_daily_template(date, stake_per_bet=1.0)


# ──── Entry point ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run the full daily GRSA analytics pipeline."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Race date in YYYY-MM-DD format (default: today)",
    )
    args = parser.parse_args()

    date = args.date
    if len(date) == 8 and date.isdigit():
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

    log("=" * 60)
    log(f"DAILY PIPELINE — {date}")
    log("=" * 60)

    check_model()
    run_pipeline(date)

    log("=" * 60)
    log("PIPELINE COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
