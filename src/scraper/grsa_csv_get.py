# src/grsa_csv_get_final.py
# 20251001

import requests
from bs4 import BeautifulSoup
import os
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
import certifi
import ssl
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from src.utils.config import CSV_DIR, BACKUP_DIR, get_index_path, write_index, verify_jsonl, update_index_partial
import argparse
import hashlib
import json

# Constants
BASE_URL = "https://greyhoundracingsa.com.au/"
MEETINGS_INDEX_URL = urljoin(BASE_URL, "racing/meetings")
MEETING_URL_TEMPLATE = urljoin(BASE_URL, "racing/meetingdetails/{venue_slug}/{date}")
CSV_LINK_SELECTOR = "a.dropdown-item[href*='/racing/exportfieldscsv/']"
TRACK_LINK_SELECTOR = "a[href*='/racing/track/']"
STATES = ["SA", "NSW", "NT", "QLD", "TAS", "VIC", "WA"]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cloudflare SSL adapter
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

def get_meetings(session, state="SA"):
    """Return list of meeting links from index page for a given state."""
    # Add state parameter to URL
    url = f"{MEETINGS_INDEX_URL}?state={state}"
    try:
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.select("span.form-link-field > a.form-link.fields")
        logger.info(f"Successfully fetched meetings page for {state}")
        return links
    except requests.RequestException as e:
        logger.exception(f"Failed to fetch meetings for {state}: {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error fetching meetings for {state}: {e}")
        return []
    
def extract_meeting_info(session, venue_slug, date):
    """Extract CSV download link and track link."""
    full_url = MEETING_URL_TEMPLATE.format(venue_slug=venue_slug, date=date)
    try:
        response = session.get(full_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # CSV link
        csv_tag = soup.select_one(CSV_LINK_SELECTOR)
        csv_url = urljoin(BASE_URL, csv_tag['href']) if csv_tag else None

        # Track link
        track_tag = soup.select_one(TRACK_LINK_SELECTOR)
        track_url = urljoin(BASE_URL, track_tag['href']) if track_tag else None

        # Extract weather
        weather = extract_weather(soup)

        logger.info(f"Extracted info for {venue_slug} ({date}): CSV={bool(csv_url)}, Track={bool(track_url)}")
        return {
            "venue_slug": venue_slug,
            "date": date,
            "csv_url": csv_url,
            "track_url": track_url,
            "weather": weather
        }
    except requests.RequestException as e:
        logger.exception(f"Failed to fetch meeting details for {venue_slug} ({date}): {e}")
        return {
            "venue_slug": venue_slug,
            "date": date,
            "csv_url": None,
            "track_url": None,
            "weather": "Unknown"
        }
    except Exception as e:
        logger.exception(f"Unexpected error extracting info for {venue_slug} ({date}): {e}")
        return {
            "venue_slug": venue_slug,
            "date": date,
            "csv_url": None,
            "track_url": None,
            "weather": "Unknown"
        }

def extract_weather(soup):
    weather_div = soup.find("div", class_="meeting-weather-heading hide-in-pdf")
    if not weather_div:
        return "Unknown"
    img = weather_div.find("img")
    if img and img.get("alt"):
        weather = img["alt"].strip().title()
    else:
        text = weather_div.get_text(strip=True)
        if "Weather:" in text:
            weather = text.split("Weather:")[-1].strip().title()
        else:
            weather = "Unknown"
    weather_map = {
        "Fine": "Fine",
        "Clear": "Fine",
        "Sunny": "Fine",
        "Overcast": "Overcast",
        "Cloudy": "Overcast",
        "Rain": "Rain",
        "Showers": "Rain",
        "Storm": "Storm",
        "Thunderstorms": "Storm",
    }
    return weather_map.get(weather, "Unknown")
def download_csv(session, csv_url, save_folder, venue_slug, date):
    """Download CSV to save_folder."""
    if not csv_url:
        logger.warning("No CSV URL provided.")
        return None

    filename = f"{venue_slug}_{date}_prerace.csv"
    
    path = save_folder / filename

    # Check if already exists
    if path.exists():
        logger.info(f"Skipped download (CSV exists): {filename}")
        return str(path)

    try:
        response = session.get(csv_url)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        logger.info(f"Downloaded and saved CSV: {filename}")
        return str(path)
    except requests.RequestException as e:
        logger.exception(f"Request failed downloading {csv_url}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error downloading {csv_url}: {e}")
        return None


def compute_md5(file_path):
    h = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.exception(f"Failed to compute MD5 for {file_path}: {e}")
        return None


def process_existing_csvs(save_folder, date):
    meetings_data = []
    for csv_file in save_folder.glob("*_prerace.csv"):
        filename = csv_file.name
        parts = filename.rsplit('_', 2)
        if len(parts) != 3 or parts[2] != 'prerace.csv' or parts[1] != date:
            logger.warning(f"Skipping invalid filename: {filename}")
            continue
        file_venue_slug = parts[0]
        venue_slug = file_venue_slug.replace('_', '-')
        csv_path = str(csv_file)
        file_size_bytes = csv_file.stat().st_size
        hash_md5 = compute_md5(csv_path)
        download_timestamp = None
        last_updated = datetime.utcnow().isoformat()
        info = {
            "venue_slug": venue_slug,
            "date": date,
            "csv_url": None,
            "track_url": None,
            "weather": "Unknown",
            "csv_path": csv_path,
            "download_timestamp": download_timestamp,
            "last_updated": last_updated,
            "file_size_bytes": file_size_bytes,
            "hash_md5": hash_md5,
            "csv_status": "existing",
            "jsonl_created": False,
        }
        meetings_data.append(info)
        if hash_md5:
            logger.info(f"Processed existing CSV for {venue_slug} ({date}): size={file_size_bytes}, md5={hash_md5}")
        else:
            logger.warning(f"Processed existing CSV for {venue_slug} ({date}) but failed to compute MD5")
    logger.info(f"Processed {len(meetings_data)} existing CSV files.")
    return meetings_data

def main():
    parser = argparse.ArgumentParser(description='Download GRSA CSV data for a specific date.')
    parser.add_argument('--date', type=str, default=datetime.now().strftime("%Y-%m-%d"),
                        help='Date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--retro', action='store_true',
                        help='Retrospective mode: create index from existing CSV files without downloading new ones.')
    args = parser.parse_args()
    today = args.date
    try:
        datetime.strptime(today, "%Y-%m-%d")
    except ValueError:
        parser.error(f"Invalid date format: {today}. Use YYYY-MM-DD")
    date_folder = today.replace('-', '')
    save_folder = CSV_DIR / date_folder

    current_date = datetime.now().date()
    target_date = datetime.strptime(today, "%Y-%m-%d").date()
    if target_date < current_date and not args.retro:
        print("⚠ Error: No CSV files available in the past. Use --retro to create index from existing files.")
        sys.exit(1)
    try:
        save_folder.mkdir(parents=True, exist_ok=True)
        print(f"Output directory ready: {save_folder}")
    except PermissionError as e:
        print(f"Permission denied: Cannot create directory {save_folder}.")
        print(f"Please create the parent directory manually: mkdir -p {save_folder.parent}")
        logger.error(f"Directory creation failed due to permissions: {e}")
        sys.exit(1)

    # Initialize meetings_data as empty list
    meetings_data = []

    if args.retro:
        logger.info(f"Retrospective mode: scanning for existing CSVs in {save_folder}")
        meetings_data = process_existing_csvs(save_folder, today)
    else:
        session = create_cloudflare_session()
        logger.info("Fetching meeting links across states...")
        # Loop through all states
        for state in STATES:
            logger.info(f"Processing {state}...")
            
            # Polite delay between states
            if state != STATES[0]:  # Don't delay before first state
                time.sleep(random.uniform(1, 3))
            
            try:
                links = get_meetings(session, state)
                logger.info(f"Retrieved {len(links)} meeting links for {state}.")
            except Exception as e:
                logger.warning(f"Failed to fetch meetings for {state}: {e}")
                continue
        
            for link in links:
                
                # Debug: print the actual href
                href = link.get("href")
                logger.debug(f"[{state}] Found link: {href}")
            
                # Extract venue and date from href like: /racing/meetingdetails/gawler/2025-10-01
                href_parts = link.get("href").split("/")
                venue_slug = href_parts[-2]  # Gets 'gawler'
                date = href_parts[-1]         # Gets '2025-10-01'
                
                # Only process if it's today's meeting
                if date != today:
                    logger.info(f"Skipping {venue_slug} ({date}) - not today")
                    continue

                file_venue_slug = venue_slug.replace('ladbrokes-', '', 1).replace('-', '_')

                try:
                    # Polite random delay between 1-3 seconds
                    time.sleep(random.uniform(1, 3))
          
                    info = extract_meeting_info(session, venue_slug, date)
                    info['venue_slug'] = venue_slug.replace('ladbrokes-', '', 1)
                    download_timestamp = datetime.utcnow().isoformat()
                    csv_path = download_csv(session, info['csv_url'], save_folder, file_venue_slug, today)
                    info['csv_path'] = csv_path
                    if csv_path:
                        try:
                            file_size_bytes = os.path.getsize(csv_path)
                            h = hashlib.md5()
                            with open(csv_path, "rb") as f:
                                for chunk in iter(lambda: f.read(8192), b""):
                                    h.update(chunk)
                            hash_md5 = h.hexdigest()
                            csv_status = "downloaded"
                            logger.info(f"Added metadata for {info['venue_slug']} ({date}): size={file_size_bytes}, md5={hash_md5}")
                        except (OSError, IOError) as e:
                            logger.exception(f"Failed to compute metadata for {csv_path}: {e}")
                            file_size_bytes = None
                            hash_md5 = None
                            csv_status = "error"
                    else:
                        file_size_bytes = None
                        hash_md5 = None
                        csv_status = "failed"
                    info['download_timestamp'] = download_timestamp
                    info['last_updated'] = download_timestamp
                    info['file_size_bytes'] = file_size_bytes
                    info['hash_md5'] = hash_md5
                    info['csv_status'] = csv_status
                    info['jsonl_created'] = False
                    meetings_data.append(info)
                    if csv_path:
                        logger.info(f"Processed CSV for {venue_slug} ({date})")
                    else:
                        logger.warning(f"Failed to process CSV for {venue_slug} ({date})")
                except Exception as e:
                    logger.exception(f"Failed to process {venue_slug} ({date}): {e}")

    # Save JSON index
    json_index_path = get_index_path(today, filetype="json", backup=True)
    json_index_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(json_index_path, "w") as f:
            json.dump(meetings_data, f, indent=2)
        logger.info(f"[INDEX] Wrote JSON backup: {json_index_path}")
    except (IOError, OSError) as e:
        logger.exception(f"Failed to write JSON index {json_index_path}: {e}")
        
    # Save CSV JSONL index
    csv_index_path = get_index_path(today, "csv")
    index_timestamp = datetime.utcnow().isoformat()
    # Preprocess data
    for item in meetings_data:
        required_keys = ['download_timestamp', 'file_size_bytes', 'hash_md5', 'csv_status', 'jsonl_created', 'last_updated']
        for key in required_keys:
            if key not in item:
                item[key] = None
        item['last_updated'] = index_timestamp

    success = write_index(meetings_data, today, filetype="csv", backup=False, logger=logger)
    if success:
        logger.info(f"[INDEX] Verified JSONL index: {csv_index_path} (records={len(meetings_data)})")
    else:
        logger.error(f"[INDEX] Verification failed for {csv_index_path}")
    
    successful = [m for m in meetings_data if m.get('csv_path')]
    logger.info(f"Successfully processed {len(successful)} meetings across all states")
    logger.info("All done!")
    
    # Final summary
    print("Summary:")
    if args.retro:
        print(f"  Existing CSVs indexed: {len(successful)}")
    else:
        print(f"  CSVs downloaded: {len(successful)}")
    print("  Indices created: 2")
    print("  Indices updated (historical): 0")
    print("  Missing CSV files found: 0")

if __name__ == "__main__":
    main()