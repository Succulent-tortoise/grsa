#!/usr/bin/env python3
"""
20250929
version 1 - initial version
added polite delay
saved file here - /home/matthew/vault/DMP/data/processed/json/track_statistics/
updated to extract all grades for each distance in averages section
extract winning box history for all boxes

Greyhound Racing Track Statistics Scraper

This script scrapes track statistics from greyhoundracingsa.com.au
and outputs the data in JSONL format.

Usage:
    python3 greyhound_track_scraper.py <track_name>

Example:
    python3 greyhound_track_scraper.py angle-park
    
future updates:
1 - update current jsonl file rather than overwrite     
2 - nameing convention to match other files ie track_statistics_venue_YYYY-MM-DD.jsonl
"""

import argparse
import json
import re
import sys
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Any
from glob import glob
import os

from bs4 import BeautifulSoup

# Import the SSL handling function from ssl_utils
import sys
sys.path.append('/home/matt_sent/projects/grsa/src/utils')
from ssl_utils import fetch
from pathlib import Path

# Import for HTTP error handling
import requests


class GreyhoundTrackScraper:
    """Scraper for greyhound racing track statistics."""
    
    def __init__(self, track_name: str):
        """Initialize the scraper with track name."""
        self.track_name = track_name
        self.track_name_slug = track_name.lower().replace(' ', '-')
        self.base_url = "https://greyhoundracingsa.com.au/racing/track"
        self.url = f"{self.base_url}/{self.track_name_slug}"
        # Removed unused session initialization
    
    def fetch_page(self) -> BeautifulSoup:
        """Fetch and parse the track page with retry for 500 errors."""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"Fetching data from: {self.url} (attempt {attempt + 1}/{max_retries})")
                # Polite delay before making a request
                time.sleep(random.uniform(1, 3))  # wait between 1–3 seconds
                
                # Use the SSL-handling fetch function
                response = fetch(self.url, timeout=30)
                html_content = response.text
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Check if we got blocked by Cloudflare or similar
                if "attention required" in soup.get_text().lower() or "blocked" in soup.get_text().lower():
                    raise ValueError("Access blocked by website security. Please try again later.")
                
                return soup
                
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 500:
                    if attempt < max_retries - 1:
                        print(f"500 Server Error on attempt {attempt + 1}, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise ValueError(f"Server error 500 after {max_retries} retries - skipping track: {str(e)}")
                else:
                    raise ValueError(f"HTTP error {getattr(e.response, 'status_code', 'unknown')}: {str(e)}")
            except Exception as e:
                if '500' in str(e).lower():
                    if attempt < max_retries - 1:
                        print(f"500 Server Error detected on attempt {attempt + 1}, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise ValueError(f"Server error 500 after {max_retries} retries - skipping track: {str(e)}")
                else:
                    raise ValueError(f"Request failed: {str(e)}")
    
    def extract_track_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract basic track information."""
        track_info = {}
        
        # Extract track name from profile-info section
        profile_info = soup.find('div', class_='profile-info')
        if profile_info:
            h2_tag = profile_info.find('h2')
            if h2_tag:
                # Remove any img tags and get clean text
                for img in h2_tag.find_all('img'):
                    img.decompose()
                track_info['track_name'] = h2_tag.get_text(strip=True)
            
            # Extract location
            details_div = profile_info.find('div', class_='details')
            if details_div:
                location_span = details_div.find('span')
                if location_span:
                    location_text = location_span.get_text(strip=True)
                    # Extract location after "Location:"
                    if 'Location:' in location_text:
                        track_info['location'] = location_text.split('Location:', 1)[1].strip()
        
        return track_info
    
    def extract_sectional_times(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract best 1st sectional times."""
        sectional_times = []
        
        # Find the sectional times section
        sectional_section = soup.find('div', class_='section-sub-title first-sectional')
        if sectional_section:
            # Find the next table after this section
            table = sectional_section.find_next('table', class_='data-table')
            if table:
                tbody = table.find('tbody')
                if tbody:
                    for row in tbody.find_all('tr'):
                        cells = row.find_all('td')
                        if len(cells) >= 5:
                            # Extract box number from img alt attribute
                            box_img = cells[3].find('img')
                            box_number = box_img.get('alt', '') if box_img else ''
                            
                            # Extract winner name from link
                            winner_link = cells[2].find('a')
                            winner_name = winner_link.get_text(strip=True) if winner_link else cells[2].get_text(strip=True)
                            
                            sectional_times.append({
                                'distance': cells[0].get_text(strip=True),
                                'time': cells[1].get_text(strip=True),
                                'winner': winner_name,
                                'box': box_number,
                                'date': cells[4].get_text(strip=True)
                            })
        
        return sectional_times
    
    def extract_finishing_times(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract best finishing times."""
        finishing_times = []
        
        # Find the finishing times section
        finishing_section = soup.find('div', class_='section-sub-title finishing')
        if finishing_section:
            # Find the next table after this section
            table = finishing_section.find_next('table', class_='data-table')
            if table:
                tbody = table.find('tbody')
                if tbody:
                    for row in tbody.find_all('tr'):
                        cells = row.find_all('td')
                        if len(cells) >= 5:
                            # Extract box number from img alt attribute
                            box_img = cells[3].find('img')
                            box_number = box_img.get('alt', '') if box_img else ''
                            
                            # Extract winner name from link
                            winner_link = cells[2].find('a')
                            winner_name = winner_link.get_text(strip=True) if winner_link else cells[2].get_text(strip=True)
                            
                            finishing_times.append({
                                'distance': cells[0].get_text(strip=True),
                                'time': cells[1].get_text(strip=True),
                                'winner': winner_name,
                                'box': box_number,
                                'date': cells[4].get_text(strip=True)
                            })
        
        return finishing_times
    
    def extract_averages_by_distance_grade(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract averages by distance and grade."""
        averages = []

        # Find the averages section
        averages_section = soup.find('div', class_='average-track-distance-grade')
        if averages_section:
            table = averages_section.find('table', class_='data-table')
            if table:
                tbody = table.find('tbody')
                if tbody:
                    current_distance = None
                    for row in tbody.find_all('tr'):
                        cells = row.find_all('td')
                        # Check if this row has a distance (rowspan)
                        if row.has_attr('class') and 'distance-row' in row['class']:
                            if len(cells) >= 4:
                                current_distance = cells[0].get_text(strip=True) or "N/A"
                                grade = cells[1].get_text(strip=True) or "N/A"
                                sectional_time = cells[2].get_text(strip=True) or "N/A"
                                finishing_time = cells[3].get_text(strip=True) or "N/A"
                                averages.append({
                                    'distance': current_distance,
                                    'grade': grade,
                                    'first_sectional_time': sectional_time,
                                    'finishing_time': finishing_time
                                })
                        else:
                            # This row continues the same distance with different grade
                            if current_distance and len(cells) >= 3:
                                grade = cells[0].get_text(strip=True) or "N/A"
                                sectional_time = cells[1].get_text(strip=True) or "N/A"
                                finishing_time = cells[2].get_text(strip=True) or "N/A"
                                averages.append({
                                    'distance': current_distance,
                                    'grade': grade,
                                    'first_sectional_time': sectional_time,
                                    'finishing_time': finishing_time
                                })

        return averages
    
    def extract_winning_box_history(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract winning box history."""
        box_history = []

        # Find all box history tables (one for each box)
        box_sections = soup.find_all('div', class_='box-history-table')

        for box_section in box_sections:
            table = box_section.find('table', class_='data-table')
            if table:
                # Find all thead elements (each represents a different box)
                theads = table.find_all('thead')

                for thead in theads:
                    # Get distance headers from this thead
                    distance_headers = []
                    box_number = '1'  # Default
                    header_row = thead.find('tr')
                    if header_row:
                        th_tags = header_row.find_all('th')[1:]  # Skip first column (box)
                        distance_headers = [th.get_text(strip=True) for th in th_tags]

                        # Extract box number from header image
                        box_img = header_row.find('img')
                        if box_img:
                            box_number = box_img.get('alt', '1')

                    # Find the tbody that immediately follows this thead
                    tbody = thead.find_next_sibling('tbody')
                    if tbody:
                        rows = tbody.find_all('tr')

                        # Extract data for each metric
                        metrics = ['starts', 'wins', 'win_percent', 'plc', 'plc_percent']
                        for i, metric in enumerate(metrics):
                            if i < len(rows):
                                cells = rows[i].find_all('td')[1:]  # Skip first column (metric name)
                                for j, distance in enumerate(distance_headers):
                                    if j < len(cells):
                                        value = cells[j].get_text(strip=True) or "N/A"
                                        box_history.append({
                                            'starting_box': box_number,
                                            'distance': distance,
                                            'metric': metric,
                                            'value': value
                                        })

        return box_history
    
    def scrape_track_data(self) -> Dict[str, Any]:
        """Main method to scrape all track data."""
        soup = self.fetch_page()
        
        # Extract all data sections
        track_info = self.extract_track_info(soup)
        sectional_times = self.extract_sectional_times(soup)
        finishing_times = self.extract_finishing_times(soup)
        averages = self.extract_averages_by_distance_grade(soup)
        box_history = self.extract_winning_box_history(soup)
        
        # Combine all data
        track_data = {
            'track_info': track_info,
            'best_sectional_times': sectional_times,
            'best_finishing_times': finishing_times,
            'averages_by_distance_grade': averages,
            'winning_box_history': box_history,
            'scraped_at': datetime.now().isoformat(),
            'source_url': self.url
        }
        
        return track_data
    
    def save_to_jsonl(self, data: Dict[str, Any], clean_name: str, today: str, old_files: List[str]) -> str:
        """Save data to JSONL file with proper naming convention, deleting old files on success."""
        # Create filename with passed clean_name and today
        filename = f"track_statistics_{clean_name}_{today}.jsonl"
        
        # Define the target directory
        output_dir = Path("/media/matt_sent/vault/dishlicker_data/data/track_statistics/")
        
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Full path for the file
        full_path = output_dir / filename
        
        # Delete old files (if any, excluding self if somehow matching)
        for old_path_str in old_files:
            old_path = Path(old_path_str)
            if str(old_path) != str(full_path):
                try:
                    os.remove(old_path)
                    print(f"Deleted old file: {old_path}")
                except OSError as e:
                    print(f"Warning: Could not delete old file {old_path}: {e}")
        
        # Write data to JSONL file
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=None)
            f.write('\n')
        
        return str(full_path)


def main():
    """Main function to run the scraper."""
    parser = argparse.ArgumentParser(
        description='Scrape greyhound racing track statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 greyhound_track_scraper.py angle-park
  python3 greyhound_track_scraper.py murray-bridge
  python3 greyhound_track_scraper.py --all
        """
    )
    parser.add_argument('--all', action='store_true', help='Scrape all tracks from venue mapping JSON')
    parser.add_argument('track_name', nargs='?', help='Track name (e.g., angle-park, murray-bridge)')
    
    args = parser.parse_args()
    
    json_path = Path("/media/matt_sent/vault/dishlicker_data/data/track_statistics/venue_stub_mapping.json")
    output_dir = Path("/media/matt_sent/vault/dishlicker_data/data/track_statistics/")
    
    # Load venues mapping
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            venues = json.load(f)
    except Exception as e:
        print(f"❌ Error loading venue mapping JSON: {e}", file=sys.stderr)
        sys.exit(1)
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    if args.all:
        if args.track_name:
            print("Warning: --all specified, ignoring track_name argument")
        
        track_slugs = list(venues.keys())
        successful = 0
        skipped = 0
        total = len(track_slugs)
        for i, slug in enumerate(track_slugs, 1):
            if slug not in venues:
                print(f"❌ Skipping invalid slug: {slug}")
                continue
            
            track_info = venues[slug]
            track_name = track_info['track_name']
            clean_name = re.sub(r'[^\w\s-]', '', track_name).title()
            clean_name = re.sub(r'[-\s]+', '_', clean_name)
            expected_filename = f"track_statistics_{clean_name}_{today}.jsonl"
            expected_path = output_dir / expected_filename
            
            # Check if today's file already exists
            if expected_path.exists():
                print(f"⏭️ Skipping {slug} - already scraped today ({expected_filename})")
                skipped += 1
                continue
            
            old_files = glob(str(output_dir / f"track_statistics_{clean_name}_*.jsonl"))
            
            try:
                scraper = GreyhoundTrackScraper(slug)
                
                # Scrape data
                print(f"Scraping data for track {i}/{total}: {slug}")
                track_data = scraper.scrape_track_data()
                
                # Override track name from mapping for consistency
                track_data['track_info']['track_name'] = track_name
                
                # Save to JSONL file (deletes old on success)
                filename = scraper.save_to_jsonl(track_data, clean_name, today, old_files)
                
                print(f"✅ Data successfully scraped and saved to: {filename}")
                print(f"📊 Track: {track_name}")
                print(f"📍 Location: {track_data['track_info'].get('location', 'Unknown')}")
                print(f"🏁 Sectional times: {len(track_data['best_sectional_times'])} records")
                print(f"🏆 Finishing times: {len(track_data['best_finishing_times'])} records")
                print(f"📈 Averages: {len(track_data['averages_by_distance_grade'])} records")
                print(f"📦 Box history: {len(track_data['winning_box_history'])} records")
                
                successful += 1
                
                # Polite delay between scrapes (longer for batch)
                if i < total:
                    delay = random.uniform(3, 5)
                    print(f"⏳ Waiting {delay:.1f} seconds before next scrape...")
                    time.sleep(delay)
                    
            except ValueError as e:
                print(f"❌ ValueError for {slug}: {e} (keeping old file if exists)", file=sys.stderr)
                continue
            except Exception as e:
                print(f"❌ Unexpected error for {slug}: {e} (keeping old file if exists)", file=sys.stderr)
                continue
        
        print(f"\n🎉 Batch scraping completed: {successful} successful, {skipped} skipped (already current), {total - successful - skipped} failed")
        
    else:
        if not args.track_name:
            parser.error("track_name argument is required when not using --all")
        
        slug = args.track_name
        if slug not in venues:
            print(f"❌ Invalid track slug: {slug} not found in venue mapping", file=sys.stderr)
            sys.exit(1)
        
        track_info = venues[slug]
        track_name = track_info['track_name']
        clean_name = re.sub(r'[^\w\s-]', '', track_name).title()
        clean_name = re.sub(r'[-\s]+', '_', clean_name)
        expected_filename = f"track_statistics_{clean_name}_{today}.jsonl"
        expected_path = output_dir / expected_filename
        
        # Check if today's file already exists
        if expected_path.exists():
            print(f"⏭️ Skipping {slug} - already scraped today ({expected_filename})")
            sys.exit(0)
        
        old_files = glob(str(output_dir / f"track_statistics_{clean_name}_*.jsonl"))
        
        try:
            # Create scraper instance
            scraper = GreyhoundTrackScraper(slug)
            
            # Scrape data
            print(f"Scraping data for track: {slug}")
            track_data = scraper.scrape_track_data()
            
            # Override track name from mapping for consistency
            track_data['track_info']['track_name'] = track_name
            
            # Save to JSONL file (deletes old on success)
            filename = scraper.save_to_jsonl(track_data, clean_name, today, old_files)
            
            print(f"✅ Data successfully scraped and saved to: {filename}")
            print(f"📊 Track: {track_name}")
            print(f"📍 Location: {track_data['track_info'].get('location', 'Unknown')}")
            print(f"🏁 Sectional times: {len(track_data['best_sectional_times'])} records")
            print(f"🏆 Finishing times: {len(track_data['best_finishing_times'])} records")
            print(f"📈 Averages: {len(track_data['averages_by_distance_grade'])} records")
            print(f"📦 Box history: {len(track_data['winning_box_history'])} records")
            
        except ValueError as e:
            print(f"❌ Error: {e} (keeping old file if exists)", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error: {e} (keeping old file if exists)", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()