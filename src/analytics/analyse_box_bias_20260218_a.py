#!/usr/bin/env python3
"""
Box Bias Analyser - Understand track patterns before building filters
Analyses all track statistics to identify meaningful box biases
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics
from datetime import datetime

# Configuration
TRACK_STATS_DIR = Path("/media/matt_sent/vault/dishlicker_data/data/track_statistics")
OUTPUT_FILE = Path("/media/matt_sent/vault/dishlicker_data/data/track_statistics/box_bias_analysis_report.txt")

# Filtering parameters
MIN_STARTS_PER_BOX = 100  # Minimum historical starts to be considered reliable
MIN_VALID_DISTANCE = 200  # Minimum valid race distance in meters (filters out garbage data)
MAX_VALID_DISTANCE = 900  # Maximum valid race distance in meters

# Venue name standardisation - handle aliases
VENUE_ALIASES = {
    "DARWIN": "WINELLE",
    "WINELLE": "WINELLE",
    "MEADOWS": "MEADOWS-MEP",
    "MEADOWS-MEP": "MEADOWS-MEP",
    "THE-MEADOWS": "MEADOWS-MEP",
    "Q1-LAKESIDE": "Q1-LAKESIDE",
    "LADBROKES-Q1": "Q1-LAKESIDE",
    "LADBROKES-Q1-LAKESIDE": "Q1-LAKESIDE",
    "Q2-PARKLANDS": "Q2-PARKLANDS",
    "LADBROKES-Q2": "Q2-PARKLANDS",
    "LADBROKES-Q2-PARKLANDS": "Q2-PARKLANDS",
    "Q-STRAIGHT": "Q-STRAIGHT",
    "LADBROKES-STRAIGHT": "Q-STRAIGHT",
    "LADBROKES-Q-STRAIGHT": "Q-STRAIGHT",
}


def load_venue_mapping() -> Dict[str, str]:
    """Load venue stub mapping from JSON file"""
    mapping_file = TRACK_STATS_DIR / "venue_stub_mapping.json"
    
    if not mapping_file.exists():
        print(f"⚠️  Venue mapping file not found: {mapping_file}")
        print("   Will use basic normalization only")
        return {}
    
    try:
        with open(mapping_file, 'r') as f:
            stub_data = json.load(f)
        
        # Create normalized mapping: "albion park" -> "ALBION-PARK"
        mapping = {}
        for stub, info in stub_data.items():
            track_name = info.get('track_name', stub)
            # Normalize the track name for lookup
            normalized_key = track_name.lower().replace(' ', '-').replace('_', '-')
            # Store as uppercase with hyphens
            mapping[normalized_key] = stub.upper()
        
        print(f"✅ Loaded {len(mapping)} venue mappings from venue_stub_mapping.json")
        return mapping
    
    except Exception as e:
        print(f"❌ Error loading venue mapping: {e}")
        return {}


def normalise_venue_name(venue: str, venue_mapping: Dict[str, str]) -> str:
    """Normalise venue names using mapping file and aliases"""
    # First normalize the input
    venue_normalized = venue.lower().strip().replace(' ', '-').replace('_', '-')
    
    # Try venue mapping first
    if venue_normalized in venue_mapping:
        return venue_mapping[venue_normalized]
    
    # Then try aliases (convert to upper for alias lookup)
    venue_upper = venue.upper().strip().replace(' ', '-').replace('_', '-')
    if venue_upper in VENUE_ALIASES:
        return VENUE_ALIASES[venue_upper]
    
    # Default: return uppercase with hyphens
    return venue_upper


def load_track_statistics() -> Dict:
    """Load all track statistics files"""
    all_stats = defaultdict(lambda: defaultdict(dict))
    venue_files = {}
    
    print("Loading track statistics files...")
    
    if not TRACK_STATS_DIR.exists():
        print(f"❌ Directory not found: {TRACK_STATS_DIR}")
        return None
    
    # Load venue mapping
    venue_mapping = load_venue_mapping()
    
    # Find all track statistics files
    stat_files = list(TRACK_STATS_DIR.glob("track_statistics_*.jsonl"))
    
    if not stat_files:
        print(f"❌ No track statistics files found in {TRACK_STATS_DIR}")
        return None
    
    print(f"Found {len(stat_files)} track statistics files\n")
    
    # Pattern to match: track_statistics_{VENUE}_{DATE}.jsonl
    filename_pattern = re.compile(r'^track_statistics_(.+?)_(\d{4}-\d{2}-\d{2})\.jsonl$')
    
    for file_path in stat_files:
        try:
            match = filename_pattern.match(file_path.name)
            
            if not match:
                print(f"⚠️  Skipping file with unexpected format: {file_path.name}")
                continue
            
            venue_raw = match.group(1)  # Everything between prefix and date
            date = match.group(2)  # The date (just for tracking data freshness)
            
            venue = normalise_venue_name(venue_raw, venue_mapping)
            
            # Track which file we're using for each venue (newest by date)
            if venue not in venue_files or date > venue_files[venue]['date']:
                venue_files[venue] = {'file': file_path.name, 'date': date}
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract box bias data
            if 'winning_box_history' in data:
                entries_processed = 0
                for entry in data['winning_box_history']:
                    box = entry.get('starting_box')
                    distance = entry.get('distance', '').replace('m', '')
                    metric = entry.get('metric')
                    value = entry.get('value', '').replace('%', '')
                    
                    if box and distance and metric:
                        key = f"{venue}_{distance}"
                        
                        if box not in all_stats[key]:
                            all_stats[key][box] = {}
                        
                        all_stats[key][box][metric] = value
                        entries_processed += 1
                
                if entries_processed == 0 and venue == "GAWLER":  # Debug one venue
                    print(f"DEBUG - No entries processed for GAWLER. Sample entry:")
                    if data['winning_box_history']:
                        print(f"  {data['winning_box_history'][0]}")
        
        except Exception as e:
            print(f"❌ Error loading {file_path.name}: {e}")
            continue
    
    print("="*80)
    print("TRACK STATISTICS FILES LOADED")
    print("="*80)
    for venue in sorted(venue_files.keys()):
        info = venue_files[venue]
        print(f"{venue:<30} | File: {info['file']:<50} | Date: {info['date']}")
    
    return all_stats


def analyse_box_bias(all_stats: Dict) -> Dict:
    """Analyse box bias patterns across all venues"""
    
    bias_analysis = []
    filtered_out = {
        'invalid_distance': 0,
        'low_sample_size': 0,
        'total_kept': 0
    }
    
    for venue_dist, boxes in all_stats.items():
        venue, distance = venue_dist.rsplit('_', 1)
        
        # Filter out invalid distances
        try:
            distance_int = int(distance)
            if distance_int < MIN_VALID_DISTANCE or distance_int > MAX_VALID_DISTANCE:
                filtered_out['invalid_distance'] += 1
                continue
        except ValueError:
            filtered_out['invalid_distance'] += 1
            continue
        
        # Get number of boxes (typically 8, but could vary)
        num_boxes = len(boxes)
        if num_boxes == 0:
            continue
        
        # First pass: collect all boxes and calculate total starts
        all_boxes_data = []
        total_starts = 0
        
        for box_num in sorted(boxes.keys(), key=lambda x: int(x)):
            box_info = boxes[box_num]
            
            # Get win percentage
            win_pct_str = box_info.get('win_percent', '0%')
            try:
                win_pct = float(win_pct_str)
            except:
                continue
            
            # Get sample size
            starts_str = box_info.get('starts', '0')
            try:
                starts = int(starts_str.replace(',', ''))
            except:
                starts = 0
            
            # Filter out low sample sizes
            if starts < MIN_STARTS_PER_BOX:
                filtered_out['low_sample_size'] += 1
                continue
            
            all_boxes_data.append({
                'box_num': int(box_num),
                'win_pct': win_pct,
                'starts': starts
            })
            total_starts += starts
        
        # Skip if no valid boxes
        if not all_boxes_data or total_starts == 0:
            continue
        
        # Second pass: calculate bias based on actual usage distribution
        box_data = []
        for box_info in all_boxes_data:
            # Calculate expected win rate based on proportion of total starts
            expected_win_rate = (box_info['starts'] / total_starts) * 100
            
            # Calculate bias relative to this box's expected rate
            if expected_win_rate > 0:
                bias = ((box_info['win_pct'] - expected_win_rate) / expected_win_rate) * 100
            else:
                bias = 0
            
            box_data.append({
                'box': box_info['box_num'],
                'win_pct': box_info['win_pct'],
                'bias': bias,
                'starts': box_info['starts'],
                'expected_win_rate': expected_win_rate
            })
        
        # Only include venue+distance if we have at least some valid boxes
        if box_data:
            bias_analysis.append({
                'venue': venue,
                'distance': distance,
                'num_boxes': num_boxes,
                'total_starts': total_starts,
                'boxes': box_data
            })
            filtered_out['total_kept'] += len(box_data)
    
    # Print filtering summary
    print(f"\n📊 DATA FILTERING SUMMARY:")
    print(f"   Filtered out {filtered_out['invalid_distance']} venue+distance combos (invalid distance <{MIN_VALID_DISTANCE}m or >{MAX_VALID_DISTANCE}m)")
    print(f"   Filtered out {filtered_out['low_sample_size']} boxes (less than {MIN_STARTS_PER_BOX} starts)")
    print(f"   ✅ Kept {filtered_out['total_kept']} reliable box+venue+distance combinations")
    print(f"   📐 Using actual usage distribution for bias calculation (not assuming equal field sizes)\n")
    
    return bias_analysis





def save_json_lookup(bias_analysis, json_filename):
    """Convert bias analysis to nested JSON lookup structure: venue -> distance -> box -> bias info"""
    lookup = {}
    for entry in bias_analysis:
        venue = entry['venue']
        distance = entry['distance']
        if venue not in lookup:
            lookup[venue] = {}
        if distance not in lookup[venue]:
            lookup[venue][distance] = {}
        for box in entry['boxes']:
            box_num = box['box']
            lookup[venue][distance][box_num] = {
                'win_pct': box['win_pct'],
                'bias': box['bias'],
                'starts': box['starts'],
                'expected_win_rate': box['expected_win_rate']
            }
    with open(json_filename, 'w') as f:
        json.dump(lookup, f, indent=2)


def check_venue_naming_issues(bias_analysis: List[Dict]):
    """Check for potential venue naming issues"""
    
    print("\n" + "="*100)
    print("🔍 VENUE NAMING CHECK")
    print("="*100)
    
    venues = set(entry['venue'] for entry in bias_analysis)
    
    print(f"\nFound {len(venues)} unique venue names:")
    for venue in sorted(venues):
        distances = [entry['distance'] for entry in bias_analysis if entry['venue'] == venue]
        print(f"  • {venue:<30} | Distances: {', '.join(sorted(set(distances)))}m")
    
    # Check for potential duplicates (similar names)
    print("\n⚠️  Potential naming issues to review:")
    potential_issues = []
    
    venue_list = sorted(venues)
    for i, v1 in enumerate(venue_list):
        for v2 in venue_list[i+1:]:
            # Check if names are very similar (simple heuristic)
            if v1.replace('-', '').replace(' ', '') in v2.replace('-', '').replace(' ', '') or \
               v2.replace('-', '').replace(' ', '') in v1.replace('-', '').replace(' ', ''):
                potential_issues.append((v1, v2))
    
    if potential_issues:
        for v1, v2 in potential_issues:
            print(f"  ⚠️  '{v1}' and '{v2}' might be the same venue")
    else:
        print("  ✅ No obvious naming conflicts detected")
    
    print("\n" + "="*100)


def main():
    print("\n" + "="*100)
    print("BOX BIAS ANALYSER - TRACK STATISTICS ANALYSIS")
    print("="*100)
    print("\nThis script analyses historical box bias data to help you understand:")
    print("  1. Which boxes at which venues have strong advantages/disadvantages")
    print("  2. What thresholds make sense for 'strong' vs 'weak' biases")
    print("  3. Any venue naming issues that need to be resolved")
    
    # Load data
    all_stats = load_track_statistics()
    
    if not all_stats:
        print("\n❌ Failed to load track statistics. Exiting.")
        return
    
    # Analyse bias patterns
    bias_analysis = analyse_box_bias(all_stats)
    
    # Open output file
    print(f"\n📝 Writing detailed analysis to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        # Write to file
        f.write("="*100 + "\n")
        f.write("BOX BIAS ANALYSIS - DETAILED REPORT\n")
        f.write("="*100 + "\n\n")
        
        # Write all venue+distance details
        f.write("="*100 + "\n")
        f.write("BOX BIAS BY VENUE AND DISTANCE\n")
        f.write("="*100 + "\n")
        
        for entry in sorted(bias_analysis, key=lambda x: (x['venue'], int(x['distance']))):
            venue = entry['venue']
            distance = entry['distance']
            boxes = entry['boxes']
            total_starts = entry['total_starts']
            
            f.write(f"\n{venue} - {distance}m (Total starts: {total_starts:,})\n")
            f.write("-" * 100 + "\n")
            
            for box in boxes:
                bias_str = f"{box['bias']:+.1f}%"
                
                if box['bias'] > 30:
                    flag = "🟢 STRONG+"
                elif box['bias'] > 15:
                    flag = "🟡 MODERATE+"
                elif box['bias'] < -15:
                    flag = "🔴 WEAK"
                elif box['bias'] < -5:
                    flag = "🟠 MODERATE-"
                else:
                    flag = "⚪ NEUTRAL"
                
                f.write(f"  Box {box['box']}: {box['win_pct']:>5.1f}% win rate | Bias: {bias_str:>7} | "
                        f"Starts: {box['starts']:>6,} | {flag}\n")
    
    print(f"✅ Detailed venue-by-venue analysis saved to: {OUTPUT_FILE}")

    # Generate JSON lookup file
    json_filename = TRACK_STATS_DIR / f"box_bias_lookup_{datetime.now().strftime('%Y-%m-%d')}.json"
    save_json_lookup(bias_analysis, json_filename)
    print(f"✅ JSON lookup file saved to: {json_filename}")
    
    # Print summary statistics to console only
    print_summary_statistics(bias_analysis)
    
    # Check for naming issues
    check_venue_naming_issues(bias_analysis)
    
    print("\n✅ Analysis complete!")
    print(f"\n📄 Full report saved to: {OUTPUT_FILE}")
    print("\nNext steps:")
    print("  1. Review the report file for detailed bias by venue")
    print("  2. Review the thresholds shown above")
    print("  3. Check if any venue names need to be added to VENUE_ALIASES")
    print("  4. Run box_bias_builder.py to create the lookup file")
    print("  5. Integrate with bet_smart.py")


def print_summary_statistics(bias_analysis: List[Dict]):
    """Print summary statistics to console"""
    
    # Collect all bias values for statistical summary
    all_biases = []
    strong_positive_biases = []
    strong_negative_biases = []
    
    for entry in bias_analysis:
        for box in entry['boxes']:
            all_biases.append(box['bias'])
            
            if box['bias'] > 30:
                strong_positive_biases.append((entry['venue'], entry['distance'], box['box'], box['bias'], box['starts']))
            elif box['bias'] < -30:
                strong_negative_biases.append((entry['venue'], entry['distance'], box['box'], box['bias'], box['starts']))
    
    # Statistical summary
    print("\n" + "="*100)
    print("STATISTICAL SUMMARY OF BOX BIASES")
    print("="*100)
    
    if all_biases:
        print(f"\nTotal venue+distance combinations analysed: {len(bias_analysis)}")
        print(f"Total box+venue+distance data points: {len(all_biases)}")
        print(f"Mean bias: {statistics.mean(all_biases):+.1f}%")
        print(f"Median bias: {statistics.median(all_biases):+.1f}%")
        print(f"Std deviation: {statistics.stdev(all_biases):.1f}%")
        print(f"Min bias: {min(all_biases):+.1f}%")
        print(f"Max bias: {max(all_biases):+.1f}%")
        
        # Percentiles
        sorted_biases = sorted(all_biases)
        p10 = sorted_biases[int(len(sorted_biases) * 0.10)]
        p90 = sorted_biases[int(len(sorted_biases) * 0.90)]
        
        print(f"\n10th percentile (weakest biases): {p10:+.1f}%")
        print(f"90th percentile (strongest biases): {p90:+.1f}%")
    
    # Top strong positive biases
    print("\n" + "="*100)
    print("🟢 TOP 20 STRONGEST POSITIVE BIASES (Boxes that win MORE than expected)")
    print("="*100)
    
    strong_positive_biases.sort(key=lambda x: x[3], reverse=True)
    for i, (venue, dist, box, bias, starts) in enumerate(strong_positive_biases[:20], 1):
        print(f"{i:>2}. {venue:<25} {dist:>4}m Box {box} | Bias: {bias:+.1f}% | Starts: {starts:>6,}")
    
    # Top strong negative biases
    print("\n" + "="*100)
    print("🔴 TOP 20 STRONGEST NEGATIVE BIASES (Boxes that win LESS than expected)")
    print("="*100)
    
    strong_negative_biases.sort(key=lambda x: x[3])
    for i, (venue, dist, box, bias, starts) in enumerate(strong_negative_biases[:20], 1):
        print(f"{i:>2}. {venue:<25} {dist:>4}m Box {box} | Bias: {bias:+.1f}% | Starts: {starts:>6,}")
    
    # Recommendations
    print("\n" + "="*100)
    print("📊 RECOMMENDED THRESHOLDS FOR bet_smart.py")
    print("="*100)
    
    if all_biases:
        print(f"\nBased on the distribution of {len(all_biases)} data points:")
        print(f"\n🟢 STRONG POSITIVE BIAS: Box bias ≥ +30%")
        print(f"   → Boost confidence in these bets")
        print(f"   → Found in {len([b for b in all_biases if b >= 30])} cases")
        
        print(f"\n🟡 MODERATE POSITIVE BIAS: Box bias between +15% and +30%")
        print(f"   → Slightly favour these bets")
        print(f"   → Found in {len([b for b in all_biases if 15 <= b < 30])} cases")
        
        print(f"\n⚪ NEUTRAL: Box bias between -15% and +15%")
        print(f"   → No adjustment needed")
        print(f"   → Found in {len([b for b in all_biases if -15 <= b < 15])} cases")
        
        print(f"\n🟠 MODERATE NEGATIVE BIAS: Box bias between -30% and -15%")
        print(f"   → Be cautious with these bets")
        print(f"   → Found in {len([b for b in all_biases if -30 < b <= -15])} cases")
        
        print(f"\n🔴 STRONG NEGATIVE BIAS: Box bias ≤ -30%")
        print(f"   → Consider skipping these bets")
        print(f"   → Found in {len([b for b in all_biases if b <= -30])} cases")
        
        print(f"\n⚠️  NOTE: Some extreme biases (+700%, -100%) are based on tiny sample sizes")
        print(f"   We should filter these out and focus on distances with 100+ starts per box")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()
