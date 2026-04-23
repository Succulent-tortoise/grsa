#!/usr/bin/env python3
"""
Smart Betting Recommender with Sweet Spot Targeting and Box Bias Filtering
Uses proven profitable niches identified from historical analysis
Enhanced with track-specific box bias data
provides cut and paste text block for the bet smart script - updated 8/2/26
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import os
import requests
from dotenv import load_dotenv


# Configuration
PREDICTIONS_DIR = Path("/media/matt_sent/vault/dishlicker_data/data/predictions")
RESULTS_DIR = Path("/media/matt_sent/vault/dishlicker_data/data/results")
OUTPUT_DIR = Path("/media/matt_sent/vault/dishlicker_data/data/bets")
TRACK_STATS_DIR = Path("/media/matt_sent/vault/dishlicker_data/data/track_statistics")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Venue name standardisation (matches analyse_box_bias.py)
VENUE_ALIASES = {
    "DARWIN": "WINNELLIE",
    "WINNELLIE": "WINNELLIE",
    "MEADOWS": "MEADOWS-MEP",
    "MEADOWS-MEP": "MEADOWS-MEP",
    "THE-MEADOWS": "MEADOWS-MEP",
    "Q1-LAKESIDE": "LADBROKES-Q1-LAKESIDE",
    "LADBROKES-Q1": "LADBROKES-Q1-LAKESIDE",
    "Q2-PARKLANDS": "LADBROKES-Q2-PARKLANDS",
    "LADBROKES-Q2": "LADBROKES-Q2-PARKLANDS",
    "Q-STRAIGHT": "LADBROKES-Q-STRAIGHT",
    "LADBROKES-STRAIGHT": "LADBROKES-Q-STRAIGHT",
}

@dataclass
class SweetSpot:
    venue: str
    distance: int
    grade: str
    priority: int
    historical_bets: int
    win_rate: float
    roi: float
    
    def matches(self, bet_venue: str, bet_distance: int, bet_grade: str) -> bool:
        return (self.venue.lower() == bet_venue.lower() and 
                self.distance == bet_distance and 
                self.grade.lower() == bet_grade.lower())

@dataclass
class BettingRecommendation:
    venue: str
    date: str
    race_number: int
    race_name: str
    time: str
    runner_name: str
    box: int
    trainer: str
    winner_prob: float
    odds: float
    confidence_gap: float
    distance: int
    grade: str
    sweet_spot: SweetSpot = None
    second_best_name: str = None
    box_bias_value: float = None
    box_bias_category: str = None
    box_bias_starts: int = None
    
    def __str__(self):
        if self.sweet_spot:
            flag = f"🎯 SWEET SPOT #{self.sweet_spot.priority}"
            detail = f" (Historical: {self.sweet_spot.historical_bets} bets, {self.sweet_spot.win_rate:.0%} win, {self.sweet_spot.roi:+.0f}% ROI)"
        else:
            flag = "✅ GOOD BET"
            detail = ""
        
        base_str = (f"{flag} - {self.venue.upper()} Race {self.race_number} @ {self.time}\n"
                    f"  Runner: {self.runner_name} (Box {self.box}) - {self.trainer}\n"
                    f"  Distance: {self.distance}m | Grade: {self.grade}\n"
                    f"  Model Confidence: {self.winner_prob:.1%} | Gap to 2nd: {self.confidence_gap:.1%}\n"
                    f"  Odds: ${self.odds:.2f}{detail}")
        
        # Add box bias info if available
        if self.box_bias_category:
            bias_icons = {
                "STRONG+": "🟢",
                "MODERATE+": "🟡",
                "NEUTRAL": "⚪",
                "MODERATE-": "🟠",
                "WEAK": "🔴"
            }
            icon = bias_icons.get(self.box_bias_category, "⚪")
            base_str += f"\n  Box Bias: {self.box_bias_value:+.1f}% ({icon} {self.box_bias_category}) [{self.box_bias_starts:,} starts]"
        
        return base_str


class SmartBettingRecommender:
    def __init__(self, date: str):
        self.date = date
        self.date_formatted = date.replace('-', '')
        
        # Load box bias data
        self.box_bias_data = self.load_box_bias()
        self.box_bias_stats = {
            'priority_1_kept': 0,
            'priority_2_skipped': 0,
            'good_bets_skipped': 0,
            'no_data': 0
        }
        
        # Define Sweet Spots (from edge finder analysis - updated 2025-12-04)
        self.sweet_spots = [
            # TOP TIER - Priority 1 (Highest ROI: 96-202%)
            SweetSpot("CASINO", 300, "M", 1, 7, 0.714, 202.1),
            SweetSpot("TAREE", 300, "M", 2, 24, 0.375, 135.6),
            SweetSpot("GRAFTON", 450, "4/5", 3, 5, 1.000, 128.0),
            SweetSpot("TOWNSVILLE", 380, "M", 4, 15, 0.733, 123.0),
            SweetSpot("BROKEN-HILL", 375, "MX", 5, 5, 0.600, 96.0),
            
            # HIGH TIER - Priority 2 (Good ROI: 61-85%)
            SweetSpot("Q1-LAKESIDE", 457, "M", 6, 6, 1.000, 85.0),
            SweetSpot("DUBBO", 318, "M", 7, 9, 0.444, 83.6),
            SweetSpot("GUNNEDAH", 340, "NG1-4", 8, 6, 1.000, 80.0),
            SweetSpot("Q1-LAKESIDE", 457, "4/5", 9, 7, 0.714, 77.9),
            SweetSpot("NOWRA", 520, "MX", 10, 10, 0.700, 76.0),
            SweetSpot("Q2-PARKLANDS", 520, "M5", 11, 10, 0.900, 71.0),
            SweetSpot("MOUNT-GAMBIER", 400, "OPEN", 12, 5, 0.600, 68.0),
            SweetSpot("GUNNEDAH", 431, "4/5", 13, 6, 0.667, 67.5),
            SweetSpot("GOULBURN", 440, "M", 14, 5, 1.000, 63.2),
            SweetSpot("MURRAY-BRIDGE-STRAIGHT", 300, "M", 15, 8, 0.750, 61.0),
        ]
        
        # Fallback filters (if no sweet spots available)
        self.profitable_venues = ["NORTHAM", "TAREE", "MURRAY-BRIDGE-STRAIGHT", "MANDURAH", "Q2-PARKLANDS"]
        self.profitable_distances = [297, 516, 431, 512, 457, 300, 366]
        
        # Minimum criteria
        self.min_odds = 2.50
        self.min_confidence_gap = 0.10
        
        self.recommendations = []
    
    def load_box_bias(self) -> Optional[Dict]:
        """Load the most recent box bias data file"""
        print("\n📊 Loading box bias data...")
        
        if not TRACK_STATS_DIR.exists():
            print(f"   ⚠️  Track statistics directory not found: {TRACK_STATS_DIR}")
            print("   Continuing without box bias filtering")
            return None
        
        # Find all box bias files
        bias_files = list(TRACK_STATS_DIR.glob("box_bias_lookup_*.json"))
        
        if not bias_files:
            print("   ⚠️  No box bias files found")
            print("   Continuing without box bias filtering")
            return None
        
        # Extract dates and find newest
        file_dates = []
        for file_path in bias_files:
            try:
                date_str = file_path.stem.replace("box_bias_lookup_", "")
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                file_dates.append((date_obj, file_path))
            except ValueError:
                print(f"   ⚠️  Skipping file with invalid date format: {file_path.name}")
                continue
        
        if not file_dates:
            print("   ⚠️  No valid box bias files found")
            print("   Continuing without box bias filtering")
            return None
        
        # Sort by date and get newest
        file_dates.sort(reverse=True, key=lambda x: x[0])
        newest_date, newest_file = file_dates[0]
        
        print(f"   Found {len(file_dates)} box bias file(s)")
        print(f"   ✅ Using most recent: {newest_file.name} (from {newest_date.strftime('%Y-%m-%d')})")
        
        try:
            with open(newest_file, 'r') as f:
                data = json.load(f)
            print(f"   ✅ Loaded box bias data for {len(data)} venues\n")
            return data
        except Exception as e:
            print(f"   ❌ Error loading box bias file: {e}")
            print("   Continuing without box bias filtering\n")
            return None
    
    def normalise_venue_name(self, venue: str) -> str:
        """Normalise venue name to match box bias data"""
        venue_upper = venue.upper().strip().replace(' ', '-').replace('_', '-')
        return VENUE_ALIASES.get(venue_upper, venue_upper)
    
    def get_box_bias(self, venue: str, distance: int, box: int) -> Optional[Dict]:
        """Look up box bias for a specific venue+distance+box combination"""
        if not self.box_bias_data:
            return None
        
        # Normalise venue name
        venue_norm = self.normalise_venue_name(venue)
        
        # Navigate nested structure
        if venue_norm not in self.box_bias_data:
            return None
        
        distance_str = str(distance)
        if distance_str not in self.box_bias_data[venue_norm]:
            return None
        
        box_str = str(box)
        if box_str not in self.box_bias_data[venue_norm][distance_str]:
            return None
        
        return self.box_bias_data[venue_norm][distance_str][box_str]
    
    def categorise_bias(self, bias_value: float) -> str:
        """Convert numeric bias into category"""
        if bias_value >= 30:
            return "STRONG+"
        elif bias_value >= 15:
            return "MODERATE+"
        elif bias_value >= -5:
            return "NEUTRAL"
        elif bias_value >= -15:
            return "MODERATE-"
        else:
            return "WEAK"
    
    def find_venues_for_date(self) -> List[str]:
        """Find all venues with prediction data for the date"""
        predictions_dir = PREDICTIONS_DIR / self.date_formatted
        
        if not predictions_dir.exists():
            return []
        
        venues = set()
        for file in predictions_dir.glob("*_predictions.jsonl"):
            venue = file.stem.replace(f"_{self.date}_predictions", "")
            venues.add(venue)
        
        return sorted(venues)
    
    def load_predictions(self, venue: str) -> List[Dict]:
        """Load prediction data for a venue"""
        pred_file = PREDICTIONS_DIR / self.date_formatted / f"{venue}_{self.date}_predictions.jsonl"
        
        if not pred_file.exists():
            return []
        
        predictions = []
        with open(pred_file, 'r') as f:
            for line in f:
                predictions.append(json.loads(line))
        
        return predictions
    
    def analyze_race(self, race: Dict, venue: str) -> BettingRecommendation:
        """Analyze a race and generate recommendation if it matches criteria"""
        # Find predicted winner
        predicted_winner = None
        for runner in race['runners']:
            if runner.get('predicted_winner'):
                predicted_winner = runner
                break
        
        if not predicted_winner or predicted_winner.get('is_scratched'):
            return None
        
        # Calculate confidence gap
        all_probs = [(r['name'], r.get('winner_prob', 0)) for r in race['runners'] 
                     if not r.get('is_scratched')]
        all_probs.sort(key=lambda x: x[1], reverse=True)
        
        second_best_name = all_probs[1][0] if len(all_probs) >= 2 else "None"
        second_best_prob = all_probs[1][1] if len(all_probs) >= 2 else 0
        
        winner_prob = predicted_winner['winner_prob']
        odds = predicted_winner.get('odds')
        confidence_gap = winner_prob - second_best_prob
        
        # Apply minimum filters
        if odds is None or odds < self.min_odds:
            return None
        if confidence_gap < self.min_confidence_gap:
            return None
        
        # Get race details
        distance = race.get('distance', 0)
        grade = race.get('grade', 'Unknown')
        box = predicted_winner['drawn_box']
        
        # Check if this matches a sweet spot
        matching_sweet_spot = None
        for sweet_spot in self.sweet_spots:
            if sweet_spot.matches(venue, distance, grade):
                matching_sweet_spot = sweet_spot
                break
        
        # Get box bias data
        box_bias_info = self.get_box_bias(venue, distance, box)
        box_bias_value = None
        box_bias_category = None
        box_bias_starts = None
        
        if box_bias_info:
            box_bias_value = box_bias_info['bias']
            box_bias_category = self.categorise_bias(box_bias_value)
            box_bias_starts = box_bias_info['starts']
        else:
            self.box_bias_stats['no_data'] += 1
        
        # Apply box bias filtering based on sweet spot priority
        if matching_sweet_spot:
            if matching_sweet_spot.priority == 1:
                # Priority 1: Keep regardless of box bias (trust proven edge)
                self.box_bias_stats['priority_1_kept'] += 1
            elif matching_sweet_spot.priority == 2:
                # Priority 2: Skip if weak box bias
                if box_bias_category in ['WEAK', 'MODERATE-']:
                    self.box_bias_stats['priority_2_skipped'] += 1
                    return None
        else:
            # Good bets (non-sweet-spots): Only keep if positive bias
            if box_bias_category in ['WEAK', 'MODERATE-', 'NEUTRAL', None]:
                self.box_bias_stats['good_bets_skipped'] += 1
                return None
            
            # Also check if it's still a good bet by venue/distance
            if venue.upper() not in self.profitable_venues:
                return None
            if distance not in self.profitable_distances:
                return None
        
        rec = BettingRecommendation(
            venue=venue,
            date=self.date,
            race_number=race['race_number'],
            race_name=race['race_name'],
            time=race.get('time', 'N/A'),
            runner_name=predicted_winner['name'],
            box=box,
            trainer=predicted_winner['trainer'],
            winner_prob=winner_prob,
            odds=odds,
            confidence_gap=confidence_gap,
            distance=distance,
            grade=grade,
            sweet_spot=matching_sweet_spot,
            second_best_name=second_best_name,
            box_bias_value=box_bias_value,
            box_bias_category=box_bias_category,
            box_bias_starts=box_bias_starts
        )
        
        return rec
    
    def generate_recommendations(self) -> Tuple[List[BettingRecommendation], List[BettingRecommendation]]:
        """Generate betting recommendations for the date"""
        venues = self.find_venues_for_date()
        
        if not venues:
            return [], []
        
        all_recommendations = []
        
        for venue in venues:
            predictions = self.load_predictions(venue)
            
            for race in predictions:
                rec = self.analyze_race(race, venue)
                if rec:
                    all_recommendations.append(rec)
        
        # Separate into sweet spots and good bets
        sweet_spot_bets = [r for r in all_recommendations if r.sweet_spot]
        good_bets = [r for r in all_recommendations if not r.sweet_spot]
        
        # Sort sweet spots by priority, then by odds
        sweet_spot_bets.sort(key=lambda x: (x.sweet_spot.priority, -x.odds))
        
        # Sort good bets by value score
        good_bets.sort(key=lambda x: x.winner_prob * x.odds, reverse=True)
        
        return sweet_spot_bets, good_bets
    
    def print_report(self, sweet_spot_bets: List[BettingRecommendation], 
                     good_bets: List[BettingRecommendation]):
        """Print betting recommendations report"""
        total_bets = len(sweet_spot_bets) + len(good_bets)
        
        if total_bets == 0:
            print("\n" + "="*80)
            print("NO BETTING OPPORTUNITIES TODAY")
            print("="*80)
            print("\nNo races matched the profitable conditions identified in analysis.")
            print("\nCriteria:")
            print("  - Sweet Spot combinations (15 specific venue+distance+grade combos)")
            print("  - OR Profitable venues + distances with 10%+ confidence gap")
            print("  - Minimum odds: $2.50")
            if self.box_bias_data:
                print("  - Box bias filtering applied (Priority 2 and Good Bets only)")
            print("\nRecommendation: Wait for better opportunities tomorrow.")
            return
        
        print("\n" + "="*80)
        print(f"SMART BETTING RECOMMENDATIONS - {self.date}")
        print("="*80)
        
        print(f"\nTOTAL OPPORTUNITIES: {total_bets}")
        print(f"  🎯 Sweet Spot Bets: {len(sweet_spot_bets)} (Historical: 30-100% ROI)")
        print(f"  ✅ Good Bets: {len(good_bets)} (Profitable venue/distance combos)")
        
        # Box bias filtering summary
        if self.box_bias_data:
            print(f"\n📊 BOX BIAS FILTERING APPLIED:")
            print(f"  • Priority 1 sweet spots kept: {self.box_bias_stats['priority_1_kept']} (no filtering)")
            print(f"  • Priority 2 sweet spots skipped: {self.box_bias_stats['priority_2_skipped']} (weak box bias)")
            print(f"  • Good bets skipped: {self.box_bias_stats['good_bets_skipped']} (neutral/negative box bias)")
            print(f"  • No box bias data: {self.box_bias_stats['no_data']} (treated as neutral)")
        
        # Sweet spot bets
        if sweet_spot_bets:
            print("\n" + "="*80)
            print("🎯 SWEET SPOT BETS (Highest Priority - Proven Profitable)")
            print("="*80)
            
            for i, rec in enumerate(sweet_spot_bets, 1):
                print(f"\n{i}. {rec}")
        
        # Good bets
        if good_bets:
            print("\n" + "="*80)
            print("✅ GOOD BETS (Profitable Conditions)")
            print("="*80)
            
            for i, rec in enumerate(good_bets, len(sweet_spot_bets) + 1):
                print(f"\n{i}. {rec}")
        
        # Strategy summary
        print("\n" + "="*80)
        print("📋 TODAY'S BETTING STRATEGY")
        print("="*80)
        
        if sweet_spot_bets:
            print(f"\n✅ RECOMMENDED: Bet the {len(sweet_spot_bets)} Sweet Spot race(s)")
            print("   These match proven profitable combinations from historical analysis")
            print(f"   Expected: 30-100% ROI based on historical performance")
            
            if len(sweet_spot_bets) >= 3:
                print(f"\n   Stake: $1 per race = ${len(sweet_spot_bets[:3])} total (top 3)")
            else:
                print(f"\n   Stake: $1 per race = ${len(sweet_spot_bets)} total")
        
        if good_bets and not sweet_spot_bets:
            print(f"\n⚠️  MODERATE: {len(good_bets)} Good Bet(s) available")
            print("   No sweet spots today, but these match profitable conditions")
            print(f"   Expected: 10-30% ROI")
            print(f"\n   Consider betting top 2-3: ${min(3, len(good_bets))} total")
        elif good_bets and sweet_spot_bets:
            print(f"\n💡 OPTIONAL: {len(good_bets)} additional Good Bet(s)")
            print("   Only if you want more action beyond sweet spots")
        
        print("\n" + "="*80)
        print("⚠️  IMPORTANT REMINDERS")
        print("="*80)
        print("  • Check for late scratchings before betting")
        print("  • These recommendations are based on historical data")
        print("  • Past performance doesn't guarantee future results")
        if self.box_bias_data:
            print("  • Box bias data helps identify track-specific advantages")
        print("  • Start with small stakes ($1-2) until you build confidence")
        print("  • Track results to validate the strategy over time")
        print("="*80)
    
    def save_betting_slip(self, sweet_spot_bets: List[BettingRecommendation], 
                          good_bets: List[BettingRecommendation]):
        """Save betting slip to file"""
        filename = OUTPUT_DIR / f"betting_slip_{self.date}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"SMART BETTING SLIP - {self.date}\n")
            f.write("="*80 + "\n\n")
            
            if sweet_spot_bets:
                f.write("🎯 SWEET SPOT BETS (Priority)\n")
                f.write("-"*80 + "\n\n")
                
                for i, rec in enumerate(sweet_spot_bets, 1):
                    f.write(f"{i}. {rec.venue.upper()} Race {rec.race_number} @ {rec.time}\n")
                    f.write(f"   Box {rec.box}: {rec.runner_name}\n")
                    f.write(f"   Sweet Spot: {rec.sweet_spot.venue} + {rec.sweet_spot.distance}m + Grade {rec.sweet_spot.grade}\n")
                    f.write(f"   Historical: {rec.sweet_spot.historical_bets} bets, {rec.sweet_spot.win_rate:.0%} win, {rec.sweet_spot.roi:+.0f}% ROI\n")
                    f.write(f"   Odds: ${rec.odds:.2f} | Confidence: {rec.winner_prob:.1%}\n")
                    if rec.box_bias_category:
                        f.write(f"   Box Bias: {rec.box_bias_value:+.1f}% ({rec.box_bias_category}) [{rec.box_bias_starts:,} starts]\n")
                    f.write(f"   Stake: $1.00\n\n")
            
            if good_bets:
                f.write("\n✅ GOOD BETS (Optional)\n")
                f.write("-"*80 + "\n\n")
                
                for i, rec in enumerate(good_bets, len(sweet_spot_bets) + 1):
                    f.write(f"{i}. {rec.venue.upper()} Race {rec.race_number} @ {rec.time}\n")
                    f.write(f"   Box {rec.box}: {rec.runner_name}\n")
                    f.write(f"   {rec.distance}m | Grade {rec.grade}\n")
                    f.write(f"   Odds: ${rec.odds:.2f} | Confidence: {rec.winner_prob:.1%}\n")
                    if rec.box_bias_category:
                        f.write(f"   Box Bias: {rec.box_bias_value:+.1f}% ({rec.box_bias_category}) [{rec.box_bias_starts:,} starts]\n")
                    f.write(f"   Stake: $1.00\n\n")
            
            total = len(sweet_spot_bets) + len(good_bets)
            f.write(f"\nTotal Recommended Stake: ${total:.2f}\n")
        
        print(f"\n💾 Betting slip saved to: {filename}")
        self.send_to_telegram(filename, total, "txt")

        # Also save a summary JSON
        summary_filename = OUTPUT_DIR / f"betting_summary_{self.date}.json"
        summary = {
            "date": self.date,
            "total_bets": len(sweet_spot_bets) + len(good_bets),
            "sweet_spot_bets": len(sweet_spot_bets),
            "good_bets": len(good_bets),
            "total_stake": len(sweet_spot_bets) + len(good_bets),
            "box_bias_applied": self.box_bias_data is not None,
            "bets": []
        }

        for rec in sweet_spot_bets + good_bets:
            bet_data = {
                "venue": rec.venue,
                "race": rec.race_number,
                "time": rec.time,
                "runner": rec.runner_name,
                "box": rec.box,
                "odds": rec.odds,
                "type": "sweet_spot" if rec.sweet_spot else "good_bet",
                "sweet_spot_priority": rec.sweet_spot.priority if rec.sweet_spot else None
            }
            if rec.box_bias_category:
                bet_data["box_bias"] = {
                    "value": rec.box_bias_value,
                    "category": rec.box_bias_category,
                    "starts": rec.box_bias_starts
                }
            summary["bets"].append(bet_data)

        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
 
        print(f"💾 Summary JSON saved to: {summary_filename}")
 
    def send_to_telegram(self, filename: Path, total_bets: int, file_type: str = "txt"):
        """Send the betting file to Telegram bot."""
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            print("⚠️  Telegram config missing (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID). Skipping send.")
            return
        
        if not filename.exists():
            print(f"❌ File does not exist: {filename}")
            return
        
        if filename.stat().st_size == 0:
            print(f"❌ File is empty: {filename}")
            return
        
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
            if file_type == "txt":
                caption = f"Smart Betting Slip - {self.date}\nTotal opportunities: {total_bets}"
            else:
                caption = f"Betting Summary JSON - {self.date}\nTotal opportunities: {total_bets}"
            
            with open(filename, 'rb') as f:
                mime_type = 'text/plain' if file_type == "txt" else 'application/json'
                files = {'document': (filename.name, f, mime_type)}
                data = {'chat_id': chat_id, 'caption': caption}
                response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200 and response.json().get('ok'):
                print(f"📱 {file_type.upper()} sent to Telegram successfully!")
            else:
                print(f"❌ Failed to send {file_type}: {response.text}")
        except Exception as e:
            print(f"❌ Telegram send error: {e}")


def main():
    import sys
    
    script_dir = Path(__file__).parent
    env_path = script_dir.parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    
    # Check if dates provided as command line arguments
    if len(sys.argv) > 1:
        dates_to_analyze = sys.argv[1:]
    else:
        # Default to today
        dates_to_analyze = [datetime.now().strftime("%Y-%m-%d")]
    
    print("\n" + "="*80)
    print("SMART BETTING RECOMMENDER - SWEET SPOT TARGETING + BOX BIAS")
    print("="*80)
    
    # Analyze each date
    for date in dates_to_analyze:
        print(f"\n{'='*80}")
        print(f"ANALYZING DATE: {date}")
        print(f"{'='*80}")
        print("\nUsing proven profitable conditions from historical analysis:")
        print("  • 15 Sweet Spot combinations (61-202% historical ROI)")
        print("  • Profitable venues and distances")
        print("  • Minimum 10% confidence gap (standouts only)")
        print("
