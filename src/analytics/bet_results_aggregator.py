#!/usr/bin/env python3
"""
bet_results_aggregator.py

Aggregate betting summary JSON files with race results (JSONL) and append results to a
master CSV for ongoing analysis. Then generate a summary ODS file for LibreOffice.

Drop-in script for: /home/matthew/vault/grsa_prod/src/analytics/

Design notes:
- Processes all betting_summary_*.json files under BETS_DIR
- Looks for results under RESULTS_DIR/YYYYMMDD/*.jsonl where YYYYMMDD comes from bet['date']
- Uses a $1 stake per bet (STAKE)
- Appends to OUTPUT_CSV, skipping duplicates based on (date, venue, race, runner)
- Marks unmatched / scratched bets and leaves them in the CSV for manual intervention
- Generates summary.ods with daily breakdown and overall totals

Dependencies: pandas, odfpy (install: pip install pandas odfpy)
"""

import os
import glob
import json
import re
from datetime import datetime
from typing import List, Dict, Optional

try:
    import pandas as pd
except Exception as e:
    raise ImportError("pandas is required. Install with: pip install pandas")

try:
    from odf.opendocument import OpenDocumentSpreadsheet
    from odf.style import Style, TableColumnProperties
    from odf.table import Table, TableColumn, TableRow, TableCell
    from odf.text import P
except Exception as e:
    raise ImportError("odfpy is required for ODS generation. Install with: pip install odfpy")

# ---------- CONFIGURATION ----------
BETS_DIR = "/media/matt_sent/vault/dishlicker_data/data/bets/"
RESULTS_DIR = "/media/matt_sent/vault/dishlicker_data/data/results/"
OUTPUT_DIR = "/media/matt_sent/vault/dishlicker_data/data/analysis/"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "bet_results.csv")
SUMMARY_ODS = os.path.join(OUTPUT_DIR, "bet_results_summary.ods")
STAKE = 1.0  # standard $1 stake per bet

# ---------- Utilities ----------

def _normalize_name(name: Optional[str]) -> Optional[str]:
    """Normalize names for comparison: lower, strip, remove punctuation, normalize whitespace."""
    if name is None:
        return None
    s = name.strip().lower()
    # remove punctuation except spaces (apostrophes and dots often cause mismatches)
    s = re.sub(r"[\'\"\.,:;!\?()\[\]\/\\-]", "", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _date_to_folder(date_str: str) -> str:
    """Convert 'YYYY-MM-DD' to 'YYYYMMDD' for results folder naming.
    If parsing fails, fallback to removing non-digits."""
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime("%Y%m%d")
    except Exception:
        return re.sub(r"\D", "", date_str)


# ---------- Load betting files ----------

def load_betting_files(bets_dir: str) -> List[Dict]:
    """Find all betting_summary_*.json files and return a flat list of bet dicts.

    Each returned dict contains normalized venue/runner and the source file name.
    """
    pattern = os.path.join(bets_dir, "betting_summary_*.json")
    files = sorted(glob.glob(pattern))
    all_bets = []

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load betting file {fp}: {e}")
            continue

        summary_date = summary.get("date")
        bets = summary.get("bets", [])
        for b in bets:
            normalized = {
                "date": summary_date,
                "venue": _normalize_name(b.get("venue")),
                "race": str(b.get("race")) if b.get("race") is not None else None,
                "time": b.get("time"),
                "runner": _normalize_name(b.get("runner")),
                "box": b.get("box"),
                "odds": float(b.get("odds")) if b.get("odds") is not None else None,
                "type": b.get("type"),
                "sweet_spot_priority": b.get("sweet_spot_priority"),
                "source_file": os.path.basename(fp),
                # placeholders for result enrichment
                "final_position": None,
                "odds_final": None,
                "is_scratched": None,
                "finishing_time": None,
                "trainer": None,
                "margin": None,
            }
            all_bets.append(normalized)

    print(f"Loaded {len(all_bets)} bets from {len(files)} betting summary files.")
    return all_bets


# ---------- Load race results for a date ----------

def load_results_for_date(date_str: str, results_dir: str) -> List[Dict]:
    """Load all JSONL result files for a given date folder (YYYYMMDD).

    Returns a list of race result dicts (one per race line in the JSONL files).
    """
    folder = _date_to_folder(date_str)
    search_dir = os.path.join(results_dir, folder)
    if not os.path.isdir(search_dir):
        # no results folder for this date
        return []

    jsonl_files = sorted(glob.glob(os.path.join(search_dir, "*_results.jsonl")))
    results = []

    for jf in jsonl_files:
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        results.append(r)
                    except json.JSONDecodeError:
                        print(f"[WARN] Failed to decode JSON line in {jf}: {line[:80]}")
        except Exception as e:
            print(f"[WARN] Failed to read results file {jf}: {e}")

    return results


# ---------- Matching logic ----------

def match_bet_to_result(bet: Dict, results: List[Dict]) -> Dict:
    """Attempt to match one bet to its race/result and enrich the bet dict with result fields.

    If no match is found, returns the bet with status 'unmatched' (status assigned in calculate_profit)
    """
    if not results:
        bet["_match_note"] = "no_results_for_date"
        return bet

    # find candidate race
    matched_race = None
    for race in results:
        # normalise race['venue'] and race_number
        rvenue = _normalize_name(race.get("venue"))
        rrn = str(race.get("race_number")) if race.get("race_number") is not None else None
        if rvenue == bet.get("venue") and rrn == bet.get("race"):
            matched_race = race
            break

    if not matched_race:
        bet["_match_note"] = "no_matching_race"
        return bet

    # find runner within race
    runners = matched_race.get("runners", [])
    matched_runner = None
    for r in runners:
        rname = _normalize_name(r.get("name"))
        if rname == bet.get("runner"):
            matched_runner = r
            break

    if not matched_runner:
        # attempt fuzzy fallback: match by drawn_box/run_box if available and bet.box present
        if bet.get("box") is not None:
            for r in runners:
                if r.get("drawn_box") == bet.get("box") or r.get("run_box") == bet.get("box"):
                    matched_runner = r
                    break

    if not matched_runner:
        bet["_match_note"] = "no_matching_runner"
        # include basic race-level info for debugging
        bet["race_name"] = matched_race.get("race_name")
        return bet

    # enrich
    bet["final_position"] = matched_runner.get("final_position")
    bet["odds_final"] = matched_runner.get("odds_final")
    bet["is_scratched"] = matched_runner.get("is_scratched")
    bet["finishing_time"] = matched_runner.get("finishing_time")
    bet["trainer"] = matched_runner.get("trainer")
    bet["margin"] = matched_runner.get("margin")
    bet["race_name"] = matched_race.get("race_name")
    bet["results_source_file"] = matched_race.get("_source_file") if matched_race.get("_source_file") else None

    return bet


# ---------- Profit / status calculation ----------

def calculate_profit(bet: Dict) -> Dict:
    """Compute status and profit for a single enriched bet dict."""
    # default
    bet["status"] = "unmatched"
    bet["profit"] = None
    bet["roi"] = None

    # handle scratched
    if bet.get("is_scratched") is True:
        bet["status"] = "scratched"
        bet["profit"] = 0.0
        bet["roi"] = 0.0
        return bet

    # if we have position info
    pos = bet.get("final_position")
    if pos is None:
        bet["status"] = "unmatched"
        return bet

    try:
        pos_int = int(pos)
    except Exception:
        bet["status"] = "unmatched"
        return bet

    if pos_int == 1:
        bet["status"] = "won"
        # Use the *selected* odds for profit calculation per your spec (stake = $1)
        if bet.get("odds") is not None:
            bet["profit"] = round((float(bet["odds"]) - 1.0) * STAKE, 2)
        else:
            # fallback to final odds if selected odds missing
            if bet.get("odds_final") is not None:
                bet["profit"] = round((float(bet["odds_final"]) - 1.0) * STAKE, 2)
            else:
                bet["profit"] = None
    else:
        bet["status"] = "lost"
        bet["profit"] = -STAKE

    # roi
    if bet["profit"] is not None:
        bet["roi"] = round(bet["profit"] / STAKE, 4)

    return bet


# ---------- CSV update / append ----------

def update_csv(output_path: str, bet_results: List[Dict]) -> None:
    """Append bet_results to CSV at output_path, avoiding duplicates using unique key.

    Unique key: (date, venue, race, runner)
    """
    if not bet_results:
        print("No bet results to write.")
        return

    df_new = pd.DataFrame(bet_results)

    # ensure columns ordering and presence
    cols = [
        "date", "venue", "race", "time", "runner", "box", "odds",
        "final_position", "odds_final", "is_scratched", "finishing_time",
        "trainer", "margin", "status", "profit", "roi", "type",
        "sweet_spot_priority", "source_file", "race_name", "_match_note"
    ]
    # keep only existing columns
    cols = [c for c in cols if c in df_new.columns]
    df_new = df_new[cols]

    # Create unique key
    df_new["_unique_key"] = df_new.apply(
        lambda r: f"{r.get('date') or ''}|||{r.get('venue') or ''}|||{r.get('race') or ''}|||{r.get('runner') or ''}", 
        axis=1
    )

    if os.path.exists(output_path):
        try:
            df_existing = pd.read_csv(output_path, dtype=str)
            # ensure _unique_key exists in existing (recreate if missing)
            if "_unique_key" not in df_existing.columns:
                df_existing["_unique_key"] = df_existing.apply(
                    lambda r: f"{r.get('date') or ''}|||{_normalize_name(r.get('venue') or '') or ''}|||{r.get('race') or ''}|||{_normalize_name(r.get('runner') or '') or ''}", 
                    axis=1
                )

            # normalize existing keys to same format as new (venue/runner normalized)
            existing_keys = set(df_existing["_unique_key"].tolist())
            df_new_to_append = df_new[~df_new["_unique_key"].isin(existing_keys)].copy()

            if df_new_to_append.empty:
                print("No new records to append — all bets already present in CSV.")
                return

            # append using pandas
            df_new_to_append.drop(columns=["_unique_key"], inplace=True, errors="ignore")
            df_new_to_append.to_csv(output_path, mode="a", header=False, index=False)
            print(f"Appended {len(df_new_to_append)} new records to {output_path}")

        except Exception as e:
            print(f"[ERROR] Failed to append to existing CSV {output_path}: {e}")
    else:
        # write new file with header
        try:
            df_new.drop(columns=["_unique_key"], inplace=True, errors="ignore")
            df_new.to_csv(output_path, index=False)
            print(f"Wrote {len(df_new)} records to new CSV {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write CSV {output_path}: {e}")


# ---------- ODS Summary Generation ----------

def update_summary_ods(csv_path: str, output_ods: str) -> None:
    """Generate LibreOffice-compatible ODS summary from bet_results CSV.
    
    Creates a spreadsheet with:
    - Daily breakdown (Date, Total Bets, Wins, Scratched, Unmatched, Profit, ROI)
    - Overall summary row
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV file not found: {csv_path}. Skipping ODS generation.")
        return

    print(f"\nGenerating LibreOffice summary ODS...")

    try:
        df = pd.read_csv(csv_path)
        
        # Convert profit to numeric, handling any issues
        df['profit'] = pd.to_numeric(df['profit'], errors='coerce').fillna(0)
        
        # Group by date
        daily_summary = df.groupby('date').agg({
            'runner': 'count',  # total bets
            'status': [
                lambda s: (s == 'won').sum(),  # wins
                lambda s: (s == 'scratched').sum(),  # scratched
                lambda s: (s == 'unmatched').sum()  # unmatched
            ],
            'profit': 'sum'
        }).reset_index()
        
        # Flatten column names
        daily_summary.columns = ['date', 'total_bets', 'wins', 'scratched', 'unmatched', 'profit']
        
        # Calculate ROI per day
        daily_summary['roi'] = (daily_summary['profit'] / (daily_summary['total_bets'] * STAKE) * 100).round(2)
        daily_summary['profit'] = daily_summary['profit'].round(2)
        
        # Sort by date
        daily_summary = daily_summary.sort_values('date')
        
        # Calculate overall totals
        total_bets = int(daily_summary['total_bets'].sum())
        total_wins = int(daily_summary['wins'].sum())
        total_scratched = int(daily_summary['scratched'].sum())
        total_unmatched = int(daily_summary['unmatched'].sum())
        total_profit = round(daily_summary['profit'].sum(), 2)
        overall_roi = round((total_profit / (total_bets * STAKE) * 100), 2) if total_bets > 0 else 0.0
        
        # Create ODS document
        doc = OpenDocumentSpreadsheet()
        
        # Define column width style
        width_style = Style(name="colwidth", family="table-column")
        width_style.addElement(TableColumnProperties(columnwidth="3.0cm"))
        doc.automaticstyles.addElement(width_style)
        
        # Create table
        table = Table(name="Betting Summary")
        
        # Add columns
        for _ in range(7):
            table.addElement(TableColumn(stylename=width_style))
        
        # Header row
        headers = ["Date", "Total Bets", "Wins", "Scratched", "Unmatched", "Profit (AUD)", "ROI (%)"]
        tr = TableRow()
        for h in headers:
            cell = TableCell()
            cell.addElement(P(text=h))
            tr.addElement(cell)
        table.addElement(tr)
        
        # Daily data rows
        for _, row in daily_summary.iterrows():
            tr = TableRow()
            values = [
                str(row['date']),
                str(int(row['total_bets'])),
                str(int(row['wins'])),
                str(int(row['scratched'])),
                str(int(row['unmatched'])),
                f"{row['profit']:.2f}",
                f"{row['roi']:.2f}"
            ]
            for v in values:
                cell = TableCell()
                cell.addElement(P(text=v))
                tr.addElement(cell)
            table.addElement(tr)
        
        # Overall summary row
        tr = TableRow()
        overall_values = [
            "Overall",
            str(total_bets),
            str(total_wins),
            str(total_scratched),
            str(total_unmatched),
            f"{total_profit:.2f}",
            f"{overall_roi:.2f}"
        ]
        for v in overall_values:
            cell = TableCell()
            cell.addElement(P(text=v))
            tr.addElement(cell)
        table.addElement(tr)
        
        doc.spreadsheet.addElement(table)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_ods), exist_ok=True)
        
        # Save document
        doc.save(output_ods)
        print(f"✓ Wrote LibreOffice summary to {output_ods}")
        print(f"  - {len(daily_summary)} days of betting data")
        print(f"  - Overall: {total_bets} bets, {total_wins} wins, Profit: ${total_profit:.2f}, ROI: {overall_roi:.2f}%")
        
    except Exception as e:
        print(f"[ERROR] Failed to generate ODS summary: {e}")
        import traceback
        traceback.print_exc()


# ---------- Summary printing ----------

def summarize_results(bet_results: List[Dict]) -> None:
    """Print terminal summary of betting results."""
    df = pd.DataFrame(bet_results)
    total = len(df)
    wins = len(df[df["status"] == "won"]) if "status" in df.columns else 0
    scratched = len(df[df["status"] == "scratched"]) if "status" in df.columns else 0
    unmatched = len(df[df["status"] == "unmatched"]) if "status" in df.columns else 0
    profit_sum = df["profit"].astype(float).sum() if "profit" in df.columns and not df["profit"].isnull().all() else 0.0
    roi = (profit_sum / (total * STAKE)) * 100 if total > 0 else 0.0

    print("\nSummary:")
    print("--------")
    print(f"Total bets processed: {total}")
    print(f"Wins: {wins}")
    print(f"Scratched: {scratched}")
    print(f"Unmatched: {unmatched}")
    print(f"Net profit (AUD): {profit_sum:.2f}")
    print(f"Overall ROI (%): {roi:.2f}%")


# ---------- Main orchestration ----------

def main():
    print("Starting bet results aggregation...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and process bets
    bets = load_betting_files(BETS_DIR)
    aggregated = []

    for bet in bets:
        results = load_results_for_date(bet.get("date"), RESULTS_DIR)
        enriched = match_bet_to_result(bet, results)
        finalized = calculate_profit(enriched)
        aggregated.append(finalized)

    # Update CSV
    update_csv(OUTPUT_CSV, aggregated)
    
    # Print summary
    summarize_results(aggregated)
    
    # Generate ODS summary
    update_summary_ods(OUTPUT_CSV, SUMMARY_ODS)
    
    print("\n✓ Aggregation complete!")


if __name__ == "__main__":
    main()
