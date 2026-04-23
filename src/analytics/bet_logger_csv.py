import json
import os
import pandas as pd
from datetime import datetime

# Paths
BASE_INPUT_DIR = "/media/matt_sent/vault/dishlicker_data/data/bets/"
DAILY_OUTPUT_DIR = "/media/matt_sent/vault/dishlicker_data/data/logs/bets/daily/"
MONTHLY_OUTPUT_DIR = "/media/matt_sent/vault/dishlicker_data/data/logs/bets/monthly/"

# Ensure directories exist
os.makedirs(DAILY_OUTPUT_DIR, exist_ok=True)
os.makedirs(MONTHLY_OUTPUT_DIR, exist_ok=True)

def generate_daily_template(date, stake_per_bet=None):
    """Generate editable daily CSV from JSON."""
    input_filename = f"betting_summary_{date}.json"
    input_path = os.path.join(BASE_INPUT_DIR, input_filename)
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if data['date'] != date:
        print(f"Warning: File date {data['date']} does not match input {date}")
    
    bets = data['bets']
    print(f"Loaded {len(bets)} recommended bets for {date}.")
    
    # Prepare rows for DataFrame
    rows = []
    for bet in bets:
        row = {
            'date': date,
            'venue': bet['venue'],
            'race': bet['race'],
            'time': bet['time'],
            'runner': bet['runner'],
            'box': bet['box'],
            'recommended_odds': bet['odds'],
            'actual_odds': bet['odds'],  # Default to recommended
            'stake': stake_per_bet if stake_per_bet else '',  # Blank if not set
            'placed': 'No',  # Default
            'type': bet['type'],
            'sweet_spot_priority': bet.get('sweet_spot_priority', ''),
            'outcome': '',  # Blank for post-race
            'return_amount': '',  # Optional manual override
            'profit': 0.0  # Will be computed on import
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    daily_filename = f"{date}_bets.csv"
    daily_path = os.path.join(DAILY_OUTPUT_DIR, daily_filename)
    df.to_csv(daily_path, index=False)
    print(f"Daily template generated: {daily_path}")
    print("Edit in Excel: Set 'stake', 'actual_odds', 'placed' (Yes/No), 'outcome' (won/lost), and optionally 'return_amount'.")
    print("Then run in 'import' mode to finalize and append to monthly log.")

def import_daily_to_monthly(date):
    """Import edited daily CSV, compute profits, append to monthly CSV."""
    daily_filename = f"{date}_bets.csv"
    daily_path = os.path.join(DAILY_OUTPUT_DIR, daily_filename)
    
    if not os.path.exists(daily_path):
        print(f"Daily CSV not found: {daily_path}. Generate template first.")
        return
    
    df_daily = pd.read_csv(daily_path)
    
    # Compute profit for each row
    for idx, row in df_daily.iterrows():
        if row['placed'] != 'Yes':
            df_daily.at[idx, 'profit'] = 0.0
            df_daily.at[idx, 'return_amount'] = 0.0
            continue
        
        stake = row['stake']
        if pd.isna(stake) or stake <= 0:
            print(f"Warning: Invalid stake for {row['runner']}. Skipping.")
            continue
        
        outcome = row['outcome'].lower()
        if outcome == 'won':
            if pd.notna(row['return_amount']) and row['return_amount'] > 0:
                return_amt = row['return_amount']
            else:
                return_amt = stake * row['actual_odds']
                df_daily.at[idx, 'return_amount'] = return_amt
            df_daily.at[idx, 'profit'] = return_amt - stake
        elif outcome == 'lost':
            df_daily.at[idx, 'return_amount'] = 0.0
            df_daily.at[idx, 'profit'] = -stake
        else:
            print(f"Warning: Invalid outcome '{outcome}' for {row['runner']}. Profit set to 0.")
            df_daily.at[idx, 'profit'] = 0.0
    
    # Save updated daily (with computed fields)
    df_daily.to_csv(daily_path, index=False)
    
    # Append to monthly
    month = date[:4] + date[5:7]  # YYYYMM
    monthly_filename = f"{month}_bets.csv"
    monthly_path = os.path.join(MONTHLY_OUTPUT_DIR, monthly_filename)
    
    if os.path.exists(monthly_path):
        df_monthly = pd.read_csv(monthly_path)
        # Avoid duplicates: unique key = date + venue + race + runner
        df_monthly['unique_key'] = df_monthly['date'].astype(str) + '_' + df_monthly['venue'] + '_' + df_monthly['race'].astype(str) + '_' + df_monthly['runner']
        df_daily['unique_key'] = df_daily['date'].astype(str) + '_' + df_daily['venue'] + '_' + df_daily['race'].astype(str) + '_' + df_daily['runner']
        new_rows = df_daily[~df_daily['unique_key'].isin(df_monthly['unique_key'])]
        if not new_rows.empty:
            df_combined = pd.concat([df_monthly, new_rows], ignore_index=True)
            df_combined.to_csv(monthly_path, index=False)
            print(f"Appended {len(new_rows)} new rows to monthly log: {monthly_path}")
        else:
            print("No new rows to append (all already exist).")
    else:
        df_daily.to_csv(monthly_path, index=False)
        print(f"Created monthly log: {monthly_path}")
    
    # Clean up temp unique_key column if saved
    if os.path.exists(monthly_path):
        df_temp = pd.read_csv(monthly_path)
        if 'unique_key' in df_temp.columns:
            df_temp.drop('unique_key', axis=1, inplace=True)
            df_temp.to_csv(monthly_path, index=False)

def generate_report(monthly_path):
    """Simple console report for metrics (load monthly CSV)."""
    if not os.path.exists(monthly_path):
        print("Monthly CSV not found.")
        return
    
    df = pd.read_csv(monthly_path)
    placed_df = df[df['placed'] == 'Yes'].copy()
    
    if placed_df.empty:
        print("No placed bets in log.")
        return
    
    total_placed = len(placed_df)
    total_wins = len(placed_df[placed_df['outcome'] == 'won'])
    total_stake = placed_df['stake'].sum()
    total_profit = placed_df['profit'].sum()
    strike_rate = (total_wins / total_placed) * 100 if total_placed > 0 else 0
    roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0
    
    print(f"\nMonthly Report ({os.path.basename(monthly_path)}):")
    print(f"  Total Placed Bets: {total_placed}")
    print(f"  Wins: {total_wins}")
    print(f"  Strike Rate: {strike_rate:.1f}%")
    print(f"  Total Stake: ${total_stake:.2f}")
    print(f"  Total Profit: ${total_profit:.2f}")
    print(f"  ROI: {roi:.1f}%")
    
    # Example breakdowns
    by_type = placed_df.groupby('type').agg({
        'profit': 'sum',
        'stake': 'sum',
        'outcome': lambda x: (x == 'won').sum()
    }).round(2)
    print("\nBreakdown by Type:")
    print(by_type)
    
    by_venue = placed_df.groupby('venue')['profit'].sum().round(2)
    print("\nProfit by Venue:")
    print(by_venue)

def main():
    print("Bet Logger CSV Edition")
    mode = input("Mode (generate/import/report): ").strip().lower()
    
    date = input("Enter date (YYYY-MM-DD): ").strip()
    if not date:
        print("Date required.")
        return
    
    if mode == 'generate':
        stake = input("Enter default stake per bet (optional, e.g., 2.0; leave blank to edit in CSV): ").strip()
        stake_per_bet = float(stake) if stake else None
        generate_daily_template(date, stake_per_bet)
    
    elif mode == 'import':
        import_daily_to_monthly(date)
    
    elif mode == 'report':
        month = date[:4] + date[5:7]
        monthly_path = os.path.join(MONTHLY_OUTPUT_DIR, f"{month}_bets.csv")
        generate_report(monthly_path)
    
    else:
        print("Invalid mode. Use 'generate', 'import', or 'report'.")

if __name__ == "__main__":
    main()