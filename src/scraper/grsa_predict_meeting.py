### Forked version from /media/matt_sent/vault/dishlicker_data/models/predict_meeting.py

import pandas as pd
import json
import os
import logging
import traceback
import joblib
import argparse
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, "/home/matt_sent/projects/grsa/src/utils")

from src.utils.config import JSONL_DIR, PREDICTIONS_DIR, TRACK_STATISTICS_DIR, INDEX_DIR

# ===============================
# ARGPARSE SETUP
# ===============================
parser = argparse.ArgumentParser(description="Predict greyhound meeting outcomes")
parser.add_argument(
    "--venue",
    type=str,
    default=None,
    help="Venue name (e.g., angle-park). If not provided, processes all venues for the date.",
)
parser.add_argument(
    "--date",
    type=str,
    required=True,
    help="Meeting date in YYYY-MM-DD or YYYYMMDD format (required)",
)
parser.add_argument(
    "--csv",
    action="store_true",
    help="Also export predictions as CSV",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Force reprocessing of all venues regardless of index status",
)
args = parser.parse_args()

venue_arg = args.venue
date_str = args.date
export_csv = args.csv
force = args.force

# Parse date to handle both formats
if len(date_str) == 8:
    dt = datetime.strptime(date_str, "%Y%m%d")
elif len(date_str) == 10:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
else:
    raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD or YYYYMMDD.")

file_date = dt.strftime("%Y-%m-%d")
folder_date = dt.strftime("%Y%m%d")

# ===============================
# PATHS
# ===============================
jsonl_dir = JSONL_DIR
predictions_dir = PREDICTIONS_DIR
track_stats_dir = TRACK_STATISTICS_DIR
index_dir = INDEX_DIR
model_path = Path("/media/matt_sent/vault/dishlicker_data/models/random_forest_baseline.pkl")

# Ensure output directory exists
(predictions_dir / folder_date).mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load model once
logger.info(f"Loading model from: {model_path}")
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")
clf = joblib.load(model_path)
expected_features = clf.feature_names_in_
logger.info(f"Model expects {len(expected_features)} features")

model_version = "random_forest_baseline_1.0"

def needs_processing(venue, pred_file, force, index_jsonl):
    if force:
        return True
    if not index_jsonl.exists():
        return True
    try:
        # Normalize venue name for comparison (handle both formats)
        venue_normalized = venue.replace("-", "_")
        
        with open(index_jsonl, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line.strip())
                entry_venue = entry.get('venue_slug', '').replace("-", "_")
                
                if entry_venue == venue_normalized:
                    created = entry.get('prediction_created', False)
                    file_exists = pred_file.exists()
                    if created and file_exists:
                        logger.info(f"Found existing prediction for {venue} in index")
                        return False
                    return True
        # Venue not found in index, needs processing
        logger.info(f"Venue {venue} not found in index, will process")
        return True
    except Exception as e:
        logger.warning(f"Could not read index JSONL for {venue}: {e}. Treating as needs processing.")
        return True

def update_index_jsonl(venue, meta_updates, index_jsonl):
    if not index_jsonl.exists():
        logger.warning(f"Index JSONL does not exist: {index_jsonl}. Cannot update.")
        return False
    
    # Normalize venue name for comparison
    venue_normalized = venue.replace("-", "_")
    
    tmp_path = index_jsonl.with_suffix('.jsonl.tmp')
    try:
        entries = []
        updated = False
        
        with open(index_jsonl, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line.strip())
                entry_venue_normalized = entry.get('venue_slug', '').replace("-", "_")
                
                if entry_venue_normalized == venue_normalized:
                    entry.update(meta_updates)
                    updated = True
                
                entries.append(entry)
        
        if not updated:
            logger.warning(f"Venue {venue} not found in index JSONL. Cannot update.")
            return False
        
        # Write atomically
        with open(tmp_path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        os.rename(tmp_path, index_jsonl)
        return True
        
    except Exception as e:
        logger.error(f"Failed to update index JSONL for {venue}: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False

# ===============================
# DISCOVER VENUES TO PROCESS
# ===============================
date_folder = jsonl_dir / folder_date
if not date_folder.exists():
    raise FileNotFoundError(f"Date folder not found: {date_folder}")

if venue_arg:
    # Single venue mode
    venues_to_process = [venue_arg]
    logger.info(f"Processing single venue: {venue_arg}")
else:
    # Multi-venue mode - discover all prerace files
    index_jsonl = index_dir / f"index_{file_date}_csv.jsonl"
    prerace_files = [f for f in os.listdir(date_folder) if f.endswith(f"_{file_date}_prerace.jsonl")]
    if not prerace_files:
        raise FileNotFoundError(f"No prerace files found in {date_folder}")
    
    venues_to_process = [f.replace(f"_{file_date}_prerace.jsonl", "") for f in prerace_files]
    logger.info(f"Found {len(venues_to_process)} venues to process: {venues_to_process}")

# ===============================
# PROCESS EACH VENUE
# ===============================
successful_venues = 0
failed_venues = 0
skipped_venues = 0

for idx, venue in enumerate(venues_to_process, 1):
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing venue {idx}/{len(venues_to_process)}: {venue}")
    logger.info(f"{'=' * 60}")

    output_jsonl = predictions_dir / folder_date / f"{venue}_{file_date}_predictions.jsonl"
    index_jsonl = index_dir / f"index_{file_date}_csv.jsonl"

    if not needs_processing(venue, output_jsonl, force, index_jsonl):
        logger.info(f"Skipped {venue}: prediction already exists and index confirms.")
        skipped_venues += 1
        continue

    logger.info(f"Processing {venue}...")

    try:
        prerace_file = date_folder / f"{venue}_{file_date}_prerace.jsonl"
        results_file = date_folder / f"{venue}_{file_date}_results.jsonl"
        old_pred_file = predictions_dir / folder_date / f"{venue}_{file_date}_predictions.csv"

        if not prerace_file.exists():
            raise FileNotFoundError(f"Pre-race file not found: {prerace_file}")

        # Load pre-race data
        logger.info(f"Loading pre-race data: {prerace_file}")
        with open(prerace_file, "r") as f:
            prerace_races = [json.loads(line) for line in f]

        # Load results if exists
        final_results = {}
        if results_file.exists():
            logger.info(f"Loading results data: {results_file}")
            with open(results_file, "r") as f:
                results_races = [json.loads(line) for line in f]
            # Create mapping: race_number -> list of runners with final_position, odds_final
            for race in results_races:
                race_num = race["race_number"]
                final_results[race_num] = race.get("runners", [])
        else:
            logger.info("No results file found, skipping final results.")

        # Load old predictions if exists
        old_predictions = {}
        if old_pred_file.exists():
            logger.info(f"Loading old predictions: {old_pred_file}")
            old_df = pd.read_csv(old_pred_file)
            for _, row in old_df.iterrows():
                key = (row["race_number"], row["drawn_box"])
                old_predictions[key] = {
                    "old_winner_prob": row["winner_prob"],
                    "old_predicted_winner": row["predicted_winner"],
                }
        else:
            logger.info("No old predictions CSV found, generating new.")

        # ===============================
        # LOAD TRACK STATISTICS
        # ===============================
        venue_map = {
            "the-gardens": "Ladbrokes_Gardens",
            "ladbrokes-q1-lakeside": "Q1_Lakeside",
            "ladbrokes-q2-parklands": "Q2_Parklands",
            "ladbrokes-q-straight": "Q_Straight",
            "angle-park": "Angle_Park",
            "hobart": "Hobart",
            "meadows": "The_Gardens",  # Hardcoded fix for meadows
            "meadows": "Meadows_Mep" # Hardcoded fix for meadows anomaly where stats are filed under The_Gardens
            # Add more as needed
        }

        track_name_for_file = venue_map.get(venue)
        if not track_name_for_file:
            track_name_for_file = venue.replace("-", "_").title()

        track_stats_files = [
            f for f in os.listdir(track_stats_dir)
            if f.startswith(f"track_statistics_{track_name_for_file}")
        ]

        if not track_stats_files:
            raise FileNotFoundError(f"No track statistics file found for venue: {venue}")

        track_stats_file = track_stats_dir / track_stats_files[0]
        logger.info(f"Using track stats file: {track_stats_file}")

        track_stats_df = pd.read_json(track_stats_file, lines=True)

        # ===============================
        # FLATTEN PRE-RACE FOR PROCESSING
        # ===============================
        races = []
        for race_data in prerace_races:
            venue_name = race_data["venue"]
            race_date_str = race_data["date"]
            race_number = race_data["race_number"]
            race_name = race_data["race_name"]
            grade = race_data.get("grade")
            distance = race_data.get("distance")

            for runner in race_data["runners"]:
                runner_record = {
                    "venue": venue_name,
                    "race_date": race_date_str,
                    "race_number": race_number,
                    "race_name": race_name,
                    "grade": grade,
                    "distance_m": distance,
                    **runner,
                }
                races.append(runner_record)

        race_df = pd.DataFrame(races)

        # ===============================
        # FEATURE ENGINEERING
        # ===============================
        # Handle minor track/distance adjustments
        race_df.loc[
            (race_df["venue"].str.lower() == "sale") & (race_df["distance_m"] == 510),
            "distance_m",
        ] = 520

        # Normalize grades
        grade_map = {"5": "5th Grade", "Maiden": "Maiden", "FFA": "FFA, INV"}
        race_df["grade_normalized"] = (
            race_df["grade"].map(grade_map).fillna(race_df["grade"]).fillna("Non Graded")
        )

        # Create date features
        race_df["race_date_dt"] = pd.to_datetime(race_df["race_date"])
        race_df["race_day"] = race_df["race_date_dt"].dt.day
        race_df["race_month"] = race_df["race_date_dt"].dt.month
        race_df["race_year"] = race_df["race_date_dt"].dt.year

        # Create run_box_for_merge (same as drawn_box for new races)
        race_df["run_box_for_merge"] = race_df["drawn_box"]

        # Prepare track stats for merge
        track_stats_df["track_name"] = track_stats_df["track_info"].apply(
            lambda x: x["track_name"]
        )

        track_stats_df = track_stats_df.explode("averages_by_distance_grade")

        def extract_track_stats(x):
            if pd.notnull(x) and isinstance(x, dict):
                return pd.Series(
                    {
                        "distance": int(x["distance"].replace("m", "")),
                        "grade_normalized": x["grade"],
                        "avg_finishing_time": x.get("avg_finishing_time"),
                        "first_sectional_time": x.get("first_sectional_time"),
                        "box_plc": x.get("box_plc"),
                        "box_plc_percent": x.get("box_plc_percent"),
                        "box_position_changed": x.get("box_position_changed"),
                        "box_number": x.get("box_number"),
                        "box_starts": x.get("box_starts"),
                        "box_win_percent": x.get("box_win_percent"),
                        "box_wins": x.get("box_wins"),
                    }
                )
            return pd.Series(
                {
                    "distance": None,
                    "grade_normalized": None,
                    "avg_finishing_time": None,
                    "first_sectional_time": None,
                    "box_plc": None,
                    "box_plc_percent": None,
                    "box_position_changed": None,
                    "box_number": None,
                    "box_starts": None,
                    "box_win_percent": None,
                    "box_wins": None,
                }
            )

        track_stats_expanded = track_stats_df["averages_by_distance_grade"].apply(
            extract_track_stats
        )
        track_stats_df = pd.concat([track_stats_df, track_stats_expanded], axis=1)

        # Merge track stats
        merged_df = pd.merge(
            race_df,
            track_stats_df,
            how="left",
            left_on=["venue", "distance_m", "grade_normalized"],
            right_on=["track_name", "distance", "grade_normalized"],
        )

        # ===============================
        # PREDICT
        # ===============================
        available_features = set(merged_df.columns)
        missing_features = set(expected_features) - available_features

        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            for feature in missing_features:
                merged_df[feature] = 0

        X = merged_df[expected_features]
        merged_df["winner_prob"] = clf.predict_proba(X)[:, 1]
        merged_df["predicted_winner"] = merged_df["winner_prob"] == merged_df.groupby(
            "race_number"
        )["winner_prob"].transform("max")

        # Add old predictions if available
        if old_predictions:
            def add_old_pred(row):
                key = (row["race_number"], row["drawn_box"])
                if key in old_predictions:
                    row["old_winner_prob"] = old_predictions[key]["old_winner_prob"]
                    row["old_predicted_winner"] = old_predictions[key]["old_predicted_winner"]
                else:
                    row["old_winner_prob"] = None
                    row["old_predicted_winner"] = None
                return row

            merged_df = merged_df.apply(add_old_pred, axis=1)

        # ===============================
        # MERGE FINAL RESULTS IF AVAILABLE
        # ===============================
        if final_results:
            def add_final_results(row):
                race_num = row["race_number"]
                box = row["drawn_box"]
                if race_num in final_results:
                    for res_runner in final_results[race_num]:
                        if res_runner.get("drawn_box") == box:
                            row["final_position"] = res_runner.get("final_position")
                            row["odds_final"] = res_runner.get("odds_final")
                            break
                else:
                    row["final_position"] = None
                    row["odds_final"] = None
                return row

            merged_df = merged_df.apply(add_final_results, axis=1)
        else:
            merged_df["final_position"] = None
            merged_df["odds_final"] = None

        # ===============================
        # RECONSTRUCT NESTED STRUCTURE FOR JSONL
        # ===============================
        predictions_races = []
        for race_data in prerace_races:
            race_num = race_data["race_number"]
            race_runners = race_data["runners"]
            # Get predictions for this race
            race_preds = merged_df[merged_df["race_number"] == race_num].to_dict("records")
            # Map back to runners by drawn_box
            runner_map = {r["drawn_box"]: r for r in race_runners}
            updated_runners = []
            for pred in race_preds:
                box = pred["drawn_box"]
                if box in runner_map:
                    runner = runner_map[box].copy()
                    # Add prediction fields
                    runner["winner_prob"] = pred["winner_prob"]
                    runner["predicted_winner"] = pred["predicted_winner"]
                    if "old_winner_prob" in pred:
                        runner["old_winner_prob"] = pred["old_winner_prob"]
                        runner["old_predicted_winner"] = pred["old_predicted_winner"]
                    runner["final_position"] = pred["final_position"]
                    runner["odds_final"] = pred["odds_final"]
                    updated_runners.append(runner)
            
            # Ensure all runners are included, even without preds (shouldn't happen)
            for box in runner_map:
                if not any(r["drawn_box"] == box for r in updated_runners):
                    runner = runner_map[box].copy()
                    # Default values
                    runner["winner_prob"] = None
                    runner["predicted_winner"] = None
                    runner["old_winner_prob"] = None
                    runner["old_predicted_winner"] = None
                    runner["final_position"] = None
                    runner["odds_final"] = None
                    updated_runners.append(runner)
            
            updated_runners.sort(key=lambda x: x["drawn_box"])
            race_data["runners"] = updated_runners
            predictions_races.append(race_data)

        # ===============================
        # OUTPUT JSONL
        # ===============================
        output_jsonl = predictions_dir / folder_date / f"{venue}_{file_date}_predictions.jsonl"
        logger.info(f"Saving predictions JSONL: {output_jsonl}")
        with open(output_jsonl, "w") as f:
            for race in predictions_races:
                f.write(json.dumps(race) + "\n")
        logger.info("✅ JSONL predictions saved.")

        # Optional CSV output
        csv_path = None
        if export_csv:
            output_cols = [
                "race_number",
                "race_name",
                "drawn_box",
                "name",
                "trainer",
                "winner_prob",
                "predicted_winner",
            ]
            if old_predictions:
                output_cols += ["old_winner_prob", "old_predicted_winner"]
            output_cols += ["final_position", "odds_final"]

            summary_df = merged_df[output_cols].sort_values(
                ["race_number", "winner_prob"], ascending=[True, False]
            )
            output_csv = predictions_dir / folder_date / f"{venue}_{file_date}_predictions.csv"
            logger.info(f"Saving predictions CSV: {output_csv}")
            summary_df.to_csv(output_csv, index=False)
            logger.info("✅ CSV predictions saved.")
            csv_path = str(output_csv)
        else:
            csv_path = None

        # Update index
        meta_updates = {
            "prediction_created": True,
            "prediction_path": str(output_jsonl),
            "csv_path": csv_path,
            "prediction_model_version": model_version,
            "prediction_timestamp": datetime.utcnow().isoformat() + "Z",
            "prediction_status": "success"
        }
        
        if update_index_jsonl(venue, meta_updates, index_jsonl):
            logger.info("✅ Index updated successfully.")
        else:
            logger.error("Failed to update index.")
            # But still count as success for processing, since prediction generated

        logger.info(f"\nVenue {venue} processing complete.")
        successful_venues += 1

    except Exception as e:
        logger.error(f"❌ Error processing {venue}: {str(e)}")
        logger.error(traceback.format_exc())

        # Update index on failure
        meta_updates = {
            "prediction_created": False,
            "prediction_status": "failed",
            "error": str(e),
            "prediction_timestamp": datetime.utcnow().isoformat() + "Z",
            "prediction_model_version": model_version,
            "traceback": traceback.format_exc()
        }
        
        if update_index_jsonl(venue, meta_updates, index_jsonl):
            logger.info("❌ Index updated with failure status.")
        else:
            logger.error("Failed to update index on failure.")

        failed_venues += 1
        continue

# ===============================
# SUMMARY
# ===============================
logger.info(f"\n{'=' * 60}")
logger.info(f"PROCESSING COMPLETE")
logger.info(f"{'=' * 60}")
logger.info(f"Successfully processed: {successful_venues} venues")
logger.info(f"Skipped: {skipped_venues} venues")
logger.info(f"Failed: {failed_venues} venues")
logger.info(f"Output directory: {predictions_dir / folder_date}")
