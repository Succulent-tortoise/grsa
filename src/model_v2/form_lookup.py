import os
import json
import glob
from datetime import datetime, timedelta
from src.model_v2.features import ewma_position, trend_slope, form_volatility, days_since_last_run, num_races, best_t_d_rank

class ForbiddenFeatureError(Exception):
    pass

def load_runner_history(runner_name, race_date):
    date_obj = datetime.strptime(race_date, '%Y%m%d')
    start_date = (date_obj - timedelta(days=30)).strftime('%Y%m%d')
    jsonl_files = glob.glob(f'/media/matt_sent/vault/dishlicker_data/data/jsonl/{start_date}/*.jsonl')
    prior_races = []

    for file in jsonl_files:
        with open(file, 'r') as f:
            for line in f:
                race_data = json.loads(line)
                if race_data['name'] == runner_name:
                    prior_races.append(race_data)

    return prior_races

def compute_form_features(prior_races):
    last_race = prior_races[-1] if prior_races else None
    features = {
        'ewma_position': ewma_position(prior_races),
        'trend_slope': trend_slope(prior_races),
        'form_volatility': form_volatility(prior_races),
        'days_since_last_run': days_since_last_run(last_race['date']) if last_race else None,
        'num_races': num_races(prior_races)
    }

    # Check for forbidden odds field
    if 'odds' in features:
        raise ForbiddenFeatureError('Forbidden feature detected: odds')

    return features

def get_runner_form(runner_name, race_date):
    prior_races = load_runner_history(runner_name, race_date)
    if len(prior_races) < 2:
        return {'error': 'insufficient form'}

    form_features = compute_form_features(prior_races)
    return form_features
