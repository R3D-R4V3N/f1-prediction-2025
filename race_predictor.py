# General F1 race predictor with improved features
import os
import warnings

import fastf1
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')

# Create cache directory
CACHE_DIR = 'f1_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# 2025 driver lineup
DRIVERS_2025 = [
    {'DriverNumber': 16, 'Abbreviation': 'LEC', 'FullName': 'Charles Leclerc', 'Team': 'Ferrari'},
    {'DriverNumber': 44, 'Abbreviation': 'HAM', 'FullName': 'Lewis Hamilton', 'Team': 'Ferrari'},
    {'DriverNumber': 63, 'Abbreviation': 'RUS', 'FullName': 'George Russell', 'Team': 'Mercedes'},
    {'DriverNumber': 72, 'Abbreviation': 'ANT', 'FullName': 'Andrea Kimi Antonelli', 'Team': 'Mercedes'},
    {'DriverNumber': 1, 'Abbreviation': 'VER', 'FullName': 'Max Verstappen', 'Team': 'Red Bull Racing'},
    {'DriverNumber': 40, 'Abbreviation': 'LAW', 'FullName': 'Liam Lawson', 'Team': 'Red Bull Racing'},
    {'DriverNumber': 4, 'Abbreviation': 'NOR', 'FullName': 'Lando Norris', 'Team': 'McLaren'},
    {'DriverNumber': 81, 'Abbreviation': 'PIA', 'FullName': 'Oscar Piastri', 'Team': 'McLaren'},
    {'DriverNumber': 14, 'Abbreviation': 'ALO', 'FullName': 'Fernando Alonso', 'Team': 'Aston Martin'},
    {'DriverNumber': 18, 'Abbreviation': 'STR', 'FullName': 'Lance Stroll', 'Team': 'Aston Martin'},
    {'DriverNumber': 10, 'Abbreviation': 'GAS', 'FullName': 'Pierre Gasly', 'Team': 'Alpine'},
    {'DriverNumber': 5, 'Abbreviation': 'DOO', 'FullName': 'Jack Doohan', 'Team': 'Alpine'},
    {'DriverNumber': 23, 'Abbreviation': 'ALB', 'FullName': 'Alexander Albon', 'Team': 'Williams'},
    {'DriverNumber': 55, 'Abbreviation': 'SAI', 'FullName': 'Carlos Sainz Jr.', 'Team': 'Williams'},
    {'DriverNumber': 31, 'Abbreviation': 'OCO', 'FullName': 'Esteban Ocon', 'Team': 'Haas F1 Team'},
    {'DriverNumber': 87, 'Abbreviation': 'BEA', 'FullName': 'Oliver Bearman', 'Team': 'Haas F1 Team'},
    {'DriverNumber': 27, 'Abbreviation': 'HUL', 'FullName': 'Nico Hülkenberg', 'Team': 'Kick Sauber'},
    {'DriverNumber': 50, 'Abbreviation': 'BOR', 'FullName': 'Gabriel Bortoleto', 'Team': 'Kick Sauber'},
    {'DriverNumber': 22, 'Abbreviation': 'TSU', 'FullName': 'Yuki Tsunoda', 'Team': 'VCARB'},
    {'DriverNumber': 41, 'Abbreviation': 'HAD', 'FullName': 'Isack Hadjar', 'Team': 'VCARB'}
]

DRIVERS_DF = pd.DataFrame(DRIVERS_2025)

# Simplified overtaking difficulty metrics (1=easiest, 5=hardest)
OVERTAKE_DIFFICULTY = {
    'Bahrain Grand Prix': 2,
    'Saudi Arabian Grand Prix': 2,
    'Australian Grand Prix': 3,
    'Japanese Grand Prix': 3,
    'Chinese Grand Prix': 3,
    'Miami Grand Prix': 2,
    'Emilia Romagna Grand Prix': 4,
    'Monaco Grand Prix': 5,
    'Canadian Grand Prix': 3,
    'Spanish Grand Prix': 4,
    'Austrian Grand Prix': 2,
    'British Grand Prix': 3,
    'Hungarian Grand Prix': 4,
    'Belgian Grand Prix': 2,
    'Dutch Grand Prix': 4,
    'Italian Grand Prix': 2,
    'Azerbaijan Grand Prix': 3,
    'Singapore Grand Prix': 5,
    'United States Grand Prix': 3,
    'Mexican Grand Prix': 3,
    'Brazilian Grand Prix': 2,
    'Las Vegas Grand Prix': 3,
    'Qatar Grand Prix': 2,
    'Abu Dhabi Grand Prix': 3,
}

# List of grand prix for selection (2024 schedule approximation)
GRAND_PRIX_LIST = [
    'Bahrain Grand Prix',
    'Saudi Arabian Grand Prix',
    'Australian Grand Prix',
    'Japanese Grand Prix',
    'Chinese Grand Prix',
    'Miami Grand Prix',
    'Emilia Romagna Grand Prix',
    'Monaco Grand Prix',
    'Canadian Grand Prix',
    'Spanish Grand Prix',
    'Austrian Grand Prix',
    'British Grand Prix',
    'Hungarian Grand Prix',
    'Belgian Grand Prix',
    'Dutch Grand Prix',
    'Italian Grand Prix',
    'Azerbaijan Grand Prix',
    'Singapore Grand Prix',
    'United States Grand Prix',
    'Mexican Grand Prix',
    'Brazilian Grand Prix',
    'Las Vegas Grand Prix',
    'Qatar Grand Prix',
    'Abu Dhabi Grand Prix'
]


def _load_historical_data(seasons):
    race_data = []
    for season in seasons:
        for rnd in range(1, 23):
            try:
                # Race session
                session = fastf1.get_session(season, rnd, 'R')
                session.load()

                results = session.results[['DriverNumber', 'Position', 'Points', 'GridPosition']]
                results['Season'] = season
                results['RaceNumber'] = rnd
                results['Circuit'] = session.event['EventName']

                # Weather information
                try:
                    weather = session.weather_data
                    results['AirTemp'] = weather['AirTemp'].mean()
                    results['TrackTemp'] = weather['TrackTemp'].mean()
                    results['Rainfall'] = weather['Rainfall'].max()
                except Exception:
                    results['AirTemp'] = np.nan
                    results['TrackTemp'] = np.nan
                    results['Rainfall'] = np.nan

                # Overtake difficulty
                results['OvertakingDifficulty'] = OVERTAKE_DIFFICULTY.get(
                    results['Circuit'].iloc[0], 3
                )

                # Qualifying data
                try:
                    q_session = fastf1.get_session(season, rnd, 'Q')
                    q_session.load()
                    q_results = q_session.results[['DriverNumber', 'Position', 'Q1', 'Q2', 'Q3']]

                    def _best_time(row):
                        times = []
                        for col in ['Q1', 'Q2', 'Q3']:
                            val = row[col]
                            if pd.notna(val):
                                try:
                                    times.append(pd.to_timedelta(val).total_seconds())
                                except Exception:
                                    pass
                        return min(times) if times else np.nan

                    q_results['BestQualiTime'] = q_results.apply(_best_time, axis=1)
                    q_results = q_results.rename(columns={'Position': 'QualiPosition'})
                    results = pd.merge(results, q_results[['DriverNumber', 'BestQualiTime', 'QualiPosition']], on='DriverNumber', how='left')
                except Exception:
                    results['BestQualiTime'] = np.nan
                    results['QualiPosition'] = np.nan

                # FP3 best lap
                try:
                    fp3_session = fastf1.get_session(season, rnd, 'FP3')
                    fp3_session.load()
                    best_laps = (
                        fp3_session.laps
                        .groupby('DriverNumber')['LapTime']
                        .min()
                        .dt.total_seconds()
                        .reset_index()
                        .rename(columns={'LapTime': 'FP3BestTime'})
                    )
                    results = pd.merge(results, best_laps, on='DriverNumber', how='left')
                except Exception:
                    results['FP3BestTime'] = np.nan

                race_data.append(results)
            except Exception:
                continue
    return pd.concat(race_data)


def _add_driver_team_info(full_data, seasons):
    seasons_drivers = {}
    for season in seasons:
        try:
            schedule = fastf1.get_event_schedule(season)
            first_race = schedule.iloc[0]['RoundNumber']
            session = fastf1.get_session(season, first_race, 'R')
            session.load()
            for _, row in session.results.iterrows():
                driver_number = row['DriverNumber']
                driver_team = row['TeamName']
                seasons_drivers.setdefault(season, {})[driver_number] = driver_team
        except Exception:
            continue

    full_data['HistoricalTeam'] = None
    for idx, row in full_data.iterrows():
        season = row['Season']
        driver_num = row['DriverNumber']
        if season in seasons_drivers and driver_num in seasons_drivers[season]:
            full_data.at[idx, 'HistoricalTeam'] = seasons_drivers[season][driver_num]
        else:
            full_data.at[idx, 'HistoricalTeam'] = 'Unknown Team'
    return full_data


def _engineer_features(full_data):
    full_data['Position'] = pd.to_numeric(full_data['Position'], errors='coerce').fillna(25)
    full_data['GridPosition'] = pd.to_numeric(full_data['GridPosition'], errors='coerce').fillna(25)
    full_data['AirTemp'] = pd.to_numeric(full_data['AirTemp'], errors='coerce')
    full_data['TrackTemp'] = pd.to_numeric(full_data['TrackTemp'], errors='coerce')
    full_data['Rainfall'] = pd.to_numeric(full_data['Rainfall'], errors='coerce')
    full_data['OvertakingDifficulty'] = pd.to_numeric(full_data['OvertakingDifficulty'], errors='coerce')
    full_data['BestQualiTime'] = pd.to_numeric(full_data['BestQualiTime'], errors='coerce')
    full_data['QualiPosition'] = pd.to_numeric(full_data['QualiPosition'], errors='coerce')
    full_data['FP3BestTime'] = pd.to_numeric(full_data['FP3BestTime'], errors='coerce')

    full_data.sort_values(['Season', 'RaceNumber'], inplace=True)
    full_data['ExperienceCount'] = full_data.groupby('DriverNumber').cumcount() + 1
    full_data['RecentAvgPosition'] = (
        full_data.groupby('DriverNumber')['Position']
        .rolling(window=5, min_periods=1).mean().shift().reset_index(level=0, drop=True)
    )
    full_data['RecentAvgPosition'] = full_data['RecentAvgPosition'].fillna(full_data['Position'].mean())
    full_data['RecentAvgPoints'] = (
        full_data.groupby('DriverNumber')['Points']
        .rolling(window=5, min_periods=1).mean().shift().reset_index(level=0, drop=True)
    )
    full_data['RecentAvgPoints'] = full_data['RecentAvgPoints'].fillna(0)

    team_perf = full_data.groupby(['HistoricalTeam', 'Season'])['Position'].mean().reset_index()
    team_perf = team_perf.rename(columns={'Position': 'TeamAvgPosition'})
    full_data = pd.merge(full_data, team_perf, on=['HistoricalTeam', 'Season'], how='left')
    full_data['TeamAvgPosition'] = full_data['TeamAvgPosition'].fillna(full_data['TeamAvgPosition'].mean())

    full_data['AirTemp'] = full_data['AirTemp'].fillna(full_data['AirTemp'].mean())
    full_data['TrackTemp'] = full_data['TrackTemp'].fillna(full_data['TrackTemp'].mean())
    full_data['Rainfall'] = full_data['Rainfall'].fillna(0)
    full_data['OvertakingDifficulty'] = full_data['OvertakingDifficulty'].fillna(3)
    full_data['BestQualiTime'] = full_data['BestQualiTime'].fillna(full_data['BestQualiTime'].mean())
    full_data['QualiPosition'] = full_data['QualiPosition'].fillna(20)
    full_data['FP3BestTime'] = full_data['FP3BestTime'].fillna(full_data['FP3BestTime'].mean())

    return full_data


def _encode_features(full_data):
    team_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    team_encoded = team_encoder.fit_transform(full_data[['Team']])
    team_df = pd.DataFrame(team_encoded, columns=team_encoder.get_feature_names_out(['Team']))

    circuit_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    circuit_encoded = circuit_encoder.fit_transform(full_data[['Circuit']])
    circuit_df = pd.DataFrame(circuit_encoded, columns=circuit_encoder.get_feature_names_out(['Circuit']))

    top_circuits = circuit_df.sum().sort_values(ascending=False).head(15).index
    circuit_df = circuit_df[top_circuits]

    features = pd.concat([
        full_data[[
            'GridPosition',
            'Season',
            'ExperienceCount',
            'TeamAvgPosition',
            'RecentAvgPosition',
            'RecentAvgPoints',
            'AirTemp',
            'TrackTemp',
            'Rainfall',
            'OvertakingDifficulty',
            'BestQualiTime',
            'QualiPosition',
            'FP3BestTime',
        ]],
        team_df,
        circuit_df
    ], axis=1)

    return features, team_encoder, circuit_encoder, top_circuits


def _train_model(features, target):
    param_dist = {
        'n_estimators': [150, 200, 250],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2]
    }
    base_model = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(base_model, param_dist, n_iter=5, cv=3, random_state=42, n_jobs=-1)
    search.fit(features, target)
    return search.best_estimator_


def predict_race(grand_prix):
    seasons = [2022, 2023, 2024]
    race_data = _load_historical_data(seasons)
    race_data = race_data.reset_index(drop=True)
    # Ensure DriverNumber is numeric for consistent merging
    race_data['DriverNumber'] = pd.to_numeric(race_data['DriverNumber'], errors='coerce')
    DRIVERS_DF['DriverNumber'] = pd.to_numeric(DRIVERS_DF['DriverNumber'], errors='coerce')
    race_data = pd.merge(race_data, DRIVERS_DF, on='DriverNumber', how='left')
    race_data = _add_driver_team_info(race_data, seasons)
    race_data = _engineer_features(race_data)

    features, team_enc, circuit_enc, top_circuits = _encode_features(race_data)
    target = race_data['Position']
    model = _train_model(features, target)

    default_air = race_data['AirTemp'].mean()
    default_track = race_data['TrackTemp'].mean()
    default_rain = 0.0
    default_overtake = 3
    default_best_q = race_data['BestQualiTime'].mean()
    default_qpos = race_data['QualiPosition'].mean()
    default_fp3 = race_data['FP3BestTime'].mean()

    # Prepare prediction dataframe for all drivers
    pred_rows = []
    team_strength = race_data[race_data['Season'] == 2024].groupby('HistoricalTeam')['Position'].mean().reset_index()
    team_strength = team_strength.rename(columns={'HistoricalTeam': 'Team', 'Position': 'TeamAvgPosition'})

    for d in DRIVERS_2025:
        exp_count = len(race_data[race_data['DriverNumber'] == d['DriverNumber']])
        if exp_count == 0:
            exp_count = 1
        else:
            exp_count += 23
        team_avg = team_strength[team_strength['Team'] == d['Team']]
        if len(team_avg) == 0:
            team_avg_pos = team_strength['TeamAvgPosition'].mean()
        else:
            team_avg_pos = team_avg.iloc[0]['TeamAvgPosition']
        pred_rows.append({
            'GridPosition': np.nan,
            'Season': 2025,
            'ExperienceCount': exp_count,
            'TeamAvgPosition': team_avg_pos,
            'RecentAvgPosition': 10.0,
            'RecentAvgPoints': 0.0,
            'AirTemp': default_air,
            'TrackTemp': default_track,
            'Rainfall': default_rain,
            'OvertakingDifficulty': default_overtake,
            'BestQualiTime': default_best_q,
            'QualiPosition': default_qpos,
            'FP3BestTime': default_fp3,
            'Team': d['Team'],
            'FullName': d['FullName'],
            'Abbreviation': d['Abbreviation']
        })

    pred_df = pd.DataFrame(pred_rows)

    team_weights = 1 / pred_df['TeamAvgPosition']
    team_weights = team_weights.fillna(team_weights.mean())
    team_weights = team_weights.clip(lower=0.001)
    norm_w = team_weights / team_weights.sum()
    grid_order = np.random.choice(pred_df.index, size=len(pred_df), replace=False, p=norm_w)
    grid_positions = np.ones(len(pred_df)) * 20
    grid_positions[grid_order] = np.arange(1, len(pred_df) + 1)
    pred_df['GridPosition'] = grid_positions

    team_enc_df = pd.DataFrame(team_enc.transform(pred_df[['Team']]), columns=team_enc.get_feature_names_out(['Team']))
    circuit_df = pd.DataFrame(circuit_enc.transform(pd.DataFrame({'Circuit': [grand_prix] * len(pred_df)})), columns=circuit_enc.get_feature_names_out(['Circuit']))
    circuit_df = circuit_df.reindex(columns=top_circuits, fill_value=0)

    pred_features = pd.concat([
        pred_df[[
            'GridPosition',
            'Season',
            'ExperienceCount',
            'TeamAvgPosition',
            'RecentAvgPosition',
            'RecentAvgPoints',
            'AirTemp',
            'TrackTemp',
            'Rainfall',
            'OvertakingDifficulty',
            'BestQualiTime',
            'QualiPosition',
            'FP3BestTime',
        ]].reset_index(drop=True),
        team_enc_df.reset_index(drop=True),
        circuit_df.reset_index(drop=True)
    ], axis=1)

    for col in features.columns:
        if col not in pred_features.columns:
            pred_features[col] = 0
    pred_features = pred_features[features.columns]

    preds = model.predict(pred_features)
    results = pd.DataFrame({
        'Driver': pred_df['FullName'],
        'Team': pred_df['Team'],
        'Grid': pred_df['GridPosition'],
        'Predicted_Position': preds
    }).sort_values('Predicted_Position')
    results['Final_Position'] = range(1, len(results) + 1)
    return results


if __name__ == '__main__':
    res = predict_race('Chinese Grand Prix')
    print(res[['Driver', 'Team', 'Grid', 'Final_Position']].head())
