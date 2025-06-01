# General F1 race predictor with improved features
import os
import warnings

import fastf1
import requests
from export_race_details import export_race_details, _fetch_session_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import optuna

warnings.filterwarnings('ignore')

# Create cache directory
CACHE_DIR = 'f1_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def _get_event_drivers(year: int, grand_prix: str) -> pd.DataFrame:
    """Return the driver lineup for a given event using FastF1.

    For events that haven't taken place yet the race result list is empty. In
    those cases fall back to the session entry list which is available through
    the same timing API used for lap data.
    """

    schedule = fastf1.get_event_schedule(year)
    match = schedule[schedule["EventName"].str.contains(grand_prix, case=False, na=False)]
    if match.empty:
        raise ValueError(f"Grand Prix '{grand_prix}' not found for {year}")

    round_number = int(match.iloc[0]["RoundNumber"])
    session = fastf1.get_session(year, round_number, "R")

    # First try to get the official race results which contain driver details.
    try:
        session.load(telemetry=False, laps=False, weather=False)
        if hasattr(session, "results") and not session.results.empty:
            info = session.results[["DriverNumber", "Abbreviation", "FullName", "TeamName"]].copy()
            info.rename(columns={"TeamName": "Team"}, inplace=True)
            return info
    except Exception:
        pass

    # If results are not available yet, attempt to read the entry list from the
    # timing data which is present for upcoming sessions as soon as timing data
    # exists (e.g. for practice or qualifying).
    try:
        if not session._timing_data_fetched:  # type: ignore[attr-defined]
            session.load(telemetry=False, laps=False, weather=False)
        entry_list = session.entry_list  # type: ignore[attr-defined]
        info = entry_list[["DriverNumber", "Abbreviation", "FullName", "TeamName"]].copy()
        info.rename(columns={"TeamName": "Team"}, inplace=True)
        if not info.empty:
            return info
    except Exception:
        pass

    # Final fallback: query the Ergast API for the driver lineup of the first
    # round of the season. This avoids hard coding a list while still providing
    # a reasonable default when FastF1 has no data for upcoming events.
    try:
        url = f"https://ergast.com/api/f1/{year}/1/results.json?limit=100"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if races:
            results = races[0].get("Results", [])
            rows = []
            for r in results:
                rows.append(
                    {
                        "DriverNumber": int(r.get("number", 0)),
                        "Abbreviation": r["Driver"].get("code", ""),
                        "FullName": f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                        "Team": r["Constructor"].get("name", ""),
                    }
                )
            if rows:
                return pd.DataFrame(rows)
    except Exception:
        pass

    # If all attempts fail, return an empty DataFrame instead of using a static
    # hard-coded driver list.
    return pd.DataFrame(columns=["DriverNumber", "Abbreviation", "FullName", "Team"])


def _get_qualifying_results(year: int, grand_prix: str) -> pd.DataFrame:
    """Return qualifying results with driver abbreviations and times.

    Parameters
    ----------
    year : int
        Season year.
    grand_prix : str
        Grand Prix name.

    Returns
    -------
    pd.DataFrame
        DataFrame containing ``Abbreviation``, ``GridPosition`` and best lap
        times from qualifying. ``GridPosition`` corresponds to the official
        starting grid.
    """
    schedule = fastf1.get_event_schedule(year)
    match = schedule[schedule["EventName"].str.contains(grand_prix, case=False, na=False)]
    if match.empty:
        raise ValueError(f"Grand Prix '{grand_prix}' not found for {year}")

    round_number = int(match.iloc[0]["RoundNumber"])
    session = fastf1.get_session(year, round_number, "Q")
    session.load()
    q_res = session.results[
        ["DriverNumber", "Abbreviation", "FullName", "TeamName", "Position", "Q1", "Q2", "Q3"]
    ].copy()

    def _to_seconds(val):
        if pd.isna(val):
            return None
        try:
            return pd.to_timedelta(val).total_seconds()
        except Exception:
            return None

    for col in ["Q1", "Q2", "Q3"]:
        q_res[col] = q_res[col].apply(_to_seconds)
    q_res["BestTime"] = q_res[["Q1", "Q2", "Q3"]].min(axis=1)
    q_res.rename(columns={"Position": "GridPosition", "TeamName": "Team"}, inplace=True)
    return q_res


def _get_fp3_results(year: int, grand_prix: str) -> pd.DataFrame:
    """Return FP3 best laps and weather information."""
    schedule = fastf1.get_event_schedule(year)
    match = schedule[schedule["EventName"].str.contains(grand_prix, case=False, na=False)]
    if match.empty:
        raise ValueError(f"Grand Prix '{grand_prix}' not found for {year}")

    round_number = int(match.iloc[0]["RoundNumber"])
    df = _fetch_session_data(year, round_number, "FP3")
    df = df.rename(columns={"Driver": "Abbreviation", "BestTime": "FP3BestTime"})
    return df[["Abbreviation", "FP3BestTime", "AvgAirTemp", "AvgTrackTemp", "MaxRainfall"]]

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
                    q_results['Q3Time'] = q_results['Q3'].apply(
                        lambda x: pd.to_timedelta(x).total_seconds() if pd.notna(x) else np.nan
                    )
                    q_results['GridFromQ3'] = q_results['Q3Time'].rank(method='first')
                    results = pd.merge(
                        results,
                        q_results[['DriverNumber', 'BestQualiTime', 'QualiPosition', 'GridFromQ3']],
                        on='DriverNumber',
                        how='left'
                    )
                    # Use Q3 ranking for grid, falling back to qualifying order if needed.
                    results['GridPosition'] = results['GridFromQ3'].fillna(results['QualiPosition'])
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
    full_data['Recent3AvgFinish'] = (
        full_data.groupby('DriverNumber')['Position']
        .rolling(window=3, min_periods=1).mean().shift().reset_index(level=0, drop=True)
    )
    full_data['Recent3AvgFinish'] = full_data['Recent3AvgFinish'].fillna(full_data['Position'].mean())
    full_data['Recent5AvgFinish'] = (
        full_data.groupby('DriverNumber')['Position']
        .rolling(window=5, min_periods=1).mean().shift().reset_index(level=0, drop=True)
    )
    full_data['Recent5AvgFinish'] = full_data['Recent5AvgFinish'].fillna(full_data['Position'].mean())
    full_data['QualiImprove'] = full_data['GridPosition'] - full_data['Position']
    full_data['RecentAvgPoints'] = (
        full_data.groupby('DriverNumber')['Points']
        .rolling(window=5, min_periods=1).mean().shift().reset_index(level=0, drop=True)
    )
    full_data['RecentAvgPoints'] = full_data['RecentAvgPoints'].fillna(0)

    driver_track = full_data.groupby(['DriverNumber', 'Circuit'])['Position'].agg(
        DriverAvgTrackFinish='mean',
        DriverTrackPodiums=lambda x: (x <= 3).sum(),
        DriverTrackDNFs=lambda x: (x > 20).sum(),
    ).reset_index()
    full_data = pd.merge(full_data, driver_track, on=['DriverNumber', 'Circuit'], how='left')

    full_data['TeamRecentQuali'] = (
        full_data.groupby('HistoricalTeam')['QualiPosition']
        .rolling(window=5, min_periods=1).mean().shift().reset_index(level=0, drop=True)
    )
    full_data['TeamRecentFinish'] = (
        full_data.groupby('HistoricalTeam')['Position']
        .rolling(window=5, min_periods=1).mean().shift().reset_index(level=0, drop=True)
    )
    full_data['TeamReliability'] = (
        full_data.groupby('HistoricalTeam')['Position']
        .rolling(window=5, min_periods=1)
        .apply(lambda x: (x > 20).sum())
        .shift()
        .reset_index(level=0, drop=True)
    )
    full_data['TeamRecentQuali'] = full_data['TeamRecentQuali'].fillna(full_data['QualiPosition'].mean())
    full_data['TeamRecentFinish'] = full_data['TeamRecentFinish'].fillna(full_data['Position'].mean())
    full_data['TeamReliability'] = full_data['TeamReliability'].fillna(0)

    TRACK_TYPE = {
        'Monaco Grand Prix': 'street',
        'Singapore Grand Prix': 'street',
        'Las Vegas Grand Prix': 'street',
    }
    DOWNFORCE = {
        'Monaco Grand Prix': 'high',
        'Hungarian Grand Prix': 'high',
        'Italian Grand Prix': 'low',
        'Belgian Grand Prix': 'low',
    }
    full_data['TrackType'] = full_data['Circuit'].map(TRACK_TYPE).fillna('permanent')
    full_data['Downforce'] = full_data['Circuit'].map(DOWNFORCE).fillna('medium')
    full_data['IsStreet'] = full_data['TrackType'].map({'street': 1, 'permanent': 0})
    df_level_map = {'low': 0, 'medium': 1, 'high': 2}
    full_data['DownforceLevel'] = full_data['Downforce'].map(df_level_map)

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
    full_data['IsStreet'] = full_data['IsStreet'].fillna(0)
    full_data['DownforceLevel'] = full_data['DownforceLevel'].fillna(1)

    return full_data


def _prepare_features(full_data, base_cols, team_encoder=None, circuit_encoder=None, top_circuits=None):
    """Encode categorical features and return a design matrix.

    Parameters
    ----------
    full_data : pd.DataFrame
        Data containing at least the columns in ``base_cols`` plus ``Team`` and ``Circuit``.
    base_cols : list
        Numerical columns to include directly in the feature matrix.
    team_encoder, circuit_encoder : OneHotEncoder or None
        Encoders to use. If ``None`` new encoders will be fitted.
    top_circuits : list or None
        Subset of circuit columns to keep. If ``None`` the 15 most frequent circuits are used.

    Returns
    -------
    features : pd.DataFrame
        The encoded feature matrix.
    team_encoder : OneHotEncoder
        Fitted team encoder (if a new one was created).
    circuit_encoder : OneHotEncoder
        Fitted circuit encoder.
    top_circuits : list
        Names of the circuit columns that were kept.
    """

    full_data = full_data.copy()

    if full_data.empty:
        team_cols = (
            team_encoder.get_feature_names_out(["Team"]) if team_encoder else []
        )
        circuit_cols = (
            circuit_encoder.get_feature_names_out(["Circuit"])
            if circuit_encoder
            else []
        )
        if top_circuits is not None:
            circuit_cols = [c for c in circuit_cols if c in top_circuits]
        empty_cols = base_cols + list(team_cols) + list(circuit_cols)
        return pd.DataFrame(columns=empty_cols), team_encoder, circuit_encoder, top_circuits

    # ``Team`` or ``Circuit`` columns may be missing if the calling code fails to
    # merge driver or event details correctly.  Rather than raising a KeyError
    # when one-hot encoding, provide sensible defaults so the model can still
    # run.
    if "Team" not in full_data.columns:
        full_data["Team"] = "Unknown Team"
    if "Circuit" not in full_data.columns:
        full_data["Circuit"] = "Unknown Circuit"

    # Ensure all numerical base columns are numeric to avoid object dtypes when
    # creating the feature matrix used by XGBoost. Missing values are filled with
    # ``0`` to keep shapes consistent during prediction.
    for col in base_cols:
        if col not in full_data.columns:
            full_data[col] = 0
        full_data[col] = pd.to_numeric(full_data[col], errors="coerce").fillna(0)

    if team_encoder is None:
        team_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        team_encoded = team_encoder.fit_transform(full_data[['Team']])
    else:
        team_encoded = team_encoder.transform(full_data[['Team']])
    team_df = pd.DataFrame(team_encoded, columns=team_encoder.get_feature_names_out(['Team']))

    if circuit_encoder is None:
        circuit_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        circuit_encoded = circuit_encoder.fit_transform(full_data[['Circuit']])
        circuit_df = pd.DataFrame(circuit_encoded, columns=circuit_encoder.get_feature_names_out(['Circuit']))
        if top_circuits is None:
            top_circuits = circuit_df.sum().sort_values(ascending=False).head(15).index
    else:
        circuit_encoded = circuit_encoder.transform(full_data[['Circuit']])
        circuit_df = pd.DataFrame(circuit_encoded, columns=circuit_encoder.get_feature_names_out(['Circuit']))
    if top_circuits is not None:
        circuit_df = circuit_df.reindex(columns=top_circuits, fill_value=0)

    features = pd.concat([
        full_data[base_cols].reset_index(drop=True),
        team_df.reset_index(drop=True),
        circuit_df.reset_index(drop=True)
    ], axis=1)

    return features, team_encoder, circuit_encoder, top_circuits


def _encode_features(full_data, base_cols, team_encoder=None, circuit_encoder=None, top_circuits=None):
    return _prepare_features(full_data, base_cols, team_encoder, circuit_encoder, top_circuits)


def _train_model(features, target, cv):
    """Train an XGBoost model using Bayesian optimization to minimize MAE."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            **params,
        )

        scores = []
        for train_idx, val_idx in cv.split(features):
            model.fit(features.iloc[train_idx], target.iloc[train_idx])
            preds = model.predict(features.iloc[val_idx])
            scores.append(mean_absolute_error(target.iloc[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    best_params = study.best_params
    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **best_params,
    )
    model.fit(features, target)
    return model


def predict_race(grand_prix, year=2025, export_details=False):
    seasons = list(range(2020, year + 1))
    race_data = _load_historical_data(seasons)
    race_data = race_data.reset_index(drop=True)
    # Ensure DriverNumber is numeric for consistent merging
    race_data['DriverNumber'] = pd.to_numeric(race_data['DriverNumber'], errors='coerce')
    qual_results = None
    try:
        drivers_df = _get_qualifying_results(year, grand_prix)
        drivers_df = drivers_df[drivers_df['BestTime'].notna()]
        qual_results = drivers_df.copy()
    except Exception:
        drivers_df = _get_event_drivers(year, grand_prix)
    drivers_df['DriverNumber'] = pd.to_numeric(drivers_df['DriverNumber'], errors='coerce')
    race_data = pd.merge(race_data, drivers_df, on='DriverNumber', how='left')
    race_data = _add_driver_team_info(race_data, seasons)
    race_data = _engineer_features(race_data)
    # Save the engineered dataset used for training and prediction so users can
    # inspect all input values.
    race_data.to_csv("prediction_data.csv", index=False)

    # Feature sets
    race_cols = [
        'GridPosition', 'Season', 'ExperienceCount', 'TeamAvgPosition',
        'RecentAvgPosition', 'RecentAvgPoints', 'AirTemp', 'TrackTemp',
        'Rainfall', 'OvertakingDifficulty', 'BestQualiTime',
        'QualiPosition', 'FP3BestTime', 'Recent3AvgFinish',
        'Recent5AvgFinish', 'QualiImprove', 'DriverAvgTrackFinish',
        'DriverTrackPodiums', 'DriverTrackDNFs', 'TeamRecentQuali',
        'TeamRecentFinish', 'TeamReliability', 'IsStreet',
        'DownforceLevel'
    ]
    quali_cols = [
        'Season', 'ExperienceCount', 'TeamAvgPosition', 'RecentAvgPosition',
        'RecentAvgPoints', 'AirTemp', 'TrackTemp', 'Rainfall',
        'OvertakingDifficulty', 'BestQualiTime', 'FP3BestTime',
        'TeamRecentQuali', 'IsStreet', 'DownforceLevel'
    ]

    # Encode features for both models using shared encoders
    features, team_enc, circuit_enc, top_circuits = _encode_features(
        race_data, race_cols
    )
    quali_feats, _, _, _ = _encode_features(
        race_data, quali_cols, team_enc, circuit_enc, top_circuits
    )

    cv = TimeSeriesSplit(n_splits=3)

    # Train race finish model
    target = race_data['Position']
    model = _train_model(features, target, cv)
    # Train grid prediction model
    grid_model = _train_model(quali_feats, race_data['GridPosition'], cv)

    grid_preds_hist = grid_model.predict(quali_feats)
    grid_mae = mean_absolute_error(race_data['GridPosition'], grid_preds_hist)
    race_data['PredGrid'] = grid_preds_hist

    race_feats_with_pred, _, _, _ = _encode_features(
        race_data, race_cols + ['PredGrid'], team_enc, circuit_enc, top_circuits
    )
    model = _train_model(race_feats_with_pred, target, cv)
    finish_preds_hist = model.predict(race_feats_with_pred)
    finish_mae = mean_absolute_error(race_data['Position'], finish_preds_hist)
    features = race_feats_with_pred

    if qual_results is not None and not qual_results.empty:
        default_best_q = qual_results['BestTime'].mean()
        default_qpos = qual_results['GridPosition'].mean()
    else:
        default_best_q = race_data['BestQualiTime'].mean()
        default_qpos = race_data['QualiPosition'].mean()

    if fp3_results is not None and not fp3_results.empty:
        default_air = fp3_results['AvgAirTemp'].mean()
        default_track = fp3_results['AvgTrackTemp'].mean()
        default_rain = fp3_results['MaxRainfall'].max()
        default_fp3 = fp3_results['FP3BestTime'].mean()
    else:
        default_air = race_data['AirTemp'].mean()
        default_track = race_data['TrackTemp'].mean()
        default_rain = 0.0
        default_fp3 = race_data['FP3BestTime'].mean()
    default_overtake = 3

    # Prepare prediction dataframe for all drivers
    pred_rows = []
    team_strength = (
        race_data[race_data['Season'] == year]
        .groupby('HistoricalTeam')['Position']
        .mean()
        .reset_index()
        .rename(columns={'Position': 'TeamAvgPosition'})
    )
    if team_strength.empty and year > 2020:
        team_strength = (
            race_data[race_data['Season'] == year - 1]
            .groupby('HistoricalTeam')['Position']
            .mean()
            .reset_index()
            .rename(columns={'Position': 'TeamAvgPosition'})
        )
    team_recent_quali = race_data.groupby('HistoricalTeam')['QualiPosition'].rolling(window=5, min_periods=1).mean().reset_index().rename(columns={'QualiPosition': 'TeamRecentQuali'})
    team_recent_finish = race_data.groupby('HistoricalTeam')['Position'].rolling(window=5, min_periods=1).mean().reset_index().rename(columns={'Position': 'TeamRecentFinish'})
    team_reliability = race_data.groupby('HistoricalTeam')['Position'].rolling(window=5, min_periods=1).apply(lambda x: (x > 20).sum()).reset_index().rename(columns={'Position': 'TeamReliability'})
    team_recent_quali = team_recent_quali.groupby('HistoricalTeam').last().reset_index()
    team_recent_finish = team_recent_finish.groupby('HistoricalTeam').last().reset_index()
    team_reliability = team_reliability.groupby('HistoricalTeam').last().reset_index()
    team_info = team_strength.merge(team_recent_quali, on='HistoricalTeam', how='left')
    team_info = team_info.merge(team_recent_finish, on='HistoricalTeam', how='left')
    team_info = team_info.merge(team_reliability, on='HistoricalTeam', how='left')
    team_info = team_info.rename(columns={'HistoricalTeam': 'Team'})

    driver_stats_lookup = race_data.set_index(['DriverNumber', 'Circuit'])[
        ['DriverAvgTrackFinish', 'DriverTrackPodiums', 'DriverTrackDNFs']
    ]

    if qual_results is None:
        try:
            qual_results = _get_qualifying_results(year, grand_prix)
            if not drivers_df.empty:
                qual_results = qual_results.merge(
                    drivers_df[['Abbreviation', 'FullName', 'Team', 'DriverNumber']],
                    on='Abbreviation',
                    how='left'
                )
            qual_results = qual_results[qual_results['BestTime'].notna()]
        except Exception:
            qual_results = None

    try:
        fp3_results = _get_fp3_results(year, grand_prix)
        if qual_results is not None:
            qual_results = qual_results.merge(fp3_results, on='Abbreviation', how='left')
    except Exception:
        fp3_results = None

    driver_iter = qual_results if qual_results is not None and not qual_results.empty else drivers_df
    for _, d in driver_iter.iterrows():
        exp_count = len(race_data[race_data['DriverNumber'] == d['DriverNumber']])
        if exp_count == 0:
            exp_count = 1
        else:
            exp_count += 23
        team_row = team_info[team_info['Team'] == d['Team']]
        if len(team_row) == 0:
            team_avg_pos = team_info['TeamAvgPosition'].mean()
            team_recent_q = team_info['TeamRecentQuali'].mean()
            team_recent_f = team_info['TeamRecentFinish'].mean()
            team_rel = 0.0
        else:
            team_avg_pos = team_row.iloc[0]['TeamAvgPosition']
            team_recent_q = team_row.iloc[0]['TeamRecentQuali']
            team_recent_f = team_row.iloc[0]['TeamRecentFinish']
            team_rel = team_row.iloc[0]['TeamReliability']

        stats = driver_stats_lookup.loc[(d['DriverNumber'], grand_prix)] if (d['DriverNumber'], grand_prix) in driver_stats_lookup.index else None
        if stats is None:
            avg_track = race_data['DriverAvgTrackFinish'].mean()
            podiums = 0.0
            dnfs = 0.0
        else:
            avg_track = stats['DriverAvgTrackFinish']
            podiums = stats['DriverTrackPodiums']
            dnfs = stats['DriverTrackDNFs']

        if qual_results is not None and 'GridPosition' in d and pd.notna(d['GridPosition']):
            grid_pos = int(d['GridPosition'])
            best_time = d['BestTime']
        else:
            grid_pos = np.nan
            best_time = default_best_q

        if fp3_results is not None and 'FP3BestTime' in d and pd.notna(d['FP3BestTime']):
            fp3_time = d['FP3BestTime']
        else:
            fp3_time = default_fp3
        pred_rows.append({
            'GridPosition': grid_pos,
            'Season': year,
            'ExperienceCount': exp_count,
            'TeamAvgPosition': team_avg_pos,
            'RecentAvgPosition': 10.0,
            'RecentAvgPoints': 0.0,
            'AirTemp': default_air,
            'TrackTemp': default_track,
            'Rainfall': default_rain,
            'OvertakingDifficulty': default_overtake,
            'BestQualiTime': best_time,
            'QualiPosition': grid_pos,
            'FP3BestTime': fp3_time,
            'Recent3AvgFinish': 10.0,
            'Recent5AvgFinish': 10.0,
            'QualiImprove': 0.0,
            'DriverAvgTrackFinish': avg_track,
            'DriverTrackPodiums': podiums,
            'DriverTrackDNFs': dnfs,
            'TeamRecentQuali': team_recent_q,
            'TeamRecentFinish': team_recent_f,
            'TeamReliability': team_rel,
            'IsStreet': 1 if grand_prix in ['Monaco Grand Prix','Singapore Grand Prix','Las Vegas Grand Prix'] else 0,
            'DownforceLevel': 1,
            'Team': d['Team'],
            'FullName': d['FullName'],
            'Abbreviation': d['Abbreviation']
        })

    pred_df = pd.DataFrame(pred_rows)

    if pred_df.empty:
        raise ValueError(f"No driver data available for {year} {grand_prix}")

    if qual_results is None:
        # Predict grid positions when no qualifying data is available
        quali_pred_features, _, _, _ = _encode_features(
            pred_df.assign(Circuit=grand_prix),
            quali_cols,
            team_enc,
            circuit_enc,
            top_circuits,
        )
        grid_scores = grid_model.predict(quali_pred_features)
        pred_df['GridPosition'] = pd.Series(grid_scores).rank(method='first').astype(int)
        pred_df['QualiPosition'] = pred_df['GridPosition']
        pred_df['PredGrid'] = grid_scores
    else:
        pred_df['PredGrid'] = pred_df['GridPosition']
    # Save the driver list with predicted grid positions so the user can inspect
    # the raw input given to the finish model.
    pred_df.to_csv("prediction_input.csv", index=False)

    # Encode the prediction rows using the same helper as for training. This
    # converts numerical columns to proper dtypes and creates aligned one-hot
    # encoded team and circuit features.
    race_pred_features, _, _, _ = _encode_features(
        pred_df.assign(Circuit=grand_prix),
        race_cols + ['PredGrid'],
        team_enc,
        circuit_enc,
        top_circuits,
    )

    # Ensure the column order matches the training feature matrix. Missing
    # columns (possible if new teams or circuits appear) are filled with 0 so the
    # XGBoost model receives the expected input shape.
    pred_features = race_pred_features.reindex(columns=features.columns, fill_value=0)

    preds = model.predict(pred_features)
    results = pd.DataFrame({
        'Driver': pred_df['Abbreviation'],
        'Team': pred_df['Team'],
        'Grid': pred_df['GridPosition'],
        'Predicted_Position': preds
    }).sort_values('Predicted_Position')
    results['Final_Position'] = range(1, len(results) + 1)
    # Save the final ordered predictions for transparency
    results.to_csv("prediction_results.csv", index=False)
    if export_details:
        try:
            detail_path = export_race_details(year, grand_prix)
            print(f"Saved session data to {detail_path}")
        except Exception as err:
            print(f"Could not export session data: {err}")
    print(f"Grid MAE on training data: {grid_mae:.2f}")
    print(f"Finish MAE on training data: {finish_mae:.2f}")
    return results


if __name__ == '__main__':
    res = predict_race('Chinese Grand Prix', year=2025, export_details=True)
    print(res[['Driver', 'Team', 'Grid', 'Final_Position']].head())
