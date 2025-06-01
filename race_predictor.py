# General F1 race predictor with improved features
import os
import warnings

import fastf1
import requests
from export_race_details import export_race_details, _fetch_session_data
from estimate_overtakes import average_overtakes
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr



class SeasonSplit:
    """Cross-validator that yields whole seasons as validation folds."""

    def __init__(self, seasons, season_col="Season"):
        self.seasons = list(seasons)
        self.season_col = season_col

    def split(self, X):
        seasons_sorted = [s for s in self.seasons if s in X[self.season_col].unique()]
        seasons_sorted.sort()
        for i in range(1, len(seasons_sorted)):
            train_idx = X.index[X[self.season_col].isin(seasons_sorted[:i])].to_numpy()
            val_idx = X.index[X[self.season_col] == seasons_sorted[i]].to_numpy()
            if len(train_idx) and len(val_idx):
                yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return max(0, len(self.seasons) - 1)
from xgboost import XGBRanker, plot_importance
import matplotlib.pyplot as plt
import optuna

try:
    import shap  # type: ignore
except Exception:
    shap = None

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
    df = df.rename(
        columns={
            "Driver": "Abbreviation",
            "BestTime": "FP3BestTime",
            "LongRunTime": "FP3LongRunTime",
        }
    )
    return df[
        [
            "Abbreviation",
            "FP3BestTime",
            "FP3LongRunTime",
            "AvgAirTemp",
            "AvgTrackTemp",
            "MaxRainfall",
        ]
    ]


def _load_overtake_stats(path: str = "overtake_stats.csv") -> dict:
    """Return weighted average overtake counts mapped by circuit name.

    If the CSV does not exist or is empty an empty dictionary is returned
    instead of raising an exception.  This allows the application to run
    even when no overtake statistics have been generated yet.
    """

    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    if df.empty or "Circuit" not in df.columns or "WeightedAvgOvertakes" not in df.columns:
        return {}

    return df.set_index("Circuit")["WeightedAvgOvertakes"].to_dict()

# Weighted average overtakes per circuit used for model training
OVERTAKE_AVERAGES = _load_overtake_stats()

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

# Basic circuit metadata used when FastF1 does not provide information.
# Values are approximate and measured in seconds for ``StdLapTime``.
CIRCUIT_METADATA = {
    'Bahrain Grand Prix': {'NumCorners': 15, 'DRSZones': 3, 'StdLapTime': 92},
    'Saudi Arabian Grand Prix': {'NumCorners': 27, 'DRSZones': 3, 'StdLapTime': 90},
    'Australian Grand Prix': {'NumCorners': 14, 'DRSZones': 4, 'StdLapTime': 80},
    'Japanese Grand Prix': {'NumCorners': 18, 'DRSZones': 1, 'StdLapTime': 93},
    'Chinese Grand Prix': {'NumCorners': 16, 'DRSZones': 2, 'StdLapTime': 95},
    'Miami Grand Prix': {'NumCorners': 19, 'DRSZones': 3, 'StdLapTime': 90},
    'Emilia Romagna Grand Prix': {'NumCorners': 19, 'DRSZones': 2, 'StdLapTime': 75},
    'Monaco Grand Prix': {'NumCorners': 19, 'DRSZones': 1, 'StdLapTime': 72},
    'Canadian Grand Prix': {'NumCorners': 14, 'DRSZones': 2, 'StdLapTime': 70},
    'Spanish Grand Prix': {'NumCorners': 14, 'DRSZones': 2, 'StdLapTime': 78},
    'Austrian Grand Prix': {'NumCorners': 10, 'DRSZones': 3, 'StdLapTime': 65},
    'British Grand Prix': {'NumCorners': 18, 'DRSZones': 2, 'StdLapTime': 85},
    'Hungarian Grand Prix': {'NumCorners': 14, 'DRSZones': 2, 'StdLapTime': 76},
    'Belgian Grand Prix': {'NumCorners': 19, 'DRSZones': 2, 'StdLapTime': 112},
    'Dutch Grand Prix': {'NumCorners': 14, 'DRSZones': 2, 'StdLapTime': 72},
    'Italian Grand Prix': {'NumCorners': 11, 'DRSZones': 2, 'StdLapTime': 79},
    'Azerbaijan Grand Prix': {'NumCorners': 20, 'DRSZones': 2, 'StdLapTime': 100},
    'Singapore Grand Prix': {'NumCorners': 19, 'DRSZones': 3, 'StdLapTime': 103},
    'United States Grand Prix': {'NumCorners': 20, 'DRSZones': 2, 'StdLapTime': 90},
    'Mexican Grand Prix': {'NumCorners': 17, 'DRSZones': 3, 'StdLapTime': 77},
    'Brazilian Grand Prix': {'NumCorners': 15, 'DRSZones': 2, 'StdLapTime': 72},
    'Las Vegas Grand Prix': {'NumCorners': 17, 'DRSZones': 2, 'StdLapTime': 95},
    'Qatar Grand Prix': {'NumCorners': 16, 'DRSZones': 2, 'StdLapTime': 81},
    'Abu Dhabi Grand Prix': {'NumCorners': 16, 'DRSZones': 2, 'StdLapTime': 85},
}

# Approximate circuit coordinates used for live weather lookups
CIRCUIT_COORDS = {
    'Bahrain Grand Prix': (26.0325, 50.5106),
    'Saudi Arabian Grand Prix': (21.6319, 39.1044),
    'Australian Grand Prix': (-37.8497, 144.9680),
    'Japanese Grand Prix': (34.8431, 136.5419),
    'Chinese Grand Prix': (31.3389, 121.2197),
    'Miami Grand Prix': (25.9581, -80.2389),
    'Emilia Romagna Grand Prix': (44.3439, 11.7167),
    'Monaco Grand Prix': (43.7347, 7.4200),
    'Canadian Grand Prix': (45.5000, -73.5228),
    'Spanish Grand Prix': (41.5700, 2.2600),
    'Austrian Grand Prix': (47.2197, 14.7647),
    'British Grand Prix': (52.0786, -1.0169),
    'Hungarian Grand Prix': (47.5789, 19.2486),
    'Belgian Grand Prix': (50.4372, 5.9713),
    'Dutch Grand Prix': (52.3889, 4.5400),
    'Italian Grand Prix': (45.6156, 9.2811),
    'Azerbaijan Grand Prix': (40.3725, 49.8533),
    'Singapore Grand Prix': (1.2914, 103.8630),
    'United States Grand Prix': (30.1328, -97.6411),
    'Mexican Grand Prix': (19.4042, -99.0906),
    'Brazilian Grand Prix': (-23.7036, -46.6997),
    'Las Vegas Grand Prix': (36.1215, -115.1694),
    'Qatar Grand Prix': (25.4900, 51.4540),
    'Abu Dhabi Grand Prix': (24.4672, 54.6030),
}


def fetch_weather(circuit: str, api_key: str | None = None) -> dict | None:
    """Return a short-range weather forecast for the given circuit.

    Parameters
    ----------
    circuit : str
        Circuit name matching keys in ``CIRCUIT_COORDS``.
    api_key : str | None
        OpenWeatherMap API key. If ``None`` the ``OPENWEATHER_API_KEY``
        environment variable is used.

    Returns
    -------
    dict | None
        Dictionary with ``ForecastAirTemp`` and ``ForecastPrecipChance`` or
        ``None`` if the request fails.
    """
    api_key = api_key or os.getenv("d2bc9fb8d94c258d06149c087ccd4892")
    coords = CIRCUIT_COORDS.get(circuit)
    if not api_key or not coords:
        return None
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"lat": coords[0], "lon": coords[1], "appid": api_key, "units": "metric"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        temps = []
        pops = []
        for item in data.get("list", []):
            main = item.get("main", {})
            if "temp" in main:
                temps.append(main["temp"])
            pops.append(item.get("pop", 0))
        if temps:
            air = float(np.mean(temps))
            rain = float(np.max(pops)) if pops else 0.0
            return {
                "ForecastAirTemp": air,
                "ForecastPrecipChance": rain,
            }
    except Exception as err:  # pragma: no cover - network call
        print(f"⚠️ Failed to fetch weather: {err}")
    return None


def _clean_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned historical dataset for model training.

    This removes duplicated driver entries per race and clips grid/finish
    positions to the valid range of 1-20. Rows without a driver number are
    dropped as they cannot be matched to future events.
    """
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["Season", "RaceNumber", "DriverNumber"])
    df = df[df["DriverNumber"].notna()]
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")
    if "SprintFinish" in df.columns:
        df["SprintFinish"] = pd.to_numeric(df["SprintFinish"], errors="coerce")
    if "Status" in df.columns:
        df["DidNotFinish"] = df["Status"].str.lower() != "finished"
    else:
        df["DidNotFinish"] = df["Position"] > 20
    df["Position"] = df["Position"].clip(1, 20)
    df["GridPosition"] = df["GridPosition"].clip(1, 20)
    if "SprintFinish" in df.columns:
        df["SprintFinish"] = df["SprintFinish"].clip(1, 20)
    return df


def _load_historical_data(seasons, overtake_map=None):
    if overtake_map is None:
        overtake_map = OVERTAKE_AVERAGES
    race_data = []
    for season in seasons:
        # Build the list of rounds from the official event schedule so the
        # function adapts to seasons with a varying number of races.
        try:
            schedule = fastf1.get_event_schedule(season)
            rounds = schedule["RoundNumber"].dropna().unique()
        except Exception:
            # Skip the season entirely if the schedule cannot be retrieved
            continue
        for rnd in sorted(rounds):
            rnd = int(rnd)
            try:
                # Skip the round if data is missing (e.g. cancelled races)
                # Race session
                session = fastf1.get_session(season, rnd, 'R')
                session.load()

                results = session.results[['DriverNumber', 'Position', 'Points', 'GridPosition', 'Status']]
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

                # Weighted average overtake count for this circuit
                results['WeightedAvgOvertakes'] = overtake_map.get(
                    results['Circuit'].iloc[0], np.nan
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
                    # Preserve the official GridPosition from the race result.
                    # Only fill missing values using qualifying information.
                    if 'GridPosition' in results.columns:
                        results['GridPosition'] = results['GridPosition'].fillna(results['QualiPosition'])
                        results['GridPosition'] = results['GridPosition'].fillna(results['GridFromQ3'])
                except Exception:
                    results['BestQualiTime'] = np.nan
                    results['QualiPosition'] = np.nan

                # FP3 best lap and long run pace
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

                    laps = fp3_session.laps
                    if not laps.empty:
                        max_lap = laps['LapNumber'].max()
                        long_runs = laps[laps['LapNumber'] >= max_lap - 4]
                        long_avg = (
                            long_runs.groupby('DriverNumber')['LapTime']
                            .apply(lambda s: s.dt.total_seconds().mean())
                            .reset_index()
                            .rename(columns={'LapTime': 'FP3LongRunTime'})
                        )
                        best_laps = best_laps.merge(long_avg, on='DriverNumber', how='left')
                    else:
                        best_laps['FP3LongRunTime'] = np.nan

                    results = pd.merge(results, best_laps, on='DriverNumber', how='left')
                except Exception:
                    results['FP3BestTime'] = np.nan
                    results['FP3LongRunTime'] = np.nan

                # Sprint race finish position
                try:
                    sprint_session = fastf1.get_session(season, rnd, 'S')
                    sprint_session.load()
                    sprint_res = sprint_session.results[['DriverNumber', 'Position']].rename(
                        columns={'Position': 'SprintFinish'}
                    )
                    results = results.merge(sprint_res, on='DriverNumber', how='left')
                except Exception:
                    results['SprintFinish'] = np.nan

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
    full_data['Position'] = pd.to_numeric(full_data.get('Position'), errors='coerce').fillna(25)
    full_data['GridPosition'] = pd.to_numeric(
        full_data.get('GridPosition'), errors='coerce'
    ).fillna(20)
    full_data['GridPosition'] = full_data['GridPosition'].clip(1, 20)
    full_data['AirTemp'] = pd.to_numeric(full_data.get('AirTemp'), errors='coerce')
    full_data['TrackTemp'] = pd.to_numeric(full_data.get('TrackTemp'), errors='coerce')
    full_data['Rainfall'] = pd.to_numeric(full_data.get('Rainfall'), errors='coerce')
    full_data['WeightedAvgOvertakes'] = pd.to_numeric(
        full_data.get('WeightedAvgOvertakes'), errors='coerce'
    )
    full_data['BestQualiTime'] = pd.to_numeric(full_data.get('BestQualiTime'), errors='coerce')
    full_data['QualiPosition'] = pd.to_numeric(full_data.get('QualiPosition'), errors='coerce')
    full_data['FP3BestTime'] = pd.to_numeric(full_data.get('FP3BestTime'), errors='coerce')
    full_data['FP3LongRunTime'] = pd.to_numeric(full_data.get('FP3LongRunTime'), errors='coerce')
    full_data['SprintFinish'] = pd.to_numeric(full_data.get('SprintFinish'), errors='coerce').fillna(25)
    full_data['SprintFinish'] = full_data['SprintFinish'].clip(1, 25)
    if 'GridDropCount' in full_data.columns:
        full_data['GridDropCount'] = pd.to_numeric(full_data['GridDropCount'], errors='coerce').fillna(0)
    else:
        full_data['GridDropCount'] = 0

    # Convert qualifying session columns if present
    if 'Q1Time' not in full_data.columns and 'Q1' in full_data.columns:
        full_data['Q1Time'] = pd.to_timedelta(full_data['Q1'], errors='coerce').dt.total_seconds()
    if 'Q3Time' not in full_data.columns and 'Q3' in full_data.columns:
        full_data['Q3Time'] = pd.to_timedelta(full_data['Q3'], errors='coerce').dt.total_seconds()
    if 'Q1Time' in full_data.columns:
        full_data['Q1Time'] = pd.to_numeric(full_data['Q1Time'], errors='coerce')
    if 'Q3Time' in full_data.columns:
        full_data['Q3Time'] = pd.to_numeric(full_data['Q3Time'], errors='coerce')

    # Determine DNFs if not already provided
    if 'DidNotFinish' not in full_data.columns:
        if 'Status' in full_data.columns:
            full_data['DidNotFinish'] = full_data['Status'].str.lower() != 'finished'
        else:
            full_data['DidNotFinish'] = full_data['Position'] > 20

    # Delta to fastest qualifier in the event
    if 'BestQualiTime' in full_data.columns:
        event_fastest = (
            full_data.groupby(['Season', 'RaceNumber'])['BestQualiTime']
            .transform('min')
        )
        full_data['DeltaToBestQuali'] = full_data['BestQualiTime'] - event_fastest
        def _delta_next(g):
            ordered = g.sort_values('BestQualiTime')
            delta = ordered['BestQualiTime'].diff(-1).abs()
            return delta.reindex(g.index)
        full_data['DeltaToNext'] = (
            full_data.groupby(['Season', 'RaceNumber'], group_keys=False)
            .apply(_delta_next)
            .reindex(full_data.index)
        )
    else:
        full_data['DeltaToBestQuali'] = np.nan
        full_data['DeltaToNext'] = np.nan

    # Delta to team mate qualifying time
    if 'BestQualiTime' in full_data.columns:
        team_mean_q = full_data.groupby(['Season', 'RaceNumber', 'HistoricalTeam'])['BestQualiTime'].transform('mean')
        team_size = full_data.groupby(['Season', 'RaceNumber', 'HistoricalTeam'])['BestQualiTime'].transform('size')
        full_data['DeltaToTeammateQuali'] = np.where(team_size > 1, (full_data['BestQualiTime'] - team_mean_q) * 2, np.nan)
    else:
        full_data['DeltaToTeammateQuali'] = np.nan


    # Qualifying session gain (Q3 - Q1) normalized per race
    if 'Q1Time' in full_data.columns and 'Q3Time' in full_data.columns:
        full_data['QualiSessionGain'] = full_data['Q1Time'] - full_data['Q3Time']
        full_data['QualiSessionGain'] = full_data.groupby(['Season', 'RaceNumber'])['QualiSessionGain'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
    else:
        full_data['QualiSessionGain'] = np.nan

    full_data.sort_values(['DriverNumber', 'Season', 'RaceNumber'], inplace=True)
    full_data['ExperienceCount'] = full_data.groupby('DriverNumber').cumcount() + 1
    full_data['IsRookie'] = (full_data['ExperienceCount'] == 1).astype(int)
    full_data['RaceIdxInSeason'] = (
        full_data.groupby(['Season', 'DriverNumber']).cumcount() + 1
    )
    full_data['CrossAvgFinish'] = (
        full_data.groupby('DriverNumber')['Position']
        .apply(lambda s: s.shift().rolling(window=5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    rookie_mean = full_data.loc[full_data['IsRookie'] == 1, 'Position'].mean()
    overall_mean = full_data['Position'].mean()
    full_data.loc[full_data['IsRookie'] == 1, 'CrossAvgFinish'] = (
        full_data.loc[full_data['IsRookie'] == 1, 'CrossAvgFinish'].fillna(rookie_mean)
    )
    full_data['CrossAvgFinish'] = full_data['CrossAvgFinish'].fillna(overall_mean)
    full_data['Recent3AvgFinish'] = (
        full_data.groupby('DriverNumber')['Position']
        .apply(lambda s: s.shift().rolling(window=3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    full_data['Recent3AvgFinish'] = full_data['Recent3AvgFinish'].fillna(full_data['Position'].mean())
    full_data['Recent5AvgFinish'] = (
        full_data.groupby('DriverNumber')['Position']
        .apply(lambda s: s.shift().rolling(window=5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    full_data['Recent5AvgFinish'] = full_data['Recent5AvgFinish'].fillna(full_data['Position'].mean())
    full_data['RecentAvgPoints'] = (
        full_data.groupby('DriverNumber')['Points']
        .apply(lambda s: s.shift().rolling(window=5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    full_data['RecentAvgPoints'] = full_data['RecentAvgPoints'].fillna(0)

    driver_track = full_data.groupby(['DriverNumber', 'Circuit']).agg(
        DriverAvgTrackFinish=('Position', 'mean'),
        DriverTrackPodiums=('Position', lambda x: (x <= 3).sum()),
        DriverTrackDNFs=('DidNotFinish', 'sum'),
    ).reset_index()
    full_data = pd.merge(full_data, driver_track, on=['DriverNumber', 'Circuit'], how='left')

    full_data['TeamRecentQuali'] = (
        full_data.groupby(['HistoricalTeam', 'Season'])['QualiPosition']
        .apply(lambda s: s.shift().rolling(window=5, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    full_data['TeamRecentFinish'] = (
        full_data.groupby(['HistoricalTeam', 'Season'])['Position']
        .apply(lambda s: s.shift().rolling(window=5, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    full_data['TeamReliability'] = (
        full_data.groupby(['HistoricalTeam', 'Season'])['DidNotFinish']
        .apply(lambda s: s.shift().rolling(window=5, min_periods=1).sum())
        .reset_index(level=[0, 1], drop=True)
    )
    full_data['TeamRecentQuali'] = full_data['TeamRecentQuali'].fillna(full_data['QualiPosition'].mean())
    full_data['TeamRecentFinish'] = full_data['TeamRecentFinish'].fillna(full_data['Position'].mean())
    full_data['TeamReliability'] = full_data['TeamReliability'].fillna(0)

    # Championship standings up to the previous round
    full_data.sort_values(['Season', 'RaceNumber'], inplace=True)
    full_data['DriverChampPoints'] = (
        full_data.groupby(['Season', 'DriverNumber'])['Points']
        .cumsum()
        .shift()
    )
    full_data['ConstructorChampPoints'] = (
        full_data.groupby(['Season', 'HistoricalTeam'])['Points']
        .cumsum()
        .shift()
    )
    full_data['DriverChampPoints'] = full_data['DriverChampPoints'].fillna(0)
    full_data['ConstructorChampPoints'] = full_data['ConstructorChampPoints'].fillna(0)
    full_data['DriverStanding'] = (
        full_data.groupby(['Season', 'RaceNumber'])['DriverChampPoints']
        .rank(method='dense', ascending=False)
    )
    full_data['ConstructorStanding'] = (
        full_data.groupby(['Season', 'RaceNumber'])['ConstructorChampPoints']
        .rank(method='dense', ascending=False)
    )

    # Previous season constructor ranking to provide baseline team strength
    prev_year_map = {}
    for yr in sorted(full_data['Season'].unique()):
        prev = full_data[full_data['Season'] == yr - 1]
        if prev.empty:
            continue
        final_pts = prev.groupby('HistoricalTeam')['ConstructorChampPoints'].max()
        rank = final_pts.rank(method='dense', ascending=False)
        prev_year_map[yr] = rank.to_dict()
    full_data['PrevYearConstructorRank'] = full_data.apply(
        lambda r: prev_year_map.get(r['Season'], {}).get(r['HistoricalTeam'], np.nan),
        axis=1
    )


    # Circuit metadata from FastF1
    try:
        from fastf1.circuit_info import get_circuit_info
        circuit_lengths = {}
        corners_map = {}
        drs_map = {}
        lap_map = {}
        for circ in full_data['Circuit'].unique():
            try:
                info = get_circuit_info(circ)
                length = (
                    info.get('Length')
                    or info.get('CircuitLength')
                    or info.get('circuitLength')
                )
                corners = (
                    info.get('NumberOfTurns')
                    or info.get('Turns')
                    or info.get('numCorners')
                )
                drs = (
                    info.get('NumberOfDRSZones')
                    or info.get('DRSZonesCount')
                    or info.get('drsZones')
                )
                laptime = (
                    info.get('LapTimeAvg')
                    or info.get('LapRecord')
                    or info.get('lapRecord')
                )
                if isinstance(length, str):
                    length = str(length).replace(' km', '')
                if isinstance(laptime, str):
                    try:
                        laptime = pd.to_timedelta(laptime).total_seconds()
                    except Exception:
                        laptime = pd.to_numeric(laptime, errors='coerce')
                circuit_lengths[circ] = pd.to_numeric(length, errors='coerce')
                corners_map[circ] = pd.to_numeric(corners, errors='coerce')
                drs_map[circ] = pd.to_numeric(drs, errors='coerce')
                lap_map[circ] = pd.to_numeric(laptime, errors='coerce')
            except Exception:
                meta = CIRCUIT_METADATA.get(circ, {})
                circuit_lengths[circ] = np.nan
                corners_map[circ] = meta.get('NumCorners', np.nan)
                drs_map[circ] = meta.get('DRSZones', np.nan)
                lap_map[circ] = meta.get('StdLapTime', np.nan)
        full_data['CircuitLength'] = full_data['Circuit'].map(circuit_lengths)
        full_data['NumCorners'] = full_data['Circuit'].map(corners_map)
        full_data['DRSZones'] = full_data['Circuit'].map(drs_map)
        full_data['StdLapTime'] = full_data['Circuit'].map(lap_map)
    except Exception:
        full_data['CircuitLength'] = full_data['Circuit'].map(
            lambda c: CIRCUIT_METADATA.get(c, {}).get('CircuitLength', np.nan)
        )
        full_data['NumCorners'] = full_data['Circuit'].map(
            lambda c: CIRCUIT_METADATA.get(c, {}).get('NumCorners', np.nan)
        )
        full_data['DRSZones'] = full_data['Circuit'].map(
            lambda c: CIRCUIT_METADATA.get(c, {}).get('DRSZones', np.nan)
        )
        full_data['StdLapTime'] = full_data['Circuit'].map(
            lambda c: CIRCUIT_METADATA.get(c, {}).get('StdLapTime', np.nan)
        )

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

    full_data['TeamAvgPosition'] = (
        full_data.groupby(['HistoricalTeam', 'Season'])['Position']
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    full_data['TeamAvgPosition'] = full_data['TeamAvgPosition'].fillna(
        full_data['TeamAvgPosition'].mean())

    full_data['AirTemp'] = full_data['AirTemp'].fillna(full_data['AirTemp'].mean())
    full_data['TrackTemp'] = full_data['TrackTemp'].fillna(full_data['TrackTemp'].mean())

    # Flag missing rainfall values and impute using the circuit-specific median
    # before falling back to the global median. This avoids biasing all missing
    # values toward a dry event while still providing a sensible default.
    full_data['RainfallMissing'] = full_data['Rainfall'].isna().astype(int)
    circuit_rain = full_data.groupby('Circuit')['Rainfall'].transform('median')
    full_data['Rainfall'] = full_data['Rainfall'].fillna(circuit_rain)
    full_data['Rainfall'] = full_data['Rainfall'].fillna(full_data['Rainfall'].median())
    full_data['WeightedAvgOvertakes'] = full_data['WeightedAvgOvertakes'].fillna(
        full_data['WeightedAvgOvertakes'].mean()
    )

    # Preserve information about missing qualifying times so the model can
    # learn that a driver failed to set a lap.
    full_data['MissedQuali'] = full_data['BestQualiTime'].isna().astype(int)
    full_data['BestQualiTime'] = full_data['BestQualiTime'].fillna(
        full_data['BestQualiTime'].median()
    )
    full_data['QualiPosition'] = full_data['QualiPosition'].fillna(20)
    full_data['FP3BestTime'] = full_data['FP3BestTime'].fillna(full_data['FP3BestTime'].mean())
    full_data['FP3LongRunTime'] = full_data['FP3LongRunTime'].fillna(full_data['FP3LongRunTime'].mean())
    full_data['SprintFinish'] = full_data['SprintFinish'].fillna(25)
    full_data['IsStreet'] = full_data['IsStreet'].fillna(0)
    full_data['DownforceLevel'] = full_data['DownforceLevel'].fillna(1)
    full_data['GridDropCount'] = full_data['GridDropCount'].fillna(0)
    full_data['DeltaToBestQuali'] = full_data['DeltaToBestQuali'].fillna(full_data['DeltaToBestQuali'].mean())
    full_data['DeltaToNext'] = full_data['DeltaToNext'].fillna(full_data['DeltaToNext'].mean())
    # ``0`` can indicate a perfect tie with the team mate. Replace missing
    # values with the median difference so the model does not interpret them
    # as exceptionally good laps.
    full_data['DeltaToTeammateQuali'] = full_data['DeltaToTeammateQuali'].fillna(
        full_data['DeltaToTeammateQuali'].median())
    full_data['QualiSessionGain'] = full_data['QualiSessionGain'].fillna(
        full_data['QualiSessionGain'].median())
    full_data['DidNotFinish'] = full_data['DidNotFinish'].fillna(False)
    full_data['CircuitLength'] = full_data['CircuitLength'].fillna(full_data['CircuitLength'].mean())
    full_data['NumCorners'] = full_data['NumCorners'].fillna(full_data['NumCorners'].median())
    full_data['DRSZones'] = full_data['DRSZones'].fillna(full_data['DRSZones'].median())
    full_data['StdLapTime'] = full_data['StdLapTime'].fillna(full_data['StdLapTime'].mean())
    full_data['DriverChampPoints'] = full_data['DriverChampPoints'].fillna(0)
    full_data['ConstructorChampPoints'] = full_data['ConstructorChampPoints'].fillna(0)
    full_data['DriverStanding'] = full_data['DriverStanding'].fillna(full_data['DriverStanding'].max())
    full_data['ConstructorStanding'] = full_data['ConstructorStanding'].fillna(full_data['ConstructorStanding'].max())
    full_data['PrevYearConstructorRank'] = full_data['PrevYearConstructorRank'].fillna(
        full_data['PrevYearConstructorRank'].max()
    )

    return full_data


def _prepare_features(
    full_data,
    base_cols,
    team_encoder=None,
    circuit_encoder=None,
    top_circuits=None,
    top_teams=None,
):
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
        Subset of circuit columns to keep. If ``None`` the 10 most frequent
        circuits are used.
    top_teams : list or None
        Subset of team columns to keep. If ``None`` the 12 most frequent teams
        are used.

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
    top_teams : list
        Names of the team columns that were kept.
    """

    full_data = full_data.copy()

    # Prefer the historic team name when both a current and historical team
    # column are present.  This avoids leaking 2025 line-up information into the
    # training features.
    if "HistoricalTeam" in full_data.columns:
        full_data["Team"] = full_data["HistoricalTeam"]

    if full_data.empty:
        team_cols = (
            team_encoder.get_feature_names_out(["TeamGrp"]).tolist()
            if team_encoder
            else []
        )
        team_cols = [c.replace("TeamGrp_", "Team_") for c in team_cols]
        circuit_cols = (
            circuit_encoder.get_feature_names_out(["CircuitGrp"]).tolist()
            if circuit_encoder
            else []
        )
        circuit_cols = [c.replace("CircuitGrp_", "Circuit_") for c in circuit_cols]
        if top_circuits is not None:
            circuit_cols = [c for c in circuit_cols if c in top_circuits]
        empty_cols = base_cols + team_cols + circuit_cols
        return (
            pd.DataFrame(columns=empty_cols),
            team_encoder,
            circuit_encoder,
            top_circuits,
            top_teams,
        )

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
            full_data[col] = np.nan
        full_data[col] = pd.to_numeric(full_data[col], errors="coerce")
    # Fill remaining NaNs with the column median to reduce the impact of
    # missing values on the model while keeping feature scales realistic.
    full_data[base_cols] = full_data[base_cols].fillna(full_data[base_cols].median())

    if top_teams is None:
        top_teams = (
            full_data["Team"].value_counts().nlargest(12).index.tolist()
        )

    full_data["TeamGrp"] = np.where(
        full_data["Team"].isin(top_teams), full_data["Team"], "Other"
    )

    if team_encoder is None:
        team_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        team_encoded = team_encoder.fit_transform(full_data[["TeamGrp"]])
    else:
        team_encoded = team_encoder.transform(full_data[["TeamGrp"]])
    team_df = pd.DataFrame(
        team_encoded, columns=team_encoder.get_feature_names_out(["TeamGrp"])
    )
    team_df.columns = [c.replace("TeamGrp_", "Team_") for c in team_df.columns]

    if top_circuits is None:
        top_circuits = (
            full_data["Circuit"].value_counts().nlargest(10).index.tolist()
        )

    full_data["CircuitGrp"] = np.where(
        full_data["Circuit"].isin(top_circuits), full_data["Circuit"], "Other"
    )

    if circuit_encoder is None:
        circuit_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        circuit_encoded = circuit_encoder.fit_transform(full_data[["CircuitGrp"]])
    else:
        circuit_encoded = circuit_encoder.transform(full_data[["CircuitGrp"]])
    circuit_df = pd.DataFrame(
        circuit_encoded, columns=circuit_encoder.get_feature_names_out(["CircuitGrp"])
    )
    circuit_df.columns = [c.replace("CircuitGrp_", "Circuit_") for c in circuit_df.columns]

    if top_circuits is not None:
        circuit_cols = [f"Circuit_{c}" for c in top_circuits]
        if "Circuit_Other" in circuit_df.columns:
            circuit_cols.append("Circuit_Other")
        circuit_df = circuit_df.reindex(columns=circuit_cols, fill_value=0)

    features = pd.concat([
        full_data[base_cols].reset_index(drop=True),
        team_df.reset_index(drop=True),
        circuit_df.reset_index(drop=True)
    ], axis=1)

    return features, team_encoder, circuit_encoder, top_circuits, top_teams


def _encode_features(
    full_data,
    base_cols,
    team_encoder=None,
    circuit_encoder=None,
    top_circuits=None,
    top_teams=None,
):
    return _prepare_features(
        full_data,
        base_cols,
        team_encoder,
        circuit_encoder,
        top_circuits,
        top_teams,
    )


def _rank_metrics(actual: pd.Series, preds: np.ndarray) -> dict:
    """Compute ranking-focused metrics."""

    actual_series = pd.Series(actual).reset_index(drop=True)
    pred_series = pd.Series(preds)

    rho = spearmanr(actual_series, pred_series).correlation
    pred_order = pred_series.rank(method="first").sort_values().index
    actual_order = actual_series.rank(method="first").sort_values().index

    top1 = float(pred_order[0] == actual_order[0]) if len(pred_order) else 0.0
    if len(pred_order) >= 3 and len(actual_order) >= 3:
        top3 = len(set(pred_order[:3]) & set(actual_order[:3])) / 3.0
    else:
        top3 = 0.0

    return {"spearman": rho, "top1": top1, "top3": top3}


def _train_model(features, target, cv, debug=False):
    """Train an XGBoost ranker using Bayesian optimization to maximize Spearman correlation.

    Parameters
    ----------
    features : pd.DataFrame
        Training feature matrix.
    target : pd.Series
        Target values.
    cv : cross-validator
        Cross-validation splitter.
    debug : bool, optional
        When ``True`` plot the top 10 feature importances after fitting.

    Returns
    -------
    Tuple[XGBRanker, float]
        The trained model and the best cross-validation Spearman score.
    """

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),
            'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7, 9]),
            'learning_rate': trial.suggest_categorical(
                'learning_rate', [0.01, 0.05, 0.1, 0.2]
            ),
            'subsample': trial.suggest_categorical('subsample', [0.6, 0.8, 1.0]),
            'colsample_bytree': trial.suggest_categorical(
                'colsample_bytree', [0.6, 0.8, 1.0]
            ),
            'min_child_weight': trial.suggest_categorical(
                'min_child_weight', [1, 3, 5, 7, 10]
            ),
        }
        model = XGBRanker(
            objective="rank:pairwise",
            random_state=42,
            **params,
        )

        scores = []
        splits = cv.split(features) if hasattr(cv, "split") else cv
        for train_idx, val_idx in splits:
            train_feat = features.iloc[train_idx].reset_index(drop=True)
            val_feat = features.iloc[val_idx].reset_index(drop=True)
            train_groups = (
                train_feat.groupby(["Season", "RaceNumber"], sort=False)
                .size()
                .to_list()
            )
            val_groups = (
                val_feat.groupby(["Season", "RaceNumber"], sort=False)
                .size()
                .to_list()
            )
            model.fit(
                train_feat,
                target.iloc[train_idx].reset_index(drop=True),
                group=train_groups,
                eval_set=[(val_feat, target.iloc[val_idx].reset_index(drop=True))],
                eval_group=[val_groups],
                verbose=False,
                early_stopping_rounds=20,
            )
            preds = model.predict(val_feat)
            rho = spearmanr(target.iloc[val_idx].reset_index(drop=True), preds).correlation
            scores.append(rho)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60, show_progress_bar=False)
    best_params = study.best_params
    best_score = study.best_value
    model = XGBRanker(
        objective="rank:pairwise",
        random_state=42,
        **best_params,
    )
    full_group = (
        features.groupby(["Season", "RaceNumber"], sort=False).size().to_list()
    )
    model.fit(features, target, group=full_group)
    if debug:
        plot_importance(model, max_num_features=10)
        plt.show()
    return model, best_score


def predict_race(grand_prix, year=2025, export_details=False, debug=False, compute_overtakes=True):
    # Determine the round number up front so that training data for the current
    # season can be trimmed to only completed events. This avoids leaking
    # information from future rounds into season-to-date features.
    schedule = fastf1.get_event_schedule(year)
    match = schedule[schedule["EventName"].str.contains(grand_prix, case=False, na=False)]
    if match.empty:
        raise ValueError(f"Grand Prix '{grand_prix}' not found for {year}")
    this_race_number = int(match.iloc[0]["RoundNumber"])

    # Include the target ``year`` in the loaded dataset so any completed races
    # from the ongoing season contribute to championship standings.
    seasons = list(range(2020, year + 1))

    overtake_map = _load_overtake_stats()
    if compute_overtakes:
        try:
            years_for_avg = list(range(max(2020, year - 3), year))
            avg = average_overtakes(grand_prix, years_for_avg)
            overtake_map[grand_prix] = avg
        except Exception as err:
            print(f"⚠️ Could not compute overtakes for {grand_prix}: {err}")

    race_data = _load_historical_data(seasons, overtake_map)
    race_data = _clean_historical_data(race_data).reset_index(drop=True)
    # Ensure DriverNumber is numeric for consistent merging
    race_data['DriverNumber'] = pd.to_numeric(race_data['DriverNumber'], errors='coerce')
    # Placeholder for qualifying and FP3 results of the target event.
    qual_results = None
    fp3_results = None
    race_data = _add_driver_team_info(race_data, seasons)
    # Remove any existing current-season team column to avoid mixing with
    # historical information. ``_prepare_features`` will internally use the
    # ``HistoricalTeam`` column for encoding.
    race_data = race_data.drop(columns=['Team'], errors='ignore')
    # Drop any rows from the prediction season that occur on or after the target
    # event. This keeps rolling aggregates and championship points strictly
    # limited to completed rounds.
    race_data = race_data.loc[
        ~((race_data['Season'] == year) & (race_data['RaceNumber'] >= this_race_number))
    ].reset_index(drop=True)
    race_data = _engineer_features(race_data)
    # Ensure strict chronological order before creating the feature matrix.
    race_data.sort_values(["Season", "RaceNumber"], inplace=True)
    race_data.reset_index(drop=True, inplace=True)
    # Save the engineered dataset used for training and prediction so users can
    # inspect all input values.
    race_data.to_csv("prediction_data.csv", index=False)

    # Feature sets
    race_cols = [
        'GridPosition', 'Season', 'ExperienceCount', 'IsRookie', 'TeamAvgPosition',
        'CrossAvgFinish', 'RecentAvgPoints', 'BestQualiTime', 'MissedQuali',
        'QualiPosition', 'FP3BestTime', 'FP3LongRunTime', 'DeltaToBestQuali',
        'DeltaToNext', 'SprintFinish',
        'Recent3AvgFinish', 'Recent5AvgFinish', 'DriverAvgTrackFinish',
        'DriverTrackPodiums', 'DriverTrackDNFs', 'TeamRecentQuali',
        'TeamRecentFinish', 'TeamReliability',
        'DriverChampPoints', 'ConstructorChampPoints',
        'DriverStanding', 'ConstructorStanding', 'PrevYearConstructorRank',
        'CircuitLength', 'NumCorners', 'DRSZones', 'StdLapTime',
        'IsStreet', 'DownforceLevel',
        'AirTemp', 'TrackTemp', 'Rainfall', 'RainfallMissing', 'WeightedAvgOvertakes'
    ]

    # Hold-out evaluation on the last completed season
    holdout_year = year - 1
    holdout_df = race_data[race_data['Season'] == holdout_year]
    train_df = race_data[race_data['Season'] < holdout_year]
    holdout_mae = None
    holdout_rank = None
    if not holdout_df.empty and not train_df.empty:
        (
            ho_feat,
            ho_team_enc,
            ho_circ_enc,
            ho_top_circuits,
            ho_top_teams,
        ) = _encode_features(train_df, race_cols)
        ho_val_feat, _, _, _, _ = _encode_features(
            holdout_df,
            race_cols,
            ho_team_enc,
            ho_circ_enc,
            ho_top_circuits,
            ho_top_teams,
        )
        ho_cv = SeasonSplit(sorted(train_df['Season'].unique()))
        ho_model, _ = _train_model(ho_feat, train_df['Position'], ho_cv, debug)
        ho_preds = ho_model.predict(ho_val_feat)
        holdout_mae = mean_absolute_error(holdout_df['Position'], ho_preds)
        holdout_rank = _rank_metrics(holdout_df['Position'], ho_preds)
    # Encode features for the race model
    (
        features,
        team_enc,
        circuit_enc,
        top_circuits,
        top_teams,
    ) = _encode_features(race_data, race_cols)
    # Align feature order with the chronological race data just to be safe
    features = features.loc[race_data.index].reset_index(drop=True)

    cv = SeasonSplit(sorted(race_data['Season'].unique()))

    # Train race finish model
    target = race_data['Position']
    model, cv_rho = _train_model(features, target, cv, debug)

    finish_preds_hist = model.predict(features)
    finish_mae = mean_absolute_error(race_data['Position'], finish_preds_hist)
    train_rank = _rank_metrics(race_data['Position'], finish_preds_hist)

    # Now fetch the driver line-up and session data for the target event
    try:
        drivers_df = _get_qualifying_results(year, grand_prix)
        drivers_df = drivers_df[drivers_df['BestTime'].notna()]
        qual_results = drivers_df.copy()
    except Exception:
        drivers_df = _get_event_drivers(year, grand_prix)
        qual_results = None

    drivers_df['DriverNumber'] = pd.to_numeric(drivers_df['DriverNumber'], errors='coerce')

    # Retrieve FP3 results now so default values include real session data if
    # available.
    try:
        fp3_results = _get_fp3_results(year, grand_prix)
        if qual_results is not None:
            qual_results = qual_results.merge(fp3_results, on='Abbreviation', how='left')
    except Exception:
        fp3_results = None

    if qual_results is not None and not qual_results.empty:
        default_best_q = qual_results['BestTime'].median()
        default_delta_next = qual_results['DeltaToNext'].mean()
    else:
        default_best_q = race_data['BestQualiTime'].median()
        default_delta_next = race_data['DeltaToNext'].mean()

    if fp3_results is not None and not fp3_results.empty:
        default_air = fp3_results['AvgAirTemp'].mean()
        default_track = fp3_results['AvgTrackTemp'].mean()
        default_rain = fp3_results['MaxRainfall'].max()
        default_fp3 = fp3_results['FP3BestTime'].mean()
        default_fp3_long = fp3_results['FP3LongRunTime'].mean()
    else:
        forecast = fetch_weather(grand_prix)
        hist_air = race_data['AirTemp'].mean()
        hist_track = race_data['TrackTemp'].mean()
        hist_rain = race_data['Rainfall'].median()
        default_fp3 = race_data['FP3BestTime'].mean()
        default_fp3_long = race_data['FP3LongRunTime'].mean()
        if forecast:
            f_air = forecast['ForecastAirTemp']
            f_track = f_air + 10  # rough approximation
            f_rain = forecast['ForecastPrecipChance']
            default_air = (f_air + hist_air) / 2
            default_track = (f_track + hist_track) / 2
            default_rain = (f_rain + hist_rain) / 2
        else:
            default_air = hist_air
            default_track = hist_track
            default_rain = hist_rain
    default_overtake = race_data['WeightedAvgOvertakes'].mean()

    # ``this_race_number`` was determined earlier so historical data could be
    # filtered accordingly.

    try:
        from fastf1.circuit_info import get_circuit_info
        circuit_lengths = {}
        corners_map = {}
        drs_map = {}
        lap_map = {}
        info = get_circuit_info(grand_prix)
        length = (
            info.get('Length')
            or info.get('CircuitLength')
            or info.get('circuitLength')
        )
        corners = (
            info.get('NumberOfTurns')
            or info.get('Turns')
            or info.get('numCorners')
        )
        drs = (
            info.get('NumberOfDRSZones')
            or info.get('DRSZonesCount')
            or info.get('drsZones')
        )
        laptime = (
            info.get('LapTimeAvg')
            or info.get('LapRecord')
            or info.get('lapRecord')
        )
        if isinstance(length, str):
            length = str(length).replace(' km', '')
        if isinstance(laptime, str):
            try:
                laptime = pd.to_timedelta(laptime).total_seconds()
            except Exception:
                laptime = pd.to_numeric(laptime, errors='coerce')
        circuit_lengths[grand_prix] = pd.to_numeric(length, errors='coerce')
        corners_map[grand_prix] = pd.to_numeric(corners, errors='coerce')
        drs_map[grand_prix] = pd.to_numeric(drs, errors='coerce')
        lap_map[grand_prix] = pd.to_numeric(laptime, errors='coerce')
    except Exception:
        meta = CIRCUIT_METADATA.get(grand_prix, {})
        circuit_lengths = {grand_prix: np.nan}
        corners_map = {grand_prix: meta.get('NumCorners', np.nan)}
        drs_map = {grand_prix: meta.get('DRSZones', np.nan)}
        lap_map = {grand_prix: meta.get('StdLapTime', np.nan)}

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
    df_level_map = {'low': 0, 'medium': 1, 'high': 2}
    track_type_val = TRACK_TYPE.get(grand_prix, 'permanent')
    is_street_val = 1 if track_type_val == 'street' else 0
    downforce_val = DOWNFORCE.get(grand_prix, 'medium')
    downforce_level_val = df_level_map.get(downforce_val, 1)
    circuit_length_val = circuit_lengths.get(grand_prix, np.nan)
    num_corners_val = corners_map.get(grand_prix, np.nan)
    drs_zones_val = drs_map.get(grand_prix, np.nan)
    std_lap_time_val = lap_map.get(grand_prix, np.nan)

    # Prepare prediction dataframe for all drivers
    pred_rows = []
    default_team_avg = race_data['Position'].mean()
    default_team_quali = race_data['QualiPosition'].mean()

    driver_stats_lookup = race_data.set_index(['DriverNumber', 'Circuit'])[
        ['DriverAvgTrackFinish', 'DriverTrackPodiums', 'DriverTrackDNFs']
    ]

    # Championship points and standings before the target event
    year_data = race_data[race_data['Season'] == year]
    if not year_data.empty:
        driver_pts = year_data.groupby('DriverNumber')['Points'].sum()
        constructor_pts = year_data.groupby('HistoricalTeam')['Points'].sum()
        driver_standings = driver_pts.rank(method='dense', ascending=False).astype(int)
        constructor_standings = constructor_pts.rank(method='dense', ascending=False).astype(int)
        driver_pts_map = driver_pts.to_dict()
        constructor_pts_map = constructor_pts.to_dict()
        driver_stand_map = driver_standings.to_dict()
        constructor_stand_map = constructor_standings.to_dict()
    else:
        driver_pts_map = {}
        constructor_pts_map = {}
        driver_stand_map = {}
        constructor_stand_map = {}

    same_season_teams = race_data[race_data['Season'] == year]
    if not same_season_teams.empty:
        team_strength = (
            same_season_teams.groupby('HistoricalTeam')['Position']
            .mean()
            .to_dict()
        )
    else:
        team_strength = {}

    prev_year = year - 1
    prev_data = race_data[race_data['Season'] == prev_year]
    if not prev_data.empty:
        final_pts = prev_data.groupby('HistoricalTeam')['ConstructorChampPoints'].max()
        prev_rank = final_pts.rank(method='dense', ascending=False)
        prev_rank_map = prev_rank.to_dict()
        default_prev_rank = int(prev_rank.max())
    else:
        prev_rank_map = {}
        default_prev_rank = 0

    if qual_results is not None and not qual_results.empty:
        fastest = qual_results['BestTime'].min()
        qual_results['DeltaToBestQuali'] = qual_results['BestTime'] - fastest
        qual_results = qual_results.sort_values('BestTime')
        qual_results['DeltaToNext'] = qual_results['BestTime'].diff(-1).abs()
        qual_results = qual_results.sort_index()
        team_mean = qual_results.groupby('Team')['BestTime'].transform('mean')
        team_size = qual_results.groupby('Team')['BestTime'].transform('size')
        qual_results['DeltaToTeammateQuali'] = np.where(team_size > 1,
                                                       (qual_results['BestTime'] - team_mean) * 2,
                                                       0)
        if 'Q1' in qual_results.columns and 'Q3' in qual_results.columns:
            qual_results['QualiSessionGain'] = qual_results['Q1'] - qual_results['Q3']
            std = qual_results['QualiSessionGain'].std()
            qual_results['QualiSessionGain'] = (
                (qual_results['QualiSessionGain'] - qual_results['QualiSessionGain'].mean()) / std
            ) if std != 0 else 0
        else:
            qual_results['QualiSessionGain'] = 0
        qual_results['GridDropCount'] = 0

    driver_iter = qual_results if qual_results is not None and not qual_results.empty else drivers_df
    overall_avg_pos = race_data['Position'].mean()
    rookie_avg_pos = race_data[race_data['ExperienceCount'] == 1]['Position'].mean()
    for _, d in driver_iter.iterrows():
        exp_count = len(race_data[race_data['DriverNumber'] == d['DriverNumber']]) + 1
        team_name = d['Team']
        team_same_season = race_data[
            (race_data['HistoricalTeam'] == team_name) &
            (race_data['Season'] == year)
        ].sort_values('RaceNumber')
        team_prev = team_same_season[team_same_season['RaceNumber'] < this_race_number]

        if len(team_prev) > 0:
            team_avg_pos = team_prev['Position'].mean()
        else:
            team_avg_pos = team_strength.get(team_name, default_team_avg)

        team_prev_q = team_prev['QualiPosition']
        if len(team_prev_q) > 0:
            team_recent_q = team_prev_q.tail(5).mean()
        else:
            team_recent_q = default_team_quali

        team_prev_f = team_prev['Position']
        if len(team_prev_f) > 0:
            team_recent_f = team_prev_f.tail(5).mean()
        else:
            team_recent_f = default_team_avg

        team_prev_dnf = team_prev['DidNotFinish']
        if len(team_prev_dnf) > 0:
            team_rel = team_prev_dnf.tail(5).sum()
        else:
            team_rel = 0.0

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
            # Use the same defaults as the training data when qualifying
            # information is missing so the model sees a consistent
            # distribution during training and prediction.
            grid_pos = 20
            best_time = default_best_q

        if fp3_results is not None and 'FP3BestTime' in d and pd.notna(d['FP3BestTime']):
            fp3_time = d['FP3BestTime']
        else:
            fp3_time = default_fp3

        if fp3_results is not None and 'FP3LongRunTime' in d and pd.notna(d['FP3LongRunTime']):
            fp3_long_time = d['FP3LongRunTime']
        else:
            fp3_long_time = default_fp3_long

        driver_num = int(d['DriverNumber'])
        past_races = race_data[
            (race_data['DriverNumber'] == driver_num) & (
                (race_data['Season'] < year) |
                ((race_data['Season'] == year) & (race_data['RaceNumber'] < this_race_number))
            )
        ].sort_values(['Season', 'RaceNumber'])

        if past_races.empty:
            cross_avg = rookie_avg_pos
            recent_avg_pts = 0.0
            recent3_avg = rookie_avg_pos
            recent5_avg = rookie_avg_pos
        else:
            cross_avg = past_races['Position'].tail(5).mean()
            recent_avg_pts = past_races['Points'].tail(5).mean()
            recent3_avg = past_races['Position'].tail(3).mean()
            recent5_avg = past_races['Position'].tail(5).mean()
        pred_rows.append({
            'GridPosition': grid_pos,
            'Season': year,
            'ExperienceCount': exp_count,
            'IsRookie': 1 if exp_count == 1 else 0,
            'TeamAvgPosition': team_avg_pos,
            'CrossAvgFinish': cross_avg,
            'RecentAvgPoints': recent_avg_pts,
            'BestQualiTime': best_time,
            'QualiPosition': grid_pos,
            'FP3BestTime': fp3_time,
            'FP3LongRunTime': fp3_long_time,
            'DeltaToBestQuali': d.get('DeltaToBestQuali', 0),
            'DeltaToNext': d.get('DeltaToNext', default_delta_next),
            'SprintFinish': 25,
            'Recent3AvgFinish': recent3_avg,
            'Recent5AvgFinish': recent5_avg,
            'DriverAvgTrackFinish': avg_track,
            'DriverTrackPodiums': podiums,
            'DriverTrackDNFs': dnfs,
            'TeamRecentQuali': team_recent_q,
            'TeamRecentFinish': team_recent_f,
            'TeamReliability': team_rel,
            'DriverChampPoints': driver_pts_map.get(d['DriverNumber'], 0.0),
            'ConstructorChampPoints': constructor_pts_map.get(d['Team'], 0.0),
            'DriverStanding': int(driver_stand_map.get(d['DriverNumber'], 0)),
            'ConstructorStanding': int(constructor_stand_map.get(d['Team'], 0)),
            'PrevYearConstructorRank': prev_rank_map.get(team_name, default_prev_rank),
            'CircuitLength': circuit_length_val,
            'NumCorners': num_corners_val,
            'DRSZones': drs_zones_val,
            'StdLapTime': std_lap_time_val,
            'IsStreet': is_street_val,
            'DownforceLevel': downforce_level_val,
            'AirTemp': default_air,
            'TrackTemp': default_track,
            'Rainfall': default_rain,
            'WeightedAvgOvertakes': default_overtake,
            'Team': d['Team'],
            'FullName': d['FullName'],
            'Abbreviation': d['Abbreviation']
        })

    pred_df = pd.DataFrame(pred_rows)

    if pred_df.empty:
        raise ValueError(f"No driver data available for {year} {grand_prix}")

    # Fill missing qualifying-related columns using the same defaults
    # as the historical training data so feature scaling remains
    # consistent between training and prediction phases.
    pred_df['GridPosition'] = (
        pd.to_numeric(pred_df['GridPosition'], errors='coerce')
        .fillna(20)
    )
    pred_df['GridPosition'] = pred_df['GridPosition'].clip(1, 20)
    pred_df['QualiPosition'] = (
        pd.to_numeric(pred_df['QualiPosition'], errors='coerce')
        .fillna(20)
    )
    pred_df['BestQualiTime'] = pd.to_numeric(pred_df['BestQualiTime'], errors='coerce')
    pred_df['MissedQuali'] = pred_df['BestQualiTime'].isna().astype(int)
    pred_df['BestQualiTime'] = pred_df['BestQualiTime'].fillna(
        race_data['BestQualiTime'].median()
    )
    pred_df['FP3BestTime'] = (
        pd.to_numeric(pred_df['FP3BestTime'], errors='coerce')
        .fillna(race_data['FP3BestTime'].mean())
    )
    pred_df['FP3LongRunTime'] = (
        pd.to_numeric(pred_df['FP3LongRunTime'], errors='coerce')
        .fillna(race_data['FP3LongRunTime'].mean())
    )
    pred_df['SprintFinish'] = (
        pd.to_numeric(pred_df.get('SprintFinish'), errors='coerce')
        .fillna(25)
    )
    pred_df['NumCorners'] = (
        pd.to_numeric(pred_df['NumCorners'], errors='coerce')
        .fillna(race_data['NumCorners'].median())
    )
    pred_df['DRSZones'] = (
        pd.to_numeric(pred_df['DRSZones'], errors='coerce')
        .fillna(race_data['DRSZones'].median())
    )
    pred_df['StdLapTime'] = (
        pd.to_numeric(pred_df['StdLapTime'], errors='coerce')
        .fillna(race_data['StdLapTime'].mean())
    )

    pred_df['Rainfall'] = pd.to_numeric(pred_df['Rainfall'], errors='coerce')
    pred_df['RainfallMissing'] = pred_df['Rainfall'].isna().astype(int)
    circ_rain_map = race_data.groupby('Circuit')['Rainfall'].median()
    circuit_rain = circ_rain_map.get(grand_prix, race_data['Rainfall'].median())
    pred_df['Rainfall'] = pred_df['Rainfall'].fillna(circuit_rain)
    pred_df['Rainfall'] = pred_df['Rainfall'].fillna(race_data['Rainfall'].median())

    # Save the driver list so the raw input fed to the model can be inspected.
    pred_df.to_csv("prediction_input.csv", index=False)

    # Encode the prediction rows using the same helper as for training. This
    # converts numerical columns to proper dtypes and creates aligned one-hot
    # encoded team and circuit features.
    race_pred_features, _, _, _, _ = _encode_features(
        pred_df.assign(Circuit=grand_prix),
        race_cols,
        team_enc,
        circuit_enc,
        top_circuits,
        top_teams,
    )

    # Ensure the column order matches the training feature matrix. Missing
    # columns (possible if new teams or circuits appear) are filled with 0 so the
    # XGBoost model receives the expected input shape.
    pred_features = race_pred_features.reindex(columns=features.columns, fill_value=0)

    preds = model.predict(pred_features)
    shap_values = None
    if debug and shap is not None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pred_features)
        except Exception as err:
            print(f"\u26a0\ufe0f Could not compute SHAP values: {err}")
            shap_values = None

    results = pd.DataFrame({
        'Driver': pred_df['Abbreviation'],
        'Team': pred_df['Team'],
        'Grid': pred_df['GridPosition'],
        'Predicted_Position': preds
    })
    sort_idx = results['Predicted_Position'].argsort()
    results = results.iloc[sort_idx].reset_index(drop=True)
    if shap_values is not None:
        shap_values = shap_values[sort_idx]
    results['Final_Position'] = range(1, len(results) + 1)
    # Save the final ordered predictions for transparency
    results.to_csv("prediction_results.csv", index=False)
    if export_details:
        try:
            detail_path = export_race_details(year, grand_prix)
            print(f"📁 Saved session data to {detail_path}")
        except Exception as err:
            print(f"⚠️ Could not export session data: {err}")
    details = None
    if debug:
        details = {
            'prediction_features': pred_features,
            'shap_values': shap_values,
            'feature_names': pred_features.columns.tolist(),
        }

    if holdout_mae is not None:
        print(
            f"📊 CV Spearman: {cv_rho:.2f} -- Hold-out MAE: {holdout_mae:.2f} -- Training MAE: {finish_mae:.2f}"
        )
        print(
            f"📈 Spearman \u03c1: {train_rank['spearman']:.2f} (train) / {holdout_rank['spearman']:.2f} (hold-out) "
            f"-- Top1: {train_rank['top1']*100:.0f}% / {holdout_rank['top1']*100:.0f}% "
            f"-- Top3: {train_rank['top3']*100:.0f}% / {holdout_rank['top3']*100:.0f}%"
        )
    else:
        print(f"📊 CV Spearman: {cv_rho:.2f} -- Training MAE: {finish_mae:.2f}")
        print(
            f"📈 Spearman \u03c1: {train_rank['spearman']:.2f} -- "
            f"Top1: {train_rank['top1']*100:.0f}% -- Top3: {train_rank['top3']*100:.0f}%"
        )
    return (results, details) if debug else results


if __name__ == '__main__':
    res = predict_race('Chinese Grand Prix', year=2025, export_details=True, debug=False)
    print(res[['Driver', 'Team', 'Grid', 'Final_Position']].head())
