from os import makedirs
import os
import logging
from warnings import filterwarnings

from fastf1 import Cache, get_event_schedule, get_session
import requests
from numpy import nan, mean, max, where
import pandas as pd
from pandas import DataFrame, Series, to_numeric, to_timedelta, read_csv
from sklearn.preprocessing import OneHotEncoder

from export_race_details import _fetch_session_data

logger = logging.getLogger(__name__)

filterwarnings('ignore')

# Enable caching for FastF1
CACHE_DIR = 'f1_cache'
makedirs(CACHE_DIR, exist_ok=True)
Cache.enable_cache(CACHE_DIR)


def _get_event_drivers(year: int, grand_prix: str) -> pd.DataFrame:
    """Return the driver lineup for a given event using FastF1."""
    schedule = get_event_schedule(year)
    match = schedule[schedule["EventName"].str.contains(grand_prix, case=False, na=False)]
    if match.empty:
        raise ValueError(f"Grand Prix '{grand_prix}' not found for {year}")

    round_number = int(match.iloc[0]["RoundNumber"])
    session = get_session(year, round_number, "R")

    # First try official race results which contain driver details
    try:
        session.load(telemetry=False, laps=False, weather=False)
        if hasattr(session, "results") and not session.results.empty:
            info = session.results[["DriverNumber", "Abbreviation", "FullName", "TeamName"]].copy()
            info.rename(columns={"TeamName": "Team"}, inplace=True)
            return info
    except Exception:
        pass

    # Fallback to entry list from timing API
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

    # Final fallback to Ergast API
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
                rows.append({
                    "DriverNumber": int(r.get("number", 0)),
                    "Abbreviation": r["Driver"].get("code", ""),
                    "FullName": f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                    "Team": r["Constructor"].get("name", ""),
                })
            if rows:
                return pd.DataFrame(rows)
    except Exception:
        pass

    return pd.DataFrame(columns=["DriverNumber", "Abbreviation", "FullName", "Team"])


def _get_qualifying_results(year: int, grand_prix: str) -> pd.DataFrame:
    """Return qualifying results with driver abbreviations and times."""
    schedule = get_event_schedule(year)
    match = schedule[schedule["EventName"].str.contains(grand_prix, case=False, na=False)]
    if match.empty:
        raise ValueError(f"Grand Prix '{grand_prix}' not found for {year}")

    round_number = int(match.iloc[0]["RoundNumber"])
    session = get_session(year, round_number, "Q")
    session.load()
    q_res = session.results[[
        "DriverNumber", "Abbreviation", "FullName", "TeamName", "Position", "Q1", "Q2", "Q3"
    ]].copy()

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
    schedule = get_event_schedule(year)
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
    """Return weighted average overtake counts mapped by circuit name."""
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty or "Circuit" not in df.columns or "WeightedAvgOvertakes" not in df.columns:
        return {}
    return df.set_index("Circuit")["WeightedAvgOvertakes"].to_dict()


OVERTAKE_AVERAGES = _load_overtake_stats()

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

# Canonical feature columns used for model training
race_cols = [
    'Season', 'RaceNumber', 'DriverNumber', 'GridPosition',
    'BestQualiTime', 'DeltaToBestQuali', 'DeltaToNext',
    'DeltaToTeammateQuali', 'QualiSessionGain', 'GridDropCount',
    'FP3BestTime', 'FP3LongRunTime',
    'AirTemp', 'TrackTemp', 'Rainfall', 'MissedQuali',
    'SprintFinish', 'CrossAvgFinish', 'RecentAvgPoints',
    'Recent3AvgFinish', 'Recent5AvgFinish', 'DriverAvgTrackFinish',
    'DriverTrackPodiums', 'DriverTrackDNFs', 'IsRookie',
    'PrevYearConstructorRank', 'TeamRecentQuali', 'TeamRecentFinish',
    'TeamReliability', 'TeamTier_0', 'TeamTier_1',
    'TeamTier_2', 'TeamTier_3', 'CircuitLength', 'NumCorners',
    'DRSZones', 'StdLapTime', 'IsStreet', 'DownforceLevel',
    'WeightedAvgOvertakes'
]


def fetch_weather(circuit: str, api_key: str | None = None) -> dict | None:
    """Return a short-range weather forecast for the given circuit."""
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
            air = float(mean(temps))
            rain = float(max(pops)) if pops else 0.0
            return {"ForecastAirTemp": air, "ForecastPrecipChance": rain}
    except Exception as err:  # pragma: no cover - network call
        logger.warning("Failed to fetch weather: %s", err)
    return None


def _clean_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean historical race results."""
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
        try:
            schedule = get_event_schedule(season)
            rounds = schedule["RoundNumber"].dropna().unique()
        except Exception:
            continue
        for rnd in sorted(rounds):
            rnd = int(rnd)
            try:
                session = get_session(season, rnd, 'R')
                session.load()
                results = session.results[['DriverNumber', 'Position', 'Points', 'GridPosition', 'Status']]
                results['Season'] = season
                results['RaceNumber'] = rnd
                results['Circuit'] = session.event['EventName']
                results['Date'] = session.date.strftime('%Y-%m-%d')
                try:
                    weather = session.weather_data
                    results['AirTemp'] = weather['AirTemp'].mean()
                    results['TrackTemp'] = weather['TrackTemp'].mean()
                    results['Rainfall'] = weather['Rainfall'].max()
                except Exception:
                    results['AirTemp'] = nan
                    results['TrackTemp'] = nan
                    results['Rainfall'] = nan
                results['WeightedAvgOvertakes'] = overtake_map.get(results['Circuit'].iloc[0], nan)
                try:
                    q_session = get_session(season, rnd, 'Q')
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
                        return min(times) if times else nan

                    q_results['BestQualiTime'] = q_results.apply(_best_time, axis=1)
                    q_results['Q3Time'] = q_results['Q3'].apply(
                        lambda x: pd.to_timedelta(x).total_seconds() if pd.notna(x) else nan
                    )
                    q_results['GridFromQ3'] = q_results['Q3Time'].rank(method='first')
                    results = pd.merge(
                        results,
                        q_results[['DriverNumber', 'BestQualiTime', 'GridFromQ3']],
                        on='DriverNumber',
                        how='left'
                    )
                    if 'GridPosition' in results.columns:
                        results['GridPosition'] = results['GridPosition'].fillna(results['GridFromQ3'])
                except Exception:
                    results['BestQualiTime'] = nan
                try:
                    fp3_session = get_session(season, rnd, 'FP3')
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
                        best_laps['FP3LongRunTime'] = nan
                    results = pd.merge(results, best_laps, on='DriverNumber', how='left')
                except Exception:
                    results['FP3BestTime'] = nan
                    results['FP3LongRunTime'] = nan
                try:
                    sprint_session = get_session(season, rnd, 'S')
                    sprint_session.load()
                    sprint_res = sprint_session.results[['DriverNumber', 'Position']].rename(
                        columns={'Position': 'SprintFinish'}
                    )
                    results = results.merge(sprint_res, on='DriverNumber', how='left')
                except Exception:
                    results['SprintFinish'] = nan
                race_data.append(results)
            except Exception:
                continue
    return pd.concat(race_data)


def _add_driver_team_info(full_data, seasons):
    seasons_drivers = {}
    for season in seasons:
        try:
            schedule = get_event_schedule(season)
            first_race = schedule.iloc[0]['RoundNumber']
            session = get_session(season, first_race, 'R')
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
    grid_series = (
        full_data['GridPosition'] if 'GridPosition' in full_data.columns
        else pd.Series(nan, index=full_data.index)
    )
    full_data['GridPosition'] = (
        pd.to_numeric(grid_series, errors='coerce')
        .fillna(20)
        .clip(1, 20)
    )
    full_data['AirTemp'] = pd.to_numeric(full_data.get('AirTemp'), errors='coerce')
    full_data['TrackTemp'] = pd.to_numeric(full_data.get('TrackTemp'), errors='coerce')
    full_data['Rainfall'] = pd.to_numeric(full_data.get('Rainfall'), errors='coerce')
    full_data['WeightedAvgOvertakes'] = pd.to_numeric(full_data.get('WeightedAvgOvertakes'), errors='coerce')
    full_data['BestQualiTime'] = pd.to_numeric(full_data.get('BestQualiTime'), errors='coerce')
    full_data['FP3BestTime'] = pd.to_numeric(full_data.get('FP3BestTime'), errors='coerce')
    full_data['FP3LongRunTime'] = pd.to_numeric(full_data.get('FP3LongRunTime'), errors='coerce')
    full_data['SprintFinish'] = pd.to_numeric(
        full_data.get('SprintFinish', pd.Series(nan, index=full_data.index)),
        errors='coerce'
    )
    full_data['SprintFinish'] = full_data['SprintFinish'].fillna(25)
    full_data['SprintFinish'] = full_data['SprintFinish'].clip(1, 25)
    if 'GridDropCount' in full_data.columns:
        full_data['GridDropCount'] = pd.to_numeric(full_data['GridDropCount'], errors='coerce').fillna(0)
    else:
        full_data['GridDropCount'] = 0
    if 'Q1Time' not in full_data.columns and 'Q1' in full_data.columns:
        full_data['Q1Time'] = pd.to_timedelta(full_data['Q1'], errors='coerce').dt.total_seconds()
    if 'Q3Time' not in full_data.columns and 'Q3' in full_data.columns:
        full_data['Q3Time'] = pd.to_timedelta(full_data['Q3'], errors='coerce').dt.total_seconds()
    if 'Q1Time' in full_data.columns:
        full_data['Q1Time'] = pd.to_numeric(full_data['Q1Time'], errors='coerce')
    if 'Q3Time' in full_data.columns:
        full_data['Q3Time'] = pd.to_numeric(full_data['Q3Time'], errors='coerce')
    if 'DidNotFinish' not in full_data.columns:
        if 'Status' in full_data.columns:
            full_data['DidNotFinish'] = full_data['Status'].str.lower() != 'finished'
        else:
            full_data['DidNotFinish'] = full_data['Position'] > 20
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
        full_data['DeltaToBestQuali'] = nan
        full_data['DeltaToNext'] = nan
    if 'BestQualiTime' in full_data.columns:
        team_mean_q = full_data.groupby(['Season', 'RaceNumber', 'HistoricalTeam'])['BestQualiTime'].transform('mean')
        team_size = full_data.groupby(['Season', 'RaceNumber', 'HistoricalTeam'])['BestQualiTime'].transform('size')
        full_data['DeltaToTeammateQuali'] = where(team_size > 1, (full_data['BestQualiTime'] - team_mean_q) * 2, nan)
    else:
        full_data['DeltaToTeammateQuali'] = nan
    if 'Q1Time' in full_data.columns and 'Q3Time' in full_data.columns:
        full_data['QualiSessionGain'] = full_data['Q1Time'] - full_data['Q3Time']
        full_data['QualiSessionGain'] = full_data.groupby(['Season', 'RaceNumber'])['QualiSessionGain'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
    else:
        full_data['QualiSessionGain'] = nan
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
        full_data.groupby(['HistoricalTeam', 'Season'])['GridPosition']
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
    full_data['TeamRecentQuali'] = full_data['TeamRecentQuali'].fillna(full_data['GridPosition'].mean())
    full_data['TeamRecentFinish'] = full_data['TeamRecentFinish'].fillna(full_data['Position'].mean())
    full_data['TeamReliability'] = full_data['TeamReliability'].fillna(0)
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
    prev_year_map = {}
    for yr in sorted(full_data['Season'].unique()):
        prev = full_data[full_data['Season'] == yr - 1]
        if prev.empty:
            continue
        final_pts = prev.groupby('HistoricalTeam')['ConstructorChampPoints'].max()
        rank = final_pts.rank(method='dense', ascending=False)
        prev_year_map[yr] = rank.to_dict()
    full_data['PrevYearConstructorRank'] = full_data.apply(
        lambda r: prev_year_map.get(r['Season'], {}).get(r['HistoricalTeam'], nan),
        axis=1
    )
    tier_map = {}
    for yr in sorted(full_data['Season'].unique()):
        prev = full_data[full_data['Season'] == yr - 1]
        if prev.empty:
            continue
        final_pts = prev.groupby('HistoricalTeam')['ConstructorChampPoints'].max()
        ranks = final_pts.rank(method='dense', ascending=True)
        pct = (ranks - 1) / max(ranks.max() - 1, 1)
        tiers = (pct * 4).astype(int).clip(0, 3)
        tier_map[yr] = tiers.to_dict()
    full_data['TeamTier'] = full_data.apply(
        lambda r: tier_map.get(r['Season'], {}).get(r['HistoricalTeam'], 3),
        axis=1
    ).astype(int)
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
                circuit_lengths[circ] = nan
                corners_map[circ] = meta.get('NumCorners', nan)
                drs_map[circ] = meta.get('DRSZones', nan)
                lap_map[circ] = meta.get('StdLapTime', nan)
        full_data['CircuitLength'] = full_data['Circuit'].map(circuit_lengths)
        full_data['NumCorners'] = full_data['Circuit'].map(corners_map)
        full_data['DRSZones'] = full_data['Circuit'].map(drs_map)
        full_data['StdLapTime'] = full_data['Circuit'].map(lap_map)
    except Exception:
        full_data['CircuitLength'] = full_data['Circuit'].map(
            lambda c: CIRCUIT_METADATA.get(c, {}).get('CircuitLength', nan)
        )
        full_data['NumCorners'] = full_data['Circuit'].map(
            lambda c: CIRCUIT_METADATA.get(c, {}).get('NumCorners', nan)
        )
        full_data['DRSZones'] = full_data['Circuit'].map(
            lambda c: CIRCUIT_METADATA.get(c, {}).get('DRSZones', nan)
        )
        full_data['StdLapTime'] = full_data['Circuit'].map(
            lambda c: CIRCUIT_METADATA.get(c, {}).get('StdLapTime', nan)
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
    full_data['Month'] = pd.to_datetime(full_data['Date'], errors='coerce').dt.month
    air_med = full_data.groupby(['Circuit', 'Month'])['AirTemp'].transform('median')
    full_data['AirTemp'] = full_data['AirTemp'].fillna(air_med)
    full_data['AirTemp'] = full_data['AirTemp'].fillna(full_data['AirTemp'].mean())
    track_med = full_data.groupby(['Circuit', 'Month'])['TrackTemp'].transform('median')
    full_data['TrackTemp'] = full_data['TrackTemp'].fillna(track_med)
    full_data['TrackTemp'] = full_data['TrackTemp'].fillna(full_data['TrackTemp'].mean())
    full_data['RainfallMissing'] = full_data['Rainfall'].isna().astype(int)
    rain_med = full_data.groupby(['Circuit', 'Month'])['Rainfall'].transform('median')
    full_data['Rainfall'] = full_data['Rainfall'].fillna(rain_med)
    circuit_rain = full_data.groupby('Circuit')['Rainfall'].transform('median')
    full_data['Rainfall'] = full_data['Rainfall'].fillna(circuit_rain)
    full_data['Rainfall'] = full_data['Rainfall'].fillna(full_data['Rainfall'].median())
    full_data = full_data.drop(columns=['Month', 'Date'], errors='ignore')
    full_data['WeightedAvgOvertakes'] = full_data['WeightedAvgOvertakes'].fillna(
        full_data['WeightedAvgOvertakes'].mean())
    full_data['MissedQuali'] = full_data['BestQualiTime'].isna().astype(int)
    full_data['BestQualiTime'] = full_data['BestQualiTime'].fillna(
        full_data['BestQualiTime'].median())
    full_data['FP3BestTime'] = full_data['FP3BestTime'].fillna(full_data['FP3BestTime'].mean())
    full_data['FP3LongRunTime'] = full_data['FP3LongRunTime'].fillna(full_data['FP3LongRunTime'].mean())
    full_data['SprintFinish'] = full_data['SprintFinish'].fillna(25)
    full_data['IsStreet'] = full_data['IsStreet'].fillna(0)
    full_data['DownforceLevel'] = full_data['DownforceLevel'].fillna(1)
    full_data['GridDropCount'] = full_data['GridDropCount'].fillna(0)
    full_data['DeltaToBestQuali'] = full_data['DeltaToBestQuali'].fillna(full_data['DeltaToBestQuali'].mean())
    full_data['DeltaToNext'] = full_data['DeltaToNext'].fillna(full_data['DeltaToNext'].mean())
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
        full_data['PrevYearConstructorRank'].max())

    required_columns = [
        'DeltaToTeammateQuali', 'QualiSessionGain', 'GridDropCount',
        'MissedQuali', 'SprintFinish', 'FP3LongRunTime'
    ]
    for col in required_columns:
        if col not in full_data.columns:
            full_data[col] = nan
    return full_data


def _prepare_features(
    full_data,
    base_cols=None,
    team_encoder=None,
    circuit_encoder=None,
    top_circuits=None,
):
    """Prepare numeric race features and encode categorical values.

    When ``base_cols`` is omitted the function returns only the prepared
    feature frame using :data:`race_cols` and leaves the encoders
    unmanaged.  This matches the historical behaviour relied on by the
    tests.  When ``base_cols`` is provided the function returns a tuple of
    ``(features, team_encoder, circuit_encoder, top_circuits)``.
    """
    if base_cols is None:
        features, _, _, _ = _prepare_features(
            full_data,
            race_cols,
            team_encoder,
            circuit_encoder,
            top_circuits,
        )
        return features
    full_data = full_data.copy()
    if "HistoricalTeam" in full_data.columns:
        full_data["Team"] = full_data["HistoricalTeam"]
    if full_data.empty:
        team_cols = (
            team_encoder.get_feature_names_out(["TeamTier"]).tolist()
            if team_encoder
            else [f"TeamTier_{i}" for i in range(4)]
        )
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
        )
    if "Team" not in full_data.columns:
        full_data["Team"] = "Unknown Team"
    if "Circuit" not in full_data.columns:
        full_data["Circuit"] = "Unknown Circuit"
    for col in base_cols:
        full_data[col] = to_numeric(full_data[col], errors="coerce")
    full_data[base_cols] = full_data[base_cols].fillna(full_data[base_cols].median())
    team_cols = [f"TeamTier_{i}" for i in range(4)]
    if "TeamTier" in full_data.columns:
        if team_encoder is None:
            team_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            team_encoded = team_encoder.fit_transform(full_data[["TeamTier"]])
        else:
            team_encoded = team_encoder.transform(full_data[["TeamTier"]])
        team_df = pd.DataFrame(
            team_encoded, columns=team_encoder.get_feature_names_out(["TeamTier"])
        )
        team_df.columns = [c.replace("TeamTier_", "TeamTier_") for c in team_df.columns]
        team_df = team_df.reindex(columns=team_cols, fill_value=0)
    else:
        # Assume one-hot columns already present
        for col in team_cols:
            if col not in full_data.columns:
                full_data[col] = 0
        team_df = pd.DataFrame(index=full_data.index)
    if top_circuits is None:
        top_circuits = (
            full_data["Circuit"].value_counts().nlargest(10).index.tolist()
        )
    full_data["CircuitGrp"] = where(
        full_data["Circuit"].isin(top_circuits), full_data["Circuit"], "Other"
    )
    if circuit_encoder is None:
        circuit_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
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
    return features, team_encoder, circuit_encoder, top_circuits


def _encode_features(
    full_data,
    base_cols=None,
    team_encoder=None,
    circuit_encoder=None,
    top_circuits=None,
):
    """Encode prepared features using provided or fitted encoders."""
    return _prepare_features(
        full_data,
        base_cols or race_cols,
        team_encoder,
        circuit_encoder,
        top_circuits,
    )


__all__ = [
    '_get_event_drivers',
    '_get_qualifying_results',
    '_get_fp3_results',
    '_load_overtake_stats',
    'OVERTAKE_AVERAGES',
    'GRAND_PRIX_LIST',
    'CIRCUIT_METADATA',
    'CIRCUIT_COORDS',
    'fetch_weather',
    '_clean_historical_data',
    '_load_historical_data',
    '_add_driver_team_info',
    '_engineer_features',
    '_prepare_features',
    '_encode_features',
    'race_cols',
]
