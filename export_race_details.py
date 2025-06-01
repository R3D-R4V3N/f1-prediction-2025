import os
import argparse
import time
import logging
import pandas as pd
import requests

try:
    import fastf1
except ImportError as exc:
    raise SystemExit(
        "fastf1 is required to run this script. Install via 'pip install fastf1'."
    ) from exc

logger = logging.getLogger(__name__)


def _fetch_session_data(
    year: int, round_number: int, session_code: str, retries: int = 3
) -> pd.DataFrame:
    """Load a session and return weather info plus driver lap/position data.

    The function retries when the underlying API responds with a 429 status
    code. Exponential backoff is used between attempts.
    """

    attempt = 0
    while True:
        session = fastf1.get_session(year, round_number, session_code)
        try:
            session.load()
            break
        except requests.exceptions.HTTPError as exc:  # type: ignore[attr-defined]
            status = getattr(exc.response, "status_code", None)
            if status == 429 and attempt < retries:
                wait_time = 2 ** attempt
                logger.info("Rate limit hit, retrying in %ds...", wait_time)
                time.sleep(wait_time)
                attempt += 1
                continue
            raise

    weather = session.weather_data
    avg_air = weather['AirTemp'].mean()
    avg_track = weather['TrackTemp'].mean()
    rainfall = weather['Rainfall'].max()

    def _to_seconds(val):
        if pd.isna(val):
            return None
        try:
            return pd.to_timedelta(val).total_seconds()
        except Exception:
            return None

    if session_code == 'FP3':
        df = (
            session.laps.groupby('Driver')['LapTime']
            .min()
            .dt.total_seconds()
            .reset_index()
            .rename(columns={'LapTime': 'BestTime'})
        )
        laps = session.laps
        if not laps.empty:
            max_lap = laps['LapNumber'].max()
            long_runs = laps[laps['LapNumber'] >= max_lap - 4]
            long_avg = (
                long_runs.groupby('Driver')['LapTime']
                .apply(lambda s: s.dt.total_seconds().mean())
                .reset_index()
                .rename(columns={'LapTime': 'LongRunTime'})
            )
            df = df.merge(long_avg, on='Driver', how='left')
        else:
            df['LongRunTime'] = None
    elif session_code == 'Q':
        df = session.results[['Abbreviation', 'Q1', 'Q2', 'Q3']].copy()
        for col in ['Q1', 'Q2', 'Q3']:
            df[col] = df[col].apply(_to_seconds)
        df['BestTime'] = df[['Q1', 'Q2', 'Q3']].min(axis=1)
        df.rename(columns={'Abbreviation': 'Driver'}, inplace=True)
    elif session_code == 'SQ':
        df = session.results[['Abbreviation', 'SQ1', 'SQ2', 'SQ3']].copy()
        for col in ['SQ1', 'SQ2', 'SQ3']:
            df[col] = df[col].apply(_to_seconds)
        df['BestTime'] = df[['SQ1', 'SQ2', 'SQ3']].min(axis=1)
        df.rename(columns={'Abbreviation': 'Driver'}, inplace=True)
    else:  # Race or Sprint session
        df = session.results[['Abbreviation', 'Position']].rename(
            columns={'Abbreviation': 'Driver', 'Position': 'FinishPosition'}
        )
        df['BestTime'] = None

    for col in ['Q1', 'Q2', 'Q3', 'SQ1', 'SQ2', 'SQ3']:
        if col not in df.columns:
            df[col] = None
    if 'FinishPosition' not in df.columns:
        df['FinishPosition'] = None

    df['Session'] = session_code
    df['Date'] = session.date.strftime('%Y-%m-%d')
    df['AvgAirTemp'] = avg_air
    df['AvgTrackTemp'] = avg_track
    df['MaxRainfall'] = rainfall

    return df[
        [
            'Session',
            'Date',
            'Driver',
            'BestTime',
            'LongRunTime',
            'FinishPosition',
            'Q1',
            'Q2',
            'Q3',
            'SQ1',
            'SQ2',
            'SQ3',
            'AvgAirTemp',
            'AvgTrackTemp',
            'MaxRainfall',
        ]
    ]


def export_race_details(year: int, grand_prix: str) -> str:
    """Export FP3, qualifying, sprint and race weather data for the given event."""
    cache_dir = 'f1_cache'
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    schedule = fastf1.get_event_schedule(year)
    match = schedule[schedule['EventName'].str.contains(grand_prix, case=False, na=False)]
    if match.empty:
        raise ValueError(f"Grand Prix '{grand_prix}' not found in {year} schedule")

    round_number = int(match.iloc[0]['RoundNumber'])
    event_name = match.iloc[0]['EventName']

    data_frames = []
    for code in ['FP3', 'Q', 'SQ', 'S', 'R']:
        try:
            data_frames.append(_fetch_session_data(year, round_number, code))
        except Exception as err:
            fallback = pd.DataFrame({
                'Session': [code],
                'Date': [''],
                'Driver': [None],
                'BestTime': [None],
                'FinishPosition': [None],
                'Q1': [None],
                'Q2': [None],
                'Q3': [None],
                'SQ1': [None],
                'SQ2': [None],
                'SQ3': [None],
                'AvgAirTemp': [None],
                'AvgTrackTemp': [None],
                'MaxRainfall': [None],
            })
            data_frames.append(fallback)
            logger.warning("Failed to load %s session: %s", code, err)

    df = pd.concat(data_frames, ignore_index=True)
    out_dir = 'race_details'
    os.makedirs(out_dir, exist_ok=True)
    safe_name = event_name.lower().replace(' ', '_')
    file_path = os.path.join(out_dir, f"{year}_{safe_name}.csv")
    df.to_csv(file_path, index=False)
    logger.info("Saved session data to %s", file_path)
    return file_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export race session weather data to CSV"
    )
    parser.add_argument('year', type=int, help='Season year (e.g. 2024)')
    parser.add_argument('grand_prix', help='Grand Prix name (e.g. "Monaco")')
    args = parser.parse_args()

    file_path = export_race_details(args.year, args.grand_prix)
    logger.info("Saved session data to %s", file_path)


if __name__ == '__main__':
    main()
