import os
import argparse
import pandas as pd

try:
    import fastf1
except ImportError as exc:
    raise SystemExit("fastf1 is required to run this script. Install via 'pip install fastf1'.") from exc


def _fetch_session_data(year: int, round_number: int, session_code: str) -> pd.DataFrame:
    """Load a session and return weather info plus driver lap data."""
    session = fastf1.get_session(year, round_number, session_code)
    session.load()

    weather = session.weather_data
    avg_air = weather['AirTemp'].mean()
    avg_track = weather['TrackTemp'].mean()
    rainfall = weather['Rainfall'].max()

    if session_code == 'FP3':
        best_times = (
            session.laps.groupby('Driver')['LapTime']
            .min()
            .dt.total_seconds()
            .reset_index()
            .rename(columns={'LapTime': 'BestTime'})
        )
    elif session_code == 'Q':
        q_res = session.results[['Abbreviation', 'Q1', 'Q2', 'Q3']].copy()

        def _best(row):
            times = []
            for col in ['Q1', 'Q2', 'Q3']:
                val = row[col]
                if pd.notna(val):
                    try:
                        times.append(pd.to_timedelta(val).total_seconds())
                    except Exception:
                        pass
            return min(times) if times else None

        q_res['BestTime'] = q_res.apply(_best, axis=1)
        best_times = q_res[['Abbreviation', 'BestTime']].rename(columns={'Abbreviation': 'Driver'})
    else:  # Race session
        best_times = session.results[['Abbreviation', 'Position']].rename(
            columns={'Abbreviation': 'Driver', 'Position': 'FinishPosition'}
        )
        best_times['BestTime'] = None

    best_times['Session'] = session_code
    best_times['Date'] = session.date.strftime('%Y-%m-%d')
    best_times['AvgAirTemp'] = avg_air
    best_times['AvgTrackTemp'] = avg_track
    best_times['MaxRainfall'] = rainfall
    if 'FinishPosition' not in best_times.columns:
        best_times['FinishPosition'] = None

    return best_times[
        ['Session', 'Date', 'Driver', 'BestTime', 'FinishPosition', 'AvgAirTemp', 'AvgTrackTemp', 'MaxRainfall']
    ]


def export_race_details(year: int, grand_prix: str) -> str:
    """Export FP3, qualifying and race weather data for the given event."""
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
    for code in ['FP3', 'Q', 'R']:
        try:
            data_frames.append(_fetch_session_data(year, round_number, code))
        except Exception as err:
            fallback = pd.DataFrame({
                'Session': [code],
                'Date': [''],
                'Driver': [None],
                'BestTime': [None],
                'FinishPosition': [None],
                'AvgAirTemp': [None],
                'AvgTrackTemp': [None],
                'MaxRainfall': [None],
            })
            data_frames.append(fallback)
            print(f"Failed to load {code} session: {err}")

    df = pd.concat(data_frames, ignore_index=True)
    out_dir = 'race_details'
    os.makedirs(out_dir, exist_ok=True)
    safe_name = event_name.lower().replace(' ', '_')
    file_path = os.path.join(out_dir, f"{year}_{safe_name}.csv")
    df.to_csv(file_path, index=False)
    return file_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export race session weather data to CSV")
    parser.add_argument('year', type=int, help='Season year (e.g. 2024)')
    parser.add_argument('grand_prix', help='Grand Prix name (e.g. "Monaco")')
    args = parser.parse_args()

    file_path = export_race_details(args.year, args.grand_prix)
    print(f"Saved session data to {file_path}")


if __name__ == '__main__':
    main()
