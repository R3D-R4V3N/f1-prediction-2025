import os
import argparse
import pandas as pd

try:
    import fastf1
except ImportError as exc:
    raise SystemExit("fastf1 is required to run this script. Install via 'pip install fastf1'.") from exc


def _fetch_session_data(year: int, round_number: int, session_code: str) -> dict:
    """Load a session and summarize weather/fastest lap information."""
    session = fastf1.get_session(year, round_number, session_code)
    session.load()

    weather = session.weather_data
    avg_air = weather['AirTemp'].mean()
    avg_track = weather['TrackTemp'].mean()
    rainfall = weather['Rainfall'].max()

    fastest_time = None
    fastest_driver = None
    if session_code in ['FP3', 'Q']:
        try:
            fastest_lap = session.laps.pick_fastest()
            fastest_time = fastest_lap['LapTime'].total_seconds()
            fastest_driver = fastest_lap['Driver']
        except Exception:
            pass

    return {
        'Session': session_code,
        'Date': session.date.strftime('%Y-%m-%d'),
        'AvgAirTemp': avg_air,
        'AvgTrackTemp': avg_track,
        'MaxRainfall': rainfall,
        'FastestLap(s)': fastest_time,
        'FastestDriver': fastest_driver,
    }


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

    data = []
    for code in ['FP3', 'Q', 'R']:
        try:
            data.append(_fetch_session_data(year, round_number, code))
        except Exception as err:
            data.append({
                'Session': code,
                'Date': '',
                'AvgAirTemp': None,
                'AvgTrackTemp': None,
                'MaxRainfall': None,
                'FastestLap(s)': None,
                'FastestDriver': None,
            })
            print(f"Failed to load {code} session: {err}")

    df = pd.DataFrame(data)
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
