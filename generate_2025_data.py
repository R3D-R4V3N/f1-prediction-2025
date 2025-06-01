import os
import pandas as pd

try:
    import fastf1
except ImportError as exc:
    raise SystemExit("fastf1 is required to run this script. Install via 'pip install fastf1'.") from exc

from export_race_details import _fetch_session_data

def export_full_season(year: int = 2025, output_file: str = "prediction_data_race_2025.csv") -> str:
    """Collect FP3, qualifying and race data for all events in a season."""
    cache_dir = 'f1_cache'
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    schedule = fastf1.get_event_schedule(year)
    data_frames = []
    for _, row in schedule.iterrows():
        gp = row['EventName']
        round_num = int(row['RoundNumber'])
        for code in ['FP3', 'Q', 'SQ', 'S', 'R']:
            try:
                df = _fetch_session_data(year, round_num, code)
                df['EventName'] = gp
                data_frames.append(df)
            except Exception as err:
                print(f"⚠️ Failed to load {year} {gp} {code}: {err}")

    if not data_frames:
        raise RuntimeError("No data collected for season")

    season_df = pd.concat(data_frames, ignore_index=True)
    season_df.to_csv(output_file, index=False)
    return output_file

if __name__ == "__main__":
    path = export_full_season()
    print(f"✅ Saved full season data to {path}")
