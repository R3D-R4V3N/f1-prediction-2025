import os
import statistics
import logging
from typing import Iterable, List, Dict

import numpy as np

try:
    import fastf1
except ImportError as exc:
    raise SystemExit("fastf1 is required to run this script. Install via 'pip install fastf1'.") from exc

import pandas as pd

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _count_position_changes(laps: pd.DataFrame) -> int:
    """Return the number of genuine position changes during a race.

    The input ``laps`` must contain the columns ``Driver``, ``LapNumber``,
    ``Position``, ``IsAccurate``, ``PitInTime`` and ``PitOutTime``. Only laps
    marked as accurate are considered. Laps where a driver's position is greater
    than 20 or where the driver is entering or leaving the pit lane are ignored.
    ``NaN`` values produced by this filtering are treated as no information, so
    position changes are only counted when both consecutive laps contain valid
    data for a driver.
    """

    if laps.empty:
        return 0

    valid = laps[laps["IsAccurate"]].copy()
    valid = valid[valid["Position"].le(20)]

    if "IsPitLap" in valid.columns:
        valid = valid[~valid["IsPitLap"]]
    else:
        valid = valid[valid["PitInTime"].isna() & valid["PitOutTime"].isna()]

    if "IsSafetyCar" in valid.columns:
        valid = valid[~valid["IsSafetyCar"]]
    elif "LapTime" in valid.columns and not valid["LapTime"].isna().all():
        avg_lap = (
            valid["LapTime"].dropna().dt.total_seconds().mean()
        )
        if avg_lap:
            sc_mask = valid["LapTime"].dt.total_seconds() > 2 * avg_lap
            valid = valid[~sc_mask]

    if valid.empty:
        return 0

    positions = (
        valid.pivot_table(index="LapNumber", columns="Driver", values="Position")
        .sort_index()
    )

    diffs = positions.diff()
    changes = diffs.fillna(0).astype(bool)
    return int(changes.sum().sum())


def count_overtakes(year: int, grand_prix: str) -> int:
    """Estimate overtakes in a race by counting lap position changes."""
    cache_dir = "f1_cache"
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    schedule = fastf1.get_event_schedule(year)
    match = schedule[schedule["EventName"].str.contains(grand_prix, case=False, na=False)]
    if match.empty:
        raise ValueError(f"Grand Prix '{grand_prix}' not found in {year} schedule")

    round_number = int(match.iloc[0]["RoundNumber"])
    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=False, weather=False)
    laps = session.laps[
        [
            "Driver",
            "LapNumber",
            "Position",
            "IsAccurate",
            "PitInTime",
            "PitOutTime",
        ]
    ]
    return _count_position_changes(laps)


def average_overtakes(grand_prix: str, years: Iterable[int]) -> float:
    """Compute a weighted average of overtakes over recent seasons."""
    yr_list = list(years)[-3:]
    counts: List[int] = []
    for yr in yr_list:
        try:
            counts.append(count_overtakes(yr, grand_prix))
        except Exception as err:
            logger.warning(
                "Failed to process %s %s: %s", yr, grand_prix, err
            )
    if not counts:
        raise RuntimeError("No races processed")

    weights = np.array([0.5 ** i for i in range(len(counts))])[::-1]
    weighted = (weights * np.array(counts)).sum() / weights.sum()
    return float(weighted)


def overtakes_per_year(grand_prix: str, years: Iterable[int]) -> Dict[int, int]:
    """Return the overtake count for each season."""
    counts: Dict[int, int] = {}
    for yr in years:
        try:
            counts[yr] = count_overtakes(yr, grand_prix)
        except Exception as err:
            logger.warning("Failed to process %s %s: %s", yr, grand_prix, err)
    if not counts:
        raise RuntimeError("No races processed")
    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate race overtakes using lap position changes")
    parser.add_argument("grand_prix", help="Grand Prix name, e.g. 'Monaco'")
    parser.add_argument("years", nargs="+", type=int, help="List of seasons to average")
    parser.add_argument(
        "--per-year",
        action="store_true",
        help="Display the overtakes for each supplied year",
    )
    args = parser.parse_args()

    if args.per_year:
        per_year = overtakes_per_year(args.grand_prix, args.years)
        for yr in sorted(per_year):
            logger.info(
                "Overtakes at %s %d: %d", args.grand_prix, yr, per_year[yr]
            )

    avg = average_overtakes(args.grand_prix, args.years)
    logger.info(
        "Weighted average overtakes at %s: %.1f", args.grand_prix, avg
    )

    # Update or create the CSV used by the prediction model
    out_file = "overtake_stats.csv"
    try:
        df = pd.read_csv(out_file)
    except Exception:
        df = pd.DataFrame(columns=["Circuit", "WeightedAvgOvertakes"])

    df = df[df["Circuit"] != args.grand_prix]
    df = pd.concat(
        [df, pd.DataFrame({"Circuit": [args.grand_prix], "WeightedAvgOvertakes": [avg]})],
        ignore_index=True,
    )
    df.to_csv(out_file, index=False)
