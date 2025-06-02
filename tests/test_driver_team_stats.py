import pytest
pd = pytest.importorskip("pandas")
from data_utils import _season_driver_team_stats

def test_season_driver_team_stats():
    df = pd.DataFrame({
        "Season": [2024, 2024, 2025, 2025, 2025],
        "RaceNumber": [1, 1, 1, 1, 2],
        "DriverNumber": [1, 2, 1, 2, 2],
        "Circuit": ["X", "X", "Y", "Y", "Z"],
        "HistoricalTeam": ["A", "B", "A", "B", "B"],
        "Position": [5, 6, 4, 3, 2],
        "Points": [10, 8, 12, 10, 18],
        "ConstructorChampPoints": [200, 150, 210, 160, 178],
        "DriverAvgTrackFinish": [5, 6, 4, 3, 2],
        "DriverTrackPodiums": [0, 0, 1, 1, 2],
        "DriverTrackDNFs": [0, 0, 0, 0, 0],
    })
    (
        lookup,
        driver_pts_map,
        constructor_pts_map,
        driver_stand_map,
        constructor_stand_map,
        team_strength,
        prev_rank_map,
        default_prev_rank,
    ) = _season_driver_team_stats(df, 2025)
    assert driver_pts_map[2] == 28
    assert constructor_pts_map["B"] == 28
    assert driver_stand_map[2] == 1
    assert constructor_stand_map["A"] == 2
    assert team_strength["B"] == pytest.approx(2.5)
    assert prev_rank_map["A"] == 1
    assert default_prev_rank == 2
    assert lookup.loc[(2, "Z"), "DriverAvgTrackFinish"] == 2
