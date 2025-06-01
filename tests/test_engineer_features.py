import pytest
import pandas as pd
from data_utils import _engineer_features

def test_delta_and_cross_avg_and_rookie_flag():
    data = pd.DataFrame({
        "Season": [2024, 2024, 2025, 2025],
        "RaceNumber": [1, 1, 1, 1],
        "DriverNumber": [1, 2, 1, 2],
        "Team": ["A", "A", "A", "A"],
        "BestQualiTime": [
            pd.to_timedelta("00:01:10"),
            pd.to_timedelta("00:01:11"),
            pd.to_timedelta("00:01:12"),
            pd.NaT,
        ],
        "Position": [5, 6, 3, 4],
        "Date": pd.to_datetime(["2024-03-01", "2024-03-01", "2025-03-01", "2025-03-01"]),
    })
    out = _engineer_features(data.copy())
    assert "DeltaToBestQuali" in out.columns
    assert (out["DeltaToBestQuali"].dropna() >= 0).all()
    dx = out[(out["Season"]==2024)].sort_values("DriverNumber")["DeltaToNext"].iloc[0]
    assert dx == pytest.approx(1.0, rel=1e-3)
    assert out[(out["Season"]==2024)]["IsRookie"].eq(1).all()
    assert out[(out["Season"]==2025)]["IsRookie"].eq(0).all()
    cavg = out[(out["Season"]==2025) & (out["DriverNumber"]==1)]["CrossAvgFinish"].iloc[0]
    assert cavg == pytest.approx(5.0)
