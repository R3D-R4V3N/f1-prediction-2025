import pytest
pd = pytest.importorskip("pandas")


def test_data_leakage_filter():
    df = pd.DataFrame({
        "Season": [2025, 2025, 2025, 2025],
        "RaceNumber": [5, 6, 7, 8],
        "DriverNumber": [1, 1, 1, 1],
        "Position": [3, 4, 5, 6],
    })
    this_race_number = 5
    mask_past = ~((df["Season"] == 2025) & (df["RaceNumber"] >= this_race_number))
    filtered = df.loc[mask_past]
    assert filtered.empty

    this_race_number = 7
    mask_past = ~((df["Season"] == 2025) & (df["RaceNumber"] >= this_race_number))
    filtered = df.loc[mask_past]
    assert list(filtered["RaceNumber"]) == [5, 6]
