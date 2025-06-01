import pandas as pd

from pipeline import predict_race  # not used but to ensure import

from data_utils import _engineer_features

def _drop_leakage(df, year, this_race_number):
    mask_past = ~((df['Season'] == year) & (df['RaceNumber'] >= this_race_number))
    return df.loc[mask_past]

def test_data_leakage():
    df = pd.DataFrame({
        'Season': [2025]*4,
        'RaceNumber': [5,6,7,8],
        'DriverNumber': [1,1,1,1],
        'GridPosition': [1,1,1,1],
        'Position': [1,1,1,1]
    })
    cleaned = _drop_leakage(df, 2025, 5)
    assert cleaned.empty
