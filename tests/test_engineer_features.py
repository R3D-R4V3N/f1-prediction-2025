import pandas as pd
from data_utils import _engineer_features

def test_engineer_features_basic():
    df = pd.DataFrame({
        'Season': [2024, 2024, 2024, 2024],
        'RaceNumber': [1,1,2,2],
        'DriverNumber': [44,55,44,55],
        'GridPosition': [1,2,2,1],
        'Position': [1,2,2,1],
        'BestQualiTime': [90.0, 91.0, 92.0, 93.0],
        'Q1': [90.0,91.0,92.0,93.0],
        'Q2': [89.0,90.0,91.0,92.0],
        'Q3': [88.0,89.0,90.0,91.0],
        'AirTemp': [20,20,21,21],
        'TrackTemp': [30,30,31,31],
        'Rainfall': [0,0,0,0],
        'WeightedAvgOvertakes': [10,10,10,10]
    })
    out = _engineer_features(df)
    assert (out['DeltaToBestQuali'] >= 0).all()
    assert out.loc[0,'DeltaToNext'] == 1
    assert out.loc[0,'CrossAvgFinish'] == 1
    assert out.loc[0,'IsRookie'] == 1
