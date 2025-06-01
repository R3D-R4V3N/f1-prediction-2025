import pandas as pd
from data_utils import _prepare_features, _encode_features, race_cols

def test_encode_features_unseen():
    df = pd.DataFrame({
        'Season':[2024],
        'RaceNumber':[1],
        'DriverNumber':[99],
        'GridPosition':[1],
        'BestQualiTime':[90],
        'DeltaToBestQuali':[0],
        'DeltaToNext':[0.5],
        'DeltaToTeammateQuali':[0],
        'QualiSessionGain':[0],
        'GridDropCount':[0],
        'FP3BestTime':[95],
        'FP3LongRunTime':[96],
        'AirTemp':[20],
        'TrackTemp':[30],
        'Rainfall':[0],
        'MissedQuali':[0],
        'SprintFinish':[25],
        'CrossAvgFinish':[5],
        'RecentAvgPoints':[5],
        'Recent3AvgFinish':[5],
        'Recent5AvgFinish':[5],
        'DriverAvgTrackFinish':[5],
        'DriverTrackPodiums':[0],
        'DriverTrackDNFs':[0],
        'IsRookie':[0],
        'PrevYearConstructorRank':[1],
        'TeamRecentQuali':[3],
        'TeamRecentFinish':[3],
        'TeamReliability':[0],
        'TeamTier':[1],
        'CircuitLength':[5.0],
        'NumCorners':[10],
        'DRSZones':[2],
        'StdLapTime':[80],
        'IsStreet':[0],
        'DownforceLevel':[1],
        'WeightedAvgOvertakes':[10],
        'Team':['NewTeam'],
        'Circuit':['Nowhere']
    })
    feats, team_enc, circ_enc, top_circuits = _encode_features(df, race_cols)
    assert len(feats.columns) == len(race_cols)
