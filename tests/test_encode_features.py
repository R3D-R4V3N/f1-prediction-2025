import pytest
pd = pytest.importorskip("pandas")
from data_utils import _prepare_features, _encode_features, race_cols

def test_encoding_unknown_circuit_and_team():
    df = pd.DataFrame({
        "Season": [2025],
        "RaceNumber": [1],
        "DriverNumber": [1],
        "GridPosition": [5],
        "BestQualiTime": [pd.to_timedelta("00:01:13")],
        "DeltaToBestQuali": [0.5],
        "DeltaToNext": [0.3],
        "DeltaToNext_Q3": [0.3],
        "DeltaToNext_Q2": [0.0],
        "MissedQ3": [0],
        "MissedQ2": [0],
        "DeltaToTeammateQuali": [0.2],
        "QualiSessionGain": [1],
        "GridDropCount": [0],
        "GridMissed": [0],
        "FP3BestTime": [pd.to_timedelta("00:01:15")],
        "FP3LongRunTime": [pd.to_timedelta("00:01:18")],
        "AirTemp": [25.0],
        "TrackTemp": [35.0],
        "Rainfall": [0.0],
        "MissedQuali": [0],
        "SprintFinish": [10],
        "CrossAvgFinish": [4.0],
        "RecentAvgPoints": [12.0],
        "Recent3AvgFinish": [4.5],
        "Recent5AvgFinish": [5.0],
        "DriverAvgTrackFinish": [3.0],
        "DriverTrackPodiums": [1],
        "DriverTrackDNFs": [0],
        "IsRookie": [1],
        "PrevYearConstructorRank": [2],
        "TeamRecentQuali": [6.0],
        "TeamRecentFinish": [7.0],
        "TeamReliability": [0.8],
        "DriverSeasonDNFs": [1.0],
        "TeamSeasonDNFs": [2.0],
        "TeamTier_0": [0], "TeamTier_1": [0], "TeamTier_2": [1], "TeamTier_3": [0],
        "CircuitLength": [5.424], "NumCorners": [19], "DRSZones": [2],
        "StdLapTime": [98.5], "IsStreet": [0], "DownforceLevel": [1],
        "Overtakes_CurrentYear": [30.0],
        "CircuitEmbed1": [0.0], "CircuitEmbed2": [0.0],
        "SafetyCarAvg": [3.0],
        "LikelihoodSC": [0.5],
        "HistoricalTeam": ["ImaginaryRacers"],
        "Circuit": ["Neverland GP"],
    })
    prepared = _prepare_features(df.copy())
    features, team_enc, circ_enc, top_circuits = _encode_features(prepared, race_cols)
    assert list(features.columns) == race_cols
    assert features.replace([pd.NA, pd.NaT], 0).notnull().all().all()
