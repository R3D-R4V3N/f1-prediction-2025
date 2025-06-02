import pytest
pd = pytest.importorskip("pandas")
from data_utils import _add_driver_team_info

class FakeSession:
    def __init__(self, df):
        self.results = df
    def load(self):
        pass


def test_driver_team_assignment_across_rounds(monkeypatch):
    def fake_get_event_schedule(year):
        return pd.DataFrame({'RoundNumber': [1, 2, 3]})

    def fake_get_session(year, rnd, kind):
        mapping = {
            1: pd.DataFrame({'DriverNumber': [1, 2], 'TeamName': ['A', 'B']}),
            2: pd.DataFrame({'DriverNumber': [1, 2], 'TeamName': ['A', 'B']}),
            3: pd.DataFrame({'DriverNumber': [1, 2], 'TeamName': ['C', 'B']}),
        }
        return FakeSession(mapping[rnd])

    monkeypatch.setattr('data_utils.get_event_schedule', fake_get_event_schedule)
    monkeypatch.setattr('data_utils.get_session', fake_get_session)

    df = pd.DataFrame({
        'Season': [2024, 2024, 2024, 2024, 2024, 2024],
        'RaceNumber': [1, 1, 2, 2, 3, 3],
        'DriverNumber': [1, 2, 1, 2, 1, 2],
    })

    result = _add_driver_team_info(df.copy(), [2024])
    assert list(result['HistoricalTeam']) == ['A', 'B', 'A', 'B', 'C', 'B']
