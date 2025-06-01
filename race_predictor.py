from predictor import predict_race
from data_utils import GRAND_PRIX_LIST

if __name__ == '__main__':
    res = predict_race('Chinese Grand Prix', year=2025, export_details=True, debug=False)
    print(res[['Driver', 'Team', 'Grid', 'Final_Position']].head())
