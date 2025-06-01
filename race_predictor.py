import logging

from predictor import predict_race
from data_utils import GRAND_PRIX_LIST

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    res = predict_race('Chinese Grand Prix', year=2025, export_details=True, debug=False)
    logger.info(
        "\n%s",
        res[['Driver', 'Team', 'Grid', 'Final_Position']].head(),
    )
