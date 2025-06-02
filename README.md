# F1 2025 Race Prediction

This repository contains tools to predict the outcome of Formula&nbsp;1 races in the 2025 season. The project builds machine learning models from recent seasons and provides scripts to generate training data, run predictions and export detailed session information.

## Features

- **Race prediction** using XGBoost or LightGBM models.
- **Automatic team and driver handling** for 2025 line‑ups, including rookies.
- **Weather and circuit statistics** such as air/track temperature and estimated overtakes per circuit.
- **Live weather forecasts** blended with historical averages when an `OPENWEATHER_API_KEY` is provided.
- **Historical performance metrics** like driver experience, recent form and track specific results.
- **Sprint form** including finishing positions from sprint races when applicable.
- **Team tier categories** derived from the previous season's constructor points.
- **Streamlit web interface** to quickly run predictions for any Grand Prix.
- **Data export utilities** to save FP3, qualifying, sprint and race session information.
- **Rank-based metrics** showing podium accuracy and Spearman correlation.

## How Predictions Work

The model is trained on event data from the 2022‑2025 seasons. For each race it collects:

1. **Driver and team results** from FastF1 with fallbacks to the Ergast API.
2. **Qualifying and practice times** for each driver, including the time gap to the next fastest qualifier.
3. **Weather data** (air temperature, track temperature, rainfall).
4. **Weighted average overtakes** per circuit calculated with `estimate_overtakes.py`.
5. **Championship standings and circuit metadata** such as track length.

An XGBoost regressor uses the official starting grid and all engineered features to forecast the finishing order.

The scripts keep track of team changes and compute various aggregates (recent results, driver performance at a circuit, reliability, etc.).

## Requirements

- Python 3.8+
- Packages: `fastf1`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `optuna`, `xgboost`, `streamlit`

FastF1 is required to download timing and weather data. XGBoost and Optuna provide the machine learning and optimisation components.

Install the dependencies with:

```bash
pip install fastf1 pandas numpy scikit-learn matplotlib seaborn optuna xgboost streamlit
```
If you want live weather predictions, set the `OPENWEATHER_API_KEY` environment variable to your OpenWeatherMap API key.

## Running Predictions

1. **Single race prediction**

   ```bash
   python pipeline.py --grand_prix "Monaco Grand Prix" --year 2024
   ```

   Provide the Grand Prix name and season year with `--grand_prix` and `--year`.
   Add `--retrain` to force the model to train again instead of using the cached
   version. Without any options the script predicts the Chinese Grand Prix in
   2025.

   Set `retrain=True` to force the model to be trained again instead of loading a cached version.

   ```python
   predict_race('Spanish Grand Prix', year=2025, retrain=True)
   ```

2. **Streamlit web app**

   ```bash
  streamlit run webapp.py
  ```

   Select a Grand Prix from the dropdown and the app will display the predicted finishing positions.

3. **Export race session data**

   ```bash
   python export_race_details.py 2024 "Monaco"
   ```

   Creates a CSV with FP3, qualifying, sprint and race data in the `race_details/` folder.

4. **Estimate overtakes**

   ```bash
   python estimate_overtakes.py "Monaco Grand Prix" 2022 2023 2024
   ```

   Calculates the weighted average number of genuine overtakes for the circuit and updates `overtake_stats.csv`. Use `--per-year` to also display the count for each season.

5. **Generate full season data**

   ```bash
   python generate_2025_data.py
   ```

Downloads FP3, qualifying and race information for every scheduled event in 2025.

## Evaluation Metrics

The training routine optimises **Spearman rank correlation** on finishing order
using ``XGBRanker`` by default. When ``use_regression=True`` the optimiser also
explores ``XGBRegressor`` and ``LightGBM``'s LambdaRank and scores each trial
with a weighted combination of rank correlation and mean absolute error (MAE).
During training and hold-out evaluation the script reports:

- **Spearman rank correlation** between predicted and actual results.
- **Top 1 accuracy** – percentage of races where the predicted winner matches the real winner.
- **Top 3 accuracy** – proportion of correctly predicted podium finishers.

In practice the pure ranking objective yields the highest Spearman score,
while enabling ``use_regression`` slightly reduces MAE with only a minor drop
in rank correlation.

## Repository Structure

- `pipeline.py` – core prediction orchestrator.
- `webapp.py` – Streamlit interface.
- `export_race_details.py` – utility to save detailed session CSVs.
- `estimate_overtakes.py` – script to compute overtake statistics.
- `generate_2025_data.py` – bulk data downloader for a full season.
 - `overtake_stats.csv` – optional lookup table of weighted average overtakes per circuit.
- `race_details/` – folder containing exported event data.

## Testing

Run the unit tests with:

```bash
pytest --maxfail=1 --disable-warnings -q
```

Set `LOGLEVEL=DEBUG` before running scripts to see detailed logs. Cached files
are stored in `cache/` and the feature importance plot is saved under
`model_info/`.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
