# F1 2025 Race Prediction

This repository contains tools to predict the outcome of Formula&nbsp;1 races in the 2025 season. The project builds machine learning models from recent seasons and provides scripts to generate training data, run predictions and export detailed session information.

## Features

- **Race prediction** using a single XGBoost model.
- **Automatic team and driver handling** for 2025 line‑ups, including rookies.
- **Weather and circuit statistics** such as air/track temperature and estimated overtakes per circuit.
- **Live weather forecasts** blended with historical averages when an `OPENWEATHER_API_KEY` is provided.
- **Historical performance metrics** like driver experience, recent form and track specific results.
- **Sprint form** including finishing positions from sprint races when applicable.
- **Streamlit web interface** to quickly run predictions for any Grand Prix.
- **Data export utilities** to save FP3, qualifying, sprint and race session information.
- **Rank-based metrics** showing podium accuracy and Spearman correlation.

## How Predictions Work

The model is trained on event data from the 2020‑2025 seasons. For each race it collects:

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
   python race_predictor.py
   ```

   By default this runs `predict_race` for the Chinese Grand Prix in 2025. Edit the last lines of `race_predictor.py` or call the function from Python to choose another event.

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

   Calculates the weighted average number of genuine overtakes for the circuit and updates `overtake_stats.csv`.

5. **Generate full season data**

   ```bash
   python generate_2025_data.py
   ```

Downloads FP3, qualifying and race information for every scheduled event in 2025.

## Evaluation Metrics

The training routine optimises **Spearman rank correlation** on finishing order
using ``XGBRanker`` with a pairwise ranking objective. During training and
hold-out evaluation the script also reports:

- **Spearman rank correlation** between predicted and actual results.
- **Top 1 accuracy** – percentage of races where the predicted winner matches the real winner.
- **Top 3 accuracy** – proportion of correctly predicted podium finishers.

## Repository Structure

- `race_predictor.py` – core prediction code.
- `webapp.py` – Streamlit interface.
- `export_race_details.py` – utility to save detailed session CSVs.
- `estimate_overtakes.py` – script to compute overtake statistics.
- `generate_2025_data.py` – bulk data downloader for a full season.
 - `overtake_stats.csv` – optional lookup table of weighted average overtakes per circuit.
- `race_details/` – folder containing exported event data.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
