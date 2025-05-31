# F1 2025 Race Prediction Model

This project predicts Formula 1 race results for the 2025 season using historical data and team/driver characteristics. It now includes a simple web interface where you can select any Grand Prix and view the predicted finishing order.

![Starting Grid vs Predicted Finish](grid_vs_finish.png)

## Project Overview

This project uses historical Formula 1 data from the 2022-2024 seasons to build a predictive model for 2025 races. The model incorporates:

- Historical driver performance
- Team strength assessment
- Qualifying position influence
- Driver experience factors
- Circuit-specific performance patterns
- Weather conditions and overtaking difficulty metrics
- Best qualifying and practice session times

The system handles team changes for 2025 (like Hamilton moving to Ferrari) and accommodates rookies through team performance metrics.

## Key Features

- **Data Collection**: Automated fetching of historical F1 race data using the FastF1 API
- **Feature Engineering**: Comprehensive driver and team metrics creation, including
  weather conditions, track overtaking difficulty, and detailed qualifying times
 - **Machine Learning**: Two-stage XGBoost models optimized with Bayesian search to predict qualifying and race finishing positions
- **Team Change Handling**: Sophisticated method for handling 2025 driver lineup changes
- **Visualization**: Three different visualizations of prediction results
- **Fallback Systems**: Robust data generation when API data is incomplete
- **Web Interface**: Streamlit app to select a Grand Prix and run predictions

## Visualizations

### 1. Grid Position vs Predicted Finish

![Starting Grid vs Predicted Finish](grid_vs_finish.png)

This visualization shows:

- How each driver is expected to perform relative to their qualifying position
- Points below the diagonal line indicate drivers expected to finish better than their starting position
- Points above the line show drivers predicted to lose positions during the race

### 2. Driver Performance Prediction

![Expected Finishing Position](shanghai_gp_prediction.png)

This chart displays:

- Expected finishing position for each driver
- Color-coded by team
- Lower values indicate better predicted performance

### 3. Team Performance Prediction

![Team Performance](team_performance.png)

This visualization shows:

- Average predicted finishing position by team
- Lower values indicate stronger team performance
- Teams are ranked from strongest to weakest

## Methodology

1. **Data Collection**

   - Historical race results from 2022-2024 seasons
   - Driver and team mappings
   - Circuit-specific performance patterns

2. **Feature Engineering**

   - Grid position influence
   - Team performance metrics
   - Driver experience quantification
   - Circuit-specific indicators
   - Weather measurements and overtaking difficulty
   - Qualifying and practice session times

3. **Machine Learning Model**

   - XGBoost Regressors trained in two stages (qualifying then race finish)
   - Bayesian hyperparameter optimization minimizing MAE
 - Separate qualifying model feeds predicted grid positions into the race model

4. **Prediction Generation**
   - Qualifying grid predicted using the dedicated model
   - Race position prediction using the trained model
   - Analysis of expected position changes during the race

## Technologies Used

- **Python**: Core programming language
- **FastF1**: Formula 1 data access API
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization
- **NumPy**: Numerical computing

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (install via pip):
  ```
  pip install fastf1 pandas numpy scikit-learn matplotlib seaborn
  ```
  The FastF1 package is required to access weather data and detailed qualifying
  information used in the model.

### Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/f1-shanghai-prediction.git
   cd f1-shanghai-prediction
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the prediction script for a single race:
   ```
   python shanghai_f1.py
   ```

4. Launch the Streamlit web app:
   ```
   streamlit run webapp.py
   ```

## Results

The model predicts a podium of:

1. 🥇 Max Verstappen (Red Bull Racing)
2. 🥈 Liam Lawson (Red Bull Racing)
3. 🥉 George Russell (Mercedes)

The full prediction includes expected finishing positions for all 20 drivers competing in a selected 2025 Grand Prix.

## Future Improvements

- Weather condition impact modeling
- Tire strategy optimization simulation
- Driver head-to-head performance analytics
- Race incident probability modeling
- Real-time data integration during race weekends

## Race Details Export

Use `export_race_details.py` to save session weather summaries and driver lap times for a Grand Prix.
Example:

```bash
python export_race_details.py 2024 "Monaco"
```

This creates a CSV in the `race_details` folder containing weather data plus each driver's session results. The CSV now lists best FP3 laps, detailed Q1--Q3 times, sprint shootout times, and race or sprint finishing positions when available.

The `predict_race` function can also export these details for the selected event
by passing `export_details=True`:

```python
from race_predictor import predict_race
predict_race("Monaco Grand Prix", year=2025, export_details=True)
```

## Resources

- [FastF1 Documentation](https://theoehrly.github.io/Fast-F1/)
- [Formula 1 Official Website](https://www.formula1.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## Author

Frank Ndungu

## License

This project is licensed under the MIT License - see the LICENSE file for details.
