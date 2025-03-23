import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# 🏎️ Create cache directory if it doesn't exist
cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"✅ Created cache directory: {cache_dir}")

# Enable the cache
fastf1.Cache.enable_cache(cache_dir)

# 🌟 Updated drivers for 2025 season from the official F1 site
drivers_2025 = [
    # Ferrari
    {'DriverNumber': 16, 'Abbreviation': 'LEC', 'FullName': 'Charles Leclerc', 'Team': 'Ferrari'},
    {'DriverNumber': 44, 'Abbreviation': 'HAM', 'FullName': 'Lewis Hamilton', 'Team': 'Ferrari'},
    
    # Mercedes-AMG Petronas
    {'DriverNumber': 63, 'Abbreviation': 'RUS', 'FullName': 'George Russell', 'Team': 'Mercedes'},
    {'DriverNumber': 72, 'Abbreviation': 'ANT', 'FullName': 'Andrea Kimi Antonelli', 'Team': 'Mercedes'},
    
    # Red Bull Racing
    {'DriverNumber': 1, 'Abbreviation': 'VER', 'FullName': 'Max Verstappen', 'Team': 'Red Bull Racing'},
    {'DriverNumber': 40, 'Abbreviation': 'LAW', 'FullName': 'Liam Lawson', 'Team': 'Red Bull Racing'},
    
    # McLaren
    {'DriverNumber': 4, 'Abbreviation': 'NOR', 'FullName': 'Lando Norris', 'Team': 'McLaren'},
    {'DriverNumber': 81, 'Abbreviation': 'PIA', 'FullName': 'Oscar Piastri', 'Team': 'McLaren'},
    
    # Aston Martin
    {'DriverNumber': 14, 'Abbreviation': 'ALO', 'FullName': 'Fernando Alonso', 'Team': 'Aston Martin'},
    {'DriverNumber': 18, 'Abbreviation': 'STR', 'FullName': 'Lance Stroll', 'Team': 'Aston Martin'},
    
    # Alpine
    {'DriverNumber': 10, 'Abbreviation': 'GAS', 'FullName': 'Pierre Gasly', 'Team': 'Alpine'},
    {'DriverNumber': 5, 'Abbreviation': 'DOO', 'FullName': 'Jack Doohan', 'Team': 'Alpine'},
    
    # Williams
    {'DriverNumber': 23, 'Abbreviation': 'ALB', 'FullName': 'Alexander Albon', 'Team': 'Williams'},
    {'DriverNumber': 55, 'Abbreviation': 'SAI', 'FullName': 'Carlos Sainz Jr.', 'Team': 'Williams'},
    
    # Haas
    {'DriverNumber': 31, 'Abbreviation': 'OCO', 'FullName': 'Esteban Ocon', 'Team': 'Haas F1 Team'},
    {'DriverNumber': 87, 'Abbreviation': 'BEA', 'FullName': 'Oliver Bearman', 'Team': 'Haas F1 Team'},
    
    # Kick Sauber
    {'DriverNumber': 27, 'Abbreviation': 'HUL', 'FullName': 'Nico Hülkenberg', 'Team': 'Kick Sauber'},
    {'DriverNumber': 50, 'Abbreviation': 'BOR', 'FullName': 'Gabriel Bortoleto', 'Team': 'Kick Sauber'},
    
    # Visa Cash App Racing Bulls (VCARB)
    {'DriverNumber': 22, 'Abbreviation': 'TSU', 'FullName': 'Yuki Tsunoda', 'Team': 'VCARB'},
    {'DriverNumber': 41, 'Abbreviation': 'HAD', 'FullName': 'Isack Hadjar', 'Team': 'VCARB'}
]

drivers_df = pd.DataFrame(drivers_2025)

# 🏆 Load historical data for training
# Let's get data from the 2022-2024 seasons
seasons = [2022, 2023, 2024]
race_data = []

print("🔍 Fetching historical F1 data...")
for season in seasons:
    for race in range(1, 23):  # Most seasons have fewer than 23 races
        try:
            race_obj = fastf1.get_session(season, race, 'R')
            race_obj.load()
            results = race_obj.results[['DriverNumber', 'Position', 'Points', 'GridPosition']]
            results['Season'] = season
            results['RaceNumber'] = race
            results['Circuit'] = race_obj.event['EventName']
            race_data.append(results)
            print(f"✅ Loaded {season} Race {race}: {race_obj.event['EventName']}")
        except Exception as e:
            print(f"❌ Error loading {season} Race {race}: {str(e)}")
            continue

# Add fallback data if not enough real data is loaded
if len(race_data) < 5:
    print("⚠️ Warning: Not enough race data loaded, adding synthetic data")
    # Create synthetic data based on recent season performance
    synthetic_data = []
    
    # Teams from strongest to weakest
    teams = ['Red Bull Racing', 'Ferrari', 'Mercedes', 'McLaren', 'Aston Martin', 
             'Alpine', 'Williams', 'Haas F1 Team', 'Kick Sauber', 'VCARB']
    
    # Create two seasons of synthetic data
    for season in [2023, 2024]:
        for race in range(1, 11):
            for i, team in enumerate(teams):
                # Create 2 drivers per team
                for driver_idx in range(2):
                    position_base = i + 1
                    grid_position_base = i + 1
                    
                    # Add some randomness to positions
                    position = max(1, min(20, position_base + np.random.randint(-2, 3)))
                    grid_position = max(1, min(20, grid_position_base + np.random.randint(-3, 4)))
                    
                    # Points based on position
                    points = 0
                    if position <= 10:
                        points_scale = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
                        points = points_scale[position-1]
                    
                    driver_number = 10 + i*10 + driver_idx
                    
                    # Create a synthetic race result
                    result = pd.DataFrame({
                        'DriverNumber': [driver_number],
                        'Position': [position],
                        'Points': [points],
                        'GridPosition': [grid_position],
                        'Season': [season],
                        'RaceNumber': [race],
                        'Circuit': [f'Synthetic Grand Prix {race}'],
                        'HistoricalTeam': [team]
                    })
                    synthetic_data.append(result)
    
    # Add the synthetic data to race_data
    race_data.extend(synthetic_data)
    print(f"✅ Added {len(synthetic_data)} synthetic race results")

# 📊 Combine all race data
combined_data = pd.concat(race_data)

# 🧮 Process the data to account for team changes
# For rookies and drivers who changed teams, we need special handling
print("\n🔄 Processing driver performance data with team context...")

# Get a reference dataframe of driver numbers from past seasons
seasons_drivers = {}
for season in seasons:
    try:
        schedule = fastf1.get_event_schedule(season)
        first_race = schedule.iloc[0]['RoundNumber']
        session = fastf1.get_session(season, first_race, 'R')
        session.load()
        
        for _, row in session.results.iterrows():
            driver_number = row['DriverNumber']
            driver_abbr = row['Abbreviation']
            driver_team = row['TeamName']
            
            if season not in seasons_drivers:
                seasons_drivers[season] = {}
            
            seasons_drivers[season][driver_number] = {
                'Abbreviation': driver_abbr,
                'Team': driver_team
            }
    except Exception as e:
        print(f"❌ Error loading driver reference data for {season}: {str(e)}")
        
        # Add fallback data for this season
        if season not in seasons_drivers:
            seasons_drivers[season] = {}
            
            # Add synthetic driver data
            teams = ['Red Bull Racing', 'Ferrari', 'Mercedes', 'McLaren', 'Aston Martin', 
                     'Alpine', 'Williams', 'Haas F1 Team', 'Kick Sauber', 'VCARB']
            
            for i, team in enumerate(teams):
                for driver_idx in range(2):
                    driver_number = 10 + i*10 + driver_idx
                    driver_abbr = f"D{driver_number}"
                    
                    seasons_drivers[season][driver_number] = {
                        'Abbreviation': driver_abbr,
                        'Team': team
                    }
            
            print(f"✅ Added fallback driver reference data for {season}")

# 💡 Feature Engineering for combined data
# Add team info from historical data
combined_data['HistoricalTeam'] = None
for idx, row in combined_data.iterrows():
    season = row['Season']
    driver_num = row['DriverNumber']
    
    if season in seasons_drivers and driver_num in seasons_drivers[season]:
        combined_data.at[idx, 'HistoricalTeam'] = seasons_drivers[season][driver_num]['Team']
    else:
        # If we can't find team info, use a default
        combined_data.at[idx, 'HistoricalTeam'] = 'Unknown Team'

# 🔄 Merge with current driver information
combined_data['DriverNumber'] = combined_data['DriverNumber'].astype(int)
full_data = pd.merge(combined_data, drivers_df, on='DriverNumber', how='left')

# Handle rookies and team changes
print("\n🔍 Handling rookies and team changes...")
rookies = []
team_changes = []

for driver in drivers_2025:
    # Check if driver exists in historical data
    historical_data = full_data[full_data['DriverNumber'] == driver['DriverNumber']]
    
    if len(historical_data) == 0:
        rookies.append(driver['FullName'])
    else:
        # Check if team changed
        last_historical_team = historical_data.iloc[-1]['HistoricalTeam']
        if last_historical_team != driver['Team']:
            team_changes.append(f"{driver['FullName']}: {last_historical_team} → {driver['Team']}")

if rookies:
    print(f"🆕 Rookies in 2025: {', '.join(rookies)}")
    # For rookies, we'll use average performance of new drivers in their teams
    
if team_changes:
    print(f"🔄 Team changes for 2025: {', '.join(team_changes)}")
    # For team changes, we'll adjust based on team performance differences

# Convert position to numeric (accounting for DNF, DSQ, etc.)
full_data['Position'] = pd.to_numeric(full_data['Position'], errors='coerce')
full_data['GridPosition'] = pd.to_numeric(full_data['GridPosition'], errors='coerce')

# Fill NaN values with a high number (assuming they're DNF)
full_data['Position'] = full_data['Position'].fillna(25)
full_data['GridPosition'] = full_data['GridPosition'].fillna(25)

# Add track-specific features
full_data['IsChinaGP'] = full_data['Circuit'].str.contains('China', case=False, na=False).astype(int)

# Calculate driver experience (race count up to that point)
full_data['ExperienceCount'] = full_data.groupby('DriverNumber').cumcount() + 1

# Add team strength metric based on recent performance
team_performance = full_data.groupby(['HistoricalTeam', 'Season'])['Position'].mean().reset_index()
team_performance = team_performance.rename(columns={'Position': 'TeamAvgPosition'})
full_data = pd.merge(full_data, team_performance, on=['HistoricalTeam', 'Season'], how='left')

# Fill missing team avg positions with overall average
if full_data['TeamAvgPosition'].isna().any():
    mean_position = full_data['TeamAvgPosition'].mean(skipna=True)
    full_data['TeamAvgPosition'] = full_data['TeamAvgPosition'].fillna(mean_position)

# Encode categorical variables
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
team_encoded = encoder.fit_transform(full_data[['Team']])
team_encoded_df = pd.DataFrame(team_encoded, columns=encoder.get_feature_names_out(['Team']))

circuit_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
circuit_encoded = circuit_encoder.fit_transform(full_data[['Circuit']])
circuit_encoded_df = pd.DataFrame(circuit_encoded, columns=circuit_encoder.get_feature_names_out(['Circuit']))

# 🔗 Combine all features
features = pd.concat([
    full_data[['GridPosition', 'Season', 'IsChinaGP', 'ExperienceCount', 'TeamAvgPosition']],
    team_encoded_df,
], axis=1)

# Add circuit features but limit to avoid too many dimensions
top_circuits = circuit_encoded_df.sum().sort_values(ascending=False).head(15).index
circuit_encoded_df_filtered = circuit_encoded_df[top_circuits]
features = pd.concat([features, circuit_encoded_df_filtered], axis=1)

# Target variable
target = full_data['Position']

# 🧠 Train the model
print("\n🧪 Training the prediction model...")
model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10)
model.fit(features, target)

# 🔮 Feature importance
feature_importance = pd.DataFrame({
    'Feature': features.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n🏆 Top 10 most important features for race position prediction:")
print(feature_importance.head(10))

# Calculate team strength for 2025 predictions
team_strength_2024 = full_data[full_data['Season'] == 2024]
if len(team_strength_2024) > 0:
    team_strength_2024 = team_strength_2024.groupby('HistoricalTeam')['Position'].mean().reset_index()
    team_strength_2024 = team_strength_2024.rename(columns={'HistoricalTeam': 'Team', 'Position': 'TeamAvgPosition'})
else:
    # Create fallback team strength data if 2024 data is missing
    print("⚠️ Warning: No 2024 data available, creating fallback team performance data")
    team_data = [
        {'Team': 'Red Bull Racing', 'TeamAvgPosition': 2.0},
        {'Team': 'Ferrari', 'TeamAvgPosition': 3.0},
        {'Team': 'Mercedes', 'TeamAvgPosition': 3.5},
        {'Team': 'McLaren', 'TeamAvgPosition': 4.0},
        {'Team': 'Aston Martin', 'TeamAvgPosition': 6.0},
        {'Team': 'Alpine', 'TeamAvgPosition': 8.0},
        {'Team': 'Williams', 'TeamAvgPosition': 9.0},
        {'Team': 'Haas F1 Team', 'TeamAvgPosition': 10.0},
        {'Team': 'Kick Sauber', 'TeamAvgPosition': 12.0},
        {'Team': 'VCARB', 'TeamAvgPosition': 11.0}
    ]
    team_strength_2024 = pd.DataFrame(team_data)

# 🚀 Predict Shanghai GP 2025 winner
print("\n🇨🇳 Predicting Shanghai GP 2025 Results...")

# Create prediction data frame with all drivers
shanghai_predictions = []

for driver in drivers_2025:
    # Create a row for each driver
    driver_row = pd.DataFrame({
        'GridPosition': [np.nan],  # Will be replaced with random qualifying position
        'Season': [2025],
        'IsChinaGP': [1],
        'Team': [driver['Team']],
    })
    
    # Add experience count
    exp_count = len(full_data[full_data['DriverNumber'] == driver['DriverNumber']])
    if exp_count == 0:  # Rookie
        exp_count = 1  # Start with 1 race experience
    else:
        exp_count += 23  # Add approximate number of races in 2024 season
    driver_row['ExperienceCount'] = exp_count
    
    # Add team strength
    team_avg = team_strength_2024[team_strength_2024['Team'] == driver['Team']]
    if len(team_avg) > 0:
        driver_row['TeamAvgPosition'] = team_avg.iloc[0]['TeamAvgPosition']
    else:
        # Use overall average for new teams or renamed teams
        driver_row['TeamAvgPosition'] = team_strength_2024['TeamAvgPosition'].mean()
    
    # Add driver info
    driver_row['FullName'] = driver['FullName']
    driver_row['Abbreviation'] = driver['Abbreviation']
    
    shanghai_predictions.append(driver_row)

# Combine all drivers
shanghai_df = pd.concat(shanghai_predictions)

# Simulate qualifying with weighted randomness based on team strength
team_weights = 1 / shanghai_df['TeamAvgPosition']  # Lower avg position = better team

# Check for NaN values and replace them
if team_weights.isna().any():
    # Replace NaN values with the mean of non-NaN values
    mean_weight = team_weights.mean(skipna=True)
    team_weights = team_weights.fillna(mean_weight)

# Ensure no negative or zero values that would cause issues
team_weights = team_weights.clip(lower=0.001)

# Normalize weights
normalized_weights = team_weights / team_weights.sum()

# Lower numbers are better for grid position
grid_order = np.random.choice(
    shanghai_df.index, 
    size=len(shanghai_df), 
    replace=False, 
    p=normalized_weights
)
grid_positions = np.ones(len(shanghai_df)) * 20  # Default to back
grid_positions[grid_order] = np.arange(1, len(shanghai_df) + 1)

shanghai_df['GridPosition'] = grid_positions

# 🎲 Encode the categorical variables for prediction
team_encoded_pred = encoder.transform(shanghai_df[['Team']])
team_encoded_pred_df = pd.DataFrame(
    team_encoded_pred,
    columns=encoder.get_feature_names_out(['Team'])
)

# Add circuit encoding for Shanghai
shanghai_circuit = pd.DataFrame({'Circuit': ['Chinese Grand Prix'] * len(shanghai_df)})
circuit_encoded_pred = circuit_encoder.transform(shanghai_circuit)
circuit_encoded_pred_df = pd.DataFrame(
    circuit_encoded_pred,
    columns=circuit_encoder.get_feature_names_out(['Circuit'])
)

# Filter to match the training features
circuit_encoded_pred_df = circuit_encoded_pred_df.reindex(columns=top_circuits, fill_value=0)

# FIX HERE: Check for duplicate columns before concatenation
print("\nChecking for duplicate columns before concatenation...")
columns_to_use = ['GridPosition', 'Season', 'IsChinaGP', 'ExperienceCount', 'TeamAvgPosition']
print(f"Selected columns from shanghai_df: {columns_to_use}")
print(f"team_encoded_pred_df columns: {team_encoded_pred_df.columns.tolist()}")

# Check for any overlap
overlap = set(columns_to_use) & set(team_encoded_pred_df.columns)
if overlap:
    print(f"Duplicate columns found: {overlap}")
    # Drop duplicates from one of the dataframes
    team_encoded_pred_df = team_encoded_pred_df.drop(columns=list(overlap))

# Now perform the concatenation with reset_index to ensure proper alignment
shanghai_features = pd.concat([
    shanghai_df[columns_to_use].reset_index(drop=True),
    team_encoded_pred_df.reset_index(drop=True)
], axis=1)

# Same check for circuit encoding
overlap = set(shanghai_features.columns) & set(circuit_encoded_pred_df.columns)
if overlap:
    print(f"Duplicate columns found with circuit data: {overlap}")
    circuit_encoded_pred_df = circuit_encoded_pred_df.drop(columns=list(overlap))

# Add circuit columns 
shanghai_features = pd.concat([
    shanghai_features.reset_index(drop=True),
    circuit_encoded_pred_df.reset_index(drop=True)
], axis=1)

# Ensure the column order matches the training data
for col in features.columns:
    if col not in shanghai_features.columns:
        shanghai_features[col] = 0

# Make sure columns are in same order as training
print(f"Training features columns: {features.columns.tolist()}")
print(f"Prediction features columns before reordering: {shanghai_features.columns.tolist()}")
shanghai_features = shanghai_features[features.columns]

# Make prediction
predictions = model.predict(shanghai_features)

# 🏁 Combine predictions with driver info
results = pd.DataFrame({
    'Driver': shanghai_df['FullName'],
    'Team': shanghai_df['Team'],
    'Grid': shanghai_df['GridPosition'],
    'Predicted_Position': predictions,
    'Experience': shanghai_df['ExperienceCount']
})

# Sort by predicted position
results = results.sort_values('Predicted_Position')

# Account for possible ties
results['Final_Position'] = range(1, len(results) + 1)

# 🎉 Display results
print("\n🏁 Predicted Shanghai GP 2025 Results:")
print(results[['Driver', 'Team', 'Grid', 'Final_Position']].head(10))

# 🏆 Winner
winner = results.iloc[0]
second = results.iloc[1]
third = results.iloc[2]

print(f"\n🏆 Predicted Podium for the 2025 Shanghai Grand Prix:")
print(f"🥇 1st: {winner['Driver']} ({winner['Team']}) - Starting P{int(winner['Grid'])}")
print(f"🥈 2nd: {second['Driver']} ({second['Team']}) - Starting P{int(second['Grid'])}")
print(f"🥉 3rd: {third['Driver']} ({third['Team']}) - Starting P{int(third['Grid'])}")

# 📊 Visualize predictions
plt.figure(figsize=(14, 10))
sns.set_style("whitegrid")
ax = sns.barplot(x='Predicted_Position', y='Driver', hue='Team', data=results.sort_values('Predicted_Position'))
plt.title('🏎️ 2025 Shanghai GP Prediction: Expected Finishing Position', fontsize=16)
plt.xlabel('Predicted Position (lower is better)', fontsize=12)
plt.ylabel('Driver', fontsize=12)
plt.legend(title='Team', loc='center right')
plt.tight_layout()
plt.savefig('shanghai_gp_prediction.png', dpi=300)
plt.close()

# 📊 Visualize grid vs predicted finish
plt.figure(figsize=(12, 10))
plt.scatter(results['Grid'], results['Final_Position'], c=results.index, cmap='viridis', s=100)

# Add labels for each driver
for i, row in results.iterrows():
    plt.annotate(row['Driver'].split()[-1], 
                 (row['Grid'], row['Final_Position']),
                 xytext=(5, 0), 
                 textcoords='offset points',
                 fontsize=9)

plt.title('🏎️ Starting Grid vs Predicted Finish - Shanghai GP 2025', fontsize=16)
plt.xlabel('Grid Position', fontsize=12)
plt.ylabel('Predicted Finish', fontsize=12)
plt.grid(True, alpha=0.3)
plt.plot([0, 20], [0, 20], 'r--')  # Reference line: starting = finishing
plt.xlim(0, 21)
plt.ylim(0, 21)
plt.tight_layout()
plt.savefig('grid_vs_finish.png', dpi=300)

# 📊 Team performance visualization
plt.figure(figsize=(12, 8))
team_results = results.groupby('Team')['Predicted_Position'].mean().sort_values().reset_index()

sns.barplot(x='Predicted_Position', y='Team', data=team_results)
plt.title('🏢 Team Performance Prediction - Shanghai GP 2025', fontsize=16)
plt.xlabel('Average Predicted Position (lower is better)', fontsize=12)
plt.ylabel('Team', fontsize=12)
plt.tight_layout()
plt.savefig('team_performance.png', dpi=300)

print("\n📊 Prediction visualizations saved as PNG files")
print("\n✨ Prediction complete! Note that this is a simplified model and actual results may vary.")
print(f"📅 The Shanghai Grand Prix is scheduled for March 23, 2025.")