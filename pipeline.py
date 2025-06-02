import pandas as pd
import numpy as np
import fastf1
import logging
import os
import pickle
from datetime import datetime
from sklearn.metrics import mean_absolute_error

from export_race_details import export_race_details
from estimate_overtakes import overtakes_per_year
from data_utils import (
    _load_historical_data,
    _clean_historical_data,
    _add_driver_team_info,
    _engineer_features,
    _encode_features,
    _load_overtake_stats,
    _get_event_drivers,
    _get_qualifying_results,
    _get_fp3_results,
    _get_sprint_results,
    _season_driver_team_stats,
    fetch_weather,
    CIRCUIT_METADATA,
    GRAND_PRIX_LIST,
    race_cols,
    DELTA_NEXT_PENALTY,
)
from model_utils import _train_model, _rank_metrics, SeasonSplit

logger = logging.getLogger(__name__)

try:
    import shap  # type: ignore
except Exception:
    shap = None


def predict_race(
    grand_prix,
    year=2025,
    export_details=False,
    debug=False,
    compute_overtakes=True,
    retrain=False,
):
    cache_model_path = f"cache/model_{year}.pkl"
    cache_data_path = f"cache/race_data_{year}.parquet"

    schedule = fastf1.get_event_schedule(year)
    match = schedule[schedule["EventName"].str.contains(grand_prix, case=False, na=False)]
    if match.empty:
        raise ValueError(f"Grand Prix '{grand_prix}' not found for {year}")
    this_race_number = int(match.iloc[0]["RoundNumber"])
    event_date = pd.to_datetime(match.iloc[0].get("EventDate"), errors="coerce")
    event_month = event_date.month
    event_day = event_date.day

    limit_rounds = {year: this_race_number - 1}

    if os.path.exists(cache_model_path) and not retrain:
        model = pickle.load(open(cache_model_path, "rb"))
        if os.path.exists(cache_data_path):
            race_data = pd.read_parquet(cache_data_path)
        else:
            seasons = list(range(2022, year + 1))
            overtake_map = _load_overtake_stats()
            race_data = _load_historical_data(seasons, overtake_map, limit_rounds)
            race_data = _engineer_features(race_data)
            race_data.to_parquet(cache_data_path)
        logger.info("Loaded cached model and data for %s %d", grand_prix, year)
        results = _predict_with_existing_model(model, race_data, grand_prix, year)
        return (results, None) if debug else results

    seasons = list(range(2022, year + 1))

    overtake_map = _load_overtake_stats()
    if compute_overtakes:
        try:
            years_for_avg = list(range(max(2022, year - 3), year))
            per_year = overtakes_per_year(grand_prix, years_for_avg)
            overtake_map.setdefault(grand_prix, {}).update(per_year)
        except Exception as err:
            logger.warning(
                "Could not compute overtakes for %s: %s", grand_prix, err
            )

    race_data = _load_historical_data(seasons, overtake_map, limit_rounds)
    race_data = _clean_historical_data(race_data).reset_index(drop=True)
    race_data['DriverNumber'] = pd.to_numeric(race_data['DriverNumber'], errors='coerce')
    qual_results = None
    fp3_results = None
    race_data = _add_driver_team_info(race_data, seasons)
    race_data = race_data.drop(columns=['Team'], errors='ignore')
    race_data = race_data.loc[
        ~((race_data['Season'] == year) & (race_data['RaceNumber'] >= this_race_number))
    ].reset_index(drop=True)
    race_data = _engineer_features(race_data)
    race_data.sort_values(["Season", "RaceNumber"], inplace=True)
    race_data.reset_index(drop=True, inplace=True)
    race_data.to_csv("prediction_data.csv", index=False)


    holdout_year = year - 1
    holdout_df = race_data[race_data['Season'] == holdout_year]
    train_df = race_data[race_data['Season'] < holdout_year]
    holdout_mae = None
    holdout_rank = None
    if not holdout_df.empty and not train_df.empty:
        (
            ho_feat,
            ho_team_enc,
            ho_circ_enc,
            ho_top_circuits,
        ) = _encode_features(train_df, race_cols)
        ho_val_feat, _, _, _ = _encode_features(
            holdout_df,
            race_cols,
            ho_team_enc,
            ho_circ_enc,
            ho_top_circuits,
        )
        ho_cv = SeasonSplit(sorted(train_df['Season'].unique()))
        ho_model, _ = _train_model(ho_feat, train_df['Position'], ho_cv, debug)
        ho_preds = ho_model.predict(ho_val_feat)
        holdout_mae = mean_absolute_error(holdout_df['Position'], ho_preds)
        holdout_rank = _rank_metrics(holdout_df['Position'], ho_preds)
    (
        features,
        team_enc,
        circuit_enc,
        top_circuits,
    ) = _encode_features(race_data, race_cols)
    features = features.loc[race_data.index].reset_index(drop=True)

    cv = SeasonSplit(sorted(race_data['Season'].unique()))

    target = race_data['Position']
    model, cv_rho = _train_model(features, target, cv, debug)

    # Cache the fully trained model and engineered data
    os.makedirs("cache", exist_ok=True)
    pickle.dump(model, open(f"cache/model_{year}.pkl", "wb"))
    if "race_data" in locals():
        race_data.to_parquet(f"cache/race_data_{year}.parquet")
    logger.info(
        "Cached model to %s and data to %s",
        cache_model_path,
        cache_data_path,
    )

    finish_preds_hist = model.predict(features)
    finish_mae = mean_absolute_error(race_data['Position'], finish_preds_hist)
    train_rank = _rank_metrics(race_data['Position'], finish_preds_hist)

    try:
        drivers_df = _get_qualifying_results(year, grand_prix)
        drivers_df = drivers_df[drivers_df['BestTime'].notna()]
        qual_results = drivers_df.copy()
    except Exception:
        drivers_df = _get_event_drivers(year, grand_prix)
        qual_results = None

    drivers_df['DriverNumber'] = pd.to_numeric(drivers_df['DriverNumber'], errors='coerce')

    try:
        fp3_results = _get_fp3_results(year, grand_prix)
        if qual_results is not None:
            qual_results = qual_results.merge(fp3_results, on='Abbreviation', how='left')
    except Exception:
        fp3_results = None

    if qual_results is not None and not qual_results.empty:
        default_best_q = qual_results['BestTime'].median()
        default_delta_next = qual_results.get('DeltaToNext', pd.Series()).mean()
        default_delta_q3 = qual_results.get('DeltaToNext_Q3', pd.Series()).mean()
        default_delta_q2 = qual_results.get('DeltaToNext_Q2', pd.Series()).mean()
        if pd.isna(default_delta_next):
            delta_series = (
                qual_results.sort_values('BestTime')['BestTime'].diff(-1).abs()
            )
            default_delta_next = delta_series.mean()
        if pd.isna(default_delta_q3):
            default_delta_q3 = DELTA_NEXT_PENALTY
        if pd.isna(default_delta_q2):
            default_delta_q2 = DELTA_NEXT_PENALTY
    else:
        default_best_q = race_data['BestQualiTime'].median()
        default_delta_next = race_data['DeltaToNext'].mean()
        default_delta_q3 = race_data.get('DeltaToNext_Q3', pd.Series()).mean()
        default_delta_q2 = race_data.get('DeltaToNext_Q2', pd.Series()).mean()
        if pd.isna(default_delta_q3):
            default_delta_q3 = DELTA_NEXT_PENALTY
        if pd.isna(default_delta_q2):
            default_delta_q2 = DELTA_NEXT_PENALTY

    hist_air = race_data['AirTemp'].mean()
    hist_track = race_data['TrackTemp'].mean()
    hist_rain = race_data['Rainfall'].median()
    if fp3_results is not None:
        forecast = fetch_weather(grand_prix)
        if forecast:
            f_air = forecast['ForecastAirTemp']
            f_track = f_air + 10
            f_rain = forecast['ForecastPrecipChance']
        else:
            f_air = hist_air
            f_track = hist_track
            f_rain = hist_rain

        event_date_dt = datetime(year, event_month, event_day)
        today = datetime.now()
        days_to_race = max(0, (event_date_dt - today).days)
        alpha = max(0.1, min(0.9, 1 - days_to_race / 30))

        default_air = alpha * f_air + (1 - alpha) * hist_air
        default_track = alpha * f_track + (1 - alpha) * hist_track
        default_rain = alpha * f_rain + (1 - alpha) * hist_rain
        default_fp3 = fp3_results['FP3BestTime'].mean()
        default_fp3_long = fp3_results['FP3LongRunTime'].mean()
    else:
        default_air, default_track, default_rain = hist_air, hist_track, hist_rain
        default_fp3 = race_data['FP3BestTime'].mean()
        default_fp3_long = race_data['FP3LongRunTime'].mean()
    default_overtake = race_data['Overtakes_CurrentYear'].mean()

    try:
        from fastf1.circuit_info import get_circuit_info
        circuit_lengths = {}
        corners_map = {}
        drs_map = {}
        lap_map = {}
        info = get_circuit_info(grand_prix)
        length = (
            info.get('Length')
            or info.get('CircuitLength')
            or info.get('circuitLength')
        )
        corners = (
            info.get('NumberOfTurns')
            or info.get('Turns')
            or info.get('numCorners')
        )
        drs = (
            info.get('NumberOfDRSZones')
            or info.get('DRSZonesCount')
            or info.get('drsZones')
        )
        laptime = (
            info.get('LapTimeAvg')
            or info.get('LapRecord')
            or info.get('lapRecord')
        )
        if isinstance(length, str):
            length = str(length).replace(' km', '')
        if isinstance(laptime, str):
            try:
                laptime = pd.to_timedelta(laptime).total_seconds()
            except Exception:
                laptime = pd.to_numeric(laptime, errors='coerce')
        circuit_lengths[grand_prix] = pd.to_numeric(length, errors='coerce')
        corners_map[grand_prix] = pd.to_numeric(corners, errors='coerce')
        drs_map[grand_prix] = pd.to_numeric(drs, errors='coerce')
        lap_map[grand_prix] = pd.to_numeric(laptime, errors='coerce')
    except Exception:
        meta = CIRCUIT_METADATA.get(grand_prix, {})
        circuit_lengths = {grand_prix: np.nan}
        corners_map = {grand_prix: meta.get('NumCorners', np.nan)}
        drs_map = {grand_prix: meta.get('DRSZones', np.nan)}
        lap_map = {grand_prix: meta.get('StdLapTime', np.nan)}

    TRACK_TYPE = {
        'Monaco Grand Prix': 'street',
        'Singapore Grand Prix': 'street',
        'Las Vegas Grand Prix': 'street',
    }
    DOWNFORCE = {
        'Monaco Grand Prix': 'high',
        'Hungarian Grand Prix': 'high',
        'Italian Grand Prix': 'low',
        'Belgian Grand Prix': 'low',
    }
    df_level_map = {'low': 0, 'medium': 1, 'high': 2}
    track_type_val = TRACK_TYPE.get(grand_prix, 'permanent')
    is_street_val = 1 if track_type_val == 'street' else 0
    downforce_val = DOWNFORCE.get(grand_prix, 'medium')
    downforce_level_val = df_level_map.get(downforce_val, 1)
    circuit_length_val = circuit_lengths.get(grand_prix, np.nan)
    num_corners_val = corners_map.get(grand_prix, np.nan)
    drs_zones_val = drs_map.get(grand_prix, np.nan)
    std_lap_time_val = lap_map.get(grand_prix, np.nan)

    pred_rows = []
    default_team_avg = race_data['Position'].mean()
    default_team_quali = race_data['GridPosition'].mean()

    (
        driver_stats_lookup,
        driver_pts_map,
        constructor_pts_map,
        driver_stand_map,
        constructor_stand_map,
        team_strength,
        prev_rank_map,
        default_prev_rank,
    ) = _season_driver_team_stats(race_data, year)

    if qual_results is not None and not qual_results.empty:
        fastest = qual_results['BestTime'].min()
        qual_results['DeltaToBestQuali'] = qual_results['BestTime'] - fastest
        qual_results = qual_results.sort_values('BestTime')
        qual_results['DeltaToNext'] = qual_results['BestTime'].diff(-1).abs()
        qual_results = qual_results.sort_index()
        team_mean = qual_results.groupby('Team')['BestTime'].transform('mean')
        team_size = qual_results.groupby('Team')['BestTime'].transform('size')
        qual_results['DeltaToTeammateQuali'] = np.where(team_size > 1,
                                                       (qual_results['BestTime'] - team_mean) * 2,
                                                       0)
        if 'Q1' in qual_results.columns and 'Q3' in qual_results.columns:
            qual_results['QualiSessionGain'] = qual_results['Q1'] - qual_results['Q3']
            std = qual_results['QualiSessionGain'].std()
            qual_results['QualiSessionGain'] = (
                (qual_results['QualiSessionGain'] - qual_results['QualiSessionGain'].mean()) / std
            ) if std != 0 else 0
        else:
            qual_results['QualiSessionGain'] = 0
        qual_results['GridDropCount'] = 0

    driver_iter = qual_results if qual_results is not None and not qual_results.empty else drivers_df
    overall_avg_pos = race_data['Position'].mean()
    rookie_avg_pos = race_data[race_data['ExperienceCount'] == 1]['Position'].mean()
    for _, d in driver_iter.iterrows():
        exp_count = len(race_data[race_data['DriverNumber'] == d['DriverNumber']]) + 1
        team_name = d['Team']
        team_same_season = race_data[
            (race_data['HistoricalTeam'] == team_name) &
            (race_data['Season'] == year)
        ].sort_values('RaceNumber')
        team_prev = team_same_season[team_same_season['RaceNumber'] < this_race_number]

        if len(team_prev) > 0:
            team_avg_pos = team_prev['Position'].mean()
        else:
            team_avg_pos = team_strength.get(team_name, default_team_avg)

        team_prev_q = team_prev['GridPosition']
        if len(team_prev_q) > 0:
            team_recent_q = team_prev_q.tail(5).mean()
        else:
            team_recent_q = default_team_quali

        team_prev_f = team_prev['Position']
        if len(team_prev_f) > 0:
            team_recent_f = team_prev_f.tail(5).mean()
        else:
            team_recent_f = default_team_avg

        team_prev_dnf = team_prev['DidNotFinish']
        if len(team_prev_dnf) > 0:
            team_rel = team_prev_dnf.tail(5).sum()
        else:
            team_rel = 0.0

        stats = driver_stats_lookup.loc[(d['DriverNumber'], grand_prix)] if (d['DriverNumber'], grand_prix) in driver_stats_lookup.index else None
        if stats is None:
            avg_track = race_data['DriverAvgTrackFinish'].mean()
            podiums = 0.0
            dnfs = 0.0
        else:
            avg_track = stats['DriverAvgTrackFinish']
            podiums = stats['DriverTrackPodiums']
            dnfs = stats['DriverTrackDNFs']

        if qual_results is not None and 'GridPosition' in d and pd.notna(d['GridPosition']):
            grid_pos = int(d['GridPosition'])
            best_time = d['BestTime']
        else:
            grid_pos = 20
            best_time = default_best_q

        if fp3_results is not None and 'FP3BestTime' in d and pd.notna(d['FP3BestTime']):
            fp3_time = d['FP3BestTime']
        else:
            fp3_time = default_fp3

        if fp3_results is not None and 'FP3LongRunTime' in d and pd.notna(d['FP3LongRunTime']):
            fp3_long_time = d['FP3LongRunTime']
        else:
            fp3_long_time = default_fp3_long

        driver_num = int(d['DriverNumber'])
        past_races = race_data[
            (race_data['DriverNumber'] == driver_num) & (
                (race_data['Season'] < year) |
                ((race_data['Season'] == year) & (race_data['RaceNumber'] < this_race_number))
            )
        ].sort_values(['Season', 'RaceNumber'])

        if past_races.empty:
            cross_avg = rookie_avg_pos
            recent_avg_pts = 0.0
            recent3_avg = rookie_avg_pos
            recent5_avg = rookie_avg_pos
        else:
            cross_avg = past_races['Position'].tail(5).mean()
            recent_avg_pts = past_races['Points'].tail(5).mean()
            recent3_avg = past_races['Position'].tail(3).mean()
            recent5_avg = past_races['Position'].tail(5).mean()
        pred_rows.append({
            'GridPosition': grid_pos,
            'Season': year,
            'ExperienceCount': exp_count,
            'IsRookie': 1 if exp_count == 1 else 0,
            'TeamAvgPosition': team_avg_pos,
            'CrossAvgFinish': cross_avg,
            'RecentAvgPoints': recent_avg_pts,
            'BestQualiTime': best_time,
            'FP3BestTime': fp3_time,
            'FP3LongRunTime': fp3_long_time,
            'DeltaToBestQuali': d.get('DeltaToBestQuali', 0),
            'DeltaToNext': d.get('DeltaToNext', default_delta_next),
            'DeltaToNext_Q3': d.get('DeltaToNext_Q3', default_delta_q3),
            'DeltaToNext_Q2': d.get('DeltaToNext_Q2', default_delta_q2),
            'MissedQ3': d.get('MissedQ3', 1),
            'MissedQ2': d.get('MissedQ2', 1),
            'SprintFinish': d.get('SprintFinish'),
            'HasSprint': 1 if has_sprint else 0,
            'Recent3AvgFinish': recent3_avg,
            'Recent5AvgFinish': recent5_avg,
            'DriverAvgTrackFinish': avg_track,
            'DriverTrackPodiums': podiums,
            'DriverTrackDNFs': dnfs,
            'TeamRecentQuali': team_recent_q,
            'TeamRecentFinish': team_recent_f,
            'TeamReliability': team_rel,
            'DriverChampPoints': driver_pts_map.get(d['DriverNumber'], 0.0),
            'ConstructorChampPoints': constructor_pts_map.get(d['Team'], 0.0),
            'DriverStanding': int(driver_stand_map.get(d['DriverNumber'], 0)),
            'ConstructorStanding': int(constructor_stand_map.get(d['Team'], 0)),
            'PrevYearConstructorRank': prev_rank_map.get(team_name, default_prev_rank),
            'CircuitLength': circuit_length_val,
            'NumCorners': num_corners_val,
            'DRSZones': drs_zones_val,
            'StdLapTime': std_lap_time_val,
            'IsStreet': is_street_val,
            'DownforceLevel': downforce_level_val,
            'AirTemp': default_air,
            'TrackTemp': default_track,
            'Rainfall': default_rain,
            'Overtakes_CurrentYear': default_overtake,
            'Team': d['Team'],
            'FullName': d['FullName'],
            'Abbreviation': d['Abbreviation']
        })

    pred_df = pd.DataFrame(pred_rows)

    if pred_df.empty:
        raise ValueError(f"No driver data available for {year} {grand_prix}")

    driver_sprint_mean = race_data.groupby("DriverNumber")["SprintFinish"].mean()
    team_sprint_mean = race_data.groupby("HistoricalTeam")["SprintFinish"].mean()

    pred_df['GridPosition'] = (
        pd.to_numeric(pred_df['GridPosition'], errors='coerce')
        .fillna(20)
    )
    pred_df['GridPosition'] = pred_df['GridPosition'].clip(1, 20)
    pred_df['BestQualiTime'] = pd.to_numeric(pred_df['BestQualiTime'], errors='coerce')
    pred_df['MissedQuali'] = pred_df['BestQualiTime'].isna().astype(int)
    pred_df['BestQualiTime'] = pred_df['BestQualiTime'].fillna(
        race_data['BestQualiTime'].median()
    )
    pred_df['DeltaToNext'] = pd.to_numeric(pred_df['DeltaToNext'], errors='coerce').fillna(default_delta_next)
    pred_df['DeltaToNext_Q3'] = pd.to_numeric(pred_df['DeltaToNext_Q3'], errors='coerce').fillna(default_delta_q3)
    pred_df['DeltaToNext_Q2'] = pd.to_numeric(pred_df['DeltaToNext_Q2'], errors='coerce').fillna(default_delta_q2)
    pred_df['MissedQ3'] = pd.to_numeric(pred_df['MissedQ3'], errors='coerce').fillna(1).astype(int)
    pred_df['MissedQ2'] = pd.to_numeric(pred_df['MissedQ2'], errors='coerce').fillna(1).astype(int)
    pred_df['FP3BestTime'] = (
        pd.to_numeric(pred_df['FP3BestTime'], errors='coerce')
        .fillna(race_data['FP3BestTime'].mean())
    )
    pred_df['FP3LongRunTime'] = (
        pd.to_numeric(pred_df['FP3LongRunTime'], errors='coerce')
        .fillna(race_data['FP3LongRunTime'].mean())
    )
    pred_df['SprintFinish'] = pd.to_numeric(pred_df.get('SprintFinish'), errors='coerce')
    pred_df['HasSprint'] = pred_df['HasSprint'].fillna(0).astype(int)
    pred_df['SprintFinish'] = pred_df.apply(
        lambda r: driver_sprint_mean.get(r['DriverNumber']) if pd.isna(r['SprintFinish']) else r['SprintFinish'],
        axis=1
    )
    pred_df['SprintFinish'] = pred_df.apply(
        lambda r: team_sprint_mean.get(r['Team']) if pd.isna(r['SprintFinish']) else r['SprintFinish'],
        axis=1
    )
    pred_df['SprintFinish'] = pred_df['SprintFinish'].fillna(race_data['SprintFinish'].mean())
    pred_df['SprintFinish'] = pred_df['SprintFinish'].clip(1, 20)
    pred_df['NumCorners'] = (
        pd.to_numeric(pred_df['NumCorners'], errors='coerce')
        .fillna(race_data['NumCorners'].median())
    )
    pred_df['DRSZones'] = (
        pd.to_numeric(pred_df['DRSZones'], errors='coerce')
        .fillna(race_data['DRSZones'].median())
    )
    pred_df['StdLapTime'] = (
        pd.to_numeric(pred_df['StdLapTime'], errors='coerce')
        .fillna(race_data['StdLapTime'].mean())
    )

    pred_df['Month'] = event_month

    air_map = race_data.groupby(['Circuit', 'Month'])['AirTemp'].median()
    air_val = air_map.get((grand_prix, event_month), np.nan)
    pred_df['AirTemp'] = pd.to_numeric(pred_df['AirTemp'], errors='coerce')
    pred_df['AirTemp'] = pred_df['AirTemp'].fillna(air_val)
    pred_df['AirTemp'] = pred_df['AirTemp'].fillna(race_data['AirTemp'].mean())

    track_map = race_data.groupby(['Circuit', 'Month'])['TrackTemp'].median()
    track_val = track_map.get((grand_prix, event_month), np.nan)
    pred_df['TrackTemp'] = pd.to_numeric(pred_df['TrackTemp'], errors='coerce')
    pred_df['TrackTemp'] = pred_df['TrackTemp'].fillna(track_val)
    pred_df['TrackTemp'] = pred_df['TrackTemp'].fillna(race_data['TrackTemp'].mean())

    pred_df['Rainfall'] = pd.to_numeric(pred_df['Rainfall'], errors='coerce')
    pred_df['RainfallMissing'] = pred_df['Rainfall'].isna().astype(int)
    rain_map = race_data.groupby(['Circuit', 'Month'])['Rainfall'].median()
    rain_val = rain_map.get((grand_prix, event_month), np.nan)
    pred_df['Rainfall'] = pred_df['Rainfall'].fillna(rain_val)
    circ_rain_map = race_data.groupby('Circuit')['Rainfall'].median()
    circuit_rain = circ_rain_map.get(grand_prix, race_data['Rainfall'].median())
    pred_df['Rainfall'] = pred_df['Rainfall'].fillna(circuit_rain)
    pred_df['Rainfall'] = pred_df['Rainfall'].fillna(race_data['Rainfall'].median())
    pred_df = pred_df.drop(columns=['Month'], errors='ignore')

    pred_df.to_csv("prediction_input.csv", index=False)

    race_pred_features, _, _, _ = _encode_features(
        pred_df.assign(Circuit=grand_prix),
        race_cols,
        team_enc,
        circuit_enc,
        top_circuits,
    )

    pred_features = race_pred_features.reindex(columns=features.columns, fill_value=0)

    preds = model.predict(pred_features)
    shap_values = None
    if debug and shap is not None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pred_features)
        except Exception as err:
            logger.warning("Could not compute SHAP values: %s", err)
            shap_values = None

    results = pd.DataFrame({
        'Driver': pred_df['Abbreviation'],
        'Team': pred_df['Team'],
        'Grid': pred_df['GridPosition'],
        'Predicted_Position': preds
    })
    sort_idx = results['Predicted_Position'].argsort()
    results = results.iloc[sort_idx].reset_index(drop=True)
    if shap_values is not None:
        shap_values = shap_values[sort_idx]
    results['Final_Position'] = range(1, len(results) + 1)
    results.to_csv("prediction_results.csv", index=False)
    if export_details:
        try:
            detail_path = export_race_details(year, grand_prix)
            logger.info("Saved session data to %s", detail_path)
        except Exception as err:
            logger.warning("Could not export session data: %s", err)
    details = None
    if debug:
        details = {
            'prediction_features': pred_features,
            'shap_values': shap_values,
            'feature_names': pred_features.columns.tolist(),
        }

    if holdout_mae is not None:
        logger.info(
            "CV Spearman: %.2f -- Hold-out MAE: %.2f -- Training MAE: %.2f",
            cv_rho,
            holdout_mae,
            finish_mae,
        )
        logger.info(
            "Spearman \u03c1: %.2f (train) / %.2f (hold-out) -- "
            "Top1: %.0f%% / %.0f%% -- Top3: %.0f%% / %.0f%%",
            train_rank["spearman"],
            holdout_rank["spearman"],
            train_rank["top1"] * 100,
            holdout_rank["top1"] * 100,
            train_rank["top3"] * 100,
            holdout_rank["top3"] * 100,
        )
    else:
        logger.info(
            "CV Spearman: %.2f -- Training MAE: %.2f", cv_rho, finish_mae
        )
        logger.info(
            "Spearman \u03c1: %.2f -- Top1: %.0f%% -- Top3: %.0f%%",
            train_rank["spearman"],
            train_rank["top1"] * 100,
            train_rank["top3"] * 100,
        )
    return (results, details) if debug else results


def _predict_with_existing_model(model, race_data, grand_prix, year):
    """Predict race results using a pre-trained model and cached data."""
    # Historical data may use either "Circuit" or legacy "GrandPrix" for the
    # event name. Support both to avoid KeyError when only one exists.
    gp_col = "GrandPrix" if "GrandPrix" in race_data.columns else "Circuit"

    schedule = fastf1.get_event_schedule(year)
    match = schedule[schedule["EventName"].str.contains(grand_prix, case=False, na=False)]
    if match.empty:
        raise ValueError(f"Grand Prix '{grand_prix}' not found for {year}")

    mask = (race_data["Season"] == year) & (race_data[gp_col] == grand_prix)
    if mask.any():
        this_race_number = race_data.loc[mask, "RaceNumber"].iloc[0]
    else:
        this_race_number = int(match.iloc[0]["RoundNumber"])

    mask_past = ~(
        (race_data["Season"] == year)
        & (race_data["RaceNumber"] >= this_race_number)
    )
    race_data = race_data.loc[mask_past].reset_index(drop=True)

    event_date = pd.to_datetime(match.iloc[0].get("EventDate"), errors="coerce")
    event_month = event_date.month
    event_day = event_date.day

    pred_df = _build_pred_df(
        race_data,
        grand_prix,
        year,
        this_race_number,
        event_month,
        event_day,
    )

    features, team_enc, circ_enc, top_circuits = _encode_features(race_data, race_cols)
    race_pred_features, _, _, _ = _encode_features(
        pred_df.assign(Circuit=grand_prix),
        race_cols,
        team_enc,
        circ_enc,
        top_circuits,
    )
    expected_cols = getattr(model.get_booster(), "feature_names", None)
    if expected_cols is None:
        expected_cols = features.columns
    race_pred_features = race_pred_features.reindex(columns=expected_cols, fill_value=0)
    preds = model.predict(race_pred_features)

    # Build the same result frame as during training
    results = pd.DataFrame({
        "Driver": pred_df["Abbreviation"],
        "Team": pred_df["Team"],
        "Grid": pred_df["GridPosition"],
        "Predicted_Position": preds,
    })
    sort_idx = results["Predicted_Position"].argsort()
    results = results.iloc[sort_idx].reset_index(drop=True)
    results["Final_Position"] = range(1, len(results) + 1)
    return results


def _build_pred_df(race_data, grand_prix, year, this_race_number, event_month, event_day):
    """Assemble a prediction input frame for a single event."""

    # Some cached datasets from older versions dropped the "Month" column.
    # Add it back if necessary so downstream aggregations do not fail.
    if "Month" not in race_data.columns:
        if "Date" in race_data.columns:
            race_data = race_data.assign(
                Month=pd.to_datetime(race_data["Date"], errors="coerce").dt.month
            )
        else:
            race_data = race_data.assign(Month=np.nan)
    try:
        drivers_df = _get_qualifying_results(year, grand_prix)
        drivers_df = drivers_df[drivers_df["BestTime"].notna()]
        qual_results = drivers_df.copy()
    except Exception:
        drivers_df = _get_event_drivers(year, grand_prix)
        qual_results = None

    drivers_df["DriverNumber"] = pd.to_numeric(drivers_df["DriverNumber"], errors="coerce")

    try:
        fp3_results = _get_fp3_results(year, grand_prix)
        if qual_results is not None:
            qual_results = qual_results.merge(fp3_results, on="Abbreviation", how="left")
    except Exception:
        fp3_results = None

    try:
        sprint_results = _get_sprint_results(year, grand_prix)
        has_sprint = not sprint_results.empty
        if qual_results is not None:
            qual_results = qual_results.merge(sprint_results, on="Abbreviation", how="left")
        drivers_df = drivers_df.merge(sprint_results, on="Abbreviation", how="left")
    except Exception:
        sprint_results = pd.DataFrame()
        has_sprint = False

    if qual_results is not None and not qual_results.empty:
        default_best_q = qual_results["BestTime"].median()
        default_delta_next = qual_results.get("DeltaToNext", pd.Series()).mean()
        default_delta_q3 = qual_results.get("DeltaToNext_Q3", pd.Series()).mean()
        default_delta_q2 = qual_results.get("DeltaToNext_Q2", pd.Series()).mean()
        if pd.isna(default_delta_next):
            delta_series = (
                qual_results.sort_values("BestTime")["BestTime"].diff(-1).abs()
            )
            default_delta_next = delta_series.mean()
        if pd.isna(default_delta_q3):
            default_delta_q3 = DELTA_NEXT_PENALTY
        if pd.isna(default_delta_q2):
            default_delta_q2 = DELTA_NEXT_PENALTY
    else:
        default_best_q = race_data["BestQualiTime"].median()
        default_delta_next = race_data["DeltaToNext"].mean()
        default_delta_q3 = race_data.get("DeltaToNext_Q3", pd.Series()).mean()
        default_delta_q2 = race_data.get("DeltaToNext_Q2", pd.Series()).mean()
        if pd.isna(default_delta_q3):
            default_delta_q3 = DELTA_NEXT_PENALTY
        if pd.isna(default_delta_q2):
            default_delta_q2 = DELTA_NEXT_PENALTY

    hist_air = race_data["AirTemp"].mean()
    hist_track = race_data["TrackTemp"].mean()
    hist_rain = race_data["Rainfall"].median()
    if fp3_results is not None:
        forecast = fetch_weather(grand_prix)
        if forecast:
            f_air = forecast["ForecastAirTemp"]
            f_track = f_air + 10
            f_rain = forecast["ForecastPrecipChance"]
        else:
            f_air = hist_air
            f_track = hist_track
            f_rain = hist_rain

        event_date_dt = datetime(year, event_month, event_day)
        today = datetime.now()
        days_to_race = max(0, (event_date_dt - today).days)
        alpha = max(0.1, min(0.9, 1 - days_to_race / 30))

        default_air = alpha * f_air + (1 - alpha) * hist_air
        default_track = alpha * f_track + (1 - alpha) * hist_track
        default_rain = alpha * f_rain + (1 - alpha) * hist_rain
        default_fp3 = fp3_results["FP3BestTime"].mean()
        default_fp3_long = fp3_results["FP3LongRunTime"].mean()
    else:
        default_air, default_track, default_rain = hist_air, hist_track, hist_rain
        default_fp3 = race_data["FP3BestTime"].mean()
        default_fp3_long = race_data["FP3LongRunTime"].mean()
    default_overtake = race_data["Overtakes_CurrentYear"].mean()

    try:
        from fastf1.circuit_info import get_circuit_info
        circuit_lengths = {}
        corners_map = {}
        drs_map = {}
        lap_map = {}
        info = get_circuit_info(grand_prix)
        length = (
            info.get("Length")
            or info.get("CircuitLength")
            or info.get("circuitLength")
        )
        corners = (
            info.get("NumberOfTurns")
            or info.get("Turns")
            or info.get("numCorners")
        )
        drs = (
            info.get("NumberOfDRSZones")
            or info.get("DRSZonesCount")
            or info.get("drsZones")
        )
        laptime = (
            info.get("LapTimeAvg")
            or info.get("LapRecord")
            or info.get("lapRecord")
        )
        if isinstance(length, str):
            length = str(length).replace(" km", "")
        if isinstance(laptime, str):
            try:
                laptime = pd.to_timedelta(laptime).total_seconds()
            except Exception:
                laptime = pd.to_numeric(laptime, errors="coerce")
        circuit_lengths[grand_prix] = pd.to_numeric(length, errors="coerce")
        corners_map[grand_prix] = pd.to_numeric(corners, errors="coerce")
        drs_map[grand_prix] = pd.to_numeric(drs, errors="coerce")
        lap_map[grand_prix] = pd.to_numeric(laptime, errors="coerce")
    except Exception:
        meta = CIRCUIT_METADATA.get(grand_prix, {})
        circuit_lengths = {grand_prix: np.nan}
        corners_map = {grand_prix: meta.get("NumCorners", np.nan)}
        drs_map = {grand_prix: meta.get("DRSZones", np.nan)}
        lap_map = {grand_prix: meta.get("StdLapTime", np.nan)}

    TRACK_TYPE = {
        "Monaco Grand Prix": "street",
        "Singapore Grand Prix": "street",
        "Las Vegas Grand Prix": "street",
    }
    DOWNFORCE = {
        "Monaco Grand Prix": "high",
        "Hungarian Grand Prix": "high",
        "Italian Grand Prix": "low",
        "Belgian Grand Prix": "low",
    }
    df_level_map = {"low": 0, "medium": 1, "high": 2}
    track_type_val = TRACK_TYPE.get(grand_prix, "permanent")
    is_street_val = 1 if track_type_val == "street" else 0
    downforce_val = DOWNFORCE.get(grand_prix, "medium")
    downforce_level_val = df_level_map.get(downforce_val, 1)
    circuit_length_val = circuit_lengths.get(grand_prix, np.nan)
    num_corners_val = corners_map.get(grand_prix, np.nan)
    drs_zones_val = drs_map.get(grand_prix, np.nan)
    std_lap_time_val = lap_map.get(grand_prix, np.nan)

    pred_rows = []
    default_team_avg = race_data["Position"].mean()
    default_team_quali = race_data["GridPosition"].mean()

    (
        driver_stats_lookup,
        driver_pts_map,
        constructor_pts_map,
        driver_stand_map,
        constructor_stand_map,
        team_strength,
        prev_rank_map,
        default_prev_rank,
    ) = _season_driver_team_stats(race_data, year)

    if qual_results is not None and not qual_results.empty:
        fastest = qual_results["BestTime"].min()
        qual_results["DeltaToBestQuali"] = qual_results["BestTime"] - fastest
        qual_results = qual_results.sort_values("BestTime")
        qual_results["DeltaToNext"] = qual_results["BestTime"].diff(-1).abs()
        qual_results = qual_results.sort_index()
        team_mean = qual_results.groupby("Team")["BestTime"].transform("mean")
        team_size = qual_results.groupby("Team")["BestTime"].transform("size")
        qual_results["DeltaToTeammateQuali"] = np.where(
            team_size > 1,
            (qual_results["BestTime"] - team_mean) * 2,
            0,
        )
        if "Q1" in qual_results.columns and "Q3" in qual_results.columns:
            qual_results["QualiSessionGain"] = qual_results["Q1"] - qual_results["Q3"]
            std = qual_results["QualiSessionGain"].std()
            qual_results["QualiSessionGain"] = (
                (qual_results["QualiSessionGain"] - qual_results["QualiSessionGain"].mean()) / std
            ) if std != 0 else 0
        else:
            qual_results["QualiSessionGain"] = 0
        qual_results["GridDropCount"] = 0

    driver_iter = qual_results if qual_results is not None and not qual_results.empty else drivers_df
    overall_avg_pos = race_data["Position"].mean()
    rookie_avg_pos = race_data[race_data["ExperienceCount"] == 1]["Position"].mean()
    for _, d in driver_iter.iterrows():
        exp_count = len(race_data[race_data["DriverNumber"] == d["DriverNumber"]]) + 1
        team_name = d["Team"]
        team_same_season = race_data[
            (race_data["HistoricalTeam"] == team_name)
            & (race_data["Season"] == year)
        ].sort_values("RaceNumber")
        team_prev = team_same_season[team_same_season["RaceNumber"] < this_race_number]

        if len(team_prev) > 0:
            team_avg_pos = team_prev["Position"].mean()
        else:
            team_avg_pos = team_strength.get(team_name, default_team_avg)

        team_prev_q = team_prev["GridPosition"]
        if len(team_prev_q) > 0:
            team_recent_q = team_prev_q.tail(5).mean()
        else:
            team_recent_q = default_team_quali

        team_prev_f = team_prev["Position"]
        if len(team_prev_f) > 0:
            team_recent_f = team_prev_f.tail(5).mean()
        else:
            team_recent_f = default_team_avg

        team_prev_dnf = team_prev["DidNotFinish"]
        if len(team_prev_dnf) > 0:
            team_rel = team_prev_dnf.tail(5).sum()
        else:
            team_rel = 0.0

        stats = (
            driver_stats_lookup.loc[(d["DriverNumber"], grand_prix)]
            if (d["DriverNumber"], grand_prix) in driver_stats_lookup.index
            else None
        )
        if stats is None:
            avg_track = race_data["DriverAvgTrackFinish"].mean()
            podiums = 0.0
            dnfs = 0.0
        else:
            avg_track = stats["DriverAvgTrackFinish"]
            podiums = stats["DriverTrackPodiums"]
            dnfs = stats["DriverTrackDNFs"]

        if qual_results is not None and "GridPosition" in d and pd.notna(d["GridPosition"]):
            grid_pos = int(d["GridPosition"])
            best_time = d["BestTime"]
        else:
            grid_pos = 20
            best_time = default_best_q

        if fp3_results is not None and "FP3BestTime" in d and pd.notna(d["FP3BestTime"]):
            fp3_time = d["FP3BestTime"]
        else:
            fp3_time = default_fp3

        if fp3_results is not None and "FP3LongRunTime" in d and pd.notna(d["FP3LongRunTime"]):
            fp3_long_time = d["FP3LongRunTime"]
        else:
            fp3_long_time = default_fp3_long

        driver_num = int(d["DriverNumber"])
        past_races = race_data[
            (race_data["DriverNumber"] == driver_num)
            & (
                (race_data["Season"] < year)
                | ((race_data["Season"] == year) & (race_data["RaceNumber"] < this_race_number))
            )
        ].sort_values(["Season", "RaceNumber"])

        if past_races.empty:
            cross_avg = rookie_avg_pos
            recent_avg_pts = 0.0
            recent3_avg = rookie_avg_pos
            recent5_avg = rookie_avg_pos
        else:
            cross_avg = past_races["Position"].tail(5).mean()
            recent_avg_pts = past_races["Points"].tail(5).mean()
            recent3_avg = past_races["Position"].tail(3).mean()
            recent5_avg = past_races["Position"].tail(5).mean()
        pred_rows.append(
            {
                "GridPosition": grid_pos,
                "Season": year,
                "ExperienceCount": exp_count,
                "IsRookie": 1 if exp_count == 1 else 0,
                "TeamAvgPosition": team_avg_pos,
                "CrossAvgFinish": cross_avg,
                "RecentAvgPoints": recent_avg_pts,
                "BestQualiTime": best_time,
                "FP3BestTime": fp3_time,
                "FP3LongRunTime": fp3_long_time,
                "DeltaToBestQuali": d.get("DeltaToBestQuali", 0),
                "DeltaToNext": d.get("DeltaToNext", default_delta_next),
                "DeltaToNext_Q3": d.get("DeltaToNext_Q3", default_delta_q3),
                "DeltaToNext_Q2": d.get("DeltaToNext_Q2", default_delta_q2),
                "MissedQ3": d.get("MissedQ3", 1),
                "MissedQ2": d.get("MissedQ2", 1),
                "SprintFinish": d.get("SprintFinish"),
                "HasSprint": 1 if has_sprint else 0,
                "Recent3AvgFinish": recent3_avg,
                "Recent5AvgFinish": recent5_avg,
                "DriverAvgTrackFinish": avg_track,
                "DriverTrackPodiums": podiums,
                "DriverTrackDNFs": dnfs,
                "TeamRecentQuali": team_recent_q,
                "TeamRecentFinish": team_recent_f,
                "TeamReliability": team_rel,
                "DriverChampPoints": driver_pts_map.get(d["DriverNumber"], 0.0),
                "ConstructorChampPoints": constructor_pts_map.get(d["Team"], 0.0),
                "DriverStanding": int(driver_stand_map.get(d["DriverNumber"], 0)),
                "ConstructorStanding": int(constructor_stand_map.get(d["Team"], 0)),
                "PrevYearConstructorRank": prev_rank_map.get(team_name, default_prev_rank),
                "CircuitLength": circuit_length_val,
                "NumCorners": num_corners_val,
                "DRSZones": drs_zones_val,
                "StdLapTime": std_lap_time_val,
                "IsStreet": is_street_val,
                "DownforceLevel": downforce_level_val,
                "AirTemp": default_air,
                "TrackTemp": default_track,
                "Rainfall": default_rain,
                "Overtakes_CurrentYear": default_overtake,
                "Team": d["Team"],
                "FullName": d["FullName"],
                "Abbreviation": d["Abbreviation"],
            }
        )

    pred_df = pd.DataFrame(pred_rows)

    if pred_df.empty:
        raise ValueError(f"No driver data available for {year} {grand_prix}")

    pred_df["GridPosition"] = (
        pd.to_numeric(pred_df["GridPosition"], errors="coerce").fillna(20)
    )
    pred_df["GridPosition"] = pred_df["GridPosition"].clip(1, 20)
    pred_df["BestQualiTime"] = pd.to_numeric(pred_df["BestQualiTime"], errors="coerce")
    pred_df["MissedQuali"] = pred_df["BestQualiTime"].isna().astype(int)
    pred_df["BestQualiTime"] = pred_df["BestQualiTime"].fillna(
        race_data["BestQualiTime"].median()
    )
    pred_df["DeltaToNext"] = pd.to_numeric(pred_df["DeltaToNext"], errors="coerce").fillna(default_delta_next)
    pred_df["DeltaToNext_Q3"] = pd.to_numeric(pred_df["DeltaToNext_Q3"], errors="coerce").fillna(default_delta_q3)
    pred_df["DeltaToNext_Q2"] = pd.to_numeric(pred_df["DeltaToNext_Q2"], errors="coerce").fillna(default_delta_q2)
    pred_df["MissedQ3"] = pd.to_numeric(pred_df["MissedQ3"], errors="coerce").fillna(1).astype(int)
    pred_df["MissedQ2"] = pd.to_numeric(pred_df["MissedQ2"], errors="coerce").fillna(1).astype(int)
    pred_df["FP3BestTime"] = (
        pd.to_numeric(pred_df["FP3BestTime"], errors="coerce").fillna(race_data["FP3BestTime"].mean())
    )
    pred_df["FP3LongRunTime"] = (
        pd.to_numeric(pred_df["FP3LongRunTime"], errors="coerce").fillna(
            race_data["FP3LongRunTime"].mean()
        )
    )
    pred_df["SprintFinish"] = pd.to_numeric(pred_df.get("SprintFinish"), errors="coerce")
    pred_df["HasSprint"] = pred_df["HasSprint"].fillna(0).astype(int)
    pred_df["SprintFinish"] = pred_df.apply(
        lambda r: driver_sprint_mean.get(r["DriverNumber"]) if pd.isna(r["SprintFinish"]) else r["SprintFinish"],
        axis=1,
    )
    pred_df["SprintFinish"] = pred_df.apply(
        lambda r: team_sprint_mean.get(r["Team"]) if pd.isna(r["SprintFinish"]) else r["SprintFinish"],
        axis=1,
    )
    pred_df["SprintFinish"] = pred_df["SprintFinish"].fillna(race_data["SprintFinish"].mean())
    pred_df["SprintFinish"] = pred_df["SprintFinish"].clip(1, 20)
    pred_df["NumCorners"] = (
        pd.to_numeric(pred_df["NumCorners"], errors="coerce").fillna(
            race_data["NumCorners"].median()
        )
    )
    pred_df["DRSZones"] = (
        pd.to_numeric(pred_df["DRSZones"], errors="coerce").fillna(
            race_data["DRSZones"].median()
        )
    )
    pred_df["StdLapTime"] = (
        pd.to_numeric(pred_df["StdLapTime"], errors="coerce").fillna(
            race_data["StdLapTime"].mean()
        )
    )

    pred_df["Month"] = event_month

    air_map = race_data.groupby(["Circuit", "Month"])["AirTemp"].median()
    air_val = air_map.get((grand_prix, event_month), np.nan)
    pred_df["AirTemp"] = pd.to_numeric(pred_df["AirTemp"], errors="coerce")
    pred_df["AirTemp"] = pred_df["AirTemp"].fillna(air_val)
    pred_df["AirTemp"] = pred_df["AirTemp"].fillna(race_data["AirTemp"].mean())

    track_map = race_data.groupby(["Circuit", "Month"])["TrackTemp"].median()
    track_val = track_map.get((grand_prix, event_month), np.nan)
    pred_df["TrackTemp"] = pd.to_numeric(pred_df["TrackTemp"], errors="coerce")
    pred_df["TrackTemp"] = pred_df["TrackTemp"].fillna(track_val)
    pred_df["TrackTemp"] = pred_df["TrackTemp"].fillna(race_data["TrackTemp"].mean())

    pred_df["Rainfall"] = pd.to_numeric(pred_df["Rainfall"], errors="coerce")
    pred_df["RainfallMissing"] = pred_df["Rainfall"].isna().astype(int)
    rain_map = race_data.groupby(["Circuit", "Month"])["Rainfall"].median()
    rain_val = rain_map.get((grand_prix, event_month), np.nan)
    pred_df["Rainfall"] = pred_df["Rainfall"].fillna(rain_val)
    circ_rain_map = race_data.groupby("Circuit")["Rainfall"].median()
    circuit_rain = circ_rain_map.get(grand_prix, race_data["Rainfall"].median())
    pred_df["Rainfall"] = pred_df["Rainfall"].fillna(circuit_rain)
    pred_df["Rainfall"] = pred_df["Rainfall"].fillna(race_data["Rainfall"].median())
    pred_df = pred_df.drop(columns=["Month"], errors="ignore")

    return pred_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict F1 race results')
    parser.add_argument(
        '--grand_prix',
        default='Chinese Grand Prix',
        help='Grand Prix name (e.g. "Monaco Grand Prix")',
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2025,
        help='Season year',
    )
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Retrain the model instead of using a cached one',
    )
    args = parser.parse_args()

    res = predict_race(
        args.grand_prix,
        year=args.year,
        export_details=True,
        retrain=args.retrain,
    )
    logger.info(
        "\n%s",
        res[['Driver', 'Team', 'Grid', 'Final_Position']].head(),
    )
