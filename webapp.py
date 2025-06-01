import streamlit as st
import pandas as pd
import numpy as np
from predictor import predict_race
from data_utils import GRAND_PRIX_LIST


@st.cache_data(show_spinner=False)
def _load_actual_results(year: int, grand_prix: str):
    """Return official race results if available."""
    try:
        import fastf1

        schedule = fastf1.get_event_schedule(year)
        match = schedule[schedule["EventName"].str.contains(grand_prix, case=False, na=False)]
        if match.empty:
            return None
        rnd = int(match.iloc[0]["RoundNumber"])
        session = fastf1.get_session(year, rnd, "R")
        session.load(telemetry=False, laps=False, weather=False)
        if hasattr(session, "results") and not session.results.empty:
            df = session.results[["Abbreviation", "Position"]].rename(
                columns={"Abbreviation": "Driver", "Position": "Actual_Position"}
            )
            return df
    except Exception:
        return None
    return None

st.title('F1 2025 Race Predictor')

year = st.sidebar.number_input('Season', min_value=2020, max_value=2025, value=2025)
debug_mode = st.sidebar.checkbox('Enable debug options')

gp = st.selectbox('Select a Grand Prix', GRAND_PRIX_LIST)

if st.button('Predict Results'):
    try:
        with st.spinner('Running predictions...'):
            output = predict_race(gp, year=year, compute_overtakes=True, debug=debug_mode)
        if debug_mode:
            results, details = output
        else:
            results = output

        st.subheader('Predicted Finishing Positions')
        st.dataframe(results[['Final_Position', 'Driver', 'Team', 'Grid']])

        if debug_mode:
            actual = _load_actual_results(year, gp)
            if actual is not None:
                merged = results.merge(actual, on='Driver', how='left')
                merged['Diff'] = merged['Actual_Position'] - merged['Final_Position']
                st.subheader('Prediction vs Actual')
                st.dataframe(merged[['Final_Position', 'Actual_Position', 'Diff', 'Driver', 'Team', 'Grid']])

            st.subheader('Debug Files')
            try:
                with open('prediction_data.csv', 'rb') as f:
                    st.download_button('Download prediction_data.csv', f, file_name='prediction_data.csv')
            except Exception:
                st.info('prediction_data.csv not available')
            try:
                with open('prediction_input.csv', 'rb') as f:
                    st.download_button('Download prediction_input.csv', f, file_name='prediction_input.csv')
            except Exception:
                st.info('prediction_input.csv not available')

            shap_vals = details.get('shap_values') if details else None
            if shap_vals is not None:
                feature_names = details.get('feature_names', [])
                for idx, row in results.iterrows():
                    with st.expander(f"Why {row['Driver']}?"):
                        sv = shap_vals[idx]
                        top_idx = np.argsort(np.abs(sv))[::-1][:5]
                        shap_df = pd.DataFrame({
                            'Feature': [feature_names[i] for i in top_idx],
                            'SHAP': sv[top_idx]
                        })
                        st.table(shap_df)
            else:
                st.info('SHAP analysis not available.')
    except Exception as e:
        st.error(str(e))
