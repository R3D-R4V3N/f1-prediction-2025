import streamlit as st
from race_predictor import predict_race, GRAND_PRIX_LIST

st.title('F1 2025 Race Predictor')

gp = st.selectbox('Select a Grand Prix', GRAND_PRIX_LIST)

if st.button('Predict Results'):
    try:
        with st.spinner('Running predictions...'):
            results = predict_race(gp, compute_overtakes=True)
        st.subheader('Predicted Finishing Positions')
        st.dataframe(results[['Final_Position', 'Driver', 'Team', 'Grid']])
    except Exception as e:
        st.error(str(e))
