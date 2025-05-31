import streamlit as st
from race_predictor import predict_race, GRAND_PRIX_LIST

st.title('F1 2025 Race Predictor')

gp = st.selectbox('Select a Grand Prix', GRAND_PRIX_LIST)

if st.button('Predict Results'):
    with st.spinner('Running predictions...'):
        results = predict_race(gp)
    st.subheader('Predicted Finishing Positions')
    st.dataframe(results[['Final_Position', 'Driver', 'Team', 'Grid']])
