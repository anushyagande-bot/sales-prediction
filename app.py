import streamlit as st
import pandas as pd
import joblib
# Load model
model = joblib.load("model/sales_model.pkl")

st.title("Sales Prediction using Machine Learning")

tv = st.number_input("TV Advertising Budget")
radio = st.number_input("Radio Advertising Budget")
newspaper = st.number_input("Newspaper Advertising Budget")

if st.button("Predict Sales"):

    input_data = pd.DataFrame({
        'TV': [tv],
        'Radio': [radio],
        'Newspaper': [newspaper]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Sales: {prediction[0]:.2f}")
