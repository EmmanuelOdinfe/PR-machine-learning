import streamlit as st 
import pickle
import numpy as np

with open("poly_model.pkl", "rb") as file:
    model = pickle.load(file)
    
with open("poly_transform.pkl", "rb") as file:
    poly = pickle.load(file)
    
st.title("PR Salary Prediction")
st.write("Predict salary based on Position Level")

level = st.number_input(
    "Enter Position Level",
    min_value=1.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

if st.button("Predict Salary"):
    level_poly = poly.transform(np.array([[level]]))
    prediction = model.predict(level_poly)
    
    st.success(f"Predicted Salary: ${prediction[0]:,.2f}")