import streamlit as st 
import joblib
import numpy as np 

scaler = joblib.load("C:/Users/Fatin/Documents/Car_Sale_Prediction/scaler.pkl")
model = joblib.load("C:/Users/Fatin/Documents/Car_Sale_Prediction/model.pkl")



st.title("Customer Car Price Estimator App ")

st.divider()

st.write("""This app is for giving advice to the customer on what car to purchase based on the information given""")


age = st.number_input("Enter the age", min_value=18, max_value=90, value=40, step=1)
salary = st.number_input("Enter the salary",min_value=1000,max_value=99999999,step=5000, value=3000)
networth = st.number_input("Enter net worth",min_value=0,max_value=99999999,step=20000,value=100000)

X= [age,salary,networth]

calculatebutton = st.button("Calculate")

st.divider()

if calculatebutton:

    X_2 = np.array(X)
    X_array= scaler.transform([X_2])
    prediction = model.predict(X_array)

    st.write(f"Prediction is {prediction[0]}")
    st.write("advice cars in the similiar values")
else:
    st.write("Please enter the values and press the calculate button")

