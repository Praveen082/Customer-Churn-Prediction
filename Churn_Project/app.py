import streamlit as st
import pickle
import pandas as pd

# Load Model
with open("churn_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load Scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Load Feature Names
with open("feature_names.pkl", "rb") as file:
    feature_names = pickle.load(file)

st.title("Customer Churn Prediction")

st.write("Predict whether a customer will leave the bank.")

st.success("Model Loaded Successfully!")

st.subheader("Enter Customer Details")

credit_score = st.number_input("Credit Score", 300, 900, 650)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input("Age", 18, 100, 35)

tenure = st.number_input("Tenure", 0, 10, 5)

balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)

products = st.number_input("Number of Products", 1, 4, 1)

credit_card = st.selectbox("Has Credit Card", [0, 1])

active = st.selectbox("Is Active Member", [0, 1])

salary = st.number_input("Estimated Salary", 0.0, 300000.0, 50000.0)

# Step 21: Prediction

if st.button("Predict Churn"):

    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": credit_card,
        "IsActiveMember": active,
        "EstimatedSalary": salary,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0
    }])

    input_data = input_data.reindex(
        columns=feature_names,
        fill_value=0
    )

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]

    if prediction[0] == 1:
        st.error("Customer is likely to leave!")
    else:
        st.success("Customer is likely to stay!")

    st.write("Churn Probability:", round(probability * 100, 2), "%")
