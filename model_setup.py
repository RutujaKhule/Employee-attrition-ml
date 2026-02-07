import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("attrition_model.pkl")

st.title("Employee Attrition Predictor")

st.write("Enter employee details")

# ---------------- USER INPUTS ----------------

age = st.number_input("Age", 18, 70, 30)
service = st.number_input("Length of Service", 0, 40, 5)

gender = st.selectbox("Gender", ["Male", "Female"])
dept = st.selectbox("Department", ["Sales", "HR", "IT", "Executive"])

# simple encoding (match training logic)
gender_val = 1 if gender == "Male" else 0
dept_map = {"Sales": 0, "HR": 1, "IT": 2, "Executive": 3}
dept_val = dept_map[dept]

# ---------------- CREATE FULL INPUT ----------------

input_data = pd.DataFrame({

    "EmployeeID": [0],
    "recorddate_key": [0],
    "birthdate_key": [0],
    "orighiredate_key": [0],
    "terminationdate_key": [0],

    "age": [age],
    "length_of_service": [service],

    "city_name": [0],
    "department_name": [dept_val],
    "job_title": [0],
    "store_name": [0],

    "gender_short": [gender_val],
    "gender_full": [gender_val],

    "termreason_desc": [0],
    "termtype_desc": [0],

    "STATUS_YEAR": [0],
    "BUSINESS_UNIT": [0]

})

# ---------------- PREDICTION ----------------

if st.button("Predict Attrition"):

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Result")

    if pred == 1:
        st.error("⚠ High Attrition Risk")
    else:
        st.success("✅ Likely to Stay")

    st.write(f"Risk Probability: {prob:.2%}")
