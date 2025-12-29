#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

st.title("🩺 Diabetes Predictor (Logistic Regression)")
st.write("Enter patient details and click **Predict**")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

with st.form("prediction_form"):
    st.header("Patient Information")

    Pregnancies = st.number_input("Pregnancies", min_value=0, value=2, step=1)
    Glucose = st.number_input("Glucose", min_value=30.0, value=120.0)
    BloodPressure = st.number_input("Blood Pressure", min_value=20.0, value=70.0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
    Insulin = st.number_input("Insulin", min_value=0.0, value=80.0)
    BMI = st.number_input("BMI", min_value=10.0, value=25.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
    Age = st.number_input("Age", min_value=1, value=35)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame({
        "Pregnancies": [Pregnancies],
        "Glucose": [Glucose],
        "BloodPressure": [BloodPressure],
        "SkinThickness": [SkinThickness],
        "Insulin": [Insulin],
        "BMI": [BMI],
        "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
        "Age": [Age],
    })

    proba = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f"⚠️ High risk of Diabetes — Probability: {proba*100:.2f}%")
    else:
        st.success(f"✅ Low risk of Diabetes — Probability: {(1-proba)*100:.2f}%")

    st.metric("Diabetes Probability", f"{proba*100:.2f}%")
    st.progress(min(proba, 1.0))

    with st.expander("Input data"):
        st.write(input_df)

    st.warning(
        "⚠️ This tool is for educational purposes only and should not be used for medical diagnosis."
    )


# In[ ]:




