# Diabetes Prediction using Machine Learning

## Problem Statement
Predict whether a person is diabetic based on medical parameters using machine learning.

## Dataset
Pima Indians Diabetes Dataset

## Approach
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature scaling and model training using pipelines
- Model evaluation using accuracy, precision, recall, and ROC-AUC

## Models Used
- Logistic Regression

## Deployment
The trained model is deployed using Streamlit for interactive predictions.

## Tech Stack
Python, Pandas, NumPy, scikit-learn, Streamlit

## Results
Achieved ROC-AUC of ~0.77 on test data.

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
