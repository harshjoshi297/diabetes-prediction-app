

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("diabetes.csv")


df.head()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report,roc_auc_score)

X = df.drop(columns='Outcome')
y = df['Outcome']

# Columns where 0 means missing
cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Remaining numeric columns (no imputation needed)
cols_no_impute = [col for col in X.columns if col not in cols_to_impute]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ('median_imputer', SimpleImputer(strategy='median'), cols_to_impute),
        ('pass_through', 'passthrough', cols_no_impute)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        solver='liblinear'
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
plt.title('ROC Curve')
plt.show()

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring='f1'
)

print("CV F1 scores:", cv_scores)
print("Mean CV F1:", cv_scores.mean())

# Get the trained Logistic Regression model from the pipeline
logistic_model = pipeline.named_steps['model']

# Get the feature names in the order they were processed by the ColumnTransformer
# This is important because the scaler and model operate on these transformed features
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

# Display the coefficients
print("Coefficients:")
for name, coef in zip(feature_names, logistic_model.coef_[0]):
    print(f"  {name}: {coef:.4f}")

# Display the intercept
print(f"\nIntercept: {logistic_model.intercept_[0]:.4f}")

import joblib
joblib.dump(pipeline, "model.pkl", protocol=5)


