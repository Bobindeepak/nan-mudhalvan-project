# 1. Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# 2. Load Dataset
# Example: UCI Heart Disease Dataset
df = pd.read_csv("heart.csv")  # Replace with your dataset

# 3. Basic EDA
print(df.head())
print(df.info())
print(df.describe())

# 4. Preprocess Data
X = df.drop('target', axis=1)  # 'target' is the label column (0 = no disease, 1 = disease)
y = df['target']

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. Train Model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 7. Evaluate Model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. Explainable AI with SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Plot SHAP values for first prediction
shap.plots.waterfall(shap_values[0])
shap.summary_plot(shap_values, X_test, feature_names=df.columns[:-1])

# Optional: Save the model
import joblib
joblib.dump(model, "disease_prediction_model.pkl")
