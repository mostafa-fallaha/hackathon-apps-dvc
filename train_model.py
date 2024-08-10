import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data/googleplaystore.csv')

# Data Preprocessing
def convert_size(size_str):
    if 'M' in size_str:
        return float(size_str.replace('M', ''))
    elif 'k' in size_str:
        return float(size_str.replace('k', '')) / 1000
    elif size_str == 'Varies with device':
        return np.nan
    else:
        return np.nan

data['Size'] = data['Size'].apply(convert_size)
data['Size'] = data['Size'].fillna(data['Size'].mean())

# Log-transform numerical features
data['Reviews'] = np.log1p(data['Reviews'])
data['Installs'] = np.log1p(data['Installs'])

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['Category', 'Type', 'Content Rating', 'Genres'])

# Define features and target
X = data.drop(columns=['Rating', 'App', 'Last Updated', 'Current Ver', 'Android Ver'])
y = data['Rating']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Start an MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model", "Random Forest Regressor")

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and calculate metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Log the model with an input example
    input_example = X_test[0:1]
    mlflow.sklearn.log_model(model, "model", input_example=input_example)

    # Log artifacts, e.g., plots
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs Predicted Ratings")
    plt.savefig("residuals_plot.png")
    mlflow.log_artifact("residuals_plot.png")

# Print metrics for manual review
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
