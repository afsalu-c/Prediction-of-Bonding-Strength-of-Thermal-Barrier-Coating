from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('synthetic_data.csv')

# Features and target
X = data.drop('Bonding strength', axis=1)
y = data['Bonding strength']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize SVR model
svr = SVR()

# Define hyperparameters and values to test
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5]
}

# Grid Search
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Params:", best_params)

# Make predictions with the best model
y_pred = best_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")

# Function to predict bonding strength given thickness and other parameters
def predict_bonding_strength_svr(thickness, current, voltage, power, ar, h2, ar_h2):
    input_features = np.array([[current, voltage, power, ar, h2, ar_h2, thickness]])
    input_features = scaler.transform(input_features)
    prediction = best_model.predict(input_features)[0]
    return prediction
