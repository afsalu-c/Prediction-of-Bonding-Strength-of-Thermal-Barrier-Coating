#Gradient Boost
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Initialize Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# Define hyperparameters and values to test
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Grid Search
grid_search = GridSearchCV(gbr, param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
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
def predict_bonding_strength_Gb(thickness, current, voltage, power, ar, h2, ar_h2):
    input_features = pd.DataFrame([[current, voltage, power, ar, h2, ar_h2, thickness]],
                                  columns=X.columns)
    prediction = best_model.predict(input_features)[0]
    return prediction
