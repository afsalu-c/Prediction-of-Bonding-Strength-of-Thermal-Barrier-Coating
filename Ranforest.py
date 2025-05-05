from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

# Initialize Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")

# Function to predict bonding strength given thickness and other parameters
def predict_bonding_strength_rf(thickness, current, voltage, power, ar, h2, ar_h2):
    input_features = pd.DataFrame([[current, voltage, power, ar, h2, ar_h2, thickness]],
                                  columns=X.columns)
    prediction = rf_model.predict(input_features)[0]
    return prediction
