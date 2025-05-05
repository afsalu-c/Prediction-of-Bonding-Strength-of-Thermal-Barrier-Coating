import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# Load the given dataset (manually inputted as per the provided table)
data = np.array([
    [503, 69.3, 34.71, 30, 5, 6, 200, 43.2],
    [503, 69.3, 34.71, 30, 5, 6, 400, 30.3],
    [641, 69.3, 44.42, 30, 5, 6, 200, 46.6],
    [641, 69.3, 44.42, 30, 5, 6, 400, 41.2],
    [520, 68, 35.36, 30, 6, 6, 300, 23.5],
    [665, 68, 45.22, 30, 6, 5, 300, 33.03],
    [617, 65, 40.11, 30, 3, 10, 300, 40.77],
    [617, 65, 40.11, 30, 3, 10, 400, 30.95]
])

# Column names
columns = ["Current/A", "Voltage", "Power", "Ar", "H2", "Ar/H2", "Thickness", "Bonding strength"]

# Convert to Pandas DataFrame
df = pd.DataFrame(data, columns=columns)

# Fit Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(df)

# Generate 400 synthetic samples
synthetic_data = gmm.sample(400)[0]  # Extract generated samples

# Convert to DataFrame for better readability
synthetic_df = pd.DataFrame(synthetic_data, columns=columns)

# Ensure values are within realistic limits (e.g., no negative values)
synthetic_df = synthetic_df.clip(lower=0)

# Save to CSV (optional)
synthetic_df.to_csv("synthetic_data.csv", index=False)

# Print first few rows
print(synthetic_df.head())
