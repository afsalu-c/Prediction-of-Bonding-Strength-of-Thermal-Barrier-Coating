# Prediction-of-Bonding-Strength-of-Thermal-Barrier-Coating
**Machine Learning project of BTech Minor Degree in Software Engineering**

This project develops a machine learning-based predictive framework to estimate the bonding strength of Air Plasma Sprayed (APS) Thermal Barrier Coatings (TBCs). The goal is to optimize process parameters efficiently, minimize experimental costs, and improve coating performance.

## Project Overview

* Built machine learning models to predict bonding strength based on spraying parameters like current, voltage, power, gas flow rates, and coating thickness.
* Models used:

  * Random Forest Regressor (RFR)
  * Gradient Boosting Regressor (GBR)
  * Extreme Gradient Boosting Regressor (XGBR)
  * Support Vector Regression (SVR)
* Dataset augmented using **Gaussian Mixture Model (GMM)** to generate synthetic data points for better model generalization.
* Used **SHAP (SHapley Additive Explanations)** to interpret feature importance and model behavior.


## Machine Learning Workflow

1. Data Preprocessing and Standardization
2. GMM-based Synthetic Data Generation
3. Model Training and Hyperparameter Tuning
4. Model Evaluation using MSE and R² Score
5. SHAP Analysis for Feature Importance

##  Results

* Improvement in bonding strength prediction accuracy.
* Thickness found to be a critical parameter influencing bonding strength.

## Technologies Used

* Python
* scikit-learn
* XGBoost
* SHAP
* Matplotlib
* Pandas
* Numpy

## Future Scope

* Expand to other coating types (e.g., EB-PVD coatings).
* Integrate real-time data acquisition for live prediction.
* Explore deep learning models for further accuracy improvements.


> **Developed as part of Minor Project work at TKM College of Engineering, Kollam (2025).**

