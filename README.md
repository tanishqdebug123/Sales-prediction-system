# Sales prediction system

# Project Overview
This project implements a time-series forecasting model to predict monthly sales based on historical data. The primary goal is to provide accurate sales forecasts that can help in making informed business decisions related to inventory management, resource allocation, and marketing strategies.

The model is built using an XGBoost Regressor, and the performance is evaluated using the R² score and Mean Absolute Error (MAE). The project demonstrates a complete machine learning pipeline from data preprocessing and feature engineering to model training, evaluation, and visualisation.

# Features
1) Data Preprocessing: Handles missing values, removes outliers, and prepares the data for modelling.

2) Feature Engineering: Extracts relevant features from the date column to improve model accuracy.

3) Model Training: Utilises an XGBoost Regressor for its high performance and robustness.

4) Model Evaluation: Measures the model's accuracy using R² score and Mean Absolute Error.

5) Data Visualisation: Presents the data and model predictions in an intuitive and easy-to-understand format.

# Results
The model achieved an R² score of 97.6% and a low Mean Absolute Error, indicating a high level of accuracy in predicting monthly sales over a 12-month forecast horizon. The visualizations in the notebook provide a clear comparison between the actual and predicted sales values.
