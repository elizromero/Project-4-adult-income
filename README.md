# Adult Income Prediction App

## Project Overview

This project predicts whether an individual's income exceeds $50K/year based on the Adult Census Income dataset. The project involves data cleaning, feature engineering, model training, and deployment using Flask. The application allows users to input data via a web interface and view predictions from multiple models.

## Features

Data Cleaning and Preprocessing:*Handled missing data, feature encoding, and scaling.
**Model Training:** Trained five machine learning models:
-Random Forest (primary model in deployment)
-Neural Network
-XGBoost
-SVC
-Linear Regression
**Flask Application:** Deployed as a web app with a user-friendly interface.
**Predictions Display:** Displays predictions from the Random Forest, Neural Network, and XGBoost models.

## Prerequisites
- Python 3.8 or above
- Flask
- TensorFlow
- Scikit-learn
- XGBoost
- Joblib

## Start the Flask app:
python app/CensusFlaskApp.py
Access the app in your browser at http://127.0.0.1:5000.

## Models Used

1. Random Forest: High interpretability and strong performance.
2. Neural Network: Handles non-linear relationships in data.
3. XGBoost: Gradient boosting model for tabular data.
4. SVC: Explored as part of experimentation.
5. Linear Regression: Benchmark model for comparison.

## Team Contributions

- **Data Preparation:** Preprocessed the dataset, handled categorical and numerical features.
- **Model Training:** Trained and evaluated five models for comparison.
- **Application Development:** Built the web app to deploy predictions.

## How It Works

1. **Input:** Users enter personal details such as age, workclass, education, etc., via the web interface.
2. **Processing:** Input is transformed to match the training data's format.
3. **Prediction:** Predictions from Random Forest, Neural Network, and XGBoost are displayed.
4. **Output:** Results are shown in the result.html template.
