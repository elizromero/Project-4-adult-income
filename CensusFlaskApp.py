from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import json

# Load training columns
with open('Resources/Columns/training_columns.json', 'r') as f:
    training_columns = json.load(f)

dropdown_options = {
    'sex': ['Female', 'Male'],
    'race': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'],
    'native_country': ['Canada', 'Cambodia', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Honduras', 'Hong', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia'],
    'workclass': ['Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'],
    'occupation': ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'],
    'education': ['1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Other', 'Preschool', 'Prof-school'],
    'marital_status': ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
    'relationship': ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']
}

# Initialize Flask app
app = Flask(__name__)

# Load models
rf_model = joblib.load('Resources/Models/random_forest_model.pkl')
nn_model = load_model('Resources/Models/neural_network_model.keras')
xgb_model = joblib.load('Resources/Models/xgboost_model.pkl')

# Load scaler
scaler = joblib.load('Resources/Scaler/scaler.pkl')  # Save your scaler during training

@app.route('/')
def home():
    # Pass dropdown options and defaults
    return render_template(
        'index.html',
        dropdown_options=dropdown_options,
        default_fnlwgt=189664.13,
        default_capital_gain=0,
        default_capital_loss=0
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data from the form
    user_input = {
        'age': int(request.form.get('age', 0)),  # Added default value
        'workclass': request.form.get('workclass', 'Private'),
        'education': request.form.get('education', 'HS-grad'),
        'marital-status': request.form.get('marital-status', 'Never-married'), 
        'occupation': request.form.get('occupation', 'Other-service'),
        'relationship': request.form.get('relationship', 'Not-in-family'),
        'race': request.form.get('race', 'White'),
        'sex': request.form.get('sex', 'Male'),
        'native-country': request.form.get('native-country', 'United-States'),
        'hours-per-week': int(request.form.get('hours-per-week', 40)),  # Added default value
        'capital-gain': int(request.form.get('capital-gain', 0)),  # Added default value
        'capital-loss': int(request.form.get('capital-loss', 0)),  # Added default value
        'fnlwgt': float(request.form.get('fnlwgt', 189664.13)),  # Added default value
    }

    # Debugging: Print input data to confirm structure
    print(user_input)

    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])

    # Preprocessing (if any)
    input_df['native-country'] = input_df['native-country'].str.lower()
    input_df['net-capital-gain'] = input_df['capital-gain'] - input_df['capital-loss']

    # Drop unnecessary columns
    input_df.drop(['capital-gain', 'capital-loss'], axis=1, inplace=True)

    # Apply one-hot encoding
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']  # You may need to define this list
    input_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)

    # Ensure consistency with training data columns
    for col in training_columns:  # training_columns = X_train.columns
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Align column order
    input_encoded = input_encoded[training_columns]

    # Standardize numerical features (for NN only)
    input_scaled = scaler.transform(input_encoded) if 'Neural Network' in request.form else input_encoded

    # Make predictions
    rf_prediction = rf_model.predict(input_encoded)[0]
    nn_prediction = nn_model.predict(input_scaled)[0]
    xgb_prediction = xgb_model.predict(input_encoded)[0]

    # Decode predictions
    predictions = {
        'Random Forest': '<=50K' if rf_prediction == 0 else '>50K',
        'Neural Network': '<=50K' if nn_prediction.argmax() == 0 else '>50K',
        'XGBoost': '<=50K' if xgb_prediction == 0 else '>50K',
    }

    return render_template('result.html', predictions=predictions)


if __name__ == '__main__':
    app.run(debug=True)
