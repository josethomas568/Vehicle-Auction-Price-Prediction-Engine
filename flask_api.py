import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load Model and Artifacts on startup
print("Loading model and artifacts...")
model = xgb.XGBRegressor()
model.load_model("xgb_auction_model.json")

artifacts = joblib.load("model_artifacts.pkl")
imputer = artifacts['imputer']
expected_columns = artifacts['columns']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get JSON data from the incoming request
        data = request.get_json()

        # 2. Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure the baseline mmr is provided
        if 'mmr' not in input_df.columns:
            return jsonify({'error': 'Missing required baseline value: mmr'}), 400

        base_mmr = float(input_df['mmr'].iloc[0])

        # 3. Quick Feature Alignment
        input_df = pd.get_dummies(input_df)

        # Reindex to match the training columns perfectly, filling missing dummies with 0
        X_api = input_df.reindex(columns=expected_columns, fill_value=0)

        # 4. Impute missing numeric values
        X_imputed = imputer.transform(X_api)

        # 5. Predict the Delta and calculate Final Price
        predicted_delta = model.predict(X_imputed)[0]
        final_price = base_mmr + predicted_delta

        # 6. Return the JSON response
        return jsonify({
            'status': 'success',
            'baseline_mmr': round(base_mmr, 2),
            'predicted_markup_discount': round(float(predicted_delta), 2),
            'final_predicted_price': round(float(final_price), 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
