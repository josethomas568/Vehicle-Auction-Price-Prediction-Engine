# Vehicle Auction Price Prediction Engine (REST API)

A scalable machine learning pipeline and serverless REST API deployed on Google Cloud Run, designed to predict the wholesale auction markup or discount of used vehicles. 

By implementing a **residual delta strategy** using gradient-boosted trees (XGBoost), this engine effectively bypasses dominant linear target leakage to drive a massive **74.48% improvement in RMSE** over baseline regression models (up from an initial 12% improvement).

## The Business Problem
When pricing vehicles at wholesale auctions, base valuations like the Manheim Market Report (MMR) establish a near-perfect 1:1 linear correlation with the final selling price. 

Traditional machine learning models fed both the vehicle features and the `mmr` suffer from **target leakage**. The models lazily split on the dominant linear feature (`mmr`) and fail to learn how the actual vehicle characteristics (condition, age, mileage, title status, color) influence the final bid.

## Residual Delta Strategy
To force the model to learn the value of specific vehicle traits, the architecture was redesigned to predict the **markup or discount (delta)** rather than the absolute price:

`Target (y) = Selling Price - MMR`

By stripping the base price out of the training features, the gradient-boosted trees dedicated 100% of their computational power to evaluating how the 89 engineered features drive the final auction price above or below the market average. The base `mmr` is then added back post-inference to calculate the final predicted price.

## Model Performance
Trained on a 500,000+ record wholesale auction dataset, the models were evaluated on their ability to predict the final absolute selling price.

| Model | Strategy | RMSE | Improvement |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Absolute Price (Baseline) | $5,778.36 | Baseline |
| **Linear Regression** | Residual Delta | $5,084.95 | +12.00% |
| **XGBoost Regressor** | Residual Delta | **$1,474.82** | **+74.48%** |

---

## Tech Stack & Architecture
* Data Engineering & EDA: `pandas`, `numpy`, `seaborn`, `matplotlib`
* Machine Learning: `xgboost`, `scikit-learn`
* API Development:`Flask`, `gunicorn`, `joblib`
* Cloud Deployment:`Docker`, Google Cloud Run, Artifact Registry, `gcloud` CLI

---

## Live API Endpoint
The winning XGBoost model is containerized and deployed as a serverless REST API on Google Cloud Platform. 

### Example Request
You can query the live engine using `curl` or Python's `requests` library:

```python
import requests
import json

# The live Google Cloud Run endpoint
url = 'https://<YOUR-CLOUD-RUN-URL>.run.app/predict'

# Example vehicle payload
mock_vehicle_data = {
    "year": 2020, 
    "condition": 4.5, 
    "odometer": 35000, 
    "mmr": 18500, 
    "Make_FORD": 1, 
    "Model_F-150": 1
}

response = requests.post(url, json=mock_vehicle_data)
print(json.dumps(response.json(), indent=4))
```
Developed by Jose Thomas as a demonstration of end-to-end MLOps and advanced predictive modeling.
