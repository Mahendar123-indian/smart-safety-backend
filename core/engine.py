import xgboost as xgb
import pandas as pd  # Make sure this import is at the top
import numpy as np


class SafetyAI:
    def __init__(self, model_path='models/danger_model.json'):
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def predict_danger(self, sensor_data):
        # The model expects these exact names because of how it was trained
        feature_names = ['speed', 'accel', 'jerk', 'risk_score', 'police_dist']

        # Create a DataFrame with the names so the model recognizes them
        data_df = pd.DataFrame([sensor_data], columns=feature_names)

        # Convert the DataFrame to the XGBoost DMatrix format
        data_matrix = xgb.DMatrix(data_df)

        prediction = self.model.predict(data_matrix)[0]
        return float(prediction)