import joblib
import os

# Load pre-trained XGBoost model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/burnout_model.pkl")
burnout_model = joblib.load(MODEL_PATH)

def predict_burnout(features: list[float]) -> dict:
    """
    features = [EE, DP, PA, BDI, GAD, MHC]
    Returns {"risk_score": float}.
    """
    pred = burnout_model.predict([features])[0]
    return {"risk_score": float(pred)}
