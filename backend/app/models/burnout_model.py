import joblib
import json
import os
from pathlib import Path

# Determine project root via file location
# burnout_model.py is at: <project>/backend/app/models/burnout_model.py
PROJECT_ROOT = Path(__file__).parents[3]   # up from models → app → backend → project root

# Path to the metadata file
meta_path = PROJECT_ROOT / "models" / "latest.json"
if not meta_path.exists():
    raise FileNotFoundError(f"Model metadata not found: {meta_path}")

# Load metadata
with open(meta_path, "r") as f:
    meta = json.load(f)

# Path to the actual model file
model_path = PROJECT_ROOT / meta["model"]
if not model_path.exists():
    raise FileNotFoundError(f"Burnout model not found: {model_path}")

burnout_model = joblib.load(model_path)

def predict_burnout(features: list[float]) -> dict:
    pred = burnout_model.predict([features])[0]
    return {"risk_score": float(pred)}
