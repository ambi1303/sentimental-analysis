#!/usr/bin/env python
import sys
import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 1) Make sure Python can find your FastAPI app under backend/app
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()           # .../mental_health_monitoring
BACKEND_PATH = SCRIPT_DIR / "backend"
sys.path.insert(0, str(BACKEND_PATH))                  # now `import app` works

# ──────────────────────────────────────────────────────────────────────────────
# 2) Standard imports
# ──────────────────────────────────────────────────────────────────────────────
import argparse
import json
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score

# Import your data loader
from app.utils.load_data import load_all

# ──────────────────────────────────────────────────────────────────────────────
# 3) Training function
# ──────────────────────────────────────────────────────────────────────────────
def train_burnout(monthly_df: pd.DataFrame, model_out: Path) -> dict:
    """
    Train an XGBoost model on the monthly DataFrame.
    Expects columns: EE, DP, PA, BDI, GAD, MHC
    Outputs model to `model_out` and returns metadata.
    """
    # Features & target
    X = monthly_df[["EE", "DP", "PA", "BDI", "GAD", "MHC"]]
    y = (monthly_df["EE"] >= 27).astype(int)  # binary high-risk: EE ≥ 27

    # Stratify split if possible
    from sklearn.model_selection import train_test_split
    stratify_arg = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    # Train XGBoost regressor
    model = XGBRegressor(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    auc = roc_auc_score(y_test, preds) if len(y_test.unique()) == 2 else None

    # Save model artifact
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)

    # Prepare metadata
    meta = {
        "model_path": str(model_out.relative_to(SCRIPT_DIR)),
        "rmse": rmse,
        "auc": auc
    }

    # Write metadata JSON
    latest_json = SCRIPT_DIR / "models" / "latest.json"
    latest_json.parent.mkdir(parents=True, exist_ok=True)
    with open(latest_json, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✔️ Trained model saved to {model_out}")
    print(f"✔️ Metadata saved to {latest_json}")
    print(f"   → RMSE: {rmse:.4f}")
    if auc is not None:
        print(f"   → ROC-AUC: {auc:.4f}")
    else:
        print("   → ROC-AUC: skipped (only one class present)")

    return meta

# ──────────────────────────────────────────────────────────────────────────────
# 4) CLI entrypoint
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train burnout prediction model")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models") / f"burnout_model_{pd.Timestamp.now():%Y%m%d}.pkl",
        help="Path where to save the trained model (.pkl)"
    )
    args = parser.parse_args()

    # Load data tables
    _, _, monthly_df = load_all()
    print(f"Loaded monthly data: {len(monthly_df)} rows")

    # Train & save
    train_burnout(monthly_df, args.output)

if __name__ == "__main__":
    main()
