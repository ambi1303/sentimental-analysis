# backend/app/main.py

from fastapi import FastAPI
from app.routes.checkin import router as checkin_router
from app.routes.assessment import router as assessment_router
from app.routes.prediction import router as prediction_router
from app.utils.load_data import load_all

app = FastAPI(title="Mental Health Monitoring API")

# Load data at startup
daily_df, weekly_df, monthly_df = load_all()
app.state.daily = daily_df
app.state.weekly = weekly_df
app.state.monthly = monthly_df

@app.get("/health")
def health_check():
    return {
        "data": {
            "daily_rows": len(app.state.daily),
            "weekly_rows": len(app.state.weekly),
            "monthly_rows": len(app.state.monthly),
        }
    }

app.include_router(checkin_router, prefix="/daily", tags=["daily"])
app.include_router(assessment_router, prefix="/assessment", tags=["assessment"])
app.include_router(prediction_router, prefix="/predict", tags=["prediction"])
