from fastapi import FastAPI
from app.routes.checkin import router as checkin_router
from app.routes.assessment import router as assessment_router
from app.routes.prediction import router as prediction_router

app = FastAPI(title="Mental Health Monitoring API")

app.include_router(checkin_router, prefix="/daily", tags=["daily"])
app.include_router(assessment_router, prefix="/assessment", tags=["assessment"])
app.include_router(prediction_router, prefix="/predict", tags=["prediction"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mental Health Monitoring API"}
