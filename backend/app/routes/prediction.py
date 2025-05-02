from fastapi import APIRouter
from pydantic import BaseModel
from app.models.burnout_model import predict_burnout

router = APIRouter()

class PredictionInput(BaseModel):
    user_id: int
    EE: int
    DP: int
    PA: int
    BDI: int
    GAD: int
    MHC: int

@router.post("/")
def burnout_prediction(payload: PredictionInput):
    features = [payload.EE, payload.DP, payload.PA, payload.BDI, payload.GAD, payload.MHC]
    result = predict_burnout(features)
    return {
        "user_id": payload.user_id,
        **result
    }
