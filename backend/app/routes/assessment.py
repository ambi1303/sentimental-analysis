from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Weekly
class WeeklyAssessment(BaseModel):
    user_id: int
    week: int
    avg_stress: int            # 1–10
    overwhelm_freq: str        # Never/Rarely/Sometimes/Often/Always

@router.post("/weekly")
def weekly_assessment(payload: WeeklyAssessment):
    freq_map = {"Never":0,"Rarely":1,"Sometimes":2,"Often":3,"Always":4}
    overwhelm_idx = freq_map.get(payload.overwhelm_freq, 0)
    return {
        "user_id": payload.user_id,
        "week": payload.week,
        "avg_stress": payload.avg_stress,
        "overwhelm_idx": overwhelm_idx
    }

# Monthly
class MonthlyAssessment(BaseModel):
    user_id: int
    month: str    # "YYYY-MM"
    EE: int       # Emotional Exhaustion
    DP: int       # Depersonalization
    PA: int       # Personal Accomplishment
    BDI: int      # Beck Depression Inventory-II total
    GAD: int      # GAD-7 total
    MHC: int      # MHC-SF-14 composite

@router.post("/monthly")
def monthly_assessment(payload: MonthlyAssessment):
    # In a real app, you'd save to DB here
    return payload
