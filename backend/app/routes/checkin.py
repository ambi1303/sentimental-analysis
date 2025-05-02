from fastapi import APIRouter
from pydantic import BaseModel
from app.utils.preprocess import clean_text
from app.models.sentiment_model import analyze_sentiment

router = APIRouter()

class CheckIn(BaseModel):
    user_id: int
    date: str            # "YYYY-MM-DD"
    mood_score: int      # 1–10
    feeling_text: str

@router.post("/sentiment")
def sentiment_analysis(payload: CheckIn):
    text_clean = clean_text(payload.feeling_text)
    sentiment = analyze_sentiment(text_clean)
    return {
        "user_id": payload.user_id,
        "date": payload.date,
        "mood_score": payload.mood_score,
        **sentiment
    }
