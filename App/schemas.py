from pydantic import BaseModel
from typing import List, Optional

class HabitInput(BaseModel):
    """Input data for prediction"""

    day_of_week: int
    is_weekend: int
    current_streak: int
    completion_rate_7d: float
    completion_rate_30d: float
    days_since_start: int
    frequency_encoded: int


    class Config:
        json_schema_extra = {
            "example": {
                "day_of_week": 1,
                "is_weekend": 0,
                "current_streak": 10,
                "completion_rate_7d": 0.71,
                "completion_rate_30d": 0.65,
                "days_since_start": 50,
                "frequency_encoded": 0,
            }
        }


class PredictionResponse(BaseModel):
    """Response for Prediction"""

    prediction: str
    probability: float
    probability_percent: str
    message: str


class JournalInput(BaseModel):
    text : str
    title : Optional[str] = None

class SentimentResult(BaseModel):
    sentiment : str
    confidence : float
    model_used : str
    emotions : List[str]

class JournalResponse(BaseModel):
    text : str
    title : str
    sentiment : str
    confidance : float
    emotions : List[str]
    model_used : str
    created_at   : str