from pathlib import Path
import os
from dotenv import load_dotenv
from App.__init__ import __version__

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Models" / "Model_1" / "habit_model.pkl"
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


APP_TITLE = "Habit Completion Prediction API"
APP_VERSION = __version__
APP_DESCRIPTION = """ Predict whether a user will complete their habit based on historical data.

Features:
- 7 input features
- Returns probability and prediction
- Model accuracy: 60.0%"""


#Sentiment Setting 
CONFIDENCE_THRESHOLS = 0.70
ROBERTA_RETRIES = 3
RETRY_DELAY = 2

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN nor fount in .env file")

FEATURE_NAMES =[
    'day_of_week',
    'is_weekend',
    'current_streak',
    'completion_rate_7d',
    'completion_rate_30d',
    'days_since_start',
    'frequency_encoded'
]