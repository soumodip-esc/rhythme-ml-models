from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# MODEL_PATH = BASE_DIR / "models" / "model_1" / "modelname.pkl"
MODEL_PATH = BASE_DIR / "Models" / "Model_1" / "habit_model.pkl"



APP_TITLE = "Habit Completion Prediction API"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = """ Predict whether a user will complete their habit based on historical data.

Features:
- 7 input features
- Returns probability and prediction
- Model accuracy: 60.0%"""


FEATURE_NAMES =[
    'day_of_week',
    'is_weekend',
    'current_streak',
    'completion_rate_7d',
    'completion_rate_30d',
    'days_since_start',
    'frequency_encoded'
]