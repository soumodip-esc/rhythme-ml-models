import pickle
import pandas as pd
from App.config import MODEL_PATH, FEATURE_NAMES


class HabitPredictor:
    def __init__(self):
        self.model = None

    def load_model(self):
        try:
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error! Loading Model: {e}")
            raise

    def predict(self, input_data: dict):
        if self.model is None:
            raise ValueError("Model not loaded")
    
        mapped_data = {
            "day_of_week": input_data["day_of_week"],
            "is_weekend": input_data["is_weekend"],
            "current_streak": input_data["current_streak"],
            "completion_rate_7d": input_data["completion_rate_7d"],
            "completion_rate_30d": input_data["completion_rate_30d"],
            "days_since_start": input_data["days_since_start"],
            "frequency_encoded": input_data["frequency_encoded"],
        }

        df = pd.DataFrame([mapped_data], columns=FEATURE_NAMES)

        
        prediction = self.model.predict(df)[0]  
        probability = self.model.predict_proba(df)[0][1]

        result = "complete" if prediction == 1 else "Skip"
        prob_percent = round(probability * 100, 1)

        return {
            "prediction": result, 
            "probability": round(probability, 3),  
            "probability_percent": f"{prob_percent}%",  
            "message": f"User will likely {result} the habit ({prob_percent}%)",  
        }

# Global instance
predictor = HabitPredictor()
