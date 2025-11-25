from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from App.config import APP_TITLE, APP_VERSION, APP_DESCRIPTION
from App.model import predictor
from App.schemas import HabitInput, PredictionResponse

# import uvicorn
# import pickle

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("API starting...")
    predictor.load_model()
    yield
    print("Shutting down...")
    
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan
)


@app.get("/")
def home():
    return {
        "message" : "Habit Prediction Model 1",
        "version": APP_VERSION,
        "docs":"/docs",
        "health":"/health"
    }
    
@app.get("/health")
def health_check():
    return {
        "status" : "healthy" if predictor.model is not None else "Unhealthy",
        "model-loaded" : predictor.model is not None,
        "api_version" : APP_VERSION
    }
    
@app.post("/predict", response_model = PredictionResponse)
def predict_habit(data : HabitInput):
    try:
        input_dict = data.model_dump()
        result = predictor.predict(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code = 500 , detail = str(e))


