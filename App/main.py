from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from App.config import APP_TITLE, APP_VERSION, APP_DESCRIPTION
from App.model import predictor
from App.schemas import HabitInput, PredictionResponse
from fastapi.middleware.cors import CORSMiddleware

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

origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://rythme-gamma.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],      # Important: allows OPTIONS so no 405 on preflight
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message" : "Habit Prediction Model 1",
        "version": APP_VERSION,
        "docs":"/docs",
        "health":"/o1/health"
    }
    
@app.get("/o1/health")
def health_check():
    return {
        "status" : "healthy" if predictor.model is not None else "Unhealthy",
        "model-loaded" : predictor.model is not None,
        "api_version" : APP_VERSION
    }
    
@app.post("/o1/predict", response_model = PredictionResponse)
def predict_habit(data : HabitInput):
    try:
        input_dict = data.model_dump()
        result = predictor.predict(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code = 500 , detail = str(e))


