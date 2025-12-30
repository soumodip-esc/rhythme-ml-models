from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from App.config import APP_TITLE, APP_VERSION, APP_DESCRIPTION
from App.model import predictor
from App.schemas import HabitInput, PredictionResponse, JournalInput, JournalResponse, SentimentResult
from . import sentiment
from datetime import datetime


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
    "https://rhythme-gamma.vercel.app",
    "https://rhythme-git-dev-oopsvincent-projects.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],     
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message" : "Rhythme Models 1",
        "version": APP_VERSION,
        "docs":"/docs",
        "health":"/o1/health"
    }
    
@app.get("/o1/health")
def health_check():
    return {
        "status" : "healthy" if predictor.model is not None else "Unhealthy",
        "habit-model-loaded" : predictor.model is not None,
        "api_version" : APP_VERSION,
        "vader" : "active",
        "roberta": "huggingface_api"
    }
    
@app.post("/o1/predict", response_model=PredictionResponse)
def predict_habit(data : HabitInput):
    try:
        input_dict = data.model_dump()
        result = predictor.predict(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code = 500 , detail = str(e))

@app.post("/o1/analyze", response_model=SentimentResult)
def analyze_text(data = JournalInput):
    if not data or not data.strip():
        raise  HTTPException(400, "Text can not be empty")
    
    result = sentiment.analyze(data)

    return SentimentResult(
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        model_used=result["model_used"],
        emotions=result["emotions"]
    )

@app.post("/o1/journal",response_model=JournalResponse)
def create_journal(data = JournalInput):
    if not data or not data.strip():
        raise  HTTPException(400, "Text can not be empty")
    
    result = sentiment.analyze(data)

    return JournalResponse(
        text=data,
        title="Untitled", 
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        emotions=result["emotions"],
        model_used=result["model_used"],
        created_at=datetime.now().isoformat()
    )