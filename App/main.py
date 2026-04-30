from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from App.config import APP_TITLE, APP_VERSION, APP_DESCRIPTION
from App.model import predictor
from App.schemas import (
    HabitInput, PredictionResponse,
    JournalInput, JournalResponse, SentimentResult,
    WeeklyInsightRequest, WeeklyInsightResponse,
    GoalGenerateRequest, GoalGenerateResponse
)
from App.insight_engine import generate_insights
from App.goals_engine import generate_goal_plan
from App.dependencies import verify_api_secret
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
    "https://*-oopsvincent-projects.vercel.app",
    "https://rhythme.amplecen.com",
    "https://amplecen.com",
    "https://*.amplecen.com"
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
        "health":"/o1/health",
        "journal" :"/o1/journal",
        "analyze":"/o1/analyze",
        "insights": "/o1/insights_weekly"
    }
    
@app.get("/v1/health")
def health_check():
    return {
        "status" : "healthy" if predictor.model is not None else "Unhealthy",
        "habit-model-loaded" : predictor.model is not None,
        "api_version" : APP_VERSION,
        "vader" : "active",
        "roberta": "huggingface_api"
    }
    
@app.post("/v1/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_secret)])
def predict_habit(data : HabitInput):
    try:
        input_dict = data.model_dump()
        result = predictor.predict(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code = 500 , detail = str(e))

@app.post("/v1/analyze", response_model=SentimentResult, dependencies=[Depends(verify_api_secret)])
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

@app.post("/v1/journal",response_model=JournalResponse, dependencies=[Depends(verify_api_secret)])
def create_journal(data : JournalInput):
    if not data.text or not data.text.strip():
        raise  HTTPException(400, "Text can not be empty")
    
    result = sentiment.analyze(data.text)

    return JournalResponse(
        text=data.text,
        title=data.title or "Untitled", 
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        emotions=result["emotions"],
        model_used=result["model_used"],
        created_at=datetime.now().isoformat()
    )

@app.post("/v1/insights_weekly", response_model=WeeklyInsightResponse, dependencies=[Depends(verify_api_secret)])
def weekly_insights(data: WeeklyInsightRequest):
    if not data.logs:
        raise HTTPException(status_code=400, detail="No logs provided.")
 
    result = generate_insights(data.logs)
    return WeeklyInsightResponse(
        insights=result["insights"],
        days_analyzed=result["days_analyzed"],
        message=result["message"]
    )
 
@app.post("/api/v1/goals/generate", dependencies=[Depends(verify_api_secret)])
def generate_goal(request: GoalGenerateRequest) -> GoalGenerateResponse:
    result = generate_goal_plan(request.goal_title, request.goal_description)
    return result