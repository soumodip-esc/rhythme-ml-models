# Rhythme ML Models API

Advanced machine learning services for the Rhythme journaling and habit tracking app.

## 🎯 Features

### 1. Sentiment Analysis
Analyze journal entries using hybrid VADER + RoBERTa approach for accurate sentiment detection.

### 2. Habit Prediction
Predict habit completion likelihood based on user's historical behavior patterns.

### 3. Weekly Behavioral Insights
Detect patterns across 14+ days of user data using statistical correlation math. Returns plain-English insight sentences — no ML model, no DB, pure compute.

---

## 📁 Project Structure
```
RHYTHME-ML-MODELS/
├── App/
│   ├── .env                    # Environment variables (HF_TOKEN, API_SECRET)
│   ├── config.py               # Configuration settings
│   ├── schemas.py              # Pydantic models
│   ├── sentiment.py            # Sentiment analysis logic
│   ├── insight_engine.py       # Behavioral pattern correlation math
│   ├── dependencies.py         # API secret auth
│   ├── main.py                 # FastAPI server
│   └── model.py                # Habit prediction model loader
├── Models/
│   └── Model_1/
│       ├── Habit Prediction Model.ipynb
│       └── habit_model.pkl     # Trained habit prediction model
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Environment Variables
Create `App/.env` file:
```env
HF_TOKEN=hf_your_huggingface_token_here
API_SECRET=your_secret_key_here
```

Get your HF token from: https://huggingface.co/settings/tokens

### 3. Run API Server
```bash
uvicorn App.main:app --reload
```

### 4. Access API
- **API Root**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/o1/health

---

## 📡 API Endpoints

All endpoints except `/` and `/o1/health` require the header:
```
x-api-secret: your_secret_here
```

---

### General

#### `GET /`
Returns API info and available endpoints.

#### `GET /o1/health`
Health check for all services.

**Response:**
```json
{
  "status": "healthy",
  "habit-model-loaded": true,
  "api_version": "1.0.0",
  "vader": "active",
  "roberta": "huggingface_api"
}
```

---

### Sentiment Analysis

#### `POST /o1/analyze`
Quick sentiment check on any text.

**Request:**
```json
{
  "text": "I'm so happy today! Everything is going great!"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.87,
  "model_used": "vader",
  "emotions": ["joyful"]
}
```

#### `POST /o1/journal`
Create a journal entry with full sentiment analysis.

**Request:**
```json
{
  "text": "Had an amazing day today!",
  "title": "Great Day"
}
```

**Response:**
```json
{
  "text": "Had an amazing day today!",
  "title": "Great Day",
  "sentiment": "positive",
  "confidence": 0.87,
  "emotions": ["joyful"],
  "model_used": "vader",
  "created_at": "2024-01-15T10:30:00"
}
```

---

### Habit Prediction

#### `POST /o1/predict`
Predict single habit completion.

**Request:**
```json
{
  "day_of_week": 1,
  "is_weekend": 0,
  "current_streak": 10,
  "completion_rate_7d": 0.71,
  "completion_rate_30d": 0.65,
  "days_since_start": 50,
  "frequency_encoded": 0
}
```

**Response:**
```json
{
  "prediction": "complete",
  "probability": 0.782,
  "probability_percent": "78.2%",
  "message": "User will likely complete the habit (78.2%)"
}
```

---

### Weekly Behavioral Insights

#### `POST /o1/insights/weekly`
Detects behavioral patterns from 14+ days of daily logs. Called once per week by the frontend. Nothing is saved — pure compute and return.

**How it works:**
- Frontend stores daily logs in localStorage
- Every 7 days, frontend sends all stored logs to this endpoint
- Backend runs Pearson and Point-Biserial correlation on 6–9 pairs of signals
- Returns up to 5 plain-English insight sentences ranked by strength

**Request:**
```json
{
  "logs": [
    {
      "date": "2024-04-01",
      "journaled": 1,
      "tasks_done": 7,
      "mood": 8,
      "focus_mins": 65,
      "sentiment": {
        "sentiment": "positive",
        "confidence": 0.91,
        "model_used": "roberta",
        "emotions": {}
      }
    },
    {
      "date": "2024-04-02",
      "journaled": 0,
      "tasks_done": 3,
      "mood": 4,
      "focus_mins": 20,
      "sentiment": null
    }
  ]
}
```

**Notes:**
- Minimum 14 days required
- `sentiment` is optional per day — only include it on days the user journaled
- `mood` must be between 1 and 10

**Response — patterns found:**
```json
{
  "insights": [
    "You focus longer on days you journal.",
    "Your mood tends to be higher on days you journal.",
    "You complete more tasks on days you journal.",
    "You get more done on days your mood is high.",
    "You focus for longer when you start the day in a good mood."
  ],
  "days_analyzed": 14,
  "message": null
}
```

**Response — not enough days:**
```json
{
  "insights": [],
  "days_analyzed": 10,
  "message": "Need at least 14 days of data. You sent 10."
}
```

**Response — no strong patterns:**
```json
{
  "insights": [],
  "days_analyzed": 14,
  "message": "No strong patterns found yet. Keep logging more days."
}
```

---

## 🧪 Testing

### Test Sentiment Analysis
```bash
curl -X POST "http://localhost:8000/o1/analyze" \
  -H "Content-Type: application/json" \
  -H "x-api-secret: your_secret_here" \
  -d '{"text": "I am feeling grateful and happy today!"}'
```

### Test Habit Prediction
```bash
curl -X POST "http://localhost:8000/o1/predict" \
  -H "Content-Type: application/json" \
  -H "x-api-secret: your_secret_here" \
  -d '{
    "day_of_week": 1,
    "is_weekend": 0,
    "current_streak": 10,
    "completion_rate_7d": 0.71,
    "completion_rate_30d": 0.65,
    "days_since_start": 50,
    "frequency_encoded": 0
  }'
```

### Test Weekly Insights
```bash
curl -X POST "http://localhost:8000/o1/insights/weekly" \
  -H "Content-Type: application/json" \
  -H "x-api-secret: your_secret_here" \
  -d @test_logs.json
```

Save your 14-day JSON to `test_logs.json` and run the curl above.

### Using Python
```python
import requests

headers = {
    "Content-Type": "application/json",
    "x-api-secret": "your_secret_here"
}

# Sentiment
response = requests.post(
    "http://localhost:8000/o1/analyze",
    headers=headers,
    json={"text": "I'm so happy today!"}
)
print(response.json())

# Habit prediction
response = requests.post(
    "http://localhost:8000/o1/predict",
    headers=headers,
    json={
        "day_of_week": 1,
        "is_weekend": 0,
        "current_streak": 10,
        "completion_rate_7d": 0.71,
        "completion_rate_30d": 0.65,
        "days_since_start": 50,
        "frequency_encoded": 0
    }
)
print(response.json())

# Weekly insights
response = requests.post(
    "http://localhost:8000/o1/insights/weekly",
    headers=headers,
    json={"logs": [...]}  # 14+ DailyLog objects
)
print(response.json())
```

---

## 📊 Sentiment Analysis

### How It Works

**Hybrid Approach:**
1. **VADER** (Fast) — Analyzes 85% of entries instantly (2ms)
2. **RoBERTa via Hugging Face API** (Accurate) — Handles complex cases (15%)

**Sentiment Values:** `positive`, `negative`, `neutral`

**Confidence:** 0.0 to 1.0 (higher = more confident)

**Models Used:**
- `vader` — VADER only (fast, confident)
- `vader+roberta` — Both models (complex text)
- `vader_fallback` — VADER backup (RoBERTa unavailable)

**Emotions:**
- Positive: `joyful`, `grateful`, `productive`, `calm`
- Negative: `sad`, `anxious`, `frustrated`, `tired`
- Neutral: `reflective`, `neutral`

---

## 📈 Habit Prediction

### Input Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| day_of_week | int | 0–6 | 0=Monday, 6=Sunday |
| is_weekend | int | 0–1 | 0=Weekday, 1=Weekend |
| current_streak | int | ≥0 | Consecutive completion days |
| completion_rate_7d | float | 0.0–1.0 | 7-day success rate |
| completion_rate_30d | float | 0.0–1.0 | 30-day success rate |
| days_since_start | int | ≥0 | Days since habit created |
| frequency_encoded | int | 0–3 | 0=Daily, 1=Weekly, 2=Monthly, 3=Twice |

### Model Performance
- **Algorithm**: Random Forest Classifier
- **Training Accuracy**: 73.0%
- **Test Accuracy**: 69.5%
- **Overfitting Gap**: 3.5%

---

## 🧠 Weekly Behavioral Insights

### Signal Pairs Analyzed

| Pair | Method |
|------|--------|
| journaled vs tasks_done | Point-Biserial |
| journaled vs mood | Point-Biserial |
| journaled vs focus_mins | Point-Biserial |
| mood vs tasks_done | Pearson |
| mood vs focus_mins | Pearson |
| tasks_done vs focus_mins | Pearson |
| sentiment_confidence vs mood | Pearson (if 14+ days have sentiment) |
| sentiment_confidence vs tasks_done | Pearson (if 14+ days have sentiment) |
| sentiment_confidence vs focus_mins | Pearson (if 14+ days have sentiment) |

### Rules
- Correlations weaker than ±0.35 are dropped as noise
- Remaining pairs ranked by strength, top 5 returned
- `message` is null when insights are found — only populated on errors

### Storage Note
This endpoint does not save anything. The frontend is responsible for storing daily logs in localStorage and sending all of them on each weekly request.

---

## 🔧 Configuration

### Environment Variables
```env
HF_TOKEN=hf_your_token_here
API_SECRET=your_secret_key_here
CONFIDENCE_THRESHOLD=0.70
ROBERTA_RETRIES=3
RETRY_DELAY=2
```

### Sentiment Settings (`config.py`)
- **HF_MODEL**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **CONFIDENCE_THRESHOLD**: `0.70`
- **ROBERTA_RETRIES**: `3`
- **RETRY_DELAY**: `2`

---

## 📦 Dependencies
```txt
fastapi==0.104.1
uvicorn==0.24.0
vaderSentiment==3.3.2
huggingface-hub==0.20.0
python-dotenv==1.0.0
pydantic==2.5.0
scipy
```

---

## 🚦 Error Handling

### Auth Error
```json
{ "detail": "Forbidden" }
```
HTTP `403` — wrong or missing `x-api-secret` header.

### Sentiment Errors
**Missing text:**
```json
{ "detail": "Text cannot be empty" }
```
**RoBERTa API failure:** Automatically falls back to VADER. `model_used` will be `"vader_fallback"`.

### Habit Prediction Errors
**Invalid input:**
```json
{ "detail": "Validation error: day_of_week must be 0-6" }
```

### Weekly Insights Errors
**No logs sent:**
```json
{ "detail": "No logs provided." }
```
**Under 14 days:**
```json
{
  "insights": [],
  "days_analyzed": 10,
  "message": "Need at least 14 days of data. You sent 10."
}
```

---

## 🌐 Deployment

### Local Development
```bash
uvicorn App.main:app --reload
```

### Production (Render)
1. Connect GitHub repository
2. Set environment variables: `HF_TOKEN`, `API_SECRET`
3. Deploy automatically

---

## 🔥 Feature Highlights

- ✅ **Hybrid Sentiment Analysis** — VADER (fast) + RoBERTa (accurate)
- ✅ **Automatic Fallback** — Always returns results even if APIs fail
- ✅ **Emotion Detection** — Extracts emotion tags automatically
- ✅ **Habit Prediction** — ML-powered completion forecasting
- ✅ **Weekly Behavioral Insights** — Correlation math on 14+ days of logs
- ✅ **API Secret Auth** — All sensitive endpoints protected
- ✅ **RESTful API** — Clean, documented endpoints
- ✅ **Type Safety** — Pydantic schemas with field validation
- ✅ **Interactive Docs** — Auto-generated Swagger UI at `/docs`
- ✅ **Production Ready** — Error handling, logging, retries

---

## 📄 License

MIT License

---

## 👥 Support

- Check API docs: http://localhost:8000/docs
- Review code in `App/` folder
- Check logs for detailed error messages