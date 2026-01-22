# Rhythme ML Models API

Advanced machine learning services for the Rhythme journaling and habit tracking app.

## 🎯 Features

### 1. Sentiment Analysis
Analyze journal entries using hybrid VADER + RoBERTa approach for accurate sentiment detection.

### 2. Habit Prediction
Predict habit completion likelihood based on user's historical behavior patterns.

---

## 📁 Project Structure
```
RHYTHME-ML-MODELS/
├── App/
│   ├── .env                    # Environment variables (HF_TOKEN)
│   ├── config.py               # Configuration settings
│   ├── schemas.py              # Pydantic models
│   ├── sentiment.py            # Sentiment analysis logic
│   ├── main.py                 # FastAPI server
│   └── model.py                # Database models
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
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Run API Server
```bash
cd App
python main.py
```

Or:
```bash
uvicorn App.main:app --reload
```

### 4. Access API
- **API Root**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 📡 API Endpoints

### General

#### `GET /`
Welcome page with API information

#### `GET /health`
Health check for all services

**Response:**
```json
{
  "status": "healthy",
  "vader": "active",
  "roberta": "huggingface_api"
}
```

---

### Sentiment Analysis Endpoints

#### `POST /analyze`
Analyze sentiment of text only (quick sentiment check)

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

#### `POST /journal`
Create journal entry with complete sentiment analysis

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

### Habit Prediction Endpoints

#### `POST /predict`
Predict single habit completion

**Request:**
```json
{
  "day_of_week": 1,
  "is_weekend": 0,
  "current_streak": 10,
  "completion_ratio_7d": 0.71,
  "completion_ratio_30d": 0.65,
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

#### `POST /predict/batch`
Make multiple habit predictions at once

---

## 🧪 Testing

### Test Sentiment Analysis (cURL)
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am feeling grateful and happy today!"
  }'
```

### Test Habit Prediction (cURL)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "day_of_week": 1,
    "is_weekend": 0,
    "current_streak": 10,
    "completion_ratio_7d": 0.71,
    "completion_ratio_30d": 0.65,
    "days_since_start": 50,
    "frequency_encoded": 0
  }'
```

### Using Python
```python
import requests

# Test sentiment analysis
response = requests.post(
    "http://localhost:8000/analyze",
    json={"text": "I'm so happy today!"}
)
print(response.json())

# Test habit prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "day_of_week": 1,
        "is_weekend": 0,
        "current_streak": 10,
        "completion_ratio_7d": 0.71,
        "completion_ratio_30d": 0.65,
        "days_since_start": 50,
        "frequency_encoded": 0
    }
)
print(response.json())
```

---

## 📊 Sentiment Analysis

### How It Works

**Hybrid Approach:**
1. **VADER** (Fast) - Analyzes 85% of entries instantly (2ms)
2. **RoBERTa via Hugging Face API** (Accurate) - Handles complex cases (15%)

**Sentiment Values:**
- `positive` - Positive sentiment detected
- `negative` - Negative sentiment detected
- `neutral` - Neutral sentiment detected

**Confidence:** 0.0 to 1.0 (higher = more confident)

**Models Used:**
- `vader` - VADER only (fast, confident)
- `vader+roberta` - Both models (complex text)
- `vader_fallback` - VADER backup (RoBERTa unavailable)

**Emotions:** Automatically detected emotion tags
- Positive: `joyful`, `grateful`, `productive`, `calm`
- Negative: `sad`, `anxious`, `frustrated`, `tired`
- Neutral: `reflective`, `neutral`

---

## 📈 Habit Prediction

### Input Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| day_of_week | int | 0-6 | 0=Monday, 6=Sunday |
| is_weekend | int | 0-1 | 0=Weekday, 1=Weekend |
| current_streak | int | ≥0 | Consecutive completion days |
| completion_ratio_7d | float | 0.0-1.0 | 7-day success rate |
| completion_ratio_30d | float | 0.0-1.0 | 30-day success rate |
| days_since_start | int | ≥0 | Days since habit created |
| frequency_encoded | int | 0-3 | 0=Daily, 1=Weekly, 2=Monthly, 3=Twice |

### Model Performance

- **Algorithm**: Random Forest Classifier
- **Training Accuracy**: 73.0%
- **Test Accuracy**: 69.5%
- **Overfitting Gap**: 3.5%

---

## 🔧 Configuration

### Environment Variables (`.env`)
```env
# Required: Hugging Face API Token
HF_TOKEN=hf_your_token_here

# Optional: Custom settings
CONFIDENCE_THRESHOLD=0.70
ROBERTA_RETRIES=3
RETRY_DELAY=2
```

### Sentiment Settings (`config.py`)

- **HF_MODEL**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **CONFIDENCE_THRESHOLD**: `0.70` (VADER confidence threshold)
- **ROBERTA_RETRIES**: `3` (API retry attempts)
- **RETRY_DELAY**: `2` (seconds between retries)

---

## 📦 Dependencies
```txt
fastapi==0.104.1
uvicorn==0.24.0
vaderSentiment==3.3.2
huggingface-hub==0.20.0
python-dotenv==1.0.0
pydantic==2.5.0
```

---

## 🚦 Error Handling

### Sentiment Analysis Errors

**Missing Text:**
```json
{
  "detail": "Text cannot be empty"
}
```

**RoBERTa API Failure:**
- Automatically falls back to VADER
- `model_used: "vader_fallback"` in response

### Habit Prediction Errors

**Invalid Input:**
```json
{
  "detail": "Validation error: day_of_week must be 0-6"
}
```

---

## 📝 Response Schemas

### SentimentResult
```json
{
  "sentiment": "string",
  "confidence": "float",
  "model_used": "string",
  "emotions": ["string"]
}
```

### JournalResponse
```json
{
  "text": "string",
  "title": "string",
  "sentiment": "string",
  "confidence": "float",
  "emotions": ["string"],
  "model_used": "string",
  "created_at": "string"
}
```

---

## 🌐 Deployment

### Local Development
```bash
python App/main.py
```

### Production (Supabase Edge Functions)
```bash
supabase functions deploy rhythme-ml-api
```

### Production (Railway/Render)
1. Connect GitHub repository
2. Set environment variable: `HF_TOKEN`
3. Deploy automatically

---

## 📄 License

MIT License

---

## 👥 Support

For issues or questions:
- Check API docs: http://localhost:8000/docs
- Review code in `App/` folder
- Check logs for detailed error messages

---

## 🎯 Quick Examples

### Example 1: Positive Journal Entry
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Had the best day ever! So grateful!"}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "model_used": "vader",
  "emotions": ["joyful", "grateful"]
}
```

### Example 2: Complex Sarcastic Entry
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Great, just great... another Monday."}'
```

**Response:**
```json
{
  "sentiment": "negative",
  "confidence": 0.85,
  "model_used": "vader+roberta",
  "emotions": ["frustrated"]
}
```

### Example 3: High Streak Habit Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "day_of_week": 2,
    "is_weekend": 0,
    "current_streak": 25,
    "completion_ratio_7d": 0.85,
    "completion_ratio_30d": 0.80,
    "days_since_start": 100,
    "frequency_encoded": 0
  }'
```

**Response:**
```json
{
  "prediction": "complete",
  "probability": 0.89,
  "probability_percent": "89.0%",
  "message": "User will likely complete the habit (89.0%)"
}
```

---

## 🔥 Features Highlights

- ✅ **Hybrid Sentiment Analysis** - VADER (fast) + RoBERTa (accurate)
- ✅ **Automatic Fallback** - Always returns results even if APIs fail
- ✅ **Emotion Detection** - Extracts emotion tags automatically
- ✅ **Habit Prediction** - ML-powered completion forecasting
- ✅ **RESTful API** - Clean, documented endpoints
- ✅ **Type Safety** - Pydantic schemas for validation
- ✅ **Interactive Docs** - Auto-generated Swagger UI
- ✅ **Production Ready** - Error handling, logging, retries