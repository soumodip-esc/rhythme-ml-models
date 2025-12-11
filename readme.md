# Habit Completion Prediction API - Model 1

Predict whether a user will complete their habit based on historical data.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run API
```bash
uvicorn App.main:app --reload
```

### 3. Access API
- **API Root**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/o1/health

## 📡 Endpoints

### GET `/`
Welcome page with API info

### GET `/health`
Check if API and model are running

### POST `/predict`
Make a single prediction

**Request Body:**
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
  "probablity": 0.782,
  "probablity_precent": "78.2%",
  "massage": "User will likely complete the habit (78.2%)"
}
```

### POST `/predict/batch`
Make multiple predictions at once

## 🧪 Testing

### Using cURL
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

## 📊 Input Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| day_of_week | int | 0-6 | 0=Monday, 6=Sunday |
| is_weekend | int | 0-1 | 0=Weekday, 1=Weekend |
| current_streak | int | ≥0 | Consecutive completion days |
| completion_ratio_7d | float | 0.0-1.0 | 7-day success rate |
| completion_ratio_30d | float | 0.0-1.0 | 30-day success rate |
| days_since_start | int | ≥0 | Days since habit created |
| frequency_encoded | int | 0-3 | 0=Daily, 1=Weekly, 2=Monthly, 3=Twice |

## 📈 Model Info

- **Algorithm**: Random Forest Classifier
- **Training Accuracy**: 73.0%
- **Test Accuracy**: 69.5%
- **Overfitting Gap**: 3.5%
