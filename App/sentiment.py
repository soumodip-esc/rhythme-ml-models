from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from huggingface_hub import InferenceClient
from App.config import HF_TOKEN, HF_MODEL, CONFIDENCE_THRESHOLS, ROBERTA_RETRIES, RETRY_DELAY
import time
import re

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

#Initialize vader model
vader = SentimentIntensityAnalyzer()

#Initialize Hugging Face Client Details
client = InferenceClient(provider = "hf-inference", api_key = HF_TOKEN)


def vader_sentiment(text:str):
    scores = vader.polarity_scores(text)
    compound = scores["compound"]
    confidence = abs(compound)

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, confidence

def roberta_sentiment(text:str):
    for attempt in range(1, ROBERTA_RETRIES + 1):
        try:
            result = client.text_classification(text, model=HF_MODEL)
            if not isinstance(result, list) or len(result) == 0:
                print(f"Unexpected RoBERTa Respose: {result}")
                return "error", 0.0
            
            preds = result if isinstance(result[0], dict) else result[0]
            top  = max(preds, key = lambda x:x.get("score", 0.0))

            raw_label = top.get("label", "")
            score = top.get("score", 0.0)
            final_label = LABEL_MAP.get(raw_label, raw_label)

            print(f"RoBERTa -> {final_label} ({score:.2f})")
            return final_label, score
        
        except Exception as e:
            print(f"RoBERTa API error (attempt {attempt} / {ROBERTA_RETRIES}): {e}")
            if attempt < ROBERTA_RETRIES:
                time.sleep(RETRY_DELAY)

    return "error", 0.0    
    
def analyze(text: str):
    vader_label, vader_conf = vader_sentiment(text)
    print(f"Vader -> {vader_label},{vader_conf:.2f}")
    
    if vader_conf >= CONFIDENCE_THRESHOLS:
        return {
            "sentiment": vader_label,  # ← Fixed typo
            "confidence": vader_conf,
            "model_used": "vader",
            "emotions": get_emotions(text, vader_label)
        }
    
    print(f"VADER uncertain -> Calling RoBERTa API...")
    rob_label, rob_conf = roberta_sentiment(text)
    
    if rob_label == "error":  # ← Fixed: check label not conf
         return {
            "sentiment": vader_label,
            "confidence": vader_conf,
            "model_used": "vader-fallback",
            "emotions": get_emotions(text, vader_label)
        }
    
    return {
        "sentiment": rob_label,
        "confidence": rob_conf,
        "model_used": "roberta",
        "emotions": get_emotions(text, rob_label)
    }

def get_emotions(text: str, sentiment: str):
    text_lower = text.lower()
    emotions = []

    # Tokenize text into proper words (avoids partial matching issues)
    words = set(re.findall(r'\b\w+\b', text_lower))

    emotion_keywords = {

        "positive": {

            "happy": [
                "happy", "joy", "joyful", "cheerful", "glad",
                "delighted", "content", "pleased", "smiling",
                "bliss", "great", "awesome", "good"
            ],

            "calm": [
                "calm", "relaxed", "peaceful", "comfortable",
                "chill", "serene", "balanced", "stable",
                "quiet", "easy"
            ],

            "excited": [
                "excited", "thrilled", "pumped", "enthusiastic",
                "eager", "hyped", "ecstatic", "overjoyed",
                "energetic"
            ]
        },

        "negative": {

            "sad": [
                "sad", "unhappy", "down", "depressed",
                "low", "heartbroken", "lonely",
                "miserable", "hopeless", "crying"
            ],

            "frustrated": [
                "frustrated", "angry", "irritated",
                "annoyed", "fed up", "mad",
                "resentful", "upset"
            ],

            "anxious": [
                "anxious", "worried", "stress",
                "stressed", "nervous", "tense",
                "panic", "afraid", "overthinking",
                "restless"
            ]
        }
    }

    # If sentiment is not positive or negative
    if sentiment not in emotion_keywords:
        return ["neutral"]

    # Check emotions
    for emotion, keyword_list in emotion_keywords[sentiment].items():
        for keyword in keyword_list:
            if keyword in words:
                emotions.append(emotion)
                break  # stop checking once matched

    return emotions if emotions else ["neutral"]

