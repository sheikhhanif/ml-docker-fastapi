import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent

sentiment = ['Positive', 'Negative']

def predict_sentiment(text):
    with open(f"{BASE_DIR}/vaccine_sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    return sentiment[model.predict([text])[0]]
