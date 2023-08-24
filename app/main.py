#importing modules
from fastapi import FastAPI
from app.model.model import predict_sentiment

# initiating fastapi instance
app = FastAPI()

# Defining path operation for root endpoint
@app.get("/")
def home():
    return {"message": "Vaccine Sentiment Detection"}

# Defining path operation for prediction endpoint
@app.post("/predict", description='enter your view on vaccine')
def predict(text: str):
    sentiment = predict_sentiment(text)
    return {"Sentiment": sentiment}