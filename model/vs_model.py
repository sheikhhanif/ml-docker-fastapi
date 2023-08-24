import pickle
import re
from pathlib import Path


with open('vaccine_sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)