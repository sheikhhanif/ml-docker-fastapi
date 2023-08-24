import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def load_split_data():
  dataset = pd.read_csv('/Users/hanif/Desktop/ml-docker-fastapi/model/dataset/tweets.csv')
  data = dataset.sample(frac=1, random_state=42)
  data.reset_index(inplace=True, drop=True)
  trainx, testx, trainy, testy = train_test_split(data['comments'],data['sentiment'], stratify=data['sentiment'], test_size=0.2)
  return trainx, testx, trainy, testy


def train_evaluate_model():
  # spliting data
  trainx, testx, trainy, testy = load_split_data()

  # building model pipeline
  model_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', SVC(kernel='linear')),
    ])
  
  # fitting the model
  model_pipe.fit(trainx, trainy)

  # testing for accuracy
  print('Accuracy Score')
  print(model_pipe.score(testx, testy))

  # saving the model
  with open('vaccine_sentiment_model.pkl', 'wb') as f:
      pickle.dump(model_pipe, f) 


# running the method
train_evaluate_model()


