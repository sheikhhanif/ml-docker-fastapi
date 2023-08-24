import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

def load_split_data():
  dataset = pd.read_csv('/Users/hanif/Desktop/ml-docker-fastapi/model/dataset/tweets.csv')
  data = dataset.sample(frac=1, random_state=42)
  data.reset_index(inplace=True, drop=True)
  trainx, testx, trainy, testy = train_test_split(data['comments'],data['sentiment'], stratify=data['sentiment'], test_size=0.2)
  return trainx, testx, trainy, testy


def train_evaluate_model():
  trainx, testx, trainy, testy = load_split_data()
  tfidf = TfidfVectorizer(stop_words='english')
  train_x_vector = tfidf.fit_transform(trainx)
  svc = SVC(kernel='linear')
  svc.fit(train_x_vector, trainy) 
  test_x_vector = tfidf.transform(testx)
  print('Accuracy Score')
  print(svc.score(test_x_vector, testy))


train_evaluate_model()