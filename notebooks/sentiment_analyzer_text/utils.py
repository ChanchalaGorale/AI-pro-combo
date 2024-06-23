
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(stopwords.words('english'))

with open("notebooks/sentiment_analyzer_text/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


def predict_sentiment(text):

    tokens = word_tokenize(str(text).lower())
    filtered_tokens = [token for token in tokens if token not in stop_words]

    processed_text = ' '.join(filtered_tokens)

    tfidf_vectorizer = TfidfVectorizer() 

    new = tfidf_vectorizer.fit_transform([ processed_text])

    prediction = model.predict(new)

    print(prediction)

    return prediction



