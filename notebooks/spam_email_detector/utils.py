
import streamlit as st
import pickle
import re
import string


with open("notebooks/spam_email_detector/model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open("notebooks/spam_email_detector/vectorizer.pkl", "rb") as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)



def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    return text


def predict_email(email):
    print(email)
    model=loaded_model
    vectorizer=loaded_vectorizer
    email_processed = preprocess_text(email)
    email_vectorized = vectorizer.transform([email_processed])
    prediction = model.predict(email_vectorized)
    print(prediction)
    return "Spam" if prediction == 1 else "Not Spam"



