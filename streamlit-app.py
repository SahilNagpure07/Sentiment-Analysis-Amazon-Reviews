import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

# Load the saved models
predictor = pickle.load(open(r"Models/randomforest.pkl", "rb"))
scaler = pickle.load(open(r"Models/Mscaler.pkl", "rb"))
cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
STOPWORDS = set(stopwords.words("english"))

# Define the prediction function
def prediction(predictor, scaler, cv, text_input): 
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]
    return "Positive" if y_predictions == 1 else "Negative"

# Streamlit UI
st.title("Text Sentiment Predictor")
text_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if text_input:
        result = prediction(predictor, scaler, cv, text_input)
        st.write(f"**Sentiment:** {result}")
    else:
        st.warning("Please enter some text to predict.")
