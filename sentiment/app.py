import streamlit as st
import pickle

# load model
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("Movie Review Sentiment Analysis")

st.write("Enter a movie review and the model will predict sentiment.")

# user input
review = st.text_area("Enter your review")

if st.button("Predict Sentiment"):
    
    vector = tfidf.transform([review])
    
    prediction = model.predict(vector)[0]
    
    if prediction == 1:
        st.success("Sentiment: Positive 😊")
    else:
        st.error("Sentiment: Negative 😞")