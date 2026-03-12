import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# model load
model = pickle.load(open("word_model.pkl","rb"))

# dataset load
df = pd.read_csv("abcnews-date-text.csv")

# sentence vector function
def sentence_vector(sentence):
    
    words = sentence.lower().split()
    vectors = []
    
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
            
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    
    return np.mean(vectors, axis=0)


st.title("🔎 Intelligent Semantic Search Engine")

st.write("This search engine understands meaning using Word2Vec embeddings.")

query = st.text_input("Enter your search query")

if st.button("Search"):
    
    query_vec = sentence_vector(query)
    
    results = []
    
    for text in df["headline_text"][:10000]:
        
        vec = sentence_vector(text)
        
        score = cosine_similarity([query_vec],[vec])[0][0]
        
        results.append((text,score))
        
    results = sorted(results,key=lambda x:x[1],reverse=True)
    
    st.subheader("Top Results")
    
    for text,score in results[:10]:
        
        st.write(f"{text}  |  Similarity: {score:.3f}")