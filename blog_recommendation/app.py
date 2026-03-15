from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("blogs_dataset.csv")

# Load trained Word2Vec model
model = pickle.load(open("word2vec_model.pkl","rb"))


# Sentence vector function
def blog_vector(text):

    words = text.split()

    vectors = []

    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


# Create vectors
df["vector"] = df["Content"].apply(blog_vector)


@app.route("/", methods=["GET","POST"])
def home():

    recommendations = None

    if request.method == "POST":

        title = request.form["title"]

        idx = df[df["Title"] == title].index[0]

        selected_vector = df.iloc[idx]["vector"]

        similarities = []

        for vec in df["vector"]:
            sim = cosine_similarity([selected_vector],[vec])[0][0]
            similarities.append(sim)

        df["similarity"] = similarities

        rec = df.sort_values("similarity", ascending=False)

        recommendations = rec.iloc[1:4][["Title","Content"]].to_dict(orient="records")

    titles = df["Title"].tolist()

    return render_template("index.html", titles=titles, recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)