from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# load model
model = pickle.load(open("word2vec_model.pkl", "rb"))

def sentence_vector(sentence):

    words = sentence.lower().split()
    vectors = []

    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


@app.route("/", methods=["GET","POST"])
def home():

    score = None

    if request.method == "POST":

        s1 = request.form["sentence1"]
        s2 = request.form["sentence2"]

        v1 = sentence_vector(s1)
        v2 = sentence_vector(s2)

        score = cosine_similarity([v1],[v2])[0][0]

    return render_template("index.html", score=score)


if __name__ == "__main__":
    app.run(debug=True)