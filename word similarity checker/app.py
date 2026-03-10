from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# load trained Word2Vec model
model = pickle.load(open("word2vec_model.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def home():

    results = None
    word = ""

    if request.method == "POST":

        word = request.form["word"].lower()

        if word in model.wv:

            similar_words = model.wv.most_similar(word, topn=5)

            results = []

            for w,score in similar_words:

                results.append({
                    "word": w,
                    "score": round(score*100,2)
                })

        else:
            results = "Word not in vocabulary"

    return render_template("index.html", results=results, word=word)


if __name__ == "__main__":
    app.run(debug=True)