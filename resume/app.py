from flask import Flask, render_template, request
import pickle
from pdfminer.high_level import extract_text
import os

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Upload folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Check if file part exists
    if "resume" not in request.files:
        return "No file part"

    file = request.files["resume"]

    if file.filename == "":
        return "No selected file"

    # Safe file path
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

    # Save uploaded file
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)

    # Extract text from PDF
    try:
        text = extract_text(filepath)
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

    # Vectorize
    vec = vectorizer.transform([text])

    # Predict domain
    prediction = model.predict(vec)[0]

    # Predict probability (confidence score)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0]
        predicted_index = list(model.classes_).index(prediction)
        confidence_score = round(prob[predicted_index] * 100, 2)  # percentage
    else:
        confidence_score = None  # agar model probability support nahi karta

    return render_template("resume.html", prediction=prediction, score=confidence_score)


if __name__ == "__main__":
    app.run(debug=True)