from flask import Flask, request, jsonify, render_template
import joblib
import json
import os

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("xss_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Load metrics and dataset info (if available) for display in UI
METRICS = {}
DATASET_INFO = {}
if os.path.exists("metrics.json"):
    with open("metrics.json", "r", encoding="utf-8") as f:
        METRICS = json.load(f)
if os.path.exists("dataset_info.json"):
    with open("dataset_info.json", "r", encoding="utf-8") as f:
        DATASET_INFO = json.load(f)


def prediction_label(pred):
    """
    Map raw model prediction to human-friendly label.
    Handles numeric and string label formats.
    """
    p = pred
    # If it's an array-like from numpy, take first element outside caller
    if isinstance(p, str):
        val = p.strip().lower()
        return "XSS" if val in {"1", "xss", "malicious"} else "Benign"
    try:
        # Numeric (0/1)
        if int(p) == 1:
            return "XSS"
        return "Benign"
    except Exception:
        return "Benign"


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON API endpoint.
    Expects: { "text": "..." }
    Returns: { "prediction": "XSS" | "Benign", "confidence_score": float }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    vec = tfidf.transform([text])
    raw_pred = model.predict(vec)[0]
    result = prediction_label(raw_pred)

    confidence = 0.0
    try:
        proba = model.predict_proba(vec)[0]
        confidence = float(max(proba))
    except Exception:
        pass

    return jsonify({
        "prediction": result,
        "confidence_score": confidence
    })


@app.route("/", methods=["GET"])
def home():
    """
    Render the web UI form.
    """
    return render_template("index.html", metrics=METRICS, dataset_info=DATASET_INFO)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


