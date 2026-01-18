from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and vectorizer
try:
    model = joblib.load("xss_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR = str(e)
    print(f"Error loading model: {e}")

# Load metrics and dataset info (if available)
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


@app.route("/", methods=["GET"])
def index():
    """
    API documentation endpoint.
    """
    return jsonify({
        "name": "XSS Detection API",
        "version": "1.0.0",
        "description": "Machine Learning API for detecting XSS (Cross-Site Scripting) attacks",
        "endpoints": {
            "GET /": "API documentation (this page)",
            "GET /health": "Health check endpoint",
            "POST /predict": "Predict if text contains XSS attack",
            "GET /metrics": "Get model performance metrics",
            "GET /info": "Get dataset information"
        },
        "usage": {
            "predict": {
                "method": "POST",
                "url": "/predict",
                "headers": {"Content-Type": "application/json"},
                "body": {"text": "your text to analyze"},
                "response": {
                    "prediction": "XSS or Benign",
                    "confidence": "confidence score (if available)",
                    "is_malicious": "boolean"
                }
            }
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    """
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.now().isoformat(),
        "error": MODEL_ERROR if not MODEL_LOADED else None
    }), 200 if MODEL_LOADED else 503


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON API endpoint for XSS prediction.
    Expects: { "text": "..." }
    Returns: { "prediction": "XSS" | "Benign", "is_malicious": bool }
    """
    if not MODEL_LOADED:
        return jsonify({
            "error": "Model not loaded",
            "message": MODEL_ERROR
        }), 503
    
    # Validate request
    if not request.is_json:
        return jsonify({
            "error": "Invalid request",
            "message": "Content-Type must be application/json"
        }), 400
    
    data = request.get_json()
    
    # Validate input
    if not data or "text" not in data:
        return jsonify({
            "error": "Missing required field",
            "message": "Request body must contain 'text' field"
        }), 400
    
    text = data.get("text", "")
    
    # Validate text is not empty
    if not text or not text.strip():
        return jsonify({
            "error": "Invalid input",
            "message": "Text field cannot be empty"
        }), 400
    
    try:
        # Make prediction
        vec = tfidf.transform([text])
        raw_pred = model.predict(vec)[0]
        result = prediction_label(raw_pred)
        
        # Try to get probability if available
        confidence = 0.0
        try:
            proba = model.predict_proba(vec)[0]
            confidence = float(max(proba))
        except Exception:
            pass
        
        # Determine status
        is_xss = result == "XSS"
        
        # Debugging: Print hasil prediksi di terminal Python
        label_str = "BAHAYA (XSS)" if is_xss else "AMAN"
        print(f"[RESULT] Prediksi: {label_str} | Yakin: {confidence * 100:.2f}%")
        
        # Format response sesuai request user
        response = {
            "query": text,
            "prediction": "XSS" if is_xss else "NORMAL",
            "confidence_score": f"{confidence:.2f}",
            "is_safe": not is_xss
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Get model performance metrics.
    """
    if not METRICS:
        return jsonify({
            "error": "Metrics not available",
            "message": "metrics.json file not found"
        }), 404
    
    return jsonify(METRICS), 200


@app.route("/info", methods=["GET"])
def info():
    """
    Get dataset information.
    """
    if not DATASET_INFO:
        return jsonify({
            "error": "Dataset info not available",
            "message": "dataset_info.json file not found"
        }), 404
    
    return jsonify(DATASET_INFO), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist"
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        "error": "Method not allowed",
        "message": "The HTTP method is not allowed for this endpoint"
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500


if __name__ == "__main__":
    print("=" * 50)
    print("XSS Detection API Server")
    print("=" * 50)
    print(f"Model loaded: {MODEL_LOADED}")
    print(f"Starting server on http://0.0.0.0:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
