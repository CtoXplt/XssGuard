# XSS Detection API - Panduan Penggunaan

API Flask untuk mendeteksi serangan XSS (Cross-Site Scripting) menggunakan Machine Learning.

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
./venv/bin/pip install -r requirements.txt
```

### 2. Jalankan Server
```bash
./venv/bin/python app.py
```

Server akan berjalan di: `http://localhost:5000`

## 📚 API Endpoints

### 1. **GET /** - API Documentation
Mendapatkan dokumentasi API dan daftar semua endpoint yang tersedia.

**Request:**
```bash
curl http://localhost:5000/
```

**Response:**
```json
{
  "name": "XSS Detection API",
  "version": "1.0.0",
  "description": "Machine Learning API for detecting XSS attacks",
  "endpoints": { ... }
}
```

---

### 2. **GET /health** - Health Check
Mengecek status kesehatan API dan model.

**Request:**
```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-12-16T23:18:00.123456"
}
```

---

### 3. **POST /predict** - Prediksi XSS
Endpoint utama untuk mendeteksi apakah teks mengandung serangan XSS.

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "<script>alert(\"XSS\")</script>"}'
```

**Response (XSS Terdeteksi):**
```json
{
  "query": "<script>alert(\"XSS\")</script>",
  "prediction": "XSS",
  "confidence_score": "0.95",
  "is_safe": false
}
```

**Response (Benign):**
```json
{
  "query": "Hello World",
  "prediction": "NORMAL",
  "confidence_score": "0.87",
  "is_safe": true
}
```

---

### 4. **GET /metrics** - Model Metrics
Mendapatkan metrik performa model (accuracy, precision, recall, dll).

**Request:**
```bash
curl http://localhost:5000/metrics
```

**Response:**
```json
{
  "accuracy": 0.95,
  "precision": 0.94,
  "recall": 0.96,
  "f1_score": 0.95
}
```

---

### 5. **GET /info** - Dataset Information
Mendapatkan informasi tentang dataset yang digunakan untuk training.

**Request:**
```bash
curl http://localhost:5000/info
```

**Response:**
```json
{
  "total_samples": 10000,
  "malicious_samples": 5000,
  "benign_samples": 5000,
  "features": ["text"]
}
```

---

## 💻 Contoh Penggunaan dengan Python

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:5000"

# 1. Check health
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# 2. Predict XSS
data = {
    "text": "<script>alert('XSS')</script>"
}
response = requests.post(
    f"{BASE_URL}/predict",
    headers={"Content-Type": "application/json"},
    json=data
)
result = response.json()
print(f"\nPrediction: {result['prediction']}")
print(f"Is Malicious: {result['is_malicious']}")
if 'confidence' in result:
    print(f"Confidence: {result['confidence']:.2%}")

# 3. Test benign text
data = {
    "text": "Hello World"
}
response = requests.post(
    f"{BASE_URL}/predict",
    headers={"Content-Type": "application/json"},
    json=data
)
result = response.json()
print(f"\nPrediction: {result['prediction']}")
print(f"Is Malicious: {result['is_malicious']}")
```

---

## 💻 Contoh Penggunaan dengan JavaScript

```javascript
// Base URL
const BASE_URL = "http://localhost:5000";

// Function to predict XSS
async function predictXSS(text) {
    try {
        const response = await fetch(`${BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const result = await response.json();
        console.log('Prediction:', result.prediction);
        console.log('Is Malicious:', result.is_malicious);
        console.log('Confidence:', result.confidence);
        
        return result;
    } catch (error) {
        console.error('Error:', error);
    }
}

// Test
predictXSS("<script>alert('XSS')</script>");
predictXSS("Hello World");
```

---

## 🔒 Error Handling

API mengembalikan error dengan format standar:

**400 Bad Request:**
```json
{
  "error": "Missing required field",
  "message": "Request body must contain 'text' field"
}
```

**503 Service Unavailable:**
```json
{
  "error": "Model not loaded",
  "message": "Error loading model"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Prediction failed",
  "message": "Error details..."
}
```

---

## 🌐 CORS Support

API sudah mendukung CORS, sehingga bisa diakses dari frontend web yang berbeda domain.

---

## 📝 Notes

- Semua endpoint selalu mengembalikan response dalam format JSON
- Endpoint `/predict` memerlukan `Content-Type: application/json`
- Field `confidence` hanya tersedia jika model mendukung `predict_proba()`
- Semua timestamp menggunakan format ISO 8601
