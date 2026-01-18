# 🛡️ XSS Detection API

API Flask profesional untuk mendeteksi serangan XSS (Cross-Site Scripting) menggunakan Machine Learning.

## ✨ Fitur

- ✅ **ML-Powered Detection** - Deteksi XSS menggunakan model machine learning
- 🌐 **CORS Enabled** - Dapat diakses dari frontend berbeda domain
- 📊 **Model Metrics** - Lihat performa model (accuracy, precision, recall)
- 🏥 **Health Check** - Monitor status API dan model
- 🔒 **Error Handling** - Response error yang konsisten dan informatif
- 📚 **Auto Documentation** - Dokumentasi API otomatis di endpoint root

## 📁 Struktur File

```
.
├── api.py                  # REST API server
├── app.py                  # Web UI application
├── xss_model.pkl           # Trained ML model
├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
├── metrics.json            # Model performance metrics (optional)
├── dataset_info.json       # Dataset information (optional)
├── requirements.txt        # Python dependencies
├── API_USAGE.md           # Panduan lengkap penggunaan API
└── README.md              # File ini
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
./venv/bin/pip install -r requirements.txt
```

### 2. Jalankan Server

**Untuk REST API:**
```bash
./venv/bin/python api.py
```
Server API berjalan di: **http://localhost:5000**

**Untuk Web UI:**
```bash
./venv/bin/python app.py
```
Web UI berjalan di: **http://localhost:5000**

### 3. Test API

Buka terminal baru dan test dengan curl:

```bash
# Health check
curl http://localhost:5000/health

# Predict XSS
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "<script>alert(\"XSS\")</script>"}'
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API documentation |
| GET | `/health` | Health check |
| POST | `/predict` | Predict XSS attack |
| GET | `/metrics` | Model performance metrics |
| GET | `/info` | Dataset information |

## 📖 Dokumentasi Lengkap

Lihat [API_USAGE.md](API_USAGE.md) untuk:
- Penjelasan detail setiap endpoint
- Contoh request & response
- Error handling
- Contoh kode Python & JavaScript

## 🧪 Contoh Penggunaan Cepat

### Python
```python
import requests

response = requests.post(
    "http://localhost:5000/predict",
    json={"text": "<script>alert('XSS')</script>"}
)
print(response.json())
# Output: {"prediction": "XSS", "is_malicious": true, ...}
```

### cURL
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello World"}'
# Output: {"prediction": "Benign", "is_malicious": false, ...}
```

## 🔧 Konfigurasi

Edit `app.py` untuk mengubah konfigurasi:

```python
# Port dan host
app.run(host="0.0.0.0", port=5000, debug=True)

# Model files
model = joblib.load("xss_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
```

## 📊 Response Format

### Success Response (XSS Detected)
```json
{
  "query": "<script>alert(\"XSS\")</script>",
  "prediction": "XSS",
  "confidence_score": "0.95",
  "is_safe": false
}
```

### Success Response (Benign)
```json
{
  "query": "Hello World",
  "prediction": "NORMAL",
  "confidence_score": "0.87",
  "is_safe": true
}
```

### Error Response
```json
{
  "error": "Missing required field",
  "message": "Request body must contain 'text' field"
}
```

## 🛠️ Development

### Install Dependencies
```bash
./venv/bin/pip install -r requirements.txt
```

### Run in Development Mode
```bash
./venv/bin/python app.py
```

### Run Tests
```bash
./venv/bin/python test_api.py
```

## 📦 Dependencies

- **Flask** - Web framework
- **Flask-CORS** - Cross-Origin Resource Sharing support
- **scikit-learn** - Machine learning library
- **joblib** - Model serialization

## 🔒 Security Notes

- API menggunakan CORS, pastikan konfigurasi CORS sesuai kebutuhan production
- Untuk production, disable debug mode: `app.run(debug=False)`
- Gunakan HTTPS untuk production deployment
- Implementasikan rate limiting untuk mencegah abuse

## 📝 License

Silakan disesuaikan dengan kebutuhan proyek Anda.

## 🤝 Support

Jika ada pertanyaan atau issue, silakan buat issue di repository.
# XssGuard
