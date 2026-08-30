# GraphGuard AI — Backend API Service

Production backend service for **GraphGuard AI**, an AI-powered Bitcoin transaction fraud / illicit-transaction detection system. Built using **FastAPI**, **Uvicorn**, **Pydantic v2**, **pandas**, **numpy**, **joblib**, **scikit-learn**, and **XGBoost**.

The backend operates strictly on real Bitcoin graph datasets (203,769 transactions, 234,355 graph edges) and pre-trained ML models (`outputs/phase5/neighborhood_xgboost.joblib`).

---

## 🚀 Quick Start & Installation

### Prerequisites
- Windows 10/11
- Python 3.11.9
- Existing trained artifacts in `outputs/` and raw dataset in `data/raw/`

### 1. Install Dependencies
Activate your virtual environment and install required packages:
```bash
.venv\Scripts\pip.exe install -r backend/requirements.txt
```

### 2. Run the Development Server
From the project root directory (`D:\GraphGuard-AI`), launch the server:
```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
Or using python module execution:
```bash
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

---

## 📍 Base URLs & Documentation

- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc UI**: `http://127.0.0.1:8000/redoc`

---

## 🏗️ Architecture Overview

The backend uses a clean, modular architecture:

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI app initialization, routes, lifespan & CORS
│   ├── config.py              # Centralized settings, file paths & thresholds
│   ├── schemas.py             # Pydantic v2 data models for API schemas
│   └── services/
│       ├── __init__.py
│       ├── data_service.py    # Startup loader & in-memory indexed DataFrames
│       ├── model_service.py   # Leakage-safe XGBoost inference engine
│       ├── graph_service.py   # Adjacency traversal for incoming/outgoing neighbors
│       ├── explanation_service.py # Feature importance & risk signal synthesizer
│       └── dashboard_service.py   # Dataset & model performance metric aggregator
├── requirements.txt           # Dependency requirements
└── README.md                  # System documentation
```

### Performance & Data Management
- **Startup Indexing**: Loads all CSV datasets once at application startup. Indexed DataFrames and graph adjacency lists allow O(1) instantaneous lookups.
- **Zero In-flight CSV Scans**: API requests do not perform repeated disk scanning of the 657 MB feature file or graph recalculation.
- **Zero Retraining**: Model inference executes using the pre-trained 185-feature XGBoost model (`neighborhood_xgboost.joblib`).

---

## 📡 API Endpoints Reference

### 1. Health Check
`GET /api/health`
Checks server readiness and loaded artifacts.

**Example Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "transactions_loaded": true,
  "graph_loaded": true
}
```

---

### 2. Dashboard Aggregate Stats
`GET /api/dashboard`
Returns real aggregate transaction counts, graph edges, label distribution, and model metrics.

**Example Response:**
```json
{
  "total_transactions": 203769,
  "total_edges": 234355,
  "labeled_transactions": 46564,
  "illicit_transactions": 4545,
  "licit_transactions": 42019,
  "unknown_transactions": 157205,
  "high_risk_transactions": 1284,
  "critical_risk_transactions": 386,
  "current_threshold": 0.9,
  "model_pr_auc": 0.7440705572802038,
  "model_roc_auc": 0.9334393476482471,
  "model_f1": 0.7338551859099804
}
```

---

### 3. Transactions List (Paginated)
`GET /api/transactions?page=1&page_size=20&risk_level=CRITICAL&timestep=40`
Supports pagination, risk level filtering (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), timestep filtering, and ID search.

**Example Response:**
```json
{
  "total": 386,
  "page": 1,
  "page_size": 20,
  "total_pages": 20,
  "transactions": [
    {
      "tx_id": 232022460,
      "time_step": 40,
      "risk_score": 0.9421,
      "risk_level": "CRITICAL",
      "prediction": "illicit / high-confidence alert",
      "in_degree": 1,
      "out_degree": 2
    }
  ]
}
```

---

### 4. Single Transaction Detail
`GET /api/transactions/{tx_id}`
Returns complete dataset details, graph features, degree information, and neighborhood risk features for a transaction.

**Example Response:**
```json
{
  "tx_id": 230425980,
  "time_step": 1,
  "risk_score": null,
  "risk_level": "UNKNOWN",
  "prediction": "unassessed",
  "class": "unknown",
  "graph_features": {
    "in_degree": 1,
    "out_degree": 1,
    "total_degree": 2
  },
  "neighborhood_risk_features": {},
  "degree_information": {
    "in_degree": 1,
    "out_degree": 1,
    "total_degree": 2
  }
}
```

---

### 5. Transaction Neighbors
`GET /api/transactions/{tx_id}/neighbors`
Retrieves real incoming and outgoing graph neighbors from the Bitcoin edgelist.

**Example Response:**
```json
{
  "tx_id": 230425980,
  "total_neighbors": 1,
  "incoming_count": 1,
  "outgoing_count": 0,
  "neighbors": [
    {
      "tx_id": 5530458,
      "relationship": "incoming",
      "time_step": 1,
      "risk_score": null,
      "risk_level": "UNKNOWN",
      "class": "unknown"
    }
  ]
}
```

---

### 6. Risk Explanation
`GET /api/transactions/{tx_id}/explanation`
Generates human-understandable evidence including top model feature importances, neighborhood risk signals, and structural graph indicators.

---

### 7. Real-Time Model Prediction
`POST /api/predict`
Calculates risk prediction for a target transaction using its 185-feature vector passed into `neighborhood_xgboost.joblib`.

**Example Request:**
```json
{
  "tx_id": 232022460
}
```

**Example Response:**
```json
{
  "tx_id": 232022460,
  "risk_score": 0.9421,
  "risk_level": "CRITICAL",
  "prediction": "illicit / high-confidence alert"
}
```
*Note: Returns HTTP 422 if transaction features for the specified timestep are unavailable for leakage-safe inference (e.g. timesteps 1-10).*

---

### 8. Model Evaluation Metrics
`GET /api/model/metrics`
Returns evaluated model performance metrics and threshold analysis.

---

### 9. Model Feature Importances
`GET /api/model/feature-importance?limit=20`
Returns top feature importances from Phase 5 model training.

---

### 10. Timesteps Statistics
`GET /api/timesteps`
Returns per-timestep transaction and label statistics (timesteps 1 to 49).

---

### 11. Risk Score Distribution
`GET /api/risk-distribution`
Returns counts for risk level bands (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) for dashboard visualization.

---

## 🎯 Risk Classification Bands

- `0.00 – 0.24`: **LOW** (`low risk`)
- `0.25 – 0.49`: **MEDIUM** (`medium risk`)
- `0.50 – 0.89`: **HIGH** (`review recommended`)
- `0.90 – 1.00`: **CRITICAL** (`illicit / high-confidence alert`) — *Recommended Production Alert Threshold: 0.90*
