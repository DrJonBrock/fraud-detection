# Fraud Detection API - Deployment Guide

## Overview
FastAPI service for credit card fraud detection using SMOTE Random Forest.

- **Endpoint:** `/predict` (POST)
- **Health check:** `/health`
- **Model:** `models/smote_random_forest.pkl`
- **Scaler:** `models/scaler.pkl`

## Local Testing
```bash
cd /home/lukef/.openclaw/workspace/projects/fraud-detection
source venv/bin/activate  # or Windows: venv\Scripts\activate
uvicorn src.api:app --reload
# Visit http://localhost:8000/docs for Swagger UI
```

Example request:
```json
{
  "Time": 0.0,
  "V1": -1.23,
  "V2": 0.45,
  ...
  "V28": 0.12,
  "Amount": 25.5
}
```

## Deploy to Railway

1. Push code to GitHub (if not already)
2. Create Railway project: `railway init`
3. Set environment (none required)
4. Deploy: `railway up`

Railway will build via Dockerfile and deploy automatically on git push.

## Deploy to Render

1. Create a Web Service pointing to your GitHub repo
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
4. Add health check path: `/health`
5. Optional: set Python version (3.12)

## Environment Variables
None required for MVP. Future: add monitoring, logging, database connections.

## Notes
- Model expects all 30 features (Time, V1-V28, Amount)
- Time and Amount are scaled using the saved StandardScaler
- Prediction returns: `{prediction: 0|1, label: "legitimate"|"fraud", fraud_probability: float}`
- Consider adding authentication for production use
- Set up logging and request tracking for monitoring

## Performance
- Model size: ~24.7 MB (SMOTE RF)
- Cold start latency: ~500ms-1s
- Subsequent predictions: <50ms

## Post-Deployment
- Test `/health` endpoint
- Send sample transactions (both fraud and legitimate) to verify
- Set up alerts for high fraud probability transactions
- Add rate limiting
