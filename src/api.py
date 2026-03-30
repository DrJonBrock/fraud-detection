"""
FastAPI service for fraud detection inference.
"""
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Load model and scaler at startup
MODEL_PATH = "models/smote_random_forest.pkl"
SCALER_PATH = "models/scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"Loaded model from {MODEL_PATH} and scaler from {SCALER_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to load model/scaler: {e}")

app = FastAPI(
    title="Fraud Detection API",
    description="Detect fraudulent credit card transactions",
    version="1.0.0"
)

class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds since first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount in original currency")

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    fraud_probability: float
    model_used: str = "SMOTE Random Forest"

@app.get("/")
def root():
    return {"message": "Fraud Detection API", "model": "SMOTE_RF", "status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """
    Predict whether a transaction is fraudulent.
    Returns: 1=fraud, 0=legitimate, plus probability of fraud.
    """
    try:
        # Convert input to array
        features = np.array([[transaction.Time] + [getattr(transaction, f'V{i}') for i in range(1,29)] + [transaction.Amount]])
        # Apply scaling to Time and Amount (columns 0 and 29)
        features[:, [0, -1]] = scaler.transform(features[:, [0, -1]])
        # Predict
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0, 1]
        label = "fraud" if pred == 1 else "legitimate"
        return {"prediction": int(pred), "label": label, "fraud_probability": float(proba)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
