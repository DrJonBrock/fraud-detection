"""
Evaluate a trained fraud detection model.
"""
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import sys
sys.path.append('.')
from src.preprocessing import load_data, preprocess_features

def evaluate(model_path: str, data_path: str):
    """Load a model and evaluate on test data."""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    X, y, _ = preprocess_features(df)

    # Predict
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y, y_pred, digits=4))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y, y_pred))

    roc_auc = roc_auc_score(y, y_pred_proba)
    print(f"\nROC-AUC: {roc_auc:.4f}")

    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC: {pr_auc:.4f}")

if __name__ == "__main__":
    evaluate("models/random_forest.pkl", "data/creditcard.csv")
