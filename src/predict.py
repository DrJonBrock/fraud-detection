"""
Predict fraud on new transaction data.
"""
import numpy as np
import joblib
import pandas as pd
import sys
sys.path.append('.')
from src.preprocessing import preprocess_features

def predict(model_path: str, input_csv: str, output_csv: str = None):
    """Load model and make predictions on new data."""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print(f"Loading input data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Preprocess: assumes same columns as training (excluding Class)
    if 'Class' in df.columns:
        y_true = df['Class'].values
        X = df.drop('Class', axis=1)
    else:
        y_true = None
        X = df.copy()

    # Ensure same feature order as training (except target)
    # In real usage, we'd need to store feature names from training
    # Here we assume identical columns
    X_processed, _, _ = preprocess_features(X)

    # Predict
    y_pred = model.predict(X_processed)
    y_pred_proba = model.predict_proba(X_processed)[:, 1]

    # Add predictions to dataframe
    df['Predicted_Fraud'] = y_pred
    df['Fraud_Probability'] = y_pred_proba

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    else:
        print(df[['Predicted_Fraud', 'Fraud_Probability']].head())

    if y_true is not None:
        from sklearn.metrics import classification_report, roc_auc_score
        print("\n=== Summary (if true labels available) ===")
        print(classification_report(y_true, y_pred, digits=4))
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"ROC-AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    predict("models/random_forest.pkl", "data/new_transactions.csv", "predictions.csv")
