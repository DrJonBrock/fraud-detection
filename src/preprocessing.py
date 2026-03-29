"""
Preprocessing utilities for credit card fraud detection.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Tuple, Dict, Any

def load_data(data_path: str) -> pd.DataFrame:
    """Load the creditcard.csv dataset."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} transactions. Fraud rate: {df['Class'].mean():.4%}")
    return df

def preprocess_features(df: pd.DataFrame, scaler_path: str = None) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Preprocess features:
    - Scale 'Time' and 'Amount' using StandardScaler
    - Keep V1-V28 as-is (already PCA-scaled)
    Returns X (features), y (target), scaler
    """
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Scale Time and Amount
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

    # Save scaler for inference if path provided
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    return X.values, y.values, scaler

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 42) -> Tuple:
    """Stratified split into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train fraud rate: {y_train.mean():.4%}, Test fraud rate: {y_test.mean():.4%}")
    return X_train, X_test, y_train, y_test

def apply_smote(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling to balance classes."""
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE - Train size: {len(X_resampled)}, Fraud rate: {y_resampled.mean():.4%}")
        return X_resampled, y_resampled
    except ImportError:
        print("imbalanced-learn not installed. Skipping SMOTE.")
        return X_train, y_train

if __name__ == "__main__":
    # Quick test
    df = load_data("data/creditcard.csv")
    X, y, scaler = preprocess_features(df, "models/scaler.pkl")
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Optionally SMOTE
    # X_train, y_train = apply_smote(X_train, y_train)
    print("Preprocessing complete.")
