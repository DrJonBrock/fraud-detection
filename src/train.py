"""
Train a fraud detection model.
"""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV
import os
import sys
sys.path.append('.')
from src.preprocessing import load_data, preprocess_features, split_data, apply_smote

def train_model(X_train: np.ndarray, y_train: np.ndarray, use_smote: bool = False, tune_hyperparams: bool = False) -> RandomForestClassifier:
    """Train RandomForest with optional SMOTE and hyperparameter tuning."""
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    if tune_hyperparams:
        print("Tuning hyperparameters with GridSearchCV...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"Best params: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        print("Training RandomForest with class_weight='balanced_subsample'...")
        model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC: {roc_auc:.4f}")

    # Precision-Recall AUC (more informative for imbalanced)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC: {pr_auc:.4f}")

def main():
    DATA_PATH = "data/creditcard.csv"
    MODEL_PATH = "models/random_forest.pkl"

    # Load and preprocess
    df = load_data(DATA_PATH)
    X, y, _ = preprocess_features(df, scaler_path=None)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train
    model = train_model(X_train, y_train, use_smote=False, tune_hyperparams=False)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
