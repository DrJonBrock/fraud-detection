# Fraud Detection: Research Summary (2025-03-29)

## Dataset
**Primary:** Kaggle Credit Card Fraud Detection Dataset
- Source: ULB Machine Learning Group (University of Liege)
- Size: 284,807 transactions (492 frauds, 0.172% fraud rate)
- Features: 31 total (Time, V1-V28 (PCA), Amount)
- Class: 0 = legitimate, 1 = fraudulent
- Highly imbalanced → need resampling techniques

**Download:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Alternative:** PaySim (synthetic) - 6 million transactions, more balanced (1%)

## Best Performing Approaches (2024 research)

1. **Ensemble Models**
   - Stacking ensemble (Random Forest + XGBoost + Logistic Regression)
   - Voting classifiers
   - Outperforms individual models on precision/recall tradeoff
   - Paper: MDPI 2024 - ensemble achieved highest F1-score

2. **Deep Learning**
   - Convolutional Neural Networks (CNNs) on transaction sequences
   - Autoencoders for anomaly detection (unsupervised)
   - Transformer models (newer) for sequence modeling
   - Challenge: require large data, careful regularization

3. **Hybrid Approaches**
   - Combine supervised (binary classification) + unsupervised (anomaly detection)
   - Use oversampling (SMOTE) + ensemble
   - Feature engineering: derive "hours" from Time, normalize Amount, bin V features

## Recommended Starting Point (MVP)

**Simple, fast, good baseline:**
- Algorithm: Random Forest or XGBoost
- Handle imbalance: `class_weight='balanced'` or SMOTE
- Metrics: Focus on **Recall** (catch frauds) while maintaining reasonable **Precision** (avoid false alarms)
- Evaluation: Confusion matrix, ROC-AUC, PR-AUC (better for imbalanced)

**Implementation steps:**
1. Load data (pandas)
2. Preprocess: Scale Amount and Time (StandardScaler), keep PCA features as-is
3. Split: Train (70%), Test (30%) - stratified
4. Handle imbalance: Optional SMOTE or class weights
5. Train Random Forest (n_estimators=100, max_depth=10)
6. Evaluate: classification_report, confusion matrix
7. Tune thresholds to maximize F1 or business-specific cost function

**Baseline expectations:**
- Without balancing: high recall but low precision? Actually baseline models get ~0.99+ accuracy but terrible recall due to imbalance.
- With balancing: can achieve >80% recall with acceptable precision (~0.95 PR-AUC achievable)
- Random Forest with class_weight='balanced_subsample' often reaches 0.85+ recall at 0.5 threshold

## Project Architecture

```
fraud-detection/
├── data/
│   └── creditcard.csv (downloaded from Kaggle)
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   └── (saved models)
├── requirements.txt
├── README.md
└── .gitignore
```

## Tech Stack
- Python 3.9+
- pandas, numpy
- scikit-learn
- imbalanced-learn (for SMOTE)
- xgboost or lightgbm
- matplotlib/seaborn for plots
- (optional) PyTorch/TensorFlow for deep learning later

## Metrics to Track
- **Precision**: Of predicted frauds, how many are correct?
- **Recall**: What % of actual frauds caught?
- **F1-score**: Harmonic mean
- **AUC-ROC**: Overall separability
- **Confusion matrix**: Business impact (cost of fraud vs false positive customer friction)

## Deployment Options
1. **FastAPI** REST endpoint → host on Vercel (serverless) or Railway/Render
2. **Batch scoring** (CSV upload)
3. **Streaming** (Kafka/Redis) - later stage

---

**Next:** Set up Python environment, run EDA, train baseline Random Forest, document results.
