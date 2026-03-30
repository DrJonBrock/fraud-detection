# Project: Fraud Detection Model
**Goal:** Build a credit card fraud detection system using machine learning
**Started:** 2025-03-29 (code implementation)
**Priority:** Medium (parallel to AusCarValuation)

## Status (2026-03-31 06:30 AWST)
- [x] Research existing datasets (Kaggle Credit Card Fraud Detection)
- [x] Choose ML approach (Random Forest with class balancing)
- [x] Set up project structure
- [x] Implement preprocessing pipeline (src/preprocessing.py)
- [x] Implement model training (src/train.py)
- [x] Implement evaluation metrics (classification report, confusion matrix, ROC-AUC, PR-AUC)
- [x] Implement prediction script (src/predict.py)
- [x] Define requirements.txt with all needed packages
- [x] Create Python virtual environment; installed all dependencies
- [x] Download dataset (creditcard.csv, 144MB) via Kaggle API
- [x] Train baseline Random Forest
- [x] Train SMOTE Random Forest (address class imbalance)
- [x] Evaluate models and select best
- [ ] Create FastAPI endpoint (next)
- [ ] Deploy API (Vercel/Railway)
- [ ] Document final model performance and usage

### Model Performance (Test Set, 85,443 samples)
| Model | Recall (Fraud) | Precision (Fraud) | PR-AUC | ROC-AUC |
|-------|----------------|-------------------|--------|---------|
| Baseline RF (class_weight='balanced_subsample') | 70.27% | 97.20% | 0.8133 | 0.9461 |
| SMOTE RF | 79.73% | 80.82% | **0.8207** | 0.9458 |

**Selected:** SMOTE Random Forest (improves recall by 9.5pp with slightly better PR-AUC)

## Notes
- This is a separate machine learning project
- Could be integrated with the web platform later as a separate product
- Need to handle sensitive data carefully (PCI compliance considerations)
- Potential for monetization as a SaaS for small businesses
- Environment: Python 3.12.3 with venv at `venv/` (all packages installed)

## Resources
- Primary dataset: Kaggle Credit Card Fraud Detection (ULB)
  - Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  - 284,807 transactions (492 frauds, 0.172% fraud rate)
  - Features: Time, V1-V28 (PCA), Amount
- Alternative: PaySim (synthetic) - 6 million transactions, more balanced (1%)

## Next Steps (immediate)
1. Build FastAPI endpoint (`src/api.py`) exposing predict function
2. Add model persistence metadata (scaler, feature order)
3. Write inference unit tests
4. Deploy to Vercel (serverless) or Railway (container)
5. Document API usage, request/response schema
6. Add monitoring for drift (future)

## Model Files
- `models/baseline_random_forest.pkl` (5.4 MB)
- `models/random_forest.pkl` (5.4 MB)
- `models/smote_random_forest.pkl` (24.7 MB) ← **selected**
