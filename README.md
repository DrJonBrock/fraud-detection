# Fraud Detection Model

Credit card fraud detection using machine learning. Based on the Kaggle dataset (ULB Machine Learning Group).

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in data/
```

## Usage

**Exploratory Data Analysis:**
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

**Train model:**
```bash
python src/train.py
```

**Evaluate:**
```bash
python src/evaluate.py --model models/random_forest.pkl
```

**Predict on new data:**
```bash
python src/predict.py --input new_transactions.csv --output predictions.csv
```

## Project Structure
- `data/`: Dataset (not in git)
- `notebooks/`: Jupyter notebooks for exploration
- `src/`: Python modules for preprocessing, training, prediction
- `models/`: Saved model files
- `requirements.txt`: Python dependencies

## Baseline Goal
Achieve >85% recall with <5% false positive rate on the Kaggle test set.

## Research
See `RESEARCH.md` for methodology, algorithm choices, and metrics.

---

**Note:** This is an educational/research project. For production use, consider compliance (PCI DSS), real-time requirements, and integration with payment processors.
