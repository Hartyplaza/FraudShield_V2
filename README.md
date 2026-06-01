# FraudShield V2 — Transaction Log Fraud Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=flat-square&logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi)
![PR-AUC](https://img.shields.io/badge/PR--AUC-0.9287-brightgreen?style=flat-square)
![Recall](https://img.shields.io/badge/Recall-0.9811-brightgreen?style=flat-square)

**End-to-end fraud detection on raw multi-format transaction logs**

[Live Demo](https://huggingface.co/spaces/Demerchanthart/FraudShield_V2) · [Portfolio](https://hartyplaza.github.io) · [LinkedIn](https://linkedin.com/in/hart-ofigwe)

</div>

---

## Overview

FraudShield V2 is a production-grade fraud detection system built on 10,000 raw transaction logs spanning 7 distinct log formats. The system handles everything from raw log ingestion to real-time API serving — demonstrating a complete end-to-end ML pipeline with a focus on production readiness and business impact.

The core challenge is that real-world transaction logs are messy: multiple formats, missing values, null rows, and malformed entries. FraudShield V2 parses all of this into a clean feature matrix, trains a LightGBM model with ADASYN oversampling, and serves predictions via FastAPI with per-transaction SHAP explanations.

---

## Results

| Metric | Value |
|--------|-------|
| **PR-AUC** | **0.9287** |
| **ROC-AUC** | **0.9972** |
| **F1** | **0.8125** |
| **Recall** | **0.9811** (52 / 53 fraud cases caught) |
| **Precision** | **0.6933** |
| **False Positive Rate** | **0.0153** |
| **Decision Threshold** | **0.65** |
| **Net Benefit** | **£221,727** on test set |

---

## Pipeline

```
Raw Logs (10,000 rows, 7 formats)
        │
        ▼
01_parsing.ipynb              Multi-format regex parser → parsed_logs.csv
        │
        ▼
02_eda.ipynb                  Exploratory analysis, fraud signal identification
        │
        ▼
03_label_simulation.ipynb     Deterministic rule-based fraud labels
        │
        ▼
04_feature_engineering.ipynb  28 engineered features
        │
        ▼
05_modelling.ipynb            4 models, RandomizedSearchCV, LightGBM wins
        │
        ▼
06_evaluation.ipynb           PR/ROC curves, SHAP analysis, business impact
        │
        ▼
FastAPI + Streamlit           Real-time scoring with SHAP explanations
```

---

## Key Design Decisions

### Multi-Format Log Parser
The raw dataset contains 7 distinct log schemas mixed together. A regex dispatch parser handles all 7, normalises timestamps to ISO format, and converts `None` placeholders to proper NaN. 77.7% of 10,000 rows parsed successfully — 22.3% were genuine null or malformed rows.

### Deterministic Label Simulation
Three rule combinations define fraud — each requires multiple signals to fire simultaneously:

| Rule | Signals | Fraud Narrative |
|------|---------|-----------------|
| A | Night + High amount + Risky type | Late-night large cashout or withdrawal |
| B | Night + High amount + Missing location | Large transaction at night with no traceable origin |
| C | High amount + Device switch + Missing location | Large transaction on a new device — account takeover |

No randomness in label assignment. Identical features always map to the same label, eliminating label noise and giving the model a clean learnable boundary.

### SMOTE in CV, ADASYN on Final Refit
SMOTE is used inside cross-validation folds to prevent data leakage into validation. ADASYN is applied during the final refit — it generates more synthetic fraud samples near the decision boundary where the classifier struggles most.

### Threshold Tuning
Default 0.5 threshold is wrong for imbalanced data. Swept 0.01–0.95 on the validation set. Optimal threshold: **0.65**.

---

## Feature Engineering

28 features across 6 groups:

| Group | Features |
|-------|----------|
| **Temporal** | hour, day_of_week, is_night, is_weekend, is_business_hours |
| **Amount** | amount, amount_z_score, is_high_amount, amount_vs_user_max, user_mean/std/max_amount |
| **Velocity** | txn_count_1d/7d/30d, txn_sum_1d/7d/30d |
| **Behavioural** | device_changed, location_changed, time_since_last_txn_hrs |
| **Missing flags** | location_missing, device_missing |
| **Categorical** | txn_type_encoded, location_encoded, device_encoded, log_format_encoded, currency_encoded |

---

## Model Comparison

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|-------|-----------|-------------|--------------|
| **LightGBM** | **0.9038** | **0.9287** | **0.9972** |
| XGBoost | 0.8476 | 0.8567 | 0.9933 |
| LogisticRegression | 0.0957 | 0.6294 | 0.9779 |
| RandomForest | 0.0964 | 0.5659 | 0.9818 |

---

## Project Structure

```
FraudShield_V2/
├── notebooks/
│   ├── 01_parsing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_label_simulation.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modelling.ipynb
│   └── 06_evaluation.ipynb
├── src/
│   ├── ingestion/
│   │   ├── parser.py
│   │   └── cleaner.py
│   ├── features/
│   │   ├── engineer.py
│   │   └── label_simulator.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── threshold_tuner.py
│   ├── explainability/
│   │   └── shap_analysis.py
│   └── api/
│       ├── main.py
│       └── schema.py
├── configs/
│   ├── pipeline.yaml
│   └── logging.yaml
├── scripts/
│   ├── run_ingestion.py
│   ├── run_features.py
│   ├── run_training.py
│   └── run_evaluation.py
├── tests/
│   ├── test_parser.py
│   ├── test_features.py
│   └── test_metrics.py
├── app.py
├── requirements.txt
├── setup.py
└── Makefile
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/Hartyplaza/FraudShield_V2
cd FraudShield_V2

# Install
pip install -r requirements.txt

# Run pipeline
make ingest
make features
make train
make evaluate

# Launch API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Launch dashboard
streamlit run app.py
```

---

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Score a transaction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 2,
    "day_of_week": 5,
    "is_night": 1,
    "is_weekend": 1,
    "is_business_hours": 0,
    "amount": 4800.0,
    "amount_z_score": 3.2,
    "is_high_amount": 1,
    "amount_vs_user_max": 0.96,
    "user_mean_amount": 2400.0,
    "user_std_amount": 1200.0,
    "user_max_amount": 4997.0,
    "txn_count_1d": 1,
    "txn_sum_1d": 4800.0,
    "txn_count_7d": 5,
    "txn_sum_7d": 12000.0,
    "txn_count_30d": 18,
    "txn_sum_30d": 43000.0,
    "device_changed": 1,
    "location_changed": 0,
    "time_since_last_txn_hrs": 0.5,
    "location_missing": 1,
    "device_missing": 0,
    "txn_type_encoded": 0,
    "location_encoded": 3,
    "device_encoded": 1,
    "log_format_encoded": 2,
    "currency_encoded": 1
  }'
```

**Response:**
```json
{
  "fraud_probability": 0.9990,
  "is_fraud": true,
  "risk_level": "critical",
  "threshold_used": 0.65,
  "top_features": {
    "is_high_amount": 3.2777,
    "amount": 1.7060,
    "location_missing": 1.1596,
    "amount_z_score": 0.9590,
    "hour": 0.3113
  }
}
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| **Data** | pandas, numpy |
| **ML** | LightGBM, XGBoost, scikit-learn |
| **Imbalance** | imbalanced-learn (SMOTE, ADASYN) |
| **Explainability** | SHAP |
| **API** | FastAPI, uvicorn, Pydantic |
| **Dashboard** | Streamlit, Plotly |
| **Tracking** | MLflow |
| **Testing** | pytest |

---

## Author

**Ofigwe Hart** — Data Scientist / ML Engineer

[![Portfolio](https://img.shields.io/badge/Portfolio-hartyplaza.github.io-blue?style=flat-square)](https://hartyplaza.github.io)
[![GitHub](https://img.shields.io/badge/GitHub-Hartyplaza-181717?style=flat-square&logo=github)](https://github.com/Hartyplaza)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-hart--ofigwe-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/hart-ofigwe)
