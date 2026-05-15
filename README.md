# Fraud Detection Live Monitoring Dashboard

## Overview

This project is a simulated production monitoring system for fraud detection models. It demonstrates how a machine learning model can be monitored over time for:

* Performance degradation
* Data drift
* Feature distribution shifts
* Feature importance-aware risk analysis
* Fraud prediction quality
* Retraining recommendations

The dashboard is built with Streamlit and uses the Kaggle Credit Card Fraud Detection dataset.

---

# Project Features

## Baseline Model Training

* Trains an XGBoost fraud classifier with dynamic class imbalance handling via `scale_pos_weight`
* Uses the Kaggle fraud dataset as the baseline reference
* Optimizes the prediction threshold to maximise recall subject to a minimum precision constraint
* Logs:

  * Precision
  * Recall
  * F1-score
  * ROC-AUC
  * PR-AUC
  * Optimized prediction threshold

---

## Data Drift Detection

The system compares incoming production batches against the training distribution using:

### Kolmogorov-Smirnov (KS) Test

Used to detect statistically significant distribution changes.

### Population Stability Index (PSI)

Used to quantify how much a feature distribution has shifted.

PSI thresholds:

| PSI Value   | Drift Level |
| ----------- | ----------- |
| < 0.10      | Low         |
| 0.10 – 0.25 | Moderate    |
| > 0.25      | High        |

### Drift × Importance Risk Analysis

Drift results are cross-referenced with XGBoost feature importances to produce a composite risk score:

```
Risk Score = PSI × Feature Importance
```

A feature is flagged as **High Risk** if it is both drifting and among the top 10 most important features. This means drift in a low-importance feature is treated differently from the same drift in a feature the model relies on heavily.

This analysis is used to:
* Generate more targeted monitoring alerts
* Guide feature selection during retraining — low-importance high-drift features can be dropped to reduce noise

---

## Performance Monitoring

Each incoming production batch is evaluated using the currently active model.

The dashboard tracks:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion Matrix
* Threshold trade-off analysis across the full 0.01–0.50 range
* Fraud predictions

---

## Fraud Detection Table

The dashboard includes expandable fraud prediction tables:

* Detected fraud transactions
* Fraud probabilities
* True positives
* False positives
* Missed fraud cases
* Full scored production batches

---

## Monitoring & Retraining Workflow

The monitoring system:

1. Scores incoming production batches
2. Detects drift and degradation
3. Cross-references drifted features with feature importance
4. Raises monitoring alerts (including importance-aware alerts)
5. Recommends retraining when thresholds are exceeded
6. Allows manual retraining through the dashboard
7. Filters unstable low-importance features from the retrained model

Retraining is intentionally separated from prediction monitoring to simulate realistic production workflows.

### Retraining Strategy

* Recent batches are upweighted by a factor of 4 to reflect current data patterns
* Historical data is capped at 50,000 rows to manage compute, with all fraud rows preserved
* Features that are high-drift but low-importance are dropped before retraining to reduce noise
* A validation model is trained on a holdout split before the final model is promoted

---

# Dashboard Architecture

```text
Incoming Batch
        ↓
Active Production Model (XGBoost)
        ↓
Fraud Predictions
        ↓
Performance Evaluation
        ↓
Drift Detection (KS + PSI)
        ↓
Drift × Importance Risk Analysis
        ↓
Alert Generation
        ↓
Retraining Recommendation
        ↓
Optional Model Retraining (with feature filtering)
```

---

# Project Structure

```text
project/
│
├── app.py
├── creditcard.csv
├── drift_1.csv
├── drift_2.csv
├── drift_3.csv
├── drift_4.csv
├── drift_5.csv
│
├── monitoring_state/
│   ├── current_model.pkl
│   ├── metadata.json
│   ├── monitoring_log.csv
│   ├── model_registry.csv
│   ├── training_pool.csv
│   └── baseline_metrics.json
│
└── README.md
```

---

# Monitoring State Folder

The `monitoring_state` folder stores the persistent application state.

| File                    | Purpose                                           |
| ----------------------- | ------------------------------------------------- |
| `current_model.pkl`     | Currently active fraud detection model            |
| `metadata.json`         | Tracks active model version and processed batches |
| `monitoring_log.csv`    | Monitoring history over time                      |
| `model_registry.csv`    | Registry of promoted model versions               |
| `training_pool.csv`     | Accumulated training data                         |
| `baseline_metrics.json` | Baseline benchmark metrics                        |

---

# Requirements

## Python Version

Python **3.8 or higher** is required.

To check your version:

```bash
python --version
```

## Internet Connection

The dashboard loads **IBM Plex Sans** and **IBM Plex Mono** fonts from Google Fonts at startup. An internet connection is required for the fonts to render correctly. If you are offline, the dashboard will still function but will fall back to the browser's default monospace font.

---

# Installation

## 1. Clone or Download the Project

Place all project files into a single folder.

---

## 2. Create a Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# Install Dependencies

Run:

```bash
pip install streamlit pandas numpy matplotlib scipy scikit-learn xgboost joblib
```

---

# Required Files

Ensure the following files are present in the project folder:

```text
creditcard.csv
drift_1.csv
drift_2.csv
drift_3.csv
drift_4.csv
drift_5.csv
app.py
```

## About the Drift CSV Files

The `drift_1.csv` through `drift_5.csv` files are **simulated production batches** — slices of transaction data used to test the monitoring system. Each file represents a batch of incoming transactions that the active model will score and evaluate for drift.

These files should be provided alongside the project. If you do not have them, you can generate your own by sampling from `creditcard.csv`:

```python
import pandas as pd

df = pd.read_csv("creditcard.csv")
batch = df.sample(n=5000, random_state=42)
batch.to_csv("drift_1.csv", index=False)
```

Each batch must contain the same feature columns as `creditcard.csv` and must include the `Class` column.

---

# Running the Dashboard

Open a terminal inside the project folder.

Run:

```bash
streamlit run app.py
```

If `streamlit` is not recognized:

```bash
python -m streamlit run app.py
```

---

# Accessing the Dashboard

After launching Streamlit, open:

```text
http://localhost:8501
```

---

# Using the Dashboard

## Option 1 — Simulated Production Batches

Use the built-in drift batches:

* `drift_1.csv`
* `drift_2.csv`
* `drift_3.csv`
* `drift_4.csv`
* `drift_5.csv`

Workflow:

1. Select a simulated production batch
2. Click `Process Batch`
3. Review:

   * Metrics
   * Drift and risk analysis
   * Fraud predictions
   * Alerts
4. Retrain the model if recommended

---

## Option 2 — Upload External Production Data

You can upload your own CSV production batch.

Requirements:

* Must contain the same feature columns as the training dataset
* Must include the `Class` column

Workflow:

1. Upload a CSV file
2. Select `Uploaded CSV`
3. Click `Process Batch`

---

# Monitoring Thresholds

The dashboard triggers alerts when:

## Performance Degradation

```python
Recall < 80% of baseline
F1 < 80% of baseline
PR-AUC < 90% of baseline
```

## Drift Conditions

```python
3 or more high-drift features
5 or more moderate-drift features
```

## Importance-Aware Drift Conditions

```python
2 or more high-importance features drifting  →  Critical alert
1 high-importance feature drifting           →  Warning alert
```

---

# Example Automated Alert Logic

```python
if current_metrics['Recall'] < baseline_metrics['Recall'] * 0.8:
    alert = 'Critical recall degradation'

if current_metrics['F1'] < baseline_metrics['F1'] * 0.8:
    alert = 'Critical F1 degradation'

if current_metrics['PR-AUC'] < baseline_metrics['PR-AUC'] * 0.9:
    alert = 'Warning PR-AUC degradation'

if high_drift_features >= 3:
    alert = 'Critical feature drift'

if moderate_drift_features >= 5:
    alert = 'Warning feature drift'

if high_importance_drifting_features >= 2:
    alert = 'Critical: high-importance features are drifting'

if high_importance_drifting_features == 1:
    alert = 'Warning: high-importance feature drifting'
```

---

# Technologies Used

| Technology   | Purpose                              |
| ------------ | ------------------------------------ |
| Python       | Core programming language            |
| Streamlit    | Interactive dashboard                |
| XGBoost      | Fraud detection classifier           |
| Scikit-learn | Model evaluation and data splitting  |
| Pandas       | Data manipulation                    |
| NumPy        | Numerical operations                 |
| SciPy        | Statistical drift tests (KS test)    |
| Matplotlib   | Visualizations                       |
| Joblib       | Model persistence                    |

---

# Future Improvements

Potential future extensions:

* Real-time streaming data
* Online learning models
* SHAP explainability
* Automated threshold recalibration per batch
* Drift dashboards per feature group
* Automated retraining pipelines
* Cloud deployment
* Database-backed monitoring logs
* CI/CD integration
* MLflow integration

---

# Dataset

This project uses the Kaggle Credit Card Fraud Detection dataset:

[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

# Authors

* Li Ming Huang
* Anna Lappo
* Ema Růžičková

---

# Troubleshooting

## Corrupted or Broken State

If the dashboard throws errors on startup or behaves unexpectedly after a failed retraining, the `monitoring_state/` folder may be in a broken state. To fix this, delete the folder entirely and restart the dashboard — it will rebuild from scratch automatically:

```bash
# Windows
rmdir /s /q monitoring_state

# Mac/Linux
rm -rf monitoring_state
```

Then rerun:

```bash
streamlit run app.py
```

Alternatively, use the **↺ Reset** button in the sidebar, which does the same thing from inside the dashboard.

## XGBoost Compatibility

This project uses `eval_metric="logloss"` passed directly to the `XGBClassifier` constructor. This requires **XGBoost 1.6 or higher**. If you see a warning about `eval_metric` being ignored, upgrade XGBoost:

```bash
pip install --upgrade xgboost
```

---

# Notes

* The dashboard simulates production monitoring workflows.
* Retraining is intentionally manual after alerts are triggered.
* The monitoring state is persisted locally in the `monitoring_state` folder.
* Resetting the simulation clears the monitoring state and restores the baseline model.
* The model was changed from Random Forest to XGBoost to better handle class imbalance via `scale_pos_weight`.
