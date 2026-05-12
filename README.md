# Fraud Detection Live Monitoring Dashboard

## Overview

This project is a simulated production monitoring system for fraud detection models. It demonstrates how a machine learning model can be monitored over time for:

* Performance degradation
* Data drift
* Feature distribution shifts
* Fraud prediction quality
* Retraining recommendations

The dashboard is built with Streamlit and uses the Kaggle Credit Card Fraud Detection dataset.

---

# Project Features

## Baseline Model Training

* Trains a Random Forest fraud classifier
* Uses the Kaggle fraud dataset as the baseline reference
* Logs:

  * Precision
  * Recall
  * F1-score
  * ROC-AUC
  * PR-AUC

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
3. Raises monitoring alerts
4. Recommends retraining when thresholds are exceeded
5. Allows manual retraining through the dashboard

Retraining is intentionally separated from prediction monitoring to simulate realistic production workflows.

---

# Dashboard Architecture

```text
Incoming Batch
        ↓
Active Production Model
        ↓
Fraud Predictions
        ↓
Performance Evaluation
        ↓
Drift Detection (KS + PSI)
        ↓
Alert Generation
        ↓
Retraining Recommendation
        ↓
Optional Model Retraining
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
pip install streamlit pandas numpy matplotlib scipy scikit-learn joblib
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
2. Click `Process Next Batch`
3. Review:

   * Metrics
   * Drift
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
3. Click `Process Next Batch`

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

---

# Example Automated Drift Check

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
```

---

# Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core programming language |
| Streamlit    | Interactive dashboard     |
| Scikit-learn | Machine learning          |
| Pandas       | Data manipulation         |
| NumPy        | Numerical operations      |
| SciPy        | Statistical drift tests   |
| Matplotlib   | Visualizations            |
| Joblib       | Model persistence         |

---

# Future Improvements

Potential future extensions:

* Real-time streaming data
* Online learning models
* SHAP explainability
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

# Notes

* The dashboard simulates production monitoring workflows.
* Retraining is intentionally manual after alerts are triggered.
* The monitoring state is persisted locally in the `monitoring_state` folder.
* Resetting the simulation clears the monitoring state and restores the baseline model.
