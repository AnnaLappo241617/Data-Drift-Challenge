import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Fraud Live Monitoring Dashboard", layout="wide")

BASE_DATA_FILE = "creditcard.csv"
DRIFT_FILES = [f"drift_{i}.csv" for i in range(1, 6)]
STATE_DIR = "monitoring_state"
MODEL_PATH = os.path.join(STATE_DIR, "current_model.pkl")
META_PATH = os.path.join(STATE_DIR, "metadata.json")
MONITOR_LOG_PATH = os.path.join(STATE_DIR, "monitoring_log.csv")
MODEL_REGISTRY_PATH = os.path.join(STATE_DIR, "model_registry.csv")
TRAINING_POOL_PATH = os.path.join(STATE_DIR, "training_pool.csv")
BASELINE_METRICS_PATH = os.path.join(STATE_DIR, "baseline_metrics.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_THRESHOLD = 0.05
THRESHOLD_OPTIMIZATION_METRIC = "recall_with_precision"
MIN_PRECISION_FOR_THRESHOLD = 0.10
ALPHA = 0.
MAX_HISTORICAL_RETRAIN_ROWS = 50000
RECENT_BATCH_WEIGHT = 4

os.makedirs(STATE_DIR, exist_ok=True)

# =========================================================
# HELPERS
# =========================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Hour" not in df.columns and "Time" in df.columns:
        df["Hour"] = (df["Time"] // 3600) % 24
    if "day" in df.columns:
        df = df.drop(columns=["day"])
    return df


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_psi(expected, actual, buckets=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    expected = expected[~pd.isna(expected)]
    actual = actual[~pd.isna(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    breakpoints = np.percentile(
        expected,
        np.arange(0, 100 + 100 / buckets, 100 / buckets)
    )

    # If a feature has almost no variation, percentile bins may collapse.
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-6, actual_perc)

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(psi)


def detect_feature_drift(train_df, prod_df, feature_cols, alpha=0.05):
    rows = []
    for col in feature_cols:
        train_vals = train_df[col].dropna()
        prod_vals = prod_df[col].dropna()

        if len(train_vals) == 0 or len(prod_vals) == 0:
            ks_stat, p_value, psi_value = np.nan, np.nan, np.nan
            psi_level = "Unavailable"
            drift_detected = False
        else:
            ks_stat, p_value = ks_2samp(train_vals, prod_vals)
            psi_value = calculate_psi(train_vals, prod_vals)

            if psi_value < 0.1:
                psi_level = "Low"
            elif psi_value < 0.25:
                psi_level = "Moderate"
            else:
                psi_level = "High"

            drift_detected = (p_value < alpha) or (psi_value >= 0.1)

        rows.append({
            "Feature": col,
            "KS Statistic": float(ks_stat) if not pd.isna(ks_stat) else np.nan,
            "p-value": float(p_value) if not pd.isna(p_value) else np.nan,
            "PSI": float(psi_value) if not pd.isna(psi_value) else np.nan,
            "PSI Level": psi_level,
            "Drift Detected": bool(drift_detected),
        })

    return pd.DataFrame(rows).sort_values(
        by=["Drift Detected", "PSI"],
        ascending=[False, False]
    ).reset_index(drop=True)


def evaluate_model(model, X, y, threshold=0.5):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "Precision": float(precision_score(y, y_pred, zero_division=0)),
        "Recall": float(recall_score(y, y_pred, zero_division=0)),
        "F1": float(f1_score(y, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else np.nan,
        "PR-AUC": float(average_precision_score(y, y_prob)) if len(np.unique(y)) > 1 else np.nan,
        "Predicted Fraud Count": int(y_pred.sum()),
        "Actual Fraud Count": int(y.sum()),
        "Confusion Matrix": confusion_matrix(y, y_pred).tolist(),
    }

def find_best_threshold(y_true, y_prob, metric="recall_with_precision", min_precision=0.10):
    thresholds = np.arange(0.01, 0.51, 0.01)

    best_threshold = DEFAULT_THRESHOLD
    best_score = -1

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if metric == "f1":
            score = f1
        elif metric == "recall_with_precision":
            if precision < min_precision:
                continue
            score = recall
        else:
            raise ValueError("metric must be 'f1' or 'recall_with_precision'")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold), float(best_score)

def train_random_forest(train_df: pd.DataFrame):
    train_df = prepare_features(train_df)

    X = train_df.drop(columns=["Class"])
    y = train_df["Class"]

    scale_pos_weight = (
        len(y[y == 0]) / max(len(y[y == 1]), 1)
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X, y)

    return model, X.columns.tolist()


def compute_baseline_metrics(base_df: pd.DataFrame):
    base_df = prepare_features(base_df)
    X = base_df.drop(columns=["Class"])
    y = base_df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scale_pos_weight = (
        len(y_train[y_train == 0]) /
        max(len(y_train[y_train == 1]), 1)
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    best_threshold, best_threshold_score = find_best_threshold(
        y_test,
        y_prob,
        metric=THRESHOLD_OPTIMIZATION_METRIC,
        min_precision=MIN_PRECISION_FOR_THRESHOLD,
    )

    metrics = evaluate_model(model, X_test, y_test, threshold=best_threshold)
    metrics["Best Threshold"] = best_threshold
    metrics["Best Threshold Score"] = best_threshold_score
    metrics["Threshold Optimization Metric"] = THRESHOLD_OPTIMIZATION_METRIC

    return metrics


def degradation_check(current_metrics, baseline_metrics, drift_df):
    alerts = []

    if current_metrics["Recall"] < baseline_metrics["Recall"] * 0.8:
        alerts.append("Critical: Recall dropped by more than 20% from baseline")

    if current_metrics["F1"] < baseline_metrics["F1"] * 0.8:
        alerts.append("Critical: F1 dropped by more than 20% from baseline")

    if not pd.isna(current_metrics["PR-AUC"]) and not pd.isna(baseline_metrics["PR-AUC"]):
        if current_metrics["PR-AUC"] < baseline_metrics["PR-AUC"] * 0.9:
            alerts.append("Warning: PR-AUC dropped by more than 10% from baseline")

    high_drift = int((drift_df["PSI Level"] == "High").sum())
    moderate_drift = int((drift_df["PSI Level"] == "Moderate").sum())

    if high_drift >= 3:
        alerts.append(f"Critical: {high_drift} features show high drift")
    if moderate_drift >= 5:
        alerts.append(f"Warning: {moderate_drift} features show moderate drift")

    retrain_needed = len(alerts) > 0
    return alerts, retrain_needed


def validate_batch_columns(batch_df: pd.DataFrame, feature_cols: list):
    required_cols = ["Class"] + feature_cols
    missing = [c for c in required_cols if c not in batch_df.columns]
    if missing:
        raise ValueError(f"Batch is missing required columns: {missing}")


def append_monitor_log(log_row: dict):
    monitor_log = pd.read_csv(MONITOR_LOG_PATH)
    monitor_log = pd.concat([monitor_log, pd.DataFrame([log_row])], ignore_index=True)
    monitor_log.to_csv(MONITOR_LOG_PATH, index=False)


def register_new_model(new_model_version, model_name, trained_on, activated_after_batch, validation_metrics):
    model_registry = pd.read_csv(MODEL_REGISTRY_PATH)
    model_registry = pd.concat([
        model_registry,
        pd.DataFrame([{
            "model_version": new_model_version,
            "model_name": model_name,
            "trained_on": trained_on,
            "activated_after_batch": activated_after_batch,
            "precision": validation_metrics["Precision"],
            "recall": validation_metrics["Recall"],
            "f1": validation_metrics["F1"],
            "roc_auc": validation_metrics["ROC-AUC"],
            "pr_auc": validation_metrics["PR-AUC"],
        }])
    ], ignore_index=True)
    model_registry.to_csv(MODEL_REGISTRY_PATH, index=False)


def build_retraining_dataset(training_pool: pd.DataFrame, batch_df: pd.DataFrame) -> pd.DataFrame:
    training_pool = prepare_features(training_pool)
    batch_df = prepare_features(batch_df)

    if len(training_pool) > MAX_HISTORICAL_RETRAIN_ROWS:
        fraud_history = training_pool[training_pool["Class"] == 1]
        non_fraud_history = training_pool[training_pool["Class"] == 0]

        remaining_slots = max(MAX_HISTORICAL_RETRAIN_ROWS - len(fraud_history), 0)
        sampled_non_fraud = non_fraud_history.sample(
            n=min(remaining_slots, len(non_fraud_history)),
            random_state=RANDOM_STATE,
        )

        historical_sample = pd.concat([fraud_history, sampled_non_fraud], ignore_index=True)
    else:
        historical_sample = training_pool.copy()

    weighted_recent_batch = pd.concat(
        [batch_df] * RECENT_BATCH_WEIGHT,
        ignore_index=True,
    )

    return pd.concat([historical_sample, weighted_recent_batch], ignore_index=True)


def retrain_and_promote_model(metadata, training_pool, batch_df, batch_name, threshold):
    updated_training_pool = build_retraining_dataset(training_pool, batch_df)
    updated_training_pool = prepare_features(updated_training_pool)

    X = updated_training_pool.drop(columns=["Class"])
    y = updated_training_pool["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scale_pos_weight = (
    len(y_train[y_train == 0]) /
    max(len(y_train[y_train == 1]), 1)
    )

    new_model = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    new_model.fit(X_train, y_train)
    validation_metrics = evaluate_model(new_model, X_test, y_test, threshold=threshold)

    final_model, new_feature_cols = train_random_forest(updated_training_pool)

    joblib.dump(final_model, MODEL_PATH)
    updated_training_pool.to_csv(TRAINING_POOL_PATH, index=False)

    metadata["current_model_version"] += 1
    metadata["feature_cols"] = new_feature_cols
    metadata["active_model_name"] = f"Model v{metadata['current_model_version']}"
    save_json(META_PATH, metadata)

    new_model_version = metadata["current_model_version"]

    register_new_model(
        new_model_version=new_model_version,
        model_name=metadata["active_model_name"],
        trained_on=f"weighted_recent_retraining_through_{batch_name}",
        activated_after_batch=batch_name,
        validation_metrics=validation_metrics,
    )

    return new_model_version, validation_metrics


def ensure_state_initialized():
    required_paths = [
        MODEL_PATH,
        META_PATH,
        MONITOR_LOG_PATH,
        MODEL_REGISTRY_PATH,
        TRAINING_POOL_PATH,
        BASELINE_METRICS_PATH,
    ]
    if all(os.path.exists(p) for p in required_paths):
        return

    if not os.path.exists(BASE_DATA_FILE):
        raise FileNotFoundError(f"Missing {BASE_DATA_FILE}")

    base_df = load_csv(BASE_DATA_FILE)
    base_df = prepare_features(base_df)

    model, feature_cols = train_random_forest(base_df)
    baseline_metrics = compute_baseline_metrics(base_df)

    joblib.dump(model, MODEL_PATH)
    base_df.to_csv(TRAINING_POOL_PATH, index=False)
    save_json(BASELINE_METRICS_PATH, baseline_metrics)

    metadata = {
        "current_model_version": 1,
        "feature_cols": feature_cols,
        "processed_batches": [],
        "active_model_name": "Model v1",
    }
    save_json(META_PATH, metadata)

    pd.DataFrame([
        {
            "model_version": 1,
            "model_name": "Model v1",
            "trained_on": BASE_DATA_FILE,
            "activated_after_batch": "initial_training",
            "precision": baseline_metrics["Precision"],
            "recall": baseline_metrics["Recall"],
            "f1": baseline_metrics["F1"],
            "roc_auc": baseline_metrics["ROC-AUC"],
            "pr_auc": baseline_metrics["PR-AUC"],
        }
    ]).to_csv(MODEL_REGISTRY_PATH, index=False)

    pd.DataFrame(columns=[
        "batch_name",
        "model_version_used",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "delta_precision",
        "delta_recall",
        "delta_f1",
        "delta_roc_auc",
        "delta_pr_auc",
        "drifted_features",
        "high_drift_features",
        "top_drifted_features",
        "alerts",
        "retrain_triggered",
        "new_model_version",
    ]).to_csv(MONITOR_LOG_PATH, index=False)


def reset_state():
    for p in [
        MODEL_PATH,
        META_PATH,
        MONITOR_LOG_PATH,
        MODEL_REGISTRY_PATH,
        TRAINING_POOL_PATH,
        BASELINE_METRICS_PATH,
    ]:
        if os.path.exists(p):
            os.remove(p)
    ensure_state_initialized()


def build_monitor_log_row(batch_name, model_version_used, current_metrics, baseline_metrics, drift_df, alerts, retrain_needed, new_model_version):
    top_drifted_features = drift_df[drift_df["Drift Detected"]].head(5)["Feature"].tolist()

    return {
        "batch_name": batch_name,
        "model_version_used": model_version_used,
        "precision": current_metrics["Precision"],
        "recall": current_metrics["Recall"],
        "f1": current_metrics["F1"],
        "roc_auc": current_metrics["ROC-AUC"],
        "pr_auc": current_metrics["PR-AUC"],
        "delta_precision": current_metrics["Precision"] - baseline_metrics["Precision"],
        "delta_recall": current_metrics["Recall"] - baseline_metrics["Recall"],
        "delta_f1": current_metrics["F1"] - baseline_metrics["F1"],
        "delta_roc_auc": current_metrics["ROC-AUC"] - baseline_metrics["ROC-AUC"],
        "delta_pr_auc": current_metrics["PR-AUC"] - baseline_metrics["PR-AUC"],
        "drifted_features": int(drift_df["Drift Detected"].sum()),
        "high_drift_features": int((drift_df["PSI Level"] == "High").sum()),
        "top_drifted_features": ", ".join(top_drifted_features) if top_drifted_features else "None",
        "alerts": " | ".join(alerts) if alerts else "No alerts",
        "retrain_triggered": bool(retrain_needed),
        "new_model_version": new_model_version if new_model_version is not None else "",
    }


def add_fraud_predictions(batch_df: pd.DataFrame, model, feature_cols: list, threshold: float) -> pd.DataFrame:
    scored_df = batch_df.copy()
    X = scored_df[feature_cols]
    fraud_probability = model.predict_proba(X)[:, 1]
    fraud_prediction = (fraud_probability >= threshold).astype(int)

    scored_df["Fraud Probability"] = fraud_probability
    scored_df["Fraud Detected"] = fraud_prediction
    scored_df["Detection Result"] = np.where(
        (scored_df["Fraud Detected"] == 1) & (scored_df["Class"] == 1),
        "True Positive",
        np.where(
            (scored_df["Fraud Detected"] == 1) & (scored_df["Class"] == 0),
            "False Positive",
            np.where(
                (scored_df["Fraud Detected"] == 0) & (scored_df["Class"] == 1),
                "Missed Fraud",
                "True Negative",
            ),
        ),
    )
    return scored_df


def process_batch(batch_file: str, threshold: float):
    metadata = load_json(META_PATH)
    baseline_metrics = load_json(BASELINE_METRICS_PATH)
    model = joblib.load(MODEL_PATH)
    feature_cols = metadata["feature_cols"]

    model_version_used = metadata["current_model_version"]

    training_pool = pd.read_csv(TRAINING_POOL_PATH)
    training_pool = prepare_features(training_pool)

    batch_df = load_csv(batch_file)
    batch_df = prepare_features(batch_df)
    validate_batch_columns(batch_df, feature_cols)

    X_batch = batch_df[feature_cols]
    y_batch = batch_df["Class"]
    scored_batch_df = add_fraud_predictions(batch_df, model, feature_cols, threshold)

    current_metrics = evaluate_model(model, X_batch, y_batch, threshold=threshold)
    drift_df = detect_feature_drift(
        training_pool[feature_cols],
        batch_df[feature_cols],
        feature_cols,
        alpha=ALPHA,
    )
    alerts, retrain_needed = degradation_check(current_metrics, baseline_metrics, drift_df)

    new_model_version = None
    new_model_validation_metrics = None

    # Processing a batch should only use the existing active model.
    # Retraining is recommended through alerts, but performed manually with a separate button.
    metadata["processed_batches"].append(batch_file)
    save_json(META_PATH, metadata)

    log_row = build_monitor_log_row(
        batch_name=batch_file,
        model_version_used=model_version_used,
        current_metrics=current_metrics,
        baseline_metrics=baseline_metrics,
        drift_df=drift_df,
        alerts=alerts,
        retrain_needed=retrain_needed,
        new_model_version=new_model_version,
    )
    append_monitor_log(log_row)

    return current_metrics, drift_df, alerts, retrain_needed, new_model_version, new_model_validation_metrics, scored_batch_df


def process_uploaded_batch(uploaded_file_obj, threshold: float):
    metadata = load_json(META_PATH)
    baseline_metrics = load_json(BASELINE_METRICS_PATH)
    model = joblib.load(MODEL_PATH)
    feature_cols = metadata["feature_cols"]

    model_version_used = metadata["current_model_version"]

    training_pool = pd.read_csv(TRAINING_POOL_PATH)
    training_pool = prepare_features(training_pool)

    batch_df = pd.read_csv(uploaded_file_obj)
    batch_name = uploaded_file_obj.name
    batch_df = prepare_features(batch_df)
    validate_batch_columns(batch_df, feature_cols)

    X_batch = batch_df[feature_cols]
    y_batch = batch_df["Class"]
    scored_batch_df = add_fraud_predictions(batch_df, model, feature_cols, threshold)

    current_metrics = evaluate_model(model, X_batch, y_batch, threshold=threshold)
    drift_df = detect_feature_drift(
        training_pool[feature_cols],
        batch_df[feature_cols],
        feature_cols,
        alpha=ALPHA,
    )
    alerts, retrain_needed = degradation_check(current_metrics, baseline_metrics, drift_df)

    new_model_version = None
    new_model_validation_metrics = None

    # Processing a batch should only use the existing active model.
    # Retraining is recommended through alerts, but performed manually with a separate button.
    metadata["processed_batches"].append(batch_name)
    save_json(META_PATH, metadata)

    log_row = build_monitor_log_row(
        batch_name=batch_name,
        model_version_used=model_version_used,
        current_metrics=current_metrics,
        baseline_metrics=baseline_metrics,
        drift_df=drift_df,
        alerts=alerts,
        retrain_needed=retrain_needed,
        new_model_version=new_model_version,
    )
    append_monitor_log(log_row)

    return batch_name, batch_df, current_metrics, drift_df, alerts, retrain_needed, new_model_version, new_model_validation_metrics, scored_batch_df


def latest_unprocessed_batches():
    metadata = load_json(META_PATH)
    done = set(metadata.get("processed_batches", []))
    return [f for f in DRIFT_FILES if os.path.exists(f) and f not in done]


def plot_history(history_df: pd.DataFrame):
    if history_df.empty:
        st.info("No monitoring history yet.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history_df["batch_name"], history_df["precision"], marker="o", label="Precision")
    ax.plot(history_df["batch_name"], history_df["recall"], marker="o", label="Recall")
    ax.plot(history_df["batch_name"], history_df["f1"], marker="o", label="F1")
    ax.plot(history_df["batch_name"], history_df["pr_auc"], marker="o", label="PR-AUC")
    ax.set_title("Monitoring Metrics Over Time")
    ax.set_ylabel("Metric value")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)


def plot_distribution(train_df: pd.DataFrame, prod_df: pd.DataFrame, feature: str):
    if feature not in train_df.columns or feature not in prod_df.columns:
        st.warning(f"Feature '{feature}' is not available in both datasets.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(train_df[feature].dropna(), bins=50, alpha=0.5, density=True, label="Training Pool")
    ax.hist(prod_df[feature].dropna(), bins=50, alpha=0.5, density=True, label="Incoming Batch")
    ax.set_title(f"Distribution Comparison: {feature}")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)


def display_metric_with_delta(container, label, current_value, baseline_value):
    if pd.isna(current_value) or pd.isna(baseline_value):
        container.metric(label, "N/A")
    else:
        container.metric(label, f"{current_value:.3f}", delta=f"{current_value - baseline_value:.3f}")


def display_drift_interpretation(drift_df, current_metrics, baseline_metrics):
    drifted_count = int(drift_df["Drift Detected"].sum())
    high_count = int((drift_df["PSI Level"] == "High").sum())
    top_drifted = drift_df[drift_df["Drift Detected"]].head(5)

    recall_delta = current_metrics["Recall"] - baseline_metrics["Recall"]
    f1_delta = current_metrics["F1"] - baseline_metrics["F1"]
    pr_auc_delta = current_metrics["PR-AUC"] - baseline_metrics["PR-AUC"]

    if drifted_count == 0:
        st.info("Interpretation: No statistically meaningful feature drift was detected in this batch using the KS test and PSI thresholds.")
        return

    feature_text = ", ".join(top_drifted["Feature"].tolist())
    st.markdown(
        f"""
        **Interpretation:** This batch shows drift in **{drifted_count} features**, including **{high_count} high-drift features**.  
        The strongest shifted features are: **{feature_text}**.  
        Compared with the baseline, recall changed by **{recall_delta:.3f}**, F1 changed by **{f1_delta:.3f}**, and PR-AUC changed by **{pr_auc_delta:.3f}**.  
        This links the production performance change to measurable distribution shifts in the incoming data.
        """
    )


# =========================================================
# APP BOOTSTRAP
# =========================================================
ensure_state_initialized()

metadata = load_json(META_PATH)
baseline_metrics = load_json(BASELINE_METRICS_PATH)
model_registry = pd.read_csv(MODEL_REGISTRY_PATH)
monitor_log = pd.read_csv(MONITOR_LOG_PATH)
training_pool = pd.read_csv(TRAINING_POOL_PATH)
training_pool = prepare_features(training_pool)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Live Monitoring Controls")
uploaded_file = st.sidebar.file_uploader("Upload new drift CSV", type=["csv"])
optimized_threshold = baseline_metrics.get("Best Threshold", DEFAULT_THRESHOLD)

use_auto_threshold = st.sidebar.checkbox(
    "Use optimized threshold",
    value=True
)

if use_auto_threshold:
    threshold = optimized_threshold
    st.sidebar.info(f"Using optimized threshold: {threshold:.2f}")
else:
    threshold = st.sidebar.slider(
        "Prediction threshold",
        0.01,
        0.50,
        optimized_threshold,
        0.01
    )

available_batches = latest_unprocessed_batches()
selected_batch = st.sidebar.selectbox("Next incoming batch", available_batches if available_batches else ["No batches left"])
source_mode = st.sidebar.radio("Incoming data source", ["Saved drift batch", "Uploaded CSV"], horizontal=False)

feature_candidates = ["Amount", "Time", "Hour", "V10", "V12", "V14"]
feature_candidates = [f for f in feature_candidates if f in training_pool.columns]
if not feature_candidates:
    feature_candidates = [c for c in training_pool.columns if c != "Class"][:6]
feature_option = st.sidebar.selectbox("Feature to inspect", feature_candidates)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    process_disabled = source_mode == "Saved drift batch" and selected_batch == "No batches left"
    process_clicked = st.button("Process Next Batch", type="primary", disabled=process_disabled)
with col_sb2:
    reset_clicked = st.button("Reset Simulation")

if reset_clicked:
    reset_state()
    st.rerun()

if process_clicked:
    try:
        if source_mode == "Saved drift batch":
            if selected_batch == "No batches left":
                st.sidebar.error("No saved drift batches left to process.")
            else:
                metrics, drift_df, alerts, retrain_needed, new_model_version, new_model_validation_metrics, scored_batch_df = process_batch(selected_batch, threshold)
                st.session_state["last_batch"] = selected_batch
                st.session_state["last_batch_df"] = prepare_features(load_csv(selected_batch))
                st.session_state["last_scored_batch_df"] = scored_batch_df
                st.session_state["last_metrics"] = metrics
                st.session_state["last_drift_df"] = drift_df
                st.session_state["last_alerts"] = alerts
                st.session_state["last_retrain_needed"] = retrain_needed
                st.session_state["last_new_model_version"] = new_model_version
                st.session_state["last_new_model_validation_metrics"] = new_model_validation_metrics
                st.session_state["last_retrain_source"] = "Saved drift batch"
                st.session_state["last_threshold"] = threshold
                st.rerun()
        else:
            if uploaded_file is None:
                st.sidebar.error("Please upload a CSV file first.")
            else:
                batch_name, batch_df, metrics, drift_df, alerts, retrain_needed, new_model_version, new_model_validation_metrics, scored_batch_df = process_uploaded_batch(uploaded_file, threshold)
                st.session_state["last_batch"] = batch_name
                st.session_state["last_batch_df"] = batch_df
                st.session_state["last_scored_batch_df"] = scored_batch_df
                st.session_state["last_metrics"] = metrics
                st.session_state["last_drift_df"] = drift_df
                st.session_state["last_alerts"] = alerts
                st.session_state["last_retrain_needed"] = retrain_needed
                st.session_state["last_new_model_version"] = new_model_version
                st.session_state["last_new_model_validation_metrics"] = new_model_validation_metrics
                st.session_state["last_retrain_source"] = "Uploaded CSV"
                st.session_state["last_threshold"] = threshold
                st.rerun()
    except Exception as e:
        st.sidebar.error(str(e))

# Reload after potential state updates
metadata = load_json(META_PATH)
model_registry = pd.read_csv(MODEL_REGISTRY_PATH)
monitor_log = pd.read_csv(MONITOR_LOG_PATH)
training_pool = pd.read_csv(TRAINING_POOL_PATH)
training_pool = prepare_features(training_pool)

# =========================================================
# HEADER
# =========================================================
st.title("Fraud Detection Live Monitoring Dashboard")
st.markdown(
    "This dashboard simulates production monitoring over time. Each incoming batch is scored by the **current active model**. "
    "If degradation or drift is detected, the system recommends retraining. A new model is only trained when the user confirms the retraining action."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active Model", metadata["active_model_name"])
c2.metric("Current Version", f"v{metadata['current_model_version']}")
c3.metric("Processed Batches", len(metadata.get("processed_batches", [])))
c4.metric("Remaining Batches", len(latest_unprocessed_batches()))

# =========================================================
# BASELINE PANEL
# =========================================================
st.subheader("Baseline Reference")
st.caption("Baseline metrics are calculated from the original Kaggle credit card fraud dataset using a holdout test split.")

b1, b2, b3, b4, b5 = st.columns(5)
b1.metric("Precision", f"{baseline_metrics['Precision']:.3f}")
b2.metric("Recall", f"{baseline_metrics['Recall']:.3f}")
b3.metric("F1", f"{baseline_metrics['F1']:.3f}")
b4.metric("ROC-AUC", f"{baseline_metrics['ROC-AUC']:.3f}")
b5.metric("PR-AUC", f"{baseline_metrics['PR-AUC']:.3f}")

st.caption(
    f"Optimized prediction threshold: {baseline_metrics.get('Best Threshold', DEFAULT_THRESHOLD):.2f}"
)

# =========================================================
# LATEST EVENT PANEL
# =========================================================
st.subheader("Latest Monitoring Event")

if "last_batch" not in st.session_state:
    st.info("No batch has been processed yet. Choose the next incoming batch in the sidebar and click 'Process Next Batch'.")
else:
    last_batch = st.session_state["last_batch"]
    last_metrics = st.session_state["last_metrics"]
    last_drift_df = st.session_state["last_drift_df"]
    last_alerts = st.session_state["last_alerts"]
    last_retrain_needed = st.session_state["last_retrain_needed"]
    last_new_model_version = st.session_state["last_new_model_version"]
    last_new_model_validation_metrics = st.session_state.get("last_new_model_validation_metrics")

    st.markdown(f"**Processed batch:** `{last_batch}`")

    m1, m2, m3, m4, m5 = st.columns(5)
    display_metric_with_delta(m1, "Precision", last_metrics["Precision"], baseline_metrics["Precision"])
    display_metric_with_delta(m2, "Recall", last_metrics["Recall"], baseline_metrics["Recall"])
    display_metric_with_delta(m3, "F1", last_metrics["F1"], baseline_metrics["F1"])
    display_metric_with_delta(m4, "ROC-AUC", last_metrics["ROC-AUC"], baseline_metrics["ROC-AUC"])
    display_metric_with_delta(m5, "PR-AUC", last_metrics["PR-AUC"], baseline_metrics["PR-AUC"])

    display_drift_interpretation(last_drift_df, last_metrics, baseline_metrics)

    if last_alerts:
        st.error("Degradation or drift alert detected")
        for alert in last_alerts:
            st.warning(alert)
    else:
        st.success("No degradation detected")

    if last_retrain_needed:
        st.warning("Retraining is recommended, but it has not been performed automatically.")

        if st.button("Retrain Model Now", type="primary"):
            try:
                metadata_for_retrain = load_json(META_PATH)
                training_pool_for_retrain = pd.read_csv(TRAINING_POOL_PATH)
                training_pool_for_retrain = prepare_features(training_pool_for_retrain)
                batch_for_retrain = st.session_state["last_batch_df"]
                retrain_threshold = st.session_state.get("last_threshold", threshold)

                new_model_version, new_model_validation_metrics = retrain_and_promote_model(
                    metadata=metadata_for_retrain,
                    training_pool=training_pool_for_retrain,
                    batch_df=batch_for_retrain,
                    batch_name=last_batch,
                    threshold=retrain_threshold,
                )

                st.session_state["last_new_model_version"] = new_model_version
                st.session_state["last_new_model_validation_metrics"] = new_model_validation_metrics
                st.session_state["last_retrain_needed"] = False
                st.success(f"Retraining completed. Active model updated to v{new_model_version}.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    elif last_new_model_validation_metrics is not None:
        st.success(f"Retraining completed. Active model updated to v{last_new_model_version}.")
        st.markdown("**New model validation metrics after retraining**")
        nv1, nv2, nv3, nv4, nv5 = st.columns(5)
        nv1.metric("Precision", f"{last_new_model_validation_metrics['Precision']:.3f}")
        nv2.metric("Recall", f"{last_new_model_validation_metrics['Recall']:.3f}")
        nv3.metric("F1", f"{last_new_model_validation_metrics['F1']:.3f}")
        nv4.metric("ROC-AUC", f"{last_new_model_validation_metrics['ROC-AUC']:.3f}")
        nv5.metric("PR-AUC", f"{last_new_model_validation_metrics['PR-AUC']:.3f}")
    else:
        st.info("No retraining was needed for this batch.")

    st.markdown("**Top drifted features**")
    st.dataframe(last_drift_df.head(10), use_container_width=True)

    cm = pd.DataFrame(
        last_metrics["Confusion Matrix"],
        index=["Actual Non-Fraud", "Actual Fraud"],
        columns=["Predicted Non-Fraud", "Predicted Fraud"],
    )
    st.markdown("**Confusion Matrix**")
    st.dataframe(cm, use_container_width=True)

    st.markdown("**Fraud Detection Table**")
    if "last_scored_batch_df" in st.session_state:
        scored_batch_df = st.session_state["last_scored_batch_df"].copy()
        detected_fraud_df = scored_batch_df[scored_batch_df["Fraud Detected"] == 1].copy()
        detected_fraud_df = detected_fraud_df.sort_values("Fraud Probability", ascending=False)

        st.metric("Detected Fraud Transactions", len(detected_fraud_df))

        if detected_fraud_df.empty:
            st.info("No transactions were flagged as fraud at the selected threshold.")
        else:
            display_cols = [
                col for col in [
                    "Time",
                    "Hour",
                    "Amount",
                    "Fraud Probability",
                    "Fraud Detected",
                    "Class",
                    "Detection Result",
                ] if col in detected_fraud_df.columns
            ]

            with st.expander("Expand detected fraud transactions", expanded=False):
                st.dataframe(
                    detected_fraud_df[display_cols],
                    use_container_width=True,
                    hide_index=False,
                )

            with st.expander("Expand full scored batch", expanded=False):
                st.dataframe(
                    scored_batch_df.sort_values("Fraud Probability", ascending=False),
                    use_container_width=True,
                    hide_index=False,
                )

# =========================================================
# MONITORING TIMELINE
# =========================================================
st.subheader("Monitoring Timeline")
if monitor_log.empty:
    st.info("Timeline will appear after the first batch is processed.")
else:
    st.dataframe(monitor_log, use_container_width=True)
    plot_history(monitor_log)

# =========================================================
# MODEL REGISTRY
# =========================================================
st.subheader("Model Registry")
st.caption("Each promoted model is logged with validation metrics from the accumulated training pool.")
st.dataframe(model_registry, use_container_width=True)

# =========================================================
# DISTRIBUTION VIEW
# =========================================================
st.subheader("Training Pool vs Incoming Batch Distribution")
if "last_batch_df" in st.session_state:
    current_batch_df = st.session_state["last_batch_df"]
    plot_distribution(training_pool, current_batch_df, feature_option)
else:
    st.info("Process a batch first to inspect distribution changes.")

# =========================================================
# AUTOMATED DRIFT CHECK EXPLANATION
# =========================================================
st.subheader("Automated Drift Check")
st.markdown(
    "The app uses both statistical drift and model-performance thresholds. "
    "A retraining alert is triggered when performance drops meaningfully or when multiple features show moderate/high PSI drift. Retraining is a separate manual action after the alert."
)

st.code(
    """
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

if any alert triggers:
    recommend retraining

if user confirms retraining:
    retrain current model on accumulated data and promote new model version
""",
    language="python",
)

st.caption("State is stored locally in the 'monitoring_state' folder. Reset the simulation anytime from the sidebar.")
