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
from sklearn.ensemble import RandomForestClassifier

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
DEFAULT_THRESHOLD = 0.50
ALPHA = 0.05

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

    breakpoints = np.percentile(
        expected,
        np.arange(0, 100 + 100 / buckets, 100 / buckets)
    )
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
            "KS Statistic": float(ks_stat),
            "p-value": float(p_value),
            "PSI": float(psi_value),
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


def train_random_forest(train_df: pd.DataFrame):
    train_df = prepare_features(train_df)
    X = train_df.drop(columns=["Class"])
    y = train_df["Class"]

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
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

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, threshold=DEFAULT_THRESHOLD)
    return metrics


def degradation_check(current_metrics, baseline_metrics, drift_df):
    alerts = []

    if current_metrics["Recall"] < baseline_metrics["Recall"] * 0.8:
        alerts.append("Critical: Recall dropped by more than 20% from baseline")

    if current_metrics["F1"] < baseline_metrics["F1"] * 0.8:
        alerts.append("Critical: F1 dropped by more than 20% from baseline")

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


def ensure_state_initialized():
    if all(os.path.exists(p) for p in [MODEL_PATH, META_PATH, MONITOR_LOG_PATH, MODEL_REGISTRY_PATH, TRAINING_POOL_PATH, BASELINE_METRICS_PATH]):
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
        "drifted_features",
        "high_drift_features",
        "alerts",
        "retrain_triggered",
        "new_model_version",
    ]).to_csv(MONITOR_LOG_PATH, index=False)


def reset_state():
    for p in [MODEL_PATH, META_PATH, MONITOR_LOG_PATH, MODEL_REGISTRY_PATH, TRAINING_POOL_PATH, BASELINE_METRICS_PATH]:
        if os.path.exists(p):
            os.remove(p)
    ensure_state_initialized()


def process_batch(batch_file: str, threshold: float):
    metadata = load_json(META_PATH)
    baseline_metrics = load_json(BASELINE_METRICS_PATH)
    model = joblib.load(MODEL_PATH)
    feature_cols = metadata["feature_cols"]

    training_pool = pd.read_csv(TRAINING_POOL_PATH)
    training_pool = prepare_features(training_pool)

    batch_df = load_csv(batch_file)
    batch_df = prepare_features(batch_df)

    X_batch = batch_df.drop(columns=["Class"])
    y_batch = batch_df["Class"]
    X_batch = X_batch[feature_cols]

    current_metrics = evaluate_model(model, X_batch, y_batch, threshold=threshold)
    drift_df = detect_feature_drift(
        training_pool[feature_cols],
        batch_df[feature_cols],
        feature_cols,
        alpha=ALPHA,
    )
    alerts, retrain_needed = degradation_check(current_metrics, baseline_metrics, drift_df)

    new_model_version = None

    if retrain_needed:
        updated_training_pool = pd.concat([training_pool, batch_df], ignore_index=True)
        updated_training_pool = prepare_features(updated_training_pool)

        new_model, new_feature_cols = train_random_forest(updated_training_pool)
        joblib.dump(new_model, MODEL_PATH)
        updated_training_pool.to_csv(TRAINING_POOL_PATH, index=False)

        metadata["current_model_version"] += 1
        metadata["feature_cols"] = new_feature_cols
        metadata["active_model_name"] = f"Model v{metadata['current_model_version']}"
        metadata["processed_batches"].append(batch_file)
        save_json(META_PATH, metadata)

        new_model_version = metadata["current_model_version"]

        model_registry = pd.read_csv(MODEL_REGISTRY_PATH)
        model_registry = pd.concat([
            model_registry,
            pd.DataFrame([{
                "model_version": new_model_version,
                "model_name": metadata["active_model_name"],
                "trained_on": f"accumulated_data_through_{batch_file}",
                "activated_after_batch": batch_file,
                "precision": current_metrics["Precision"],
                "recall": current_metrics["Recall"],
                "f1": current_metrics["F1"],
                "roc_auc": current_metrics["ROC-AUC"],
                "pr_auc": current_metrics["PR-AUC"],
            }])
        ], ignore_index=True)
        model_registry.to_csv(MODEL_REGISTRY_PATH, index=False)
    else:
        metadata["processed_batches"].append(batch_file)
        save_json(META_PATH, metadata)

    log_row = {
        "batch_name": batch_file,
        "model_version_used": metadata["current_model_version"] - 1 if new_model_version is not None else metadata["current_model_version"],
        "precision": current_metrics["Precision"],
        "recall": current_metrics["Recall"],
        "f1": current_metrics["F1"],
        "roc_auc": current_metrics["ROC-AUC"],
        "pr_auc": current_metrics["PR-AUC"],
        "drifted_features": int(drift_df["Drift Detected"].sum()),
        "high_drift_features": int((drift_df["PSI Level"] == "High").sum()),
        "alerts": " | ".join(alerts) if alerts else "No alerts",
        "retrain_triggered": bool(retrain_needed),
        "new_model_version": new_model_version if new_model_version is not None else "",
    }

    monitor_log = pd.read_csv(MONITOR_LOG_PATH)
    monitor_log = pd.concat([monitor_log, pd.DataFrame([log_row])], ignore_index=True)
    monitor_log.to_csv(MONITOR_LOG_PATH, index=False)

    return current_metrics, drift_df, alerts, retrain_needed, new_model_version


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
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)


def plot_distribution(train_df: pd.DataFrame, prod_df: pd.DataFrame, feature: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(train_df[feature], bins=50, alpha=0.5, density=True, label="Training Pool")
    ax.hist(prod_df[feature], bins=50, alpha=0.5, density=True, label="Incoming Batch")
    ax.set_title(f"Distribution Comparison: {feature}")
    ax.legend()
    st.pyplot(fig)


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
threshold = st.sidebar.slider("Prediction threshold", 0.01, 1.00, DEFAULT_THRESHOLD, 0.01)
available_batches = latest_unprocessed_batches()
selected_batch = st.sidebar.selectbox("Next incoming batch", available_batches if available_batches else ["No batches left"])
source_mode = st.sidebar.radio("Incoming data source", ["Saved drift batch", "Uploaded CSV"], horizontal=False)
feature_option = st.sidebar.selectbox("Feature to inspect", ["Amount", "Time", "Hour", "V10", "V12", "V14"])

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    process_clicked = st.button("Process Next Batch", type="primary", disabled=(selected_batch == "No batches left"))
with col_sb2:
    reset_clicked = st.button("Reset Simulation")

if reset_clicked:
    reset_state()
    st.rerun()

def process_uploaded_batch(uploaded_file_obj, threshold: float):
    metadata = load_json(META_PATH)
    baseline_metrics = load_json(BASELINE_METRICS_PATH)
    model = joblib.load(MODEL_PATH)
    feature_cols = metadata["feature_cols"]

    training_pool = pd.read_csv(TRAINING_POOL_PATH)
    training_pool = prepare_features(training_pool)

    batch_df = pd.read_csv(uploaded_file_obj)
    batch_name = uploaded_file_obj.name
    batch_df = prepare_features(batch_df)

    missing = [c for c in ["Class"] + feature_cols if c not in batch_df.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing required columns: {missing}")

    X_batch = batch_df.drop(columns=["Class"])
    y_batch = batch_df["Class"]
    X_batch = X_batch[feature_cols]

    current_metrics = evaluate_model(model, X_batch, y_batch, threshold=threshold)
    drift_df = detect_feature_drift(
        training_pool[feature_cols],
        batch_df[feature_cols],
        feature_cols,
        alpha=ALPHA,
    )
    alerts, retrain_needed = degradation_check(current_metrics, baseline_metrics, drift_df)

    new_model_version = None

    if retrain_needed:
        updated_training_pool = pd.concat([training_pool, batch_df], ignore_index=True)
        updated_training_pool = prepare_features(updated_training_pool)

        new_model, new_feature_cols = train_random_forest(updated_training_pool)
        joblib.dump(new_model, MODEL_PATH)
        updated_training_pool.to_csv(TRAINING_POOL_PATH, index=False)

        metadata["current_model_version"] += 1
        metadata["feature_cols"] = new_feature_cols
        metadata["active_model_name"] = f"Model v{metadata['current_model_version']}"
        metadata["processed_batches"].append(batch_name)
        save_json(META_PATH, metadata)

        new_model_version = metadata["current_model_version"]

        model_registry = pd.read_csv(MODEL_REGISTRY_PATH)
        model_registry = pd.concat([
            model_registry,
            pd.DataFrame([{
                "model_version": new_model_version,
                "model_name": metadata["active_model_name"],
                "trained_on": f"accumulated_data_through_{batch_name}",
                "activated_after_batch": batch_name,
                "precision": current_metrics["Precision"],
                "recall": current_metrics["Recall"],
                "f1": current_metrics["F1"],
                "roc_auc": current_metrics["ROC-AUC"],
                "pr_auc": current_metrics["PR-AUC"],
            }])
        ], ignore_index=True)
        model_registry.to_csv(MODEL_REGISTRY_PATH, index=False)
    else:
        metadata["processed_batches"].append(batch_name)
        save_json(META_PATH, metadata)

    log_row = {
        "batch_name": batch_name,
        "model_version_used": metadata["current_model_version"] - 1 if new_model_version is not None else metadata["current_model_version"],
        "precision": current_metrics["Precision"],
        "recall": current_metrics["Recall"],
        "f1": current_metrics["F1"],
        "roc_auc": current_metrics["ROC-AUC"],
        "pr_auc": current_metrics["PR-AUC"],
        "drifted_features": int(drift_df["Drift Detected"].sum()),
        "high_drift_features": int((drift_df["PSI Level"] == "High").sum()),
        "alerts": " | ".join(alerts) if alerts else "No alerts",
        "retrain_triggered": bool(retrain_needed),
        "new_model_version": new_model_version if new_model_version is not None else "",
    }

    monitor_log = pd.read_csv(MONITOR_LOG_PATH)
    monitor_log = pd.concat([monitor_log, pd.DataFrame([log_row])], ignore_index=True)
    monitor_log.to_csv(MONITOR_LOG_PATH, index=False)

    return batch_name, batch_df, current_metrics, drift_df, alerts, retrain_needed, new_model_version

if process_clicked:
    try:
        if source_mode == "Saved drift batch":
            if selected_batch == "No batches left":
                st.sidebar.error("No saved drift batches left to process.")
            else:
                metrics, drift_df, alerts, retrain_needed, new_model_version = process_batch(selected_batch, threshold)
                st.session_state["last_batch"] = selected_batch
                st.session_state["last_batch_df"] = prepare_features(load_csv(selected_batch))
                st.session_state["last_metrics"] = metrics
                st.session_state["last_drift_df"] = drift_df
                st.session_state["last_alerts"] = alerts
                st.session_state["last_retrain_needed"] = retrain_needed
                st.session_state["last_new_model_version"] = new_model_version
                st.rerun()
        else:
            if uploaded_file is None:
                st.sidebar.error("Please upload a CSV file first.")
            else:
                batch_name, batch_df, metrics, drift_df, alerts, retrain_needed, new_model_version = process_uploaded_batch(uploaded_file, threshold)
                st.session_state["last_batch"] = batch_name
                st.session_state["last_batch_df"] = batch_df
                st.session_state["last_metrics"] = metrics
                st.session_state["last_drift_df"] = drift_df
                st.session_state["last_alerts"] = alerts
                st.session_state["last_retrain_needed"] = retrain_needed
                st.session_state["last_new_model_version"] = new_model_version
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
    "If degradation or drift is detected, the system retrains and promotes a new active model automatically."
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
b1, b2, b3, b4, b5 = st.columns(5)
b1.metric("Precision", f"{baseline_metrics['Precision']:.3f}")
b2.metric("Recall", f"{baseline_metrics['Recall']:.3f}")
b3.metric("F1", f"{baseline_metrics['F1']:.3f}")
b4.metric("ROC-AUC", f"{baseline_metrics['ROC-AUC']:.3f}")
b5.metric("PR-AUC", f"{baseline_metrics['PR-AUC']:.3f}")

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

    st.markdown(f"**Processed batch:** `{last_batch}`")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precision", f"{last_metrics['Precision']:.3f}")
    m2.metric("Recall", f"{last_metrics['Recall']:.3f}")
    m3.metric("F1", f"{last_metrics['F1']:.3f}")
    m4.metric("ROC-AUC", f"{last_metrics['ROC-AUC']:.3f}")
    m5.metric("PR-AUC", f"{last_metrics['PR-AUC']:.3f}")

    if last_alerts:
        st.error("Degradation detected")
        for alert in last_alerts:
            st.warning(alert)
    else:
        st.success("No degradation detected")

    if last_retrain_needed:
        st.success(f"Retraining triggered. Active model updated to v{last_new_model_version}.")
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
st.code(
    """if current_metrics['Recall'] < baseline_metrics['Recall'] * 0.8:\n"
    "    alert = 'Critical recall degradation'\n"
    "if current_metrics['F1'] < baseline_metrics['F1'] * 0.8:\n"
    "    alert = 'Critical F1 degradation'\n"
    "if high_drift_features >= 3:\n"
    "    alert = 'Critical feature drift'\n"
    "if any alert triggers:\n"
    "    retrain current model on accumulated data and promote new model version\n""",
    language="python",
)

st.caption("State is stored locally in the 'monitoring_state' folder. Reset the simulation anytime from the sidebar.")
