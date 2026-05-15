import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

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
st.set_page_config(page_title="Fraud Live Monitoring Dashboard", layout="wide", page_icon="🛡️")

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
ALPHA = 0.05

MAX_HISTORICAL_RETRAIN_ROWS = 50000
RECENT_BATCH_WEIGHT = 4

os.makedirs(STATE_DIR, exist_ok=True)

# =========================================================
# GLOBAL STYLES
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, .stApp {
    background-color: #080c14;
    color: #c8d0e0;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0c1120;
    border-right: 1px solid #1a2235;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label {
    color: #7a8aaa !important;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    text-transform: none;
    font-size: 0.9rem;
    color: #c8d0e0 !important;
    letter-spacing: 0;
}
[data-testid="stSidebar"] h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #3b82f6 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f1829 0%, #0c1422 100%);
    border: 1px solid #1e2d47;
    border-radius: 6px;
    padding: 18px 20px 14px;
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
[data-testid="stMetricLabel"] {
    color: #5a6a8a !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #e8edf8 !important;
    font-size: 1.7rem !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.02em;
}
[data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Headings ── */
h1 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 300 !important;
    font-size: 1.6rem !important;
    color: #e8edf8 !important;
    letter-spacing: -0.01em;
    margin-bottom: 2px !important;
}
h2 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #3b6baf !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 2rem !important;
    margin-bottom: 0.5rem !important;
}
h3 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    color: #c8d0e0 !important;
}

/* ── Divider ── */
hr {
    border-color: #1a2235 !important;
    margin: 1.2rem 0 !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a2235;
    border-radius: 6px;
    overflow: hidden;
}
[data-testid="stDataFrame"] th {
    background-color: #0c1422 !important;
    color: #5a6a8a !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="stDataFrame"] td {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #b0bdd0 !important;
    background-color: #0a1020 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em;
    padding: 8px 20px !important;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
.stButton > button[kind="secondary"] {
    background: #1a2235 !important;
    color: #7a8aaa !important;
}

/* ── Selectbox / inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] {
    background-color: #0c1422 !important;
    border-color: #1e2d47 !important;
    border-radius: 4px !important;
    color: #c8d0e0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* ── Info / warning / error boxes ── */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.88rem !important;
    border-left-width: 3px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background-color: #0c1422 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
    color: #7a8aaa !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em;
}

/* ── Code blocks ── */
.stCode, code {
    font-family: 'IBM Plex Mono', monospace !important;
    background-color: #0c1422 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 4px !important;
    font-size: 0.82rem !important;
}

/* ── Matplotlib chart bg ── */
.stPlotlyChart, .stPyplot { border: 1px solid #1a2235; border-radius: 6px; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] p {
    color: #3b4a62 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label {
    color: #7a8aaa !important;
    font-size: 0.82rem !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background-color: #0c1422 !important;
    border: 1px dashed #1e2d47 !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CHART THEME
# =========================================================
mpl.rcParams.update({
    "figure.facecolor": "#080c14",
    "axes.facecolor": "#0a1020",
    "axes.edgecolor": "#1a2235",
    "axes.labelcolor": "#5a6a8a",
    "axes.titlecolor": "#c8d0e0",
    "xtick.color": "#5a6a8a",
    "ytick.color": "#5a6a8a",
    "text.color": "#c8d0e0",
    "grid.color": "#1a2235",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "legend.facecolor": "#0c1422",
    "legend.edgecolor": "#1a2235",
    "legend.labelcolor": "#c8d0e0",
    "font.family": "monospace",
})

CHART_COLORS = ["#3b82f6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"]

# =========================================================
# UI HELPERS
# =========================================================

def page_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="padding: 0.5rem 0 1.2rem; border-bottom: 1px solid #1a2235; margin-bottom: 1.5rem;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#3b6baf;
                    letter-spacing:0.16em; text-transform:uppercase; margin-bottom:4px;">
            Fraud Shield · Monitoring
        </div>
        <div style="font-family:'IBM Plex Sans',sans-serif; font-size:1.55rem;
                    font-weight:300; color:#e8edf8; letter-spacing:-0.01em;">
            {title}
        </div>
        {"<div style='font-size:0.85rem; color:#5a6a8a; margin-top:4px;'>" + subtitle + "</div>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; font-weight:600;
                color:#3b6baf; letter-spacing:0.14em; text-transform:uppercase;
                margin: 1.6rem 0 0.6rem; padding-bottom:6px; border-bottom:1px solid #1a2235;">
        {text}
    </div>
    """, unsafe_allow_html=True)


def status_banner():
    if "last_alerts" not in st.session_state:
        return
    alerts = st.session_state["last_alerts"]
    batch = st.session_state.get("last_batch", "—")
    if not alerts:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; background:#071a12;
                    border:1px solid #0d3324; border-left:3px solid #10b981;
                    border-radius:4px; padding:10px 16px; margin-bottom:1rem;">
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
                         color:#10b981; letter-spacing:0.1em;">● HEALTHY</span>
            <span style="font-family:'IBM Plex Sans',sans-serif; font-size:0.82rem;
                         color:#5a8a6a;">No alerts on <code style="background:none;color:#10b981">{batch}</code></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        critical = [a for a in alerts if "Critical" in a]
        is_critical = len(critical) > 0
        colour = "#ef4444" if is_critical else "#f59e0b"
        bg = "#1a0808" if is_critical else "#1a1408"
        border = "#3a1010" if is_critical else "#3a2c08"
        label = "● CRITICAL" if is_critical else "▲ WARNING"
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; background:{bg};
                    border:1px solid {border}; border-left:3px solid {colour};
                    border-radius:4px; padding:10px 16px; margin-bottom:1rem;">
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
                         color:{colour}; letter-spacing:0.1em;">{label}</span>
            <span style="font-family:'IBM Plex Sans',sans-serif; font-size:0.82rem; color:#9a7a6a;">
                {len(alerts)} alert(s) on <code style="background:none;color:{colour}">{batch}</code>
            </span>
        </div>
        """, unsafe_allow_html=True)
        for alert in alerts:
            c = "#ef4444" if "Critical" in alert else "#f59e0b"
            st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
                        color:{c}; padding:3px 0 3px 16px; border-left:2px solid {c}22;
                        margin-bottom:3px;">
                {alert}
            </div>
            """, unsafe_allow_html=True)


def kpi_row(metrics: dict, baseline: dict = None):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        if baseline and label in baseline and not pd.isna(value):
            delta = f"{value - baseline[label]:.3f}"
            col.metric(label, f"{value:.3f}", delta=delta)
        elif pd.isna(value):
            col.metric(label, "N/A")
        else:
            col.metric(label, f"{value:.3f}" if isinstance(value, float) else value)


def info_card(label: str, value: str, accent: str = "#3b82f6"):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0f1829,#0c1422);
                border:1px solid #1e2d47; border-radius:6px;
                padding:18px 20px 14px; position:relative; overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;
                    background:{accent};"></div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
                    color:#5a6a8a; letter-spacing:0.1em; text-transform:uppercase;
                    margin-bottom:6px;">{label}</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:1.5rem;
                    font-weight:600; color:#e8edf8;">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def style_drift_df(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_style(row):
        if row.get("PSI Level") == "High":
            return ["background-color:#1a0808; color:#ff6b6b"] * len(row)
        elif row.get("PSI Level") == "Moderate":
            return ["background-color:#1a1408; color:#f0c040"] * len(row)
        elif row.get("High Risk") is True:
            return ["background-color:#0d1a2a; color:#60a5fa"] * len(row)
        return [""] * len(row)
    return df.style.apply(row_style, axis=1)


# =========================================================
# HELPERS (unchanged logic)
# =========================================================

def prepare_features(df):
    df = df.copy()
    if "Hour" not in df.columns and "Time" in df.columns:
        df["Hour"] = (df["Time"] // 3600) % 24
    if "day" in df.columns:
        df = df.drop(columns=["day"])
    return df

def load_csv(path): return pd.read_csv(path)

def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_psi(expected, actual, buckets=10):
    expected = np.asarray(expected)[~pd.isna(np.asarray(expected))]
    actual = np.asarray(actual)[~pd.isna(np.asarray(actual))]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    bp = np.unique(np.percentile(expected, np.arange(0, 100 + 100/buckets, 100/buckets)))
    if len(bp) < 3:
        return 0.0
    bp[0] = -np.inf; bp[-1] = np.inf
    ep = np.histogram(expected, bins=bp)[0] / len(expected)
    ap = np.histogram(actual, bins=bp)[0] / len(actual)
    ep = np.where(ep == 0, 1e-6, ep)
    ap = np.where(ap == 0, 1e-6, ap)
    return float(np.sum((ap - ep) * np.log(ap / ep)))

def detect_feature_drift(train_df, prod_df, feature_cols, alpha=0.05):
    rows = []
    for col in feature_cols:
        tv, pv = train_df[col].dropna(), prod_df[col].dropna()
        if len(tv) == 0 or len(pv) == 0:
            rows.append({"Feature": col, "KS Statistic": np.nan, "p-value": np.nan,
                         "PSI": np.nan, "PSI Level": "Unavailable", "Drift Detected": False})
            continue
        ks_stat, p_value = ks_2samp(tv, pv)
        psi = calculate_psi(tv, pv)
        psi_level = "High" if psi >= 0.25 else ("Moderate" if psi >= 0.1 else "Low")
        rows.append({"Feature": col, "KS Statistic": float(ks_stat), "p-value": float(p_value),
                     "PSI": float(psi), "PSI Level": psi_level,
                     "Drift Detected": bool((p_value < alpha) or (psi >= 0.1))})
    return pd.DataFrame(rows).sort_values(["Drift Detected","PSI"], ascending=[False,False]).reset_index(drop=True)

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
    best_threshold, best_score = DEFAULT_THRESHOLD, -1
    for t in np.arange(0.01, 0.51, 0.01):
        y_pred = (y_prob >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        score = f1 if metric == "f1" else (rec if prec >= min_precision else -1)
        if score > best_score:
            best_score = score; best_threshold = t
    return float(best_threshold), float(best_score)

def train_fraud_model(train_df):
    train_df = prepare_features(train_df)
    X = train_df.drop(columns=["Class"]); y = train_df["Class"]
    spw = len(y[y==0]) / max(len(y[y==1]), 1)
    model = XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                          eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X, y)
    return model, X.columns.tolist()

def train_random_forest(train_df): return train_fraud_model(train_df)

def compute_baseline_metrics(base_df):
    base_df = prepare_features(base_df)
    X = base_df.drop(columns=["Class"]); y = base_df["Class"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                               random_state=RANDOM_STATE, stratify=y)
    spw = len(y_tr[y_tr==0]) / max(len(y_tr[y_tr==1]), 1)
    model = XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                          eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]
    bt, bts = find_best_threshold(y_te, y_prob, THRESHOLD_OPTIMIZATION_METRIC, MIN_PRECISION_FOR_THRESHOLD)
    metrics = evaluate_model(model, X_te, y_te, threshold=bt)
    metrics["Best Threshold"] = bt; metrics["Best Threshold Score"] = bts
    metrics["Threshold Optimization Metric"] = THRESHOLD_OPTIMIZATION_METRIC
    return metrics

def get_feature_importance(model, feature_cols):
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}
                            ).sort_values("Importance", ascending=False).reset_index(drop=True)
    return pd.DataFrame(columns=["Feature", "Importance"])

def cross_reference_drift_importance(drift_df, importance_df, top_n_important=10):
    merged = drift_df.merge(importance_df, on="Feature", how="left")
    merged["Importance"] = merged["Importance"].fillna(0)
    merged["Importance Rank"] = merged["Importance"].rank(ascending=False)
    top_features = set(importance_df.head(top_n_important)["Feature"].tolist())
    merged["High Risk"] = merged["Drift Detected"] & merged["Feature"].isin(top_features)
    merged["Risk Score"] = merged["PSI"] * merged["Importance"]
    return merged.sort_values("Risk Score", ascending=False).reset_index(drop=True)

def degradation_check(current_metrics, baseline_metrics, drift_df, importance_df=None):
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
    if importance_df is not None and not importance_df.empty:
        risk_df = cross_reference_drift_importance(drift_df, importance_df)
        high_risk = risk_df[risk_df["High Risk"]]["Feature"].tolist()
        if len(high_risk) >= 2:
            alerts.append(f"Critical: {len(high_risk)} high-importance features drifting: {', '.join(high_risk[:5])}")
        elif len(high_risk) == 1:
            alerts.append(f"Warning: High-importance feature drifting: {high_risk[0]}")
    return alerts, len(alerts) > 0

def validate_batch_columns(batch_df, feature_cols):
    missing = [c for c in ["Class"] + feature_cols if c not in batch_df.columns]
    if missing:
        raise ValueError(f"Batch is missing required columns: {missing}")

def append_monitor_log(log_row):
    ml = pd.read_csv(MONITOR_LOG_PATH)
    ml = pd.concat([ml, pd.DataFrame([log_row])], ignore_index=True)
    ml.to_csv(MONITOR_LOG_PATH, index=False)

def register_new_model(new_model_version, model_name, trained_on, activated_after_batch, validation_metrics):
    mr = pd.read_csv(MODEL_REGISTRY_PATH)
    mr = pd.concat([mr, pd.DataFrame([{
        "model_version": new_model_version, "model_name": model_name,
        "trained_on": trained_on, "activated_after_batch": activated_after_batch,
        "precision": validation_metrics["Precision"], "recall": validation_metrics["Recall"],
        "f1": validation_metrics["F1"], "roc_auc": validation_metrics["ROC-AUC"],
        "pr_auc": validation_metrics["PR-AUC"],
    }])], ignore_index=True)
    mr.to_csv(MODEL_REGISTRY_PATH, index=False)

def build_retraining_dataset(training_pool, batch_df):
    training_pool = prepare_features(training_pool)
    batch_df = prepare_features(batch_df)
    if len(training_pool) > MAX_HISTORICAL_RETRAIN_ROWS:
        fh = training_pool[training_pool["Class"]==1]
        nfh = training_pool[training_pool["Class"]==0]
        slots = max(MAX_HISTORICAL_RETRAIN_ROWS - len(fh), 0)
        historical_sample = pd.concat([fh, nfh.sample(n=min(slots, len(nfh)),
                                        random_state=RANDOM_STATE)], ignore_index=True)
    else:
        historical_sample = training_pool.copy()
    return prepare_features(pd.concat(
        [historical_sample] + [batch_df] * RECENT_BATCH_WEIGHT, ignore_index=True))

def get_stable_feature_cols(drift_df, importance_df, feature_cols):
    if drift_df is None or importance_df is None:
        return feature_cols
    risk_df = cross_reference_drift_importance(drift_df, importance_df)
    to_drop = risk_df[(risk_df["PSI Level"]=="High") & (~risk_df["High Risk"])]["Feature"].tolist()
    stable = [f for f in feature_cols if f not in to_drop]
    return stable if len(stable) >= 10 else feature_cols

def retrain_and_promote_model(metadata, training_pool, batch_df, batch_name, threshold, drift_df=None):
    updated_pool = build_retraining_dataset(training_pool, batch_df)
    temp_model, all_cols = train_fraud_model(updated_pool)
    importance_df = get_feature_importance(temp_model, all_cols)
    stable_cols = get_stable_feature_cols(drift_df, importance_df, all_cols)
    X = updated_pool[stable_cols]; y = updated_pool["Class"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                               random_state=RANDOM_STATE, stratify=y)
    spw = len(y_tr[y_tr==0]) / max(len(y_tr[y_tr==1]), 1)
    val_model = XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                              eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    val_model.fit(X_tr, y_tr)
    val_metrics = evaluate_model(val_model, X_te, y_te, threshold=threshold)
    stable_pool = updated_pool[stable_cols + ["Class"]]
    final_model, new_feature_cols = train_fraud_model(stable_pool)
    joblib.dump(final_model, MODEL_PATH)
    updated_pool.to_csv(TRAINING_POOL_PATH, index=False)
    metadata["current_model_version"] += 1
    metadata["feature_cols"] = new_feature_cols
    metadata["active_model_name"] = f"Model v{metadata['current_model_version']}"
    save_json(META_PATH, metadata)
    register_new_model(metadata["current_model_version"], metadata["active_model_name"],
                       f"weighted_recent_retraining_through_{batch_name}", batch_name, val_metrics)
    return metadata["current_model_version"], val_metrics

def build_monitor_log_row(batch_name, model_version_used, current_metrics, baseline_metrics,
                          drift_df, alerts, retrain_needed, new_model_version):
    top = drift_df[drift_df["Drift Detected"]].head(5)["Feature"].tolist()
    return {
        "batch_name": batch_name, "model_version_used": model_version_used,
        "precision": current_metrics["Precision"], "recall": current_metrics["Recall"],
        "f1": current_metrics["F1"], "roc_auc": current_metrics["ROC-AUC"],
        "pr_auc": current_metrics["PR-AUC"],
        "delta_precision": current_metrics["Precision"] - baseline_metrics["Precision"],
        "delta_recall": current_metrics["Recall"] - baseline_metrics["Recall"],
        "delta_f1": current_metrics["F1"] - baseline_metrics["F1"],
        "delta_roc_auc": current_metrics["ROC-AUC"] - baseline_metrics["ROC-AUC"],
        "delta_pr_auc": current_metrics["PR-AUC"] - baseline_metrics["PR-AUC"],
        "drifted_features": int(drift_df["Drift Detected"].sum()),
        "high_drift_features": int((drift_df["PSI Level"]=="High").sum()),
        "top_drifted_features": ", ".join(top) if top else "None",
        "alerts": " | ".join(alerts) if alerts else "No alerts",
        "retrain_triggered": bool(retrain_needed),
        "new_model_version": new_model_version if new_model_version is not None else "",
    }

def add_fraud_predictions(batch_df, model, feature_cols, threshold):
    scored = batch_df.copy()
    prob = model.predict_proba(scored[feature_cols])[:, 1]
    pred = (prob >= threshold).astype(int)
    scored["Fraud Probability"] = prob
    scored["Fraud Detected"] = pred
    scored["Detection Result"] = np.where(
        (pred==1)&(scored["Class"]==1), "True Positive",
        np.where((pred==1)&(scored["Class"]==0), "False Positive",
        np.where((pred==0)&(scored["Class"]==1), "Missed Fraud", "True Negative")))
    return scored

def process_batch(batch_file, threshold):
    metadata = load_json(META_PATH); baseline = load_json(BASELINE_METRICS_PATH)
    model = joblib.load(MODEL_PATH); feature_cols = metadata["feature_cols"]
    mv = metadata["current_model_version"]
    pool = prepare_features(pd.read_csv(TRAINING_POOL_PATH))
    batch_df = prepare_features(load_csv(batch_file))
    validate_batch_columns(batch_df, feature_cols)
    scored = add_fraud_predictions(batch_df, model, feature_cols, threshold)
    current = evaluate_model(model, batch_df[feature_cols], batch_df["Class"], threshold)
    drift_df = detect_feature_drift(pool[feature_cols], batch_df[feature_cols], feature_cols, ALPHA)
    imp_df = get_feature_importance(model, feature_cols)
    alerts, retrain = degradation_check(current, baseline, drift_df, imp_df)
    metadata["processed_batches"].append(batch_file); save_json(META_PATH, metadata)
    append_monitor_log(build_monitor_log_row(batch_file, mv, current, baseline, drift_df, alerts, retrain, None))
    return current, drift_df, alerts, retrain, None, None, scored

def process_uploaded_batch(uploaded_file_obj, threshold):
    metadata = load_json(META_PATH); baseline = load_json(BASELINE_METRICS_PATH)
    model = joblib.load(MODEL_PATH); feature_cols = metadata["feature_cols"]
    mv = metadata["current_model_version"]
    pool = prepare_features(pd.read_csv(TRAINING_POOL_PATH))
    batch_df = prepare_features(pd.read_csv(uploaded_file_obj))
    batch_name = uploaded_file_obj.name
    validate_batch_columns(batch_df, feature_cols)
    scored = add_fraud_predictions(batch_df, model, feature_cols, threshold)
    current = evaluate_model(model, batch_df[feature_cols], batch_df["Class"], threshold)
    drift_df = detect_feature_drift(pool[feature_cols], batch_df[feature_cols], feature_cols, ALPHA)
    imp_df = get_feature_importance(model, feature_cols)
    alerts, retrain = degradation_check(current, baseline, drift_df, imp_df)
    metadata["processed_batches"].append(batch_name); save_json(META_PATH, metadata)
    append_monitor_log(build_monitor_log_row(batch_name, mv, current, baseline, drift_df, alerts, retrain, None))
    return batch_name, batch_df, current, drift_df, alerts, retrain, None, None, scored

def latest_unprocessed_batches():
    done = set(load_json(META_PATH).get("processed_batches", []))
    return [f for f in DRIFT_FILES if os.path.exists(f) and f not in done]

def ensure_state_initialized():
    required = [MODEL_PATH, META_PATH, MONITOR_LOG_PATH, MODEL_REGISTRY_PATH,
                TRAINING_POOL_PATH, BASELINE_METRICS_PATH]
    if all(os.path.exists(p) for p in required):
        return
    if not os.path.exists(BASE_DATA_FILE):
        raise FileNotFoundError(f"Missing {BASE_DATA_FILE}")
    base_df = prepare_features(load_csv(BASE_DATA_FILE))
    model, feature_cols = train_fraud_model(base_df)
    baseline = compute_baseline_metrics(base_df)
    joblib.dump(model, MODEL_PATH)
    base_df.to_csv(TRAINING_POOL_PATH, index=False)
    save_json(BASELINE_METRICS_PATH, baseline)
    save_json(META_PATH, {"current_model_version": 1, "feature_cols": feature_cols,
                          "processed_batches": [], "active_model_name": "Model v1"})
    pd.DataFrame([{"model_version":1,"model_name":"Model v1","trained_on":BASE_DATA_FILE,
                   "activated_after_batch":"initial_training","precision":baseline["Precision"],
                   "recall":baseline["Recall"],"f1":baseline["F1"],"roc_auc":baseline["ROC-AUC"],
                   "pr_auc":baseline["PR-AUC"]}]).to_csv(MODEL_REGISTRY_PATH, index=False)
    pd.DataFrame(columns=["batch_name","model_version_used","precision","recall","f1","roc_auc",
                           "pr_auc","delta_precision","delta_recall","delta_f1","delta_roc_auc",
                           "delta_pr_auc","drifted_features","high_drift_features",
                           "top_drifted_features","alerts","retrain_triggered","new_model_version"]
                 ).to_csv(MONITOR_LOG_PATH, index=False)

def reset_state():
    for p in [MODEL_PATH, META_PATH, MONITOR_LOG_PATH, MODEL_REGISTRY_PATH,
              TRAINING_POOL_PATH, BASELINE_METRICS_PATH]:
        if os.path.exists(p):
            os.remove(p)
    ensure_state_initialized()

def display_metric_with_delta(container, label, current_value, baseline_value):
    if pd.isna(current_value) or pd.isna(baseline_value):
        container.metric(label, "N/A")
    else:
        container.metric(label, f"{current_value:.3f}", delta=f"{current_value - baseline_value:.3f}")

def get_feature_importance_cached(model, feature_cols):
    return get_feature_importance(model, feature_cols)

# =========================================================
# BOOTSTRAP
# =========================================================
ensure_state_initialized()
metadata = load_json(META_PATH)
baseline_metrics = load_json(BASELINE_METRICS_PATH)
model_registry = pd.read_csv(MODEL_REGISTRY_PATH)
monitor_log = pd.read_csv(MONITOR_LOG_PATH)
training_pool = prepare_features(pd.read_csv(TRAINING_POOL_PATH))

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#3b82f6;
                letter-spacing:0.16em; text-transform:uppercase; padding:1rem 0 0.5rem;">
        🛡️ Fraud Shield
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "Overview", "Metrics", "Drift Analysis", "Fraud Detection",
        "Model & Retraining", "Model Parameters", "Monitoring Logs",
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.68rem;letter-spacing:0.1em;color:#3b6baf;text-transform:uppercase;'>Batch Controls</p>", unsafe_allow_html=True)

    with st.expander("Data Source", expanded=True):
        uploaded_file = st.file_uploader("Upload production batch", type=["csv"],
                                         label_visibility="collapsed")
        available_batches = latest_unprocessed_batches()
        selected_batch = st.selectbox("Simulated batch",
                                      available_batches if available_batches else ["No batches left"],
                                      label_visibility="collapsed")
        source_mode = st.radio("Source", ["Saved drift batch", "Uploaded CSV"],
                               horizontal=False, label_visibility="collapsed")

    with st.expander("Threshold", expanded=False):
        optimized_threshold = baseline_metrics.get("Best Threshold", DEFAULT_THRESHOLD)
        use_auto = st.checkbox("Use optimized threshold", value=True)
        if use_auto:
            threshold = optimized_threshold
            st.caption(f"Optimized: {threshold:.2f}")
        else:
            threshold = st.slider("Threshold", 0.01, 0.50, optimized_threshold, 0.01)

    with st.expander("Feature Inspector", expanded=False):
        feature_candidates = ["Amount","Time","Hour","V10","V12","V14"]
        feature_candidates = [f for f in feature_candidates if f in training_pool.columns]
        if not feature_candidates:
            feature_candidates = [c for c in training_pool.columns if c != "Class"][:6]
        feature_option = st.selectbox("Feature to inspect", feature_candidates,
                                      label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        disabled = source_mode == "Saved drift batch" and selected_batch == "No batches left"
        process_clicked = st.button("▶ Process", type="primary", disabled=disabled, use_container_width=True)
    with col2:
        reset_clicked = st.button("↺ Reset", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#2a3a52; line-height:1.8;">
        Model · <span style="color:#5a7aaa">{metadata['active_model_name']}</span><br>
        Batches · <span style="color:#5a7aaa">{len(metadata.get('processed_batches',[]))}</span>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar actions ──
if reset_clicked:
    reset_state()
    st.rerun()

if process_clicked:
    try:
        if source_mode == "Saved drift batch":
            if selected_batch == "No batches left":
                st.sidebar.error("No saved drift batches left.")
            else:
                metrics, drift_df, alerts, retrain_needed, nmv, nmvm, scored = process_batch(selected_batch, threshold)
                st.session_state.update({
                    "last_batch": selected_batch,
                    "last_batch_df": prepare_features(load_csv(selected_batch)),
                    "last_scored_batch_df": scored, "last_metrics": metrics,
                    "last_drift_df": drift_df, "last_alerts": alerts,
                    "last_retrain_needed": retrain_needed,
                    "last_new_model_version": nmv,
                    "last_new_model_validation_metrics": nmvm,
                    "last_retrain_source": "Saved drift batch",
                    "last_threshold": threshold,
                })
                st.rerun()
        else:
            if uploaded_file is None:
                st.sidebar.error("Please upload a CSV file first.")
            else:
                bn, bdf, metrics, drift_df, alerts, retrain_needed, nmv, nmvm, scored = process_uploaded_batch(uploaded_file, threshold)
                st.session_state.update({
                    "last_batch": bn, "last_batch_df": bdf,
                    "last_scored_batch_df": scored, "last_metrics": metrics,
                    "last_drift_df": drift_df, "last_alerts": alerts,
                    "last_retrain_needed": retrain_needed,
                    "last_new_model_version": nmv,
                    "last_new_model_validation_metrics": nmvm,
                    "last_retrain_source": "Uploaded CSV",
                    "last_threshold": threshold,
                })
                st.rerun()
    except Exception as e:
        st.sidebar.error(str(e))

# Reload state
metadata = load_json(META_PATH)
model_registry = pd.read_csv(MODEL_REGISTRY_PATH)
monitor_log = pd.read_csv(MONITOR_LOG_PATH)
training_pool = prepare_features(pd.read_csv(TRAINING_POOL_PATH))

# =========================================================
# PAGE: OVERVIEW
# =========================================================
if page == "Overview":
    page_header("Overview", "Live production monitoring · XGBoost fraud classifier")
    status_banner()

    section_label("System Status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Model", metadata["active_model_name"])
    c2.metric("Version", f"v{metadata['current_model_version']}")
    c3.metric("Processed Batches", len(metadata.get("processed_batches", [])))
    c4.metric("Remaining Batches", len(latest_unprocessed_batches()))

    section_label("Baseline Reference")
    st.caption("Holdout test split · Original Kaggle credit card fraud dataset")
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Precision", f"{baseline_metrics['Precision']:.3f}")
    b2.metric("Recall", f"{baseline_metrics['Recall']:.3f}")
    b3.metric("F1", f"{baseline_metrics['F1']:.3f}")
    b4.metric("ROC-AUC", f"{baseline_metrics['ROC-AUC']:.3f}")
    b5.metric("PR-AUC", f"{baseline_metrics['PR-AUC']:.3f}")
    st.caption(f"Optimized threshold · {baseline_metrics.get('Best Threshold', DEFAULT_THRESHOLD):.2f}")

    if "last_metrics" in st.session_state:
        section_label("Latest Batch · vs Baseline")
        lm = st.session_state["last_metrics"]
        m1, m2, m3, m4, m5 = st.columns(5)
        display_metric_with_delta(m1, "Precision", lm["Precision"], baseline_metrics["Precision"])
        display_metric_with_delta(m2, "Recall", lm["Recall"], baseline_metrics["Recall"])
        display_metric_with_delta(m3, "F1", lm["F1"], baseline_metrics["F1"])
        display_metric_with_delta(m4, "ROC-AUC", lm["ROC-AUC"], baseline_metrics["ROC-AUC"])
        display_metric_with_delta(m5, "PR-AUC", lm["PR-AUC"], baseline_metrics["PR-AUC"])
    else:
        st.markdown("""
        <div style="background:#0c1422; border:1px dashed #1e2d47; border-radius:6px;
                    padding:2rem; text-align:center; color:#3b4a62;
                    font-family:'IBM Plex Mono',monospace; font-size:0.82rem; margin-top:1rem;">
            No batch processed yet — select a batch in the sidebar and click ▶ Process
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# PAGE: METRICS
# =========================================================
elif page == "Metrics":
    page_header("Performance Metrics", "Batch evaluation against baseline · threshold trade-off analysis")
    status_banner()

    if "last_metrics" not in st.session_state:
        st.info("Process a batch first to view performance metrics.")
    else:
        lm = st.session_state["last_metrics"]

        section_label("Current Batch vs Baseline")
        m1, m2, m3, m4, m5 = st.columns(5)
        display_metric_with_delta(m1, "Precision", lm["Precision"], baseline_metrics["Precision"])
        display_metric_with_delta(m2, "Recall", lm["Recall"], baseline_metrics["Recall"])
        display_metric_with_delta(m3, "F1", lm["F1"], baseline_metrics["F1"])
        display_metric_with_delta(m4, "ROC-AUC", lm["ROC-AUC"], baseline_metrics["ROC-AUC"])
        display_metric_with_delta(m5, "PR-AUC", lm["PR-AUC"], baseline_metrics["PR-AUC"])

        section_label("Confusion Matrix")
        cm_data = pd.DataFrame(
            lm["Confusion Matrix"],
            index=["Actual Non-Fraud", "Actual Fraud"],
            columns=["Predicted Non-Fraud", "Predicted Fraud"],
        )
        col_cm, col_gap = st.columns([1, 2])
        with col_cm:
            st.dataframe(cm_data, use_container_width=True)

        if "last_drift_df" in st.session_state:
            section_label("Performance Interpretation")
            drift_df = st.session_state["last_drift_df"]
            drifted = int(drift_df["Drift Detected"].sum())
            high = int((drift_df["PSI Level"]=="High").sum())
            rd = lm["Recall"] - baseline_metrics["Recall"]
            fd = lm["F1"] - baseline_metrics["F1"]
            pd_ = lm["PR-AUC"] - baseline_metrics["PR-AUC"]
            rc = "#10b981" if rd >= 0 else "#ef4444"
            fc = "#10b981" if fd >= 0 else "#ef4444"
            pc = "#10b981" if pd_ >= 0 else "#ef4444"
            st.markdown(f"""
            <div style="background:#0c1422; border:1px solid #1a2235; border-radius:6px; padding:1rem 1.2rem;">
                <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.88rem; color:#8a9ab8; line-height:1.7;">
                    Drift detected in <span style="color:#e8edf8; font-weight:500">{drifted} features</span>
                    including <span style="color:#ef4444">{high} high-drift</span> features.<br>
                    Recall <span style="color:{rc}; font-family:'IBM Plex Mono',monospace">{rd:+.3f}</span> ·
                    F1 <span style="color:{fc}; font-family:'IBM Plex Mono',monospace">{fd:+.3f}</span> ·
                    PR-AUC <span style="color:{pc}; font-family:'IBM Plex Mono',monospace">{pd_:+.3f}</span>
                    vs baseline.
                </div>
            </div>
            """, unsafe_allow_html=True)

        if "last_scored_batch_df" in st.session_state:
            section_label("Threshold Trade-Off")
            scored_df = st.session_state["last_scored_batch_df"]
            rows = []
            for t in np.arange(0.01, 0.51, 0.01):
                yp = (scored_df["Fraud Probability"] >= t).astype(int)
                rows.append({
                    "Threshold": round(float(t), 2),
                    "Precision": precision_score(scored_df["Class"], yp, zero_division=0),
                    "Recall": recall_score(scored_df["Class"], yp, zero_division=0),
                    "F1": f1_score(scored_df["Class"], yp, zero_division=0),
                    "Predicted Fraud Count": int(yp.sum()),
                })
            tdf = pd.DataFrame(rows)

            fig, ax = plt.subplots(figsize=(9, 4))
            for i, col in enumerate(["Precision", "Recall", "F1"]):
                ax.plot(tdf["Threshold"], tdf[col], color=CHART_COLORS[i],
                        linewidth=2, label=col)
            ax.axvline(x=threshold, color="#f59e0b", linestyle="--", linewidth=1, alpha=0.7)
            ax.text(threshold + 0.005, 0.05, f"current\n{threshold:.2f}",
                    color="#f59e0b", fontsize=7, va="bottom")
            ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
            ax.legend(); ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig)

            with st.expander("Threshold table", expanded=False):
                st.dataframe(tdf, use_container_width=True)

# =========================================================
# PAGE: DRIFT ANALYSIS
# =========================================================
elif page == "Drift Analysis":
    page_header("Drift Analysis", "KS test · PSI · Drift × Importance risk scoring")
    status_banner()

    if "last_drift_df" not in st.session_state:
        st.info("Process a batch first to view drift analysis.")
    else:
        drift_df = st.session_state["last_drift_df"]

        section_label("Drift Summary")
        d1, d2, d3 = st.columns(3)
        d1.metric("Drifted Features", int(drift_df["Drift Detected"].sum()))
        d2.metric("High Drift", int((drift_df["PSI Level"]=="High").sum()))
        d3.metric("Moderate Drift", int((drift_df["PSI Level"]=="Moderate").sum()))

        section_label("Feature Drift Table")
        st.caption("Red rows = high PSI · Yellow = moderate · Blue = high-risk (drifted + important)")
        model = joblib.load(MODEL_PATH)
        importance_df = get_feature_importance(model, metadata["feature_cols"])
        risk_df = cross_reference_drift_importance(drift_df, importance_df)
        display_cols = ["Feature","PSI","PSI Level","KS Statistic","p-value",
                        "Drift Detected","Importance","High Risk","Risk Score"]
        st.dataframe(style_drift_df(risk_df[display_cols].head(15)), use_container_width=True)

        section_label("Distribution Comparison")
        if "last_batch_df" in st.session_state:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            train_vals = training_pool[feature_option].dropna()
            batch_vals = st.session_state["last_batch_df"][feature_option].dropna()
            ax.hist(train_vals, bins=50, alpha=0.55, density=True,
                    color=CHART_COLORS[0], label="Training Pool")
            ax.hist(batch_vals, bins=50, alpha=0.55, density=True,
                    color=CHART_COLORS[1], label="Incoming Batch")
            ax.set_title(f"{feature_option}"); ax.set_ylabel("Density")
            ax.legend(); ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig)

        section_label("Fraud-Only Drift")
        if "last_batch_df" in st.session_state:
            batch_df = st.session_state["last_batch_df"]
            train_fraud = training_pool[training_pool["Class"]==1]
            batch_fraud = batch_df[batch_df["Class"]==1]
            st.caption(f"Historical fraud rows: {len(train_fraud)} · Latest batch fraud rows: {len(batch_fraud)}")
            if len(batch_fraud) < 5:
                st.warning("Very few fraud cases in latest batch — fraud-only drift results may be unstable.")
            else:
                feature_cols = metadata["feature_cols"]
                fdrift = detect_feature_drift(train_fraud[feature_cols], batch_fraud[feature_cols],
                                              feature_cols, ALPHA)
                st.dataframe(style_drift_df(fdrift.head(15)), use_container_width=True)

        section_label("Data Drift vs Concept Drift")
        st.markdown("""
        <div style="background:#0c1422; border:1px solid #1a2235; border-radius:6px;
                    padding:1rem 1.2rem; font-family:'IBM Plex Sans',sans-serif;
                    font-size:0.85rem; color:#7a8aaa; line-height:1.8;">
            <span style="color:#3b82f6; font-weight:500;">Data drift</span> —
            input feature distributions changed. Detected via KS test and PSI.<br>
            <span style="color:#06b6d4; font-weight:500;">Concept drift</span> —
            the relationship between features and fraud changed. A feature that once
            separated fraud from non-fraud may no longer do so, even if its distribution is stable.
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# PAGE: FRAUD DETECTION
# =========================================================
elif page == "Fraud Detection":
    page_header("Fraud Detection", "Scored transactions · true positives · missed fraud")
    status_banner()

    if "last_scored_batch_df" not in st.session_state:
        st.info("Process a batch first to view fraud predictions.")
    else:
        scored = st.session_state["last_scored_batch_df"].copy()
        detected = scored[scored["Fraud Detected"]==1].sort_values("Fraud Probability", ascending=False)

        section_label("Batch Summary")
        f1c, f2c, f3c, f4c = st.columns(4)
        f1c.metric("Flagged as Fraud", len(detected))
        f2c.metric("Actual Fraud", int(scored["Class"].sum()))
        f3c.metric("True Positives", int(((scored["Fraud Detected"]==1)&(scored["Class"]==1)).sum()))
        f4c.metric("Threshold", f"{st.session_state.get('last_threshold', threshold):.2f}")

        result_counts = scored["Detection Result"].value_counts()
        cols_rc = st.columns(4)
        for i, label in enumerate(["True Positive","False Positive","Missed Fraud","True Negative"]):
            count = result_counts.get(label, 0)
            colour = {"True Positive":"#10b981","False Positive":"#f59e0b",
                      "Missed Fraud":"#ef4444","True Negative":"#3b82f6"}[label]
            cols_rc[i].markdown(f"""
            <div style="background:#0c1422; border:1px solid #1a2235; border-left:3px solid {colour};
                        border-radius:4px; padding:12px 16px; text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:1.4rem;
                            font-weight:600; color:{colour};">{count}</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
                            color:#5a6a8a; letter-spacing:0.08em; margin-top:4px;">{label.upper()}</div>
            </div>
            """, unsafe_allow_html=True)

        display_cols = [c for c in ["Time","Hour","Amount","Fraud Probability",
                                     "Fraud Detected","Class","Detection Result"] if c in scored.columns]

        section_label("Flagged Transactions")
        if detected.empty:
            st.info("No transactions flagged at the current threshold.")
        else:
            with st.expander(f"Detected fraud — {len(detected)} transactions", expanded=True):
                st.dataframe(detected[display_cols], use_container_width=True)

        with st.expander("Full scored batch", expanded=False):
            st.dataframe(scored.sort_values("Fraud Probability", ascending=False),
                         use_container_width=True)

# =========================================================
# PAGE: MODEL & RETRAINING
# =========================================================
elif page == "Model & Retraining":
    page_header("Model & Retraining", "Degradation alerts · manual retraining · model registry")
    status_banner()

    section_label("Retraining Status")
    if "last_retrain_needed" not in st.session_state:
        st.info("Process a batch first to determine whether retraining is recommended.")
    elif st.session_state["last_retrain_needed"]:
        st.markdown("""
        <div style="background:#1a1408; border:1px solid #3a2c08; border-left:3px solid #f59e0b;
                    border-radius:4px; padding:12px 16px; font-family:'IBM Plex Sans',sans-serif;
                    font-size:0.88rem; color:#c8a060; margin-bottom:1rem;">
            ▲ Retraining recommended — degradation or drift thresholds exceeded.
        </div>
        """, unsafe_allow_html=True)

        if st.button("Retrain Model Now", type="primary"):
            try:
                meta_r = load_json(META_PATH)
                pool_r = prepare_features(pd.read_csv(TRAINING_POOL_PATH))
                new_v, new_vm = retrain_and_promote_model(
                    metadata=meta_r, training_pool=pool_r,
                    batch_df=st.session_state["last_batch_df"],
                    batch_name=st.session_state["last_batch"],
                    threshold=st.session_state.get("last_threshold", threshold),
                    drift_df=st.session_state.get("last_drift_df"),
                )
                st.session_state["last_new_model_version"] = new_v
                st.session_state["last_new_model_validation_metrics"] = new_vm
                st.session_state["last_retrain_needed"] = False
                st.success(f"Retraining complete — active model updated to v{new_v}.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    else:
        st.markdown("""
        <div style="background:#071a12; border:1px solid #0d3324; border-left:3px solid #10b981;
                    border-radius:4px; padding:12px 16px; font-family:'IBM Plex Sans',sans-serif;
                    font-size:0.88rem; color:#60a87a;">
            ● No retraining currently recommended.
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.get("last_new_model_validation_metrics"):
        section_label("New Model Validation Metrics")
        nm = st.session_state["last_new_model_validation_metrics"]
        nv1, nv2, nv3, nv4, nv5 = st.columns(5)
        nv1.metric("Precision", f"{nm['Precision']:.3f}")
        nv2.metric("Recall", f"{nm['Recall']:.3f}")
        nv3.metric("F1", f"{nm['F1']:.3f}")
        nv4.metric("ROC-AUC", f"{nm['ROC-AUC']:.3f}")
        nv5.metric("PR-AUC", f"{nm['PR-AUC']:.3f}")

    section_label("Model Registry")
    st.dataframe(model_registry, use_container_width=True)

# =========================================================
# PAGE: MODEL PARAMETERS
# =========================================================
elif page == "Model Parameters":
    page_header("Model Parameters", "XGBoost config · feature importance · alert thresholds")

    section_label("XGBoost Configuration")
    params = {
        "Model Type": "XGBoost Classifier",
        "n_estimators": 400,
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": "Dynamic (class imbalance ratio)",
        "Prediction Threshold": threshold,
        "Threshold Optimization": THRESHOLD_OPTIMIZATION_METRIC,
        "Min Precision for Threshold": MIN_PRECISION_FOR_THRESHOLD,
        "KS Test Alpha": ALPHA,
        "Max Historical Retrain Rows": MAX_HISTORICAL_RETRAIN_ROWS,
        "Recent Batch Weight": RECENT_BATCH_WEIGHT,
    }
    st.dataframe(pd.DataFrame(params.items(), columns=["Parameter","Value"]),
                 use_container_width=True, hide_index=True)

    section_label("Feature Importance")
    model = joblib.load(MODEL_PATH)
    importance_df = get_feature_importance(model, metadata["feature_cols"])

    if importance_df.empty:
        st.info("Feature importance not available.")
    else:
        top_imp = importance_df.head(15).sort_values("Importance")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.barh(top_imp["Feature"], top_imp["Importance"], color=CHART_COLORS[0],
                       alpha=0.85, height=0.65)
        for bar in bars:
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{bar.get_width():.4f}", va="center", ha="left",
                    fontsize=7, color="#5a6a8a")
        ax.set_xlabel("Importance"); ax.grid(True, axis="x")
        fig.tight_layout()
        st.pyplot(fig)

        with st.expander("Full importance table", expanded=False):
            st.dataframe(importance_df, use_container_width=True, hide_index=True)

    section_label("Alert Rules")
    st.code("""
# Performance degradation
if Recall   < baseline × 0.80 → Critical
if F1       < baseline × 0.80 → Critical
if PR-AUC   < baseline × 0.90 → Warning

# Drift thresholds
if high_drift_features   >= 3 → Critical
if moderate_drift_features >= 5 → Warning

# Importance-aware drift
if high_importance_drifting >= 2 → Critical
if high_importance_drifting == 1 → Warning
""", language="python")

# =========================================================
# PAGE: MONITORING LOGS
# =========================================================
elif page == "Monitoring Logs":
    page_header("Monitoring Logs", "Batch history · metric trends · model registry")

    if not monitor_log.empty:
        section_label("Metric Trends")
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, col in enumerate(["precision","recall","f1","pr_auc"]):
            ax.plot(monitor_log["batch_name"], monitor_log[col],
                    marker="o", color=CHART_COLORS[i], linewidth=2,
                    markersize=5, label=col.upper().replace("_","-"))
        ax.set_ylabel("Score"); ax.legend(); ax.grid(True)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        st.pyplot(fig)

        section_label("Batch Log")
        st.dataframe(monitor_log, use_container_width=True)
    else:
        st.markdown("""
        <div style="background:#0c1422; border:1px dashed #1e2d47; border-radius:6px;
                    padding:2rem; text-align:center; color:#3b4a62;
                    font-family:'IBM Plex Mono',monospace; font-size:0.82rem;">
            No batches processed yet — timeline will appear here.
        </div>
        """, unsafe_allow_html=True)

    section_label("Model Registry")
    st.dataframe(model_registry, use_container_width=True)

    section_label("State Storage")
    st.markdown("""
    <div style="background:#0c1422; border:1px solid #1a2235; border-radius:6px;
                padding:1rem 1.2rem; font-family:'IBM Plex Mono',monospace;
                font-size:0.78rem; color:#5a6a8a; line-height:2;">
        monitoring_state/<br>
        ├── <span style="color:#3b82f6">current_model.pkl</span> · active fraud detection model<br>
        ├── <span style="color:#3b82f6">metadata.json</span> · model version + processed batches<br>
        ├── <span style="color:#3b82f6">monitoring_log.csv</span> · full batch history<br>
        ├── <span style="color:#3b82f6">model_registry.csv</span> · promoted model versions<br>
        ├── <span style="color:#3b82f6">training_pool.csv</span> · accumulated training data<br>
        └── <span style="color:#3b82f6">baseline_metrics.json</span> · benchmark reference
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Fraud Shield · State stored in monitoring_state/ · Reset simulation from sidebar")
