import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

st.set_page_config(page_title="Customer Churn Risk Dashboard", layout="wide")
st.title("📉 Customer Churn Risk Dashboard")

xgb = joblib.load("../outputs/xgb_model.pkl")
log_reg = joblib.load("../outputs/logreg_model.pkl")
scaler = joblib.load("../outputs/scaler.pkl")

X_test = pd.read_csv("../data/X_test.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()

xgb_probs = xgb.predict_proba(X_test)[:, 1]
logreg_probs = log_reg.predict_proba(scaler.transform(X_test))[:, 1]

results = X_test.copy()
results["churn_probability"] = xgb_probs
results["actual_churn"] = y_test.values

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview & At-Risk Customers", "⚖️ Model Comparison", "💰 Business Impact", "🔍 Explain a Prediction"]
)
# TAB 1: Overview 
with tab1:
    st.sidebar.header("Filters")
    min_prob = st.sidebar.slider("Minimum churn probability", 0.0, 1.0, 0.5, 0.05)

    filtered = results[results["churn_probability"] >= min_prob].sort_values(
        "churn_probability", ascending=False
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total customers (test set)", len(results))
    col2.metric("Flagged as at-risk", len(filtered))
    col3.metric("Average churn probability", f"{results['churn_probability'].mean():.1%}")

    st.subheader("At-Risk Customers")
    st.dataframe(
        filtered[["churn_probability", "actual_churn"]].style.format({"churn_probability": "{:.1%}"})
    )

# TAB 2: Model Comparison 
with tab2:
    st.subheader("Logistic Regression vs XGBoost")

    logreg_pred = (logreg_probs >= 0.5).astype(int)
    xgb_pred = (xgb_probs >= 0.5).astype(int)

    logreg_report = classification_report(y_test, logreg_pred, output_dict=True)
    xgb_report = classification_report(y_test, xgb_pred, output_dict=True)

    comparison = pd.DataFrame({
        "Logistic Regression": {
            "Accuracy": logreg_report["accuracy"],
            "Precision (Churn)": logreg_report["1"]["precision"],
            "Recall (Churn)": logreg_report["1"]["recall"],
            "F1 (Churn)": logreg_report["1"]["f1-score"],
            "ROC-AUC": roc_auc_score(y_test, logreg_probs),
        },
        "XGBoost": {
            "Accuracy": xgb_report["accuracy"],
            "Precision (Churn)": xgb_report["1"]["precision"],
            "Recall (Churn)": xgb_report["1"]["recall"],
            "F1 (Churn)": xgb_report["1"]["f1-score"],
            "ROC-AUC": roc_auc_score(y_test, xgb_probs),
        },
    })
    st.dataframe(comparison.style.format("{:.3f}"))

    st.subheader("ROC Curve Comparison")
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probs in [("Logistic Regression", logreg_probs), ("XGBoost", xgb_probs)]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    st.info(
        "Both models perform similarly, suggesting churn in this dataset is driven largely by "
        "strong, near-linear signals (contract type, tenure, payment method) rather than complex "
        "non-linear interactions — so the added complexity of XGBoost doesn't yield a large gain here."
    )

# TAB 3: Business Impact 
with tab3:
    st.subheader("What if we target only the highest-risk customers?")

    top_pct = st.slider("Target top X% highest-risk customers", 5, 100, 20, 5)
    n_targeted = int(len(results) * top_pct / 100)

    ranked = results.sort_values("churn_probability", ascending=False)
    targeted = ranked.head(n_targeted)

    total_actual_churners = results["actual_churn"].sum()
    caught_churners = targeted["actual_churn"].sum()
    capture_rate = caught_churners / total_actual_churners if total_actual_churners > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers targeted", n_targeted)
    col2.metric("Actual churners caught", int(caught_churners))
    col3.metric("% of all churners captured", f"{capture_rate:.1%}")

    st.write(
        f"By proactively reaching out to just the top **{top_pct}%** of customers ranked by predicted "
        f"churn risk, a retention team could reach **{capture_rate:.1%}** of all customers who were "
        f"actually going to churn — instead of contacting the entire customer base."
    )

# TAB 4: Explain a Prediction 
with tab4:
    st.subheader("Explain a Prediction")
    row_idx = st.number_input(
        "Select a customer row index (from test set)", 0, len(X_test) - 1, 0
    )

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    st.write(f"Predicted churn probability: **{xgb_probs[row_idx]:.1%}**")
    st.write(f"Actual outcome: **{'Churned' if y_test.iloc[row_idx] == 1 else 'Stayed'}**")

    explanation = shap.Explanation(
        values=shap_values[row_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[row_idx],
        feature_names=X_test.columns.tolist(),
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    st.pyplot(fig)
    plt.clf()