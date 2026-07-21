import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Risk Dashboard", layout="wide")
st.title("📉 Customer Churn Risk Dashboard")

# Load model and data 
model = joblib.load("../outputs/xgb_model.pkl")
X_test = pd.read_csv("../data/X_test.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()

# Predict churn probability for every customer in test set 
probs = model.predict_proba(X_test)[:, 1]
results = X_test.copy()
results["churn_probability"] = probs
results["actual_churn"] = y_test.values

# Sidebar filters 
st.sidebar.header("Filters")
min_prob = st.sidebar.slider("Minimum churn probability", 0.0, 1.0, 0.5, 0.05)

filtered = results[results["churn_probability"] >= min_prob].sort_values(
    "churn_probability", ascending=False
)

# Top metrics 
col1, col2, col3 = st.columns(3)
col1.metric("Total customers (test set)", len(results))
col2.metric("Flagged as at-risk", len(filtered))
col3.metric("Average churn probability", f"{results['churn_probability'].mean():.1%}")

# At-risk customer table 
st.subheader("At-Risk Customers")
st.dataframe(
    filtered[["churn_probability", "actual_churn"]].style.format({"churn_probability": "{:.1%}"})
)

# Explain a single prediction 
st.subheader("Explain a Prediction")
row_idx = st.number_input(
    "Select a customer row index (from test set)", 0, len(X_test) - 1, 0
)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

st.write(f"Predicted churn probability: **{probs[row_idx]:.1%}**")
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