import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

X_test = pd.read_csv("../data/X_test.csv")
xgb = joblib.load("../outputs/xgb_model.pkl")

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

# Global feature importance - which features matter most overall
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("../outputs/shap_summary.png", dpi=120)
plt.close()
print("Saved shap_summary.png")

# Single customer's prediction
customer_idx = 0
shap.force_plot(
    explainer.expected_value, shap_values[customer_idx], X_test.iloc[customer_idx],
    matplotlib=True, show=False
)
plt.savefig("../outputs/shap_single_customer.png", dpi=120)
print("Saved shap_single_customer.png")