import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
df = pd.read_csv("../data/cleaned_churn.csv")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

df["Churn"].value_counts().plot(kind="bar", ax=axes[0,0])
axes[0,0].set_title("Churn Distribution")

sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0,1])
axes[0,1].set_title("Churn by Contract Type")

sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30, ax=axes[0,2])
axes[0,2].set_title("Tenure Distribution by Churn")

sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1,0])
axes[1,0].set_title("Monthly Charges by Churn")

sns.countplot(data=df, x="InternetService", hue="Churn", ax=axes[1,1])
axes[1,1].set_title("Churn by Internet Service")

sns.countplot(data=df, x="PaymentMethod", hue="Churn", ax=axes[1,2])
axes[1,2].tick_params(axis="x", rotation=30)
axes[1,2].set_title("Churn by Payment Method")

plt.tight_layout()
plt.savefig("../outputs/eda_overview.png", dpi=120)

print(df.groupby("Contract")["Churn"].mean().sort_values(ascending=False))
print(df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False))
print(df.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False))
print(df.groupby("Churn")["tenure"].mean())