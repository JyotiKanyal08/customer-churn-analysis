import pandas as pd

df = pd.read_csv("../data/cleaned_churn.csv")

def tenure_bucket(t):
    if t <= 12: return "0-1yr"
    elif t <= 24: return "1-2yr"
    elif t <= 48: return "2-4yr"
    else: return "4yr+"

df["tenure_group"] = df["tenure"].apply(tenure_bucket)

service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", 
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

def count_services(row):
    return sum(1 for col in service_cols if row[col] == "Yes")

df["num_services"] = df.apply(count_services, axis=1)

# Average charge per month of tenure vs actual monthly charge 
df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

df.to_csv("../data/featured_churn.csv", index=False)
print(df[["tenure_group", "num_services", "avg_monthly_spend"]].head())