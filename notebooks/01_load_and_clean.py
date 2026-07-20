import pandas as pd
import numpy as np

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Shape:", df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.head())

blank_rows = df[df["TotalCharges"] == " "]
print("\nRows with blank TotalCharges:")
print(blank_rows[["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["TotalCharges"] = df["TotalCharges"].fillna(0)

df = df.drop(columns=["customerID"])

# Encode target as 0/1
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("\nChurn rate:", df["Churn"].mean())
print(df["Churn"].value_counts())

df.to_csv("../data/cleaned_churn.csv", index=False)
print("\nSaved cleaned_churn.csv")