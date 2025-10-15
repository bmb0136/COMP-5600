from math import exp
import numpy as np
import pandas as pd

"""
# Problem 1
"""
def problem1():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Replace non-numerical features
    for col in ["Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn"]:
        df[col] = df[col].replace({"No": 0, "Yes": 1})
    df["gender"] = df["gender"].replace({"Male": 0, "Female": 1})
    df["InternetService"] = df["InternetService"].replace({"No": 0, "DSL": 1, "Fiber optic": 2})
    df["Contract"] = df["Contract"].replace({"Month-to-Month": 1, "One year": 12, "Two year": 24})
    df["PaymentMethod"] = df["PaymentMethod"].replace({"Electronic check": 1, "Mailed check": 2, "Bank transfer (automatic)": 3, "Credit card (automatic)": 4})

    # Fix blanks
    df["TotalCharges"] = df["TotalCharges"].replace({" ": 0})
    
    # Normalize unbounded columns
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        df[col] = pd.to_numeric(df[col])
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    print(df)

    # Ignored columns: Churn, customerId
    NUM_FEATURES = df.shape[1] - 2

    def logistic_regression():
        def h(theta, x):
            t = theta[0]
            for i in range(1, len(theta)):
                t += theta[i] * x
            return 1 / (1 + np.exp(-t))
        def h_prime(theta, x):
            temp = h(theta, x)
            return temp * (1 - temp)
    pass

def problem2():
    pass

def main():
    problem1()
    problem2()
