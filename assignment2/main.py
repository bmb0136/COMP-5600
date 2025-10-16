from math import exp
import numpy as np
import pandas as pd

"""
# Problem 1
"""
def problem1():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    pd.set_option('future.no_silent_downcasting', True)

    # Replace non-numerical features
    for col in ["Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn"]:
        df[col] = df[col].replace({"No": 0., "Yes": 1.})
    df["gender"] = df["gender"].replace({"Male": 0., "Female": 1.})
    df["InternetService"] = df["InternetService"].replace({"No": 0., "DSL": 1., "Fiber optic": 2.})
    df["Contract"] = df["Contract"].replace({"Month-to-month": 1., "One year": 12., "Two year": 24.})
    df["PaymentMethod"] = df["PaymentMethod"].replace({"Electronic check": 1., "Mailed check": 2., "Bank transfer (automatic)": 3., "Credit card (automatic)": 4.})

    # Fix blanks
    df["TotalCharges"] = df["TotalCharges"].replace({" ": 0.})
    
    # Fix "No internet service" and "No phone service"
    # Replace with zeros since:
    # - You cant have "OnlineSecurity" through "StreamingMovies" if you dont have internet
    # - If you have zero phone lines that's not multiple lines
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"]:
        df[col] = df[col].replace({"No internet service": 0., "No phone service": 0.})
    
    # Normalize unbounded columns
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        df[col] = pd.to_numeric(df[col])
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Ignored columns: Churn, customerId
    NUM_FEATURES = df.shape[1] - 2
    NUM_SAMPLES = df.shape[0]

    XS = np.array(df[["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"]], dtype=np.float64)
    YS = np.array(df[["Churn"]], dtype=np.float64)

    def logistic_regression():
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        theta = np.zeros(NUM_FEATURES)
        alpha = 0.0001

        for i in range(1000):
            predicted = sigmoid(np.dot(XS, theta)) 

            error = -np.mean(YS * np.log(predicted + 1e-15) + (1 - YS) * np.log(1 - predicted + 1e-15))

            gradient = np.mean(np.dot(XS.T, predicted - YS), axis=1) / NUM_SAMPLES

            theta -= alpha * gradient
            print(f"Iter {i} | error = {error} | theta = {", ".join(f"{x:.3f}" for x in theta)}")


    logistic_regression()

def problem2():
    pass

def main():
    problem1()
    problem2()
