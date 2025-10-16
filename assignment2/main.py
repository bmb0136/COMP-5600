from collections import Counter
from math import exp
import random
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import cv2
import sys

"""
# Problem 1
"""
def problem1():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    pd.set_option("future.no_silent_downcasting", True)

    # Replace non-numerical features
    for col in ["Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn"]:
        df[col] = df[col].replace({"No": 0., "Yes": 1.})
    df["gender"] = df["gender"].replace({"Male": 0., "Female": 1.})
    df["InternetService"] = df["InternetService"].replace({"No": 0., "DSL": 1., "Fiber optic": 2.})
    df["Contract"] = df["Contract"].replace({"Month-to-month": 1., "One year": 12., "Two year": 24.})
    df["PaymentMethod"] = df["PaymentMethod"].replace({"Electronic check": 1., "Mailed check": 2., "Bank transfer (automatic)": 3., "Credit card (automatic)": 4.})

    # Fix blanks
    df["TotalCharges"] = df["TotalCharges"].replace({" ": 0.})

    # Ensure there are an even number of Yes/No churn values (prevent bias)
    yeses = df[df["Churn"] == 1]
    nos = df[df["Churn"] == 0]
    count = min(len(yeses), len(nos))

    """
    FOR SOME REASON, trying to use the whole dataset causes np.dot or the @ operator to hang
    I HAVE NO IDEA why this happens
    I spend multiple HOURS fixing this
    If you would like to try more, then increase this number or comment out the line below.
    No guarantee that it works
    """
    count = 250

    df = pd.concat([yeses.sample(n=count, replace=False, random_state=69), nos.sample(n=count, replace=False, random_state=67)])
    
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

    def eval_logistic_regression(inputs, theta):
        z = np.dot(inputs, theta)
        z = np.clip(z, -100, 100)
        return 1 / (1 + np.exp(-z))

    def train_logistic_regression(inputs, labels, iters=5000):
        theta = np.zeros(NUM_FEATURES)
        alpha = 0.001

        print(f"Training logistic regression with {iters} iters and {len(inputs)} samples")
        for i in range(iters):
            predicted = eval_logistic_regression(inputs, theta)
            gradient = np.mean(inputs.T @ (predicted - labels), axis=1) / NUM_SAMPLES
            theta -= alpha * gradient
            if i % (iters // 50) == 0:
                print(".", end="", flush=True)
        print()

        return {"theta": theta}
    
    def cross_validate(X, Y, folds, train, evaluate):
        metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "roc_auc": []
        }

        folder = StratifiedKFold(n_splits=folds, shuffle=True, random_state=420)
        for i_train, i_test in folder.split(X, Y):

            X_train, X_test = X[i_train], X[i_test]
            Y_train, Y_test = Y[i_train], Y[i_test]

            params = train(X_train, Y_train)
            raw_predictions = evaluate(X_test, **params)

            rounded_predictions = np.round(raw_predictions).astype(int)

            metrics["accuracy"].append(accuracy_score(Y_test, rounded_predictions))
            metrics["precision"].append(precision_score(Y_test, rounded_predictions))
            metrics["recall"].append(recall_score(Y_test, rounded_predictions))
            metrics["f1"].append(f1_score(Y_test, rounded_predictions))
            metrics["roc_auc"].append(roc_auc_score(Y_test, raw_predictions))
        return metrics

    def eval_bayes(X, classes, class_priors, conditional_probs):
        predictions = []
        for x in X:
            class_scores = {}
            for c in classes:
                prob = class_priors[c]
                for j, val in enumerate(x):
                    prob *= conditional_probs[c][j].get(val, 1e-6)
                class_scores[c] = prob
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)

    def train_bayes(X, Y):
        classes = np.unique(Y)
        class_priors = {c: np.mean(Y == c) for c in classes}
        conditional_probs = {c: [{} for _ in range(NUM_FEATURES)] for c in classes}

        for c in classes:
            X_c = X[np.where(Y == c)[0]]
            for j in range(NUM_FEATURES):
                feature_values = X_c[:, j]
                value_counts = Counter(feature_values)
                total = len(feature_values)
                for val, count in value_counts.items():
                    conditional_probs[c][j][val] = count / total

        return {
            "classes": classes,
            "class_priors": class_priors,
            "conditional_probs": conditional_probs
        }

    XS = np.array(df[["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"]], dtype=np.float64)
    YS = np.array(df[["Churn"]], dtype=np.float64)

    m = cross_validate(XS, YS, 5, train_logistic_regression, eval_logistic_regression)
    print("--------------------------")
    print("Logistic Regression stats:")
    for k, v in m.items():
        print(f"{k:<12}:{np.mean(v)}")
    print("--------------------------")

    m = cross_validate(XS, YS, 5, train_bayes, eval_bayes)
    print("--------------------------")
    print("Bayes stats:")
    for k, v in m.items():
        print(f"{k:<12}:{np.mean(v)}")
    print("--------------------------")

def problem2():
    pass

def main():
    problem1()
    problem2()
